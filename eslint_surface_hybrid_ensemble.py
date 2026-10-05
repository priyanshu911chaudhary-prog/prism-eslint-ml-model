"""
ESLint Surface/Suppress Hybrid ML Pipeline
------------------------------------------
Architecture:
    ESLint finding
        -> feature engineering
        -> TF-IDF(message) + OneHot(categorical) + scaled numeric features
        -> Logistic Regression
        -> Random Forest
        -> XGBoost
        -> probability fusion
        -> validation-selected threshold
        -> SURFACE / SUPPRESSED

Important:
- surface_label is used ONLY to create the target y.
- surface_target, repo_id, finding_id, split, label_source are never input features.
- repo_id is deliberately excluded to reduce repository-specific leakage.
- Nominal categorical fields use OneHotEncoder, not LabelEncoder.
- LabelEncoder is used only for the target labels.
- Empty/None values are normalized to NaN and handled inside the fitted preprocessor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier


RANDOM_STATE = 42

CATEGORICAL_COLS = [
    "rule_id",
    "rule_family",
    "severity_text",
    "language",
    "path_extension",
]

DROP_COLS = [
    "finding_id",
    "repo_id",
    "split",
    "tool",
    "surface_label",
    "surface_target",
    "label_source",
]

# Numeric fields are derived after feature engineering.
BASE_NUMERIC_COLS = [
    "severity",
    "is_error",
    "is_warning",
    "message_length",
    "message_word_count",
    "start_line",
    "start_column",
    "end_line",
    "end_column",
    "finding_span_lines",
    "finding_span_columns",
    "has_fix",
    "fix_text_length",
    "fix_range_length",
    "has_suggestions",
    "suggestion_count",
    "changed_line_start",
    "changed_line_end",
    "changed_line_count",
    "finding_overlaps_change",
    "finding_change_distance",
    "file_size_lines",
    "finding_start_line_ratio",
    "pr_change_code_lines",
    "base_file_finding_count",
    "base_rule_finding_count",
    "base_file_has_findings",
    "base_rule_exists_in_file",
    "prior_findings_in_file",
    "pr_total_findings_in_file",
    "same_rule_findings_in_file",
    "same_rule_findings_in_repo",
    "path_depth",
    "path_has_test",
    "path_has_src",
    "path_has_config",
    "path_has_generated",
]

ENGINEERED_NUMERIC_COLS = [
    "change_density",
    "distance_ratio",
    "distance_per_changed_line",
    "file_finding_density",
    "rule_finding_density",
    "finding_to_change_ratio",
    "near_change",
    "severity_overlap_interaction",
    "fix_overlap_interaction",
    "suggestion_overlap_interaction",
    "change_density_overlap",
    "rule_concentration",
]

NUMERIC_COLS = BASE_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain features from raw ESLint finding metadata.

    Missing-value policy:
    - Empty/whitespace-only strings are converted to NaN.
    - Numeric feature engineering preserves NaN so the downstream numeric
      imputer can handle it using statistics learned from the training set.
    - Text is kept as NaN until the TF-IDF pipeline converts it to ``""``.
    """
    x = df.copy()

    # Treat blank strings and common null-like tokens as missing.
    missing_tokens = {"", "null", "none", "nan", "na", "n/a"}
    object_cols = x.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        # Use ordinary Python objects + np.nan instead of pandas ``pd.NA``.
        # Some sklearn imputers/estimators expect NumPy-style missing values
        # and can reject pd.NA during transform().
        values = x[col].astype(object)
        values = values.map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        missing_mask = values.map(
            lambda value: isinstance(value, str)
            and value.lower() in missing_tokens
        )
        x[col] = values.mask(missing_mask, np.nan)

    # Ensure numeric columns are numeric; invalid/blank values become NaN.
    for col in BASE_NUMERIC_COLS:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")

    x["change_density"] = x["changed_line_count"] / (x["file_size_lines"] + 1)
    x["distance_ratio"] = x["finding_change_distance"] / (x["file_size_lines"] + 1)
    x["distance_per_changed_line"] = (
        x["finding_change_distance"] / (x["changed_line_count"] + 1)
    )
    x["file_finding_density"] = (
        x["base_file_finding_count"] / (x["file_size_lines"] + 1)
    )
    x["rule_finding_density"] = (
        x["same_rule_findings_in_file"] / (x["file_size_lines"] + 1)
    )
    x["finding_to_change_ratio"] = (
        x["finding_span_lines"] / (x["changed_line_count"] + 1)
    )
    x["near_change"] = np.where(
        x["finding_change_distance"].isna(),
        np.nan,
        (x["finding_change_distance"] <= 5).astype(float),
    )

    x["severity_overlap_interaction"] = (
        x["severity"] * x["finding_overlaps_change"]
    )
    x["fix_overlap_interaction"] = x["has_fix"] * x["finding_overlaps_change"]
    x["suggestion_overlap_interaction"] = (
        x["has_suggestions"] * x["finding_overlaps_change"]
    )
    x["change_density_overlap"] = (
        x["change_density"] * x["finding_overlaps_change"]
    )
    x["rule_concentration"] = (
        x["same_rule_findings_in_file"] / (x["base_file_finding_count"] + 1)
    )

    return x


def build_preprocessor() -> ColumnTransformer:
    """
    Hybrid feature representation with production-safe missing handling:
    1. TF-IDF word unigrams + bigrams for message context.
       Missing/blank message -> empty string.
    2. One-hot encoding for nominal categories.
       Missing category -> explicit ``__MISSING__`` bucket.
    3. Median-imputed + standardized numeric/engineered features.
       Missingness indicators are added so the model can distinguish
       "unknown" from an observed value.

    All imputation statistics are learned only from the training data because
    the entire preprocessor is fitted on the training split.
    """
    text_pipeline = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
        # ColumnTransformer gives the text selector as a 2-D array after the
        # imputer; TF-IDF expects an iterable of raw strings, so flatten it.
        ("flatten", FunctionTransformer(np.ravel, validate=False)),
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.98,
                sublinear_tf=True,
                max_features=3000,
            ),
        ),
    ]

    from sklearn.pipeline import Pipeline

    return ColumnTransformer(
        transformers=[
            (
                "text_tfidf",
                Pipeline(text_pipeline),
                ["message"],
            ),
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant",
                            fill_value="__MISSING__",
                        ),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                        ),
                    ),
                ]),
                CATEGORICAL_COLS,
            ),
            (
                "numeric",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                            add_indicator=True,
                        ),
                    ),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC_COLS,
            ),
        ],
        sparse_threshold=0.3,
    )


def build_models() -> Dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            C=0.1,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.90,
            min_child_weight=2,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> Dict[str, Any]:
    pred = (probability >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_surface": float(precision_score(y_true, pred, zero_division=0)),
        "recall_surface": float(recall_score(y_true, pred, zero_division=0)),
        "f1_surface": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def choose_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, Dict[str, Any]]:
    """
    Select threshold on validation data by maximum F1.
    This is intentionally separate from model fitting.
    """
    best_threshold = 0.50
    best_score = (-1.0, -1.0)

    for threshold in np.linspace(0.20, 0.80, 121):
        m = metrics(y_true, probability, threshold)
        score = (m["f1_surface"], m["precision_surface"])
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, metrics(y_true, probability, best_threshold)


def train(csv_path: str, artifact_out: str = "eslint_surface_hybrid_ensemble.joblib"):
    df = pd.read_csv(csv_path)

    required_raw = set(CATEGORICAL_COLS + BASE_NUMERIC_COLS + [
        "message", "surface_label", "split"
    ])
    missing_columns = required_raw - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Training dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    train_df = add_features(df[df["split"] == "train"].copy())
    val_df = add_features(df[df["split"] == "validation"].copy())
    test_df = add_features(df[df["split"] == "test"].copy())

    # Target encoding only. Do NOT label-encode nominal input features.
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["surface_label"])

    y_train = label_encoder.transform(train_df["surface_label"])
    y_val = label_encoder.transform(val_df["surface_label"])
    y_test = label_encoder.transform(test_df["surface_label"])

    # Safety check: expected semantics for this dataset.
    class_map = {cls: int(i) for i, cls in enumerate(label_encoder.classes_)}
    if class_map.get("suppressed") != 0 or class_map.get("surface") != 1:
        raise ValueError(f"Unexpected target mapping: {class_map}")

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    y_train_val = label_encoder.transform(train_val_df["surface_label"])

    # Validation representation for model/weight/threshold selection.
    preprocessor_val = build_preprocessor()
    X_train = preprocessor_val.fit_transform(train_df)
    X_val = preprocessor_val.transform(val_df)

    validation_models = build_models()
    val_probabilities = {}

    for name, model in validation_models.items():
        model.fit(X_train, y_train)
        val_probabilities[name] = model.predict_proba(X_val)[:, 1]

    # Chosen after validation experiments on this dataset.
    weights = {
        "logistic_regression": 0.15,
        "random_forest": 0.40,
        "xgboost": 0.45,
    }

    ensemble_val = sum(
        weights[name] * val_probabilities[name]
        for name in weights
    )

    threshold, validation_metrics = choose_threshold(y_val, ensemble_val)

    # Refit all three models on train + validation.
    final_preprocessor = build_preprocessor()
    X_train_val = final_preprocessor.fit_transform(train_val_df)
    X_test = final_preprocessor.transform(test_df)

    final_models = build_models()
    test_probabilities = {}

    for name, model in final_models.items():
        model.fit(X_train_val, y_train_val)
        test_probabilities[name] = model.predict_proba(X_test)[:, 1]

    ensemble_test = sum(
        weights[name] * test_probabilities[name]
        for name in weights
    )

    test_metrics = metrics(y_test, ensemble_test, threshold)

    predictions = test_df[
        ["finding_id", "repo_id", "rule_id", "message"]
    ].copy()
    predictions["actual_label"] = test_df["surface_label"].values
    predictions["p_surface_lr"] = test_probabilities["logistic_regression"]
    predictions["p_surface_rf"] = test_probabilities["random_forest"]
    predictions["p_surface_xgb"] = test_probabilities["xgboost"]
    predictions["ensemble_surface_probability"] = ensemble_test
    predictions["decision"] = np.where(
        ensemble_test >= threshold, "surface", "suppressed"
    )

    predictions.to_csv("eslint_surface_hybrid_test_predictions.csv", index=False)

    artifact = {
        "preprocessor": final_preprocessor,
        "label_encoder": label_encoder,
        "models": final_models,
        "weights": weights,
        "threshold": threshold,
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "drop_cols": DROP_COLS,
    }
    joblib.dump(artifact, artifact_out)

    report = {
        "dataset_shape": list(df.shape),
        "class_distribution": df["surface_label"].value_counts().to_dict(),
        "feature_counts": {
            "numeric": len(NUMERIC_COLS),
            "categorical": len(CATEGORICAL_COLS),
            "text_column": "message",
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "ensemble_weights": weights,
        "threshold": threshold,
        "label_mapping": class_map,
        "missing_value_policy": {
            "text": "blank/None -> empty string before TF-IDF",
            "categorical": "blank/None -> __MISSING__ then OneHotEncoder",
            "numeric": "median imputation learned on fitted training data + numeric missing indicators",
        },
    }

    with open("eslint_surface_hybrid_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def predict_one(finding: Dict[str, Any], artifact_path: str):
    """
    Production-style inference for one new ESLint finding.

    Missing values are intentionally allowed. Values such as ``None``, empty
    strings, or whitespace-only strings are handled by the fitted preprocessor.
    """
    artifact = joblib.load(artifact_path)

    # Create an input row and make absent model-input columns explicit as NaN.
    # This makes partial findings safe to score instead of failing at transform().
    row = pd.DataFrame([finding])
    required_cols = ["message"] + CATEGORICAL_COLS + NUMERIC_COLS
    for col in required_cols:
        if col not in row.columns:
            row[col] = np.nan

    row = add_features(row)

    # Final sklearn-safe normalization in case a caller supplied pd.NA or
    # another pandas nullable scalar directly.
    row = row.astype(object).where(pd.notna(row), np.nan)

    X = artifact["preprocessor"].transform(row)
    models = artifact["models"]
    weights = artifact["weights"]
    threshold = artifact["threshold"]

    probabilities = {
        name: float(model.predict_proba(X)[:, 1][0])
        for name, model in models.items()
    }

    ensemble_probability = sum(
        weights[name] * probabilities[name]
        for name in probabilities
    )

    decision = "surface" if ensemble_probability >= threshold else "suppressed"

    return {
        "probabilities": probabilities,
        "ensemble_surface_probability": float(ensemble_probability),
        "threshold": float(threshold),
        "decision": decision,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to eslint_ml_features_surface_v1_cleaned.csv",
    )
    parser.add_argument(
        "--artifact",
        default="eslint_surface_hybrid_ensemble.joblib",
    )
    args = parser.parse_args()

    report = train(args.csv, args.artifact)
    print(json.dumps(report, indent=2))


def predict_one_with_missing_values(artifact_path: str):
    """Small sanity check/demo showing that blank values are accepted."""
    sample = {
        "message": "",
        "rule_id": None,
        "rule_family": "",
        "severity_text": None,
        "language": "javascript",
        "path_extension": ".js",
        "severity": None,
        "is_error": None,
        "is_warning": None,
        "changed_line_count": None,
        "finding_overlaps_change": None,
        "finding_change_distance": None,
        "file_size_lines": None,
    }
    return predict_one(sample, artifact_path)
