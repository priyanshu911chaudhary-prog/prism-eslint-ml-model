from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from eslint_surface_hybrid_ensemble import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    add_features,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = BASE_DIR / "model" / "eslint_surface_hybrid_ensemble.joblib"
ARTIFACT_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_ARTIFACT)))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "100"))


class Finding(BaseModel):
    """
    Input schema for one ESLint finding.

    Every model feature is optional because real ESLint integrations can have
    incomplete metadata. Unknown extra fields are allowed and ignored by the
    ML pipeline, which makes this API easier to integrate with raw ESLint JSON.
    """

    model_config = ConfigDict(extra="allow")

    message: str | None = None

    rule_id: str | None = None
    rule_family: str | None = None
    severity_text: str | None = None
    language: str | None = None
    path_extension: str | None = None

    severity: float | None = None
    is_error: float | None = None
    is_warning: float | None = None
    message_length: float | None = None
    message_word_count: float | None = None
    start_line: float | None = None
    start_column: float | None = None
    end_line: float | None = None
    end_column: float | None = None
    finding_span_lines: float | None = None
    finding_span_columns: float | None = None
    has_fix: float | None = None
    fix_text_length: float | None = None
    fix_range_length: float | None = None
    has_suggestions: float | None = None
    suggestion_count: float | None = None
    changed_line_start: float | None = None
    changed_line_end: float | None = None
    changed_line_count: float | None = None
    finding_overlaps_change: float | None = None
    finding_change_distance: float | None = None
    file_size_lines: float | None = None
    finding_start_line_ratio: float | None = None
    pr_change_code_lines: float | None = None
    base_file_finding_count: float | None = None
    base_rule_finding_count: float | None = None
    base_file_has_findings: float | None = None
    base_rule_exists_in_file: float | None = None
    prior_findings_in_file: float | None = None
    pr_total_findings_in_file: float | None = None
    same_rule_findings_in_file: float | None = None
    same_rule_findings_in_repo: float | None = None
    path_depth: float | None = None
    path_has_test: float | None = None
    path_has_src: float | None = None
    path_has_config: float | None = None
    path_has_generated: float | None = None


class BatchRequest(BaseModel):
    findings: List[Finding] = Field(..., min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    probabilities: Dict[str, float]
    ensemble_surface_probability: float
    threshold: float
    decision: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


class Predictor:
    def __init__(self, artifact_path: Path):
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

        artifact = joblib.load(artifact_path)

        self.preprocessor = artifact["preprocessor"]
        self.models = artifact["models"]
        self.weights = artifact["weights"]
        self.threshold = float(artifact["threshold"])
        self.artifact_path = artifact_path

    @staticmethod
    def _to_frame(finding: Dict[str, Any]) -> pd.DataFrame:
        """Convert one request into the exact feature frame expected by training."""
        row = pd.DataFrame([finding])

        # Make every production-required raw column explicit. Missing columns
        # are allowed and become NaN, then are handled by the fitted pipeline.
        required_cols = ["message"] + CATEGORICAL_COLS + NUMERIC_COLS
        for col in required_cols:
            if col not in row.columns:
                row[col] = np.nan

        row = add_features(row)

        # Convert pandas nullable scalars to ordinary NumPy NaN where needed.
        row = row.astype(object).where(pd.notna(row), np.nan)
        return row

    def predict_one(self, finding: Dict[str, Any]) -> PredictionResponse:
        row = self._to_frame(finding)
        X = self.preprocessor.transform(row)

        probabilities = {
            name: float(model.predict_proba(X)[:, 1][0])
            for name, model in self.models.items()
        }

        ensemble_probability = sum(
            self.weights[name] * probabilities[name]
            for name in probabilities
        )

        decision = (
            "surface"
            if ensemble_probability >= self.threshold
            else "suppressed"
        )

        return PredictionResponse(
            probabilities=probabilities,
            ensemble_surface_probability=float(ensemble_probability),
            threshold=self.threshold,
            decision=decision,
        )


predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = Predictor(ARTIFACT_PATH)
    yield
    predictor = None


app = FastAPI(
    title="PRism ESLint Surface Classifier",
    description=(
        "Predict whether an ESLint finding should be surfaced or suppressed "
        "using TF-IDF + one-hot + engineered numeric features with a "
        "Logistic Regression, Random Forest, and XGBoost ensemble."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False if cors_origins == ["*"] else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["meta"])
def root() -> Dict[str, str]:
    return {
        "service": "prism-eslint-surface-classifier",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health", tags=["meta"])
def health() -> Dict[str, Any]:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    return {
        "status": "healthy",
        "model_loaded": True,
        "threshold": predictor.threshold,
        "models": list(predictor.models.keys()),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(finding: Finding) -> PredictionResponse:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        return predictor.predict_one(finding.model_dump())
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Unable to score finding: {exc}",
        ) from exc


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["prediction"],
)
def predict_batch(request: BatchRequest) -> BatchPredictionResponse:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    if len(request.findings) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Maximum batch size is {MAX_BATCH_SIZE}",
        )

    try:
        predictions = [
            predictor.predict_one(finding.model_dump())
            for finding in request.findings
        ]
        return BatchPredictionResponse(predictions=predictions)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Unable to score batch: {exc}",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
