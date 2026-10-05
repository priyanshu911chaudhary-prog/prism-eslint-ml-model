# PRism ESLint Surface Classifier API

FastAPI wrapper around the trained ESLint finding classifier.

## Model pipeline

`ESLint finding -> feature engineering -> TF-IDF(message) + One-Hot(categorical) + imputed numeric features -> Logistic Regression + Random Forest + XGBoost -> weighted probability fusion -> threshold -> surface/suppressed`

The trained artifact already contains the fitted preprocessor, models, ensemble weights, and threshold. The API does **not** retrain at request time.

## Run locally

```bash
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app:app --reload
```

Open:

- `http://localhost:8000/docs`
- `http://localhost:8000/health`

## Predict one finding

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sample.json
```

Example `sample.json`:

```json
{
  "message": "'foo' is assigned a value but never used.",
  "rule_id": "no-unused-vars",
  "rule_family": "variables",
  "severity_text": "warning",
  "language": "javascript",
  "path_extension": ".js",
  "severity": 1,
  "is_error": 0,
  "is_warning": 1,
  "has_fix": 0,
  "has_suggestions": 0,
  "changed_line_count": 11,
  "finding_overlaps_change": 1,
  "finding_change_distance": 0,
  "file_size_lines": 150,
  "same_rule_findings_in_file": 1,
  "same_rule_findings_in_repo": 3,
  "path_depth": 2,
  "path_has_test": 0,
  "path_has_src": 1,
  "path_has_config": 0,
  "path_has_generated": 0
}
```

Response shape:

```json
{
  "probabilities": {
    "logistic_regression": 0.6384,
    "random_forest": 0.5878,
    "xgboost": 0.5165
  },
  "ensemble_surface_probability": 0.5633,
  "threshold": 0.505,
  "decision": "surface"
}
```

## Null / missing values

Missing model inputs are accepted. Examples:

```json
{
  "message": null,
  "rule_id": null,
  "severity": null,
  "changed_line_count": null,
  "file_size_lines": null
}
```

Text blanks are normalized before TF-IDF, categorical blanks become a dedicated missing bucket, and numeric values are median-imputed with missingness indicators using statistics learned during training.

## Batch prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"findings":[{"message":"no-unused-vars","rule_id":"no-unused-vars"},{"message":null,"rule_id":null}]}'
```

Default maximum batch size: 100. Change with `MAX_BATCH_SIZE`.

## Docker

Build:

```bash
docker build -t prism-eslint-classifier .
```

Run:

```bash
docker run --rm -p 8000:8000 prism-eslint-classifier
```

## Deployment notes

The model file is about 26 MB and is copied into the image. For cloud deployment, keep the model artifact versioned with the API image so preprocessing and model versions cannot drift.

Environment variables:

- `PORT` — HTTP port (default `8000`)
- `MODEL_PATH` — custom `.joblib` path
- `CORS_ORIGINS` — comma-separated allowed origins (default `*`)
- `MAX_BATCH_SIZE` — batch request limit (default `100`)
