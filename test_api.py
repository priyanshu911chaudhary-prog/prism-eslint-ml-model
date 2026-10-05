from fastapi.testclient import TestClient
from app import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True


def test_prediction():
    payload = {
        "message": "'foo' is assigned a value but never used.",
        "rule_id": "no-unused-vars",
        "rule_family": "variables",
        "severity_text": "warning",
        "language": "javascript",
        "path_extension": ".js",
        "severity": 1,
        "is_error": 0,
        "is_warning": 1,
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
        "path_has_generated": 0,
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert set(body["probabilities"]) == {
            "logistic_regression",
            "random_forest",
            "xgboost",
        }
        assert body["decision"] in {"surface", "suppressed"}


def test_missing_values():
    payload = {
        "message": None,
        "rule_id": None,
        "rule_family": None,
        "severity_text": None,
        "severity": None,
        "changed_line_count": None,
        "finding_overlaps_change": None,
        "finding_change_distance": None,
        "file_size_lines": None,
        "language": "javascript",
        "path_extension": ".js",
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["decision"] in {"surface", "suppressed"}
