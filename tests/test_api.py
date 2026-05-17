"""Smoke tests for the FastAPI prediction service.

Run: ``pytest tests/test_api.py -q``.

Tests use ``fastapi.testclient.TestClient`` (backed by ``httpx``) so no
live server is required.  If ``models/final_cp2.joblib`` is absent the
prediction tests are skipped — prevents CI failures when the artifact
is gitignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MODEL_PATH  # noqa: E402

_MODEL_EXISTS = MODEL_PATH.exists()

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def client():
    if not _MODEL_EXISTS:
        pytest.skip("Model artifact not found — skipping API tests")
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as c:
        yield c


_VALID_PAYLOAD = {
    "start_date": "2024-06-15",
    "campaign_objective": "Lead Generation",
    "platform": "Facebook",
    "ad_placement": "Feed",
    "device_type": "Mobile",
    "operating_system": "Android",
    "creative_format": "Image",
    "creative_size": "300x250",
    "ad_copy_length": "Medium",
    "creative_emotion": "Curiosity",
    "target_audience_age": "25-34",
    "target_audience_gender": "Male",
    "audience_interest_category": "Shoppers",
    "income_bracket": "$50K-$100K",
    "purchase_intent_score": "Medium",
    "day_of_week": "Monday",
    "industry_vertical": "E-commerce",
    "budget_tier": "Medium",
    "has_call_to_action": True,
    "retargeting_flag": False,
    "creative_age_days": 30,
    "quarter": 2,
    "hour_of_day": 14,
    "campaign_day": 10,
    "quality_score": 7,
    "impressions": 50000,
    "clicks": 500,
    "conversions": 20,
    "bounce_rate": 45.0,
    "avg_session_duration_seconds": 120.0,
    "pages_per_session": 3.5,
    "CTR": 1.0,
    "conversion_rate": 4.0,
}

# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_model_info(client):
    resp = client.get("/model")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_name" in data
    assert data["seed"] == 42


def test_predict_valid(client):
    resp = client.post("/predict", json=_VALID_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert "profit_prediction" in data
    assert isinstance(data["profit_prediction"], (int, float))


def test_predict_missing_field(client):
    bad = {k: v for k, v in _VALID_PAYLOAD.items() if k != "platform"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_batch(client):
    payloads = [_VALID_PAYLOAD, _VALID_PAYLOAD, _VALID_PAYLOAD]
    resp = client.post("/predict_batch", json=payloads)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["predictions"]) == 3


def test_predict_batch_empty(client):
    resp = client.post("/predict_batch", json=[])
    assert resp.status_code == 422


def test_predict_sanity(client):
    """API response must match direct pipeline prediction."""
    import numpy as np
    import pandas as pd

    from src.api.service import load_model

    resp = client.post("/predict", json=_VALID_PAYLOAD)
    api_value = resp.json()["profit_prediction"]

    pipe = load_model()
    row = dict(_VALID_PAYLOAD)
    row["start_date"] = pd.Timestamp(row["start_date"])
    direct_value = float(pipe.predict(pd.DataFrame([row]))[0])

    np.testing.assert_allclose(api_value, direct_value, rtol=1e-6)
