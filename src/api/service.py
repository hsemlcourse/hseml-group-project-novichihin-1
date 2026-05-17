"""Model loading and prediction logic for the FastAPI service."""

from __future__ import annotations

import warnings
from typing import Any

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline

from src.api.schemas import CampaignFeatures
from src.config import MODEL_PATH

_MODEL_CACHE: Pipeline | None = None


def load_model() -> Pipeline:
    """Load the trained pipeline from disk (cached after first call).

    The artifact is a dict ``{"pipeline": Pipeline, "best_experiment_id": str}``
    saved by ``notebooks/04_experiments_cp2.ipynb``.

    Suppresses sklearn ``InconsistentVersionWarning`` — the runtime is pinned to
    a different minor version than the one used to dump the artifact, but the
    estimators in use (``Pipeline``, ``ColumnTransformer``, ``OneHotEncoder``,
    ``StandardScaler``, ``GradientBoostingRegressor``) have a stable internal
    layout across these versions.
    """
    global _MODEL_CACHE  # noqa: PLW0603
    if _MODEL_CACHE is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            artifact = joblib.load(MODEL_PATH)
        _MODEL_CACHE = artifact["pipeline"] if isinstance(artifact, dict) else artifact
    return _MODEL_CACHE


def _payload_to_df(payload: CampaignFeatures) -> pd.DataFrame:
    """Convert a single Pydantic payload into a one-row DataFrame matching the training schema."""
    row: dict[str, Any] = payload.model_dump()
    row["start_date"] = pd.Timestamp(row["start_date"])
    return pd.DataFrame([row])


def predict_one(payload: CampaignFeatures) -> float:
    pipe = load_model()
    df = _payload_to_df(payload)
    return float(pipe.predict(df)[0])


def predict_batch(payloads: list[CampaignFeatures]) -> list[float]:
    pipe = load_model()
    rows = [p.model_dump() for p in payloads]
    df = pd.DataFrame(rows)
    df["start_date"] = pd.to_datetime(df["start_date"])
    preds = pipe.predict(df)
    return [float(v) for v in preds]
