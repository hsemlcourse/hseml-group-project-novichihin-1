"""Reproducibility helpers and shared metric utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import SEED


def set_seed(seed: int = SEED) -> None:
    """Fix all sources of randomness used in this project."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return RMSE (primary), MAE, R^2 for regression predictions."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def metrics_row(name: str, y_true: np.ndarray, y_pred: np.ndarray, split: str) -> dict[str, float | str]:
    """Wrap regression_metrics into a row suitable for an experiments DataFrame."""
    row: dict[str, float | str] = {"model": name, "split": split}
    row.update(regression_metrics(y_true, y_pred))
    return row


def append_metrics(
    table: pd.DataFrame | None,
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: str,
) -> pd.DataFrame:
    """Append a metrics row to an existing experiments table (or create a new one)."""
    row = metrics_row(name, y_true, y_pred, split)
    new = pd.DataFrame([row])
    return new if table is None else pd.concat([table, new], ignore_index=True)
