"""High-level helpers used by the baseline and experiment notebooks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.config import SEED
from src.preprocessing import build_baseline_full_pipeline, build_full_pipeline, split_xy
from src.utils import regression_metrics


def fit_and_evaluate(
    estimator: Any,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    name: str,
) -> tuple[Any, list[dict[str, float | str]]]:
    """Fit a full Pipeline(prep -> features -> estimator) and score it on val/test."""
    pipe = build_full_pipeline(estimator)
    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)

    pipe.fit(X_train, y_train)
    val_pred = pipe.predict(X_val)
    test_pred = pipe.predict(X_test)

    rows: list[dict[str, float | str]] = []
    for split, y_true, y_pred in (("val", y_val, val_pred), ("test", y_test, test_pred)):
        row: dict[str, float | str] = {"model": name, "split": split}
        row.update(regression_metrics(np.asarray(y_true), np.asarray(y_pred)))
        rows.append(row)
    return pipe, rows


def fit_and_evaluate_baseline(
    estimator: Any,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    name: str,
) -> tuple[Any, list[dict[str, float | str]]]:
    """Like :func:`fit_and_evaluate` but uses minimal prep (no calendar feature engineering)."""
    pipe = build_baseline_full_pipeline(estimator)
    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)

    pipe.fit(X_train, y_train)
    val_pred = pipe.predict(X_val)
    test_pred = pipe.predict(X_test)

    rows: list[dict[str, float | str]] = []
    for split, y_true, y_pred in (("val", y_val, val_pred), ("test", y_test, test_pred)):
        row: dict[str, float | str] = {"model": name, "split": split}
        row.update(regression_metrics(np.asarray(y_true), np.asarray(y_pred)))
        rows.append(row)
    return pipe, rows


def cv_score(
    estimator: Any,
    train_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return per-fold regression metrics for a Pipeline-wrapped estimator."""
    pipe = build_full_pipeline(estimator)
    X, y = split_xy(train_df)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    records: list[dict[str, float | int]] = []
    for fold, (tr, va) in enumerate(kf.split(X)):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[va])
        metrics = regression_metrics(np.asarray(y.iloc[va]), np.asarray(pred))
        metrics["fold"] = fold
        records.append(metrics)
    return pd.DataFrame.from_records(records)


def collect_metrics(rows: Iterable[dict[str, float | str]]) -> pd.DataFrame:
    """Convenience: turn a list of metric dicts into a sorted DataFrame."""
    df = pd.DataFrame(list(rows))
    if not df.empty:
        df = df.sort_values(["split", "RMSE"]).reset_index(drop=True)
    return df
