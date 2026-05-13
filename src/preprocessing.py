"""Feature engineering + preprocessing pipeline.

Design goals:
- Keep everything inside a single sklearn Pipeline so any fitted statistic
  (median / scaler / OHE categories) is learned only on the training subset
  and cannot leak into val/test.
- Drop trivially leaking features (components of the target) up-front in a
  `FunctionTransformer` so no downstream step can accidentally use them.
- Expose small helpers (`split_xy`, `time_or_random_split`) used by notebooks.
- Rubric baseline uses :func:`prepare_features_baseline` / :func:`build_baseline_full_pipeline`
  (no calendar-derived features from ``start_date``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BOOL_COLS,
    CAT_LOW_COLS,
    DATE_COL,
    DATE_DERIVED_COLS,
    ID_COLS,
    LEAKAGE_COLS,
    NUM_COLS,
    SEED,
    TARGET,
)


def split_xy(df: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) removing IDs and target from the feature frame."""
    drop_cols = [c for c in ID_COLS + [target] if c in df.columns]
    return df.drop(columns=drop_cols), df[target].astype(float)


def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar features from the `start_date` column; drop the raw date afterwards."""
    df = df.copy()
    if DATE_COL in df.columns:
        dt = pd.to_datetime(df[DATE_COL], errors="coerce")
        df["date_month"] = dt.dt.month.astype("float")
        df["date_dayofweek"] = dt.dt.dayofweek.astype("float")
        df["date_weekofyear"] = dt.dt.isocalendar().week.astype("float")
        df["date_is_weekend"] = (dt.dt.dayofweek >= 5).astype("float")
        df["date_dayofyear"] = dt.dt.dayofyear.astype("float")
        df = df.drop(columns=[DATE_COL])
    return df


def _cast_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Convert truthy/falsy columns to 0/1 floats so the numeric branch can handle them."""
    df = df.copy()
    for col in BOOL_COLS:
        if col in df.columns:
            series = df[col]
            if series.dtype == bool:
                df[col] = series.astype("float")
            else:
                df[col] = (
                    series.astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
                )
    return df


def _drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that would trivially reconstruct the target."""
    return df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """All transformations that do not learn parameters from data."""
    out = _drop_leakage(df)
    out = _cast_booleans(out)
    out = _add_date_features(out)
    return out


def prepare_features_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal prep for rubric baseline: leakage removal + boolean casts only.

    Deliberately skips calendar feature engineering from ``start_date`` (those are
    produced in :func:`prepare_features`). The raw ``start_date`` column is dropped
    so the downstream preprocessor does not depend on date-derived columns.
    Seasonality is still partially captured via existing columns like ``quarter``
    and ``day_of_week`` from the dataset.
    """
    out = _drop_leakage(df)
    out = _cast_booleans(out)
    if DATE_COL in out.columns:
        out = out.drop(columns=[DATE_COL])
    return out


def _ohe() -> OneHotEncoder:
    """OneHotEncoder that is sparse-safe across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    num_cols: Sequence[str] | None = None,
    cat_cols: Sequence[str] | None = None,
    bool_cols: Sequence[str] | None = None,
    date_cols: Sequence[str] | None = None,
) -> ColumnTransformer:
    """Numeric (impute+scale), categorical (impute+OHE) and pass-through boolean/date features."""
    num_cols = list(num_cols if num_cols is not None else NUM_COLS)
    cat_cols = list(cat_cols if cat_cols is not None else CAT_LOW_COLS)
    bool_cols = list(bool_cols if bool_cols is not None else BOOL_COLS)
    date_cols = list(date_cols if date_cols is not None else DATE_DERIVED_COLS)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", _ohe()),
        ]
    )
    bool_pipe = SimpleImputer(strategy="most_frequent")
    date_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
            ("bool", bool_pipe, bool_cols),
            ("date", date_pipe, date_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_baseline_preprocessor(
    num_cols: Sequence[str] | None = None,
    cat_cols: Sequence[str] | None = None,
    bool_cols: Sequence[str] | None = None,
) -> ColumnTransformer:
    """Same as :func:`build_preprocessor` but without the date-derived branch."""
    num_cols = list(num_cols if num_cols is not None else NUM_COLS)
    cat_cols = list(cat_cols if cat_cols is not None else CAT_LOW_COLS)
    bool_cols = list(bool_cols if bool_cols is not None else BOOL_COLS)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", _ohe()),
        ]
    )
    bool_pipe = SimpleImputer(strategy="most_frequent")

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
            ("bool", bool_pipe, bool_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_full_pipeline(estimator) -> Pipeline:
    """End-to-end transformer: raw DataFrame -> engineered features -> preprocessor -> estimator."""
    from sklearn.preprocessing import FunctionTransformer

    return Pipeline(
        steps=[
            ("prep", FunctionTransformer(prepare_features, validate=False)),
            ("features", build_preprocessor()),
            ("model", estimator),
        ]
    )


def build_baseline_full_pipeline(estimator) -> Pipeline:
    """Pipeline for rubric baseline: minimal feature prep + preprocessor + estimator."""
    from sklearn.preprocessing import FunctionTransformer

    return Pipeline(
        steps=[
            ("prep", FunctionTransformer(prepare_features_baseline, validate=False)),
            ("features", build_baseline_preprocessor()),
            ("model", estimator),
        ]
    )


def time_or_random_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SEED,
    *,
    time_order: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test). Use a time-based split if `start_date` is present, random otherwise.

    A time split keeps val/test after train in calendar order, which prevents the model
    from "seeing the future" during training.

    Set ``time_order=False`` to shuffle rows (70/15/15) even when ``start_date`` exists — useful for
    ablations; calendar features from ``start_date`` remain available in every fold row.
    """
    df = df.copy()
    n = len(df)

    if time_order and DATE_COL in df.columns and df[DATE_COL].notna().any():
        df = df.sort_values(DATE_COL).reset_index(drop=True)
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))
        n_train = n - n_val - n_test
        train = df.iloc[:n_train]
        val = df.iloc[n_train : n_train + n_val]
        test = df.iloc[n_train + n_val :]
        return train, val, test

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
