"""Smoke tests for the preprocessing / training pipeline.

Run: ``pytest -q``.

Tests generate a tiny synthetic DataFrame that matches the real dataset's schema,
so they do not require the Kaggle CSV to be present in ``data/raw``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CAT_LOW_COLS, LEAKAGE_COLS, NUM_COLS, TARGET  # noqa: E402
from src.data_shift import numeric_shift_table  # noqa: E402
from src.preprocessing import (  # noqa: E402
    build_baseline_full_pipeline,
    build_full_pipeline,
    prepare_features,
    prepare_features_baseline,
    split_xy,
    time_or_random_split,
)
from src.utils import regression_metrics, set_seed  # noqa: E402


def _fake_frame(n: int = 80) -> pd.DataFrame:
    """Synthetic frame mirroring the real schema (enough for smoke tests)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "campaign_id": [f"C{i:04d}" for i in range(n)],
            "campaign_objective": rng.choice(["Lead Generation", "Engagement"], size=n),
            "platform": rng.choice(["Facebook", "Google", "TikTok"], size=n),
            "ad_placement": rng.choice(["Feed", "Search", "Stories"], size=n),
            "device_type": rng.choice(["Mobile", "Desktop"], size=n),
            "operating_system": rng.choice(["Android", "iOS"], size=n),
            "creative_format": rng.choice(["Text", "Image", "Video"], size=n),
            "creative_size": rng.choice(["728x90", "320x50", "300x250"], size=n),
            "ad_copy_length": rng.choice(["Short", "Medium", "Long"], size=n),
            "has_call_to_action": rng.choice([True, False], size=n),
            "creative_emotion": rng.choice(["Curiosity", "Neutral", "Excitement"], size=n),
            "creative_age_days": rng.integers(0, 400, size=n),
            "target_audience_age": rng.choice(["18-24", "25-34", "35-44", "45-54", "65+"], size=n),
            "target_audience_gender": rng.choice(["Male", "Female", "All"], size=n),
            "audience_interest_category": rng.choice(["Shoppers", "Business Professionals"], size=n),
            "income_bracket": rng.choice(["<$50K", "$50K-$100K", ">$100K"], size=n),
            "purchase_intent_score": rng.choice(["Low", "Medium", "High"], size=n),
            "retargeting_flag": rng.choice([True, False], size=n),
            "start_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, size=n), unit="D"),
            "quarter": rng.integers(1, 5, size=n),
            "day_of_week": rng.choice(["Monday", "Tuesday", "Wednesday"], size=n),
            "hour_of_day": rng.integers(0, 24, size=n),
            "campaign_day": rng.integers(1, 60, size=n),
            "quality_score": rng.integers(1, 11, size=n),
            "actual_cpc": rng.uniform(0.1, 5.0, size=n),
            "impressions": rng.integers(1_000, 100_000, size=n),
            "clicks": rng.integers(10, 2_000, size=n),
            "conversions": rng.integers(0, 50, size=n),
            "ad_spend": rng.uniform(50, 3_000, size=n),
            "revenue": rng.uniform(50, 6_000, size=n),
            "bounce_rate": rng.uniform(20, 90, size=n),
            "avg_session_duration_seconds": rng.uniform(10, 300, size=n),
            "pages_per_session": rng.uniform(1, 10, size=n),
            "industry_vertical": rng.choice(["E-commerce", "Finance", "Gaming"], size=n),
            "budget_tier": rng.choice(["Low", "Medium", "High"], size=n),
            "CTR": rng.uniform(0.1, 5.0, size=n),
            "CPC": rng.uniform(0.1, 5.0, size=n),
            "conversion_rate": rng.uniform(0.1, 5.0, size=n),
            "CPA": rng.uniform(10, 500, size=n),
            "ROAS": rng.uniform(0.2, 5.0, size=n),
        }
    )
    df[TARGET] = df["revenue"] - df["ad_spend"] + rng.normal(0, 10, size=n)
    return df


def test_prepare_features_drops_leakage_and_adds_date():
    df = _fake_frame()
    out = prepare_features(df)
    for col in LEAKAGE_COLS:
        assert col not in out.columns, f"{col} must be dropped"
    for col in ["date_month", "date_dayofweek", "date_is_weekend"]:
        assert col in out.columns
    assert "start_date" not in out.columns


def test_preprocessing_fit_transform_shapes():
    df = _fake_frame()
    X, y = split_xy(df)
    pipe = build_full_pipeline(LinearRegression())
    pipe.fit(X, y)
    transformed = pipe[:-1].transform(X)
    assert transformed.shape[0] == len(df)
    assert not np.isnan(np.asarray(transformed)).any()
    # Every input numeric column must be present downstream.
    # We check that at least the numeric branch contributed features.
    assert transformed.shape[1] >= len(NUM_COLS)


def test_pipeline_predicts_with_expected_shape():
    df = _fake_frame()
    train_df, val_df, test_df = time_or_random_split(df, val_size=0.2, test_size=0.2)
    pipe = build_full_pipeline(LinearRegression())
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert preds.shape == y_test.shape
    metrics = regression_metrics(y_test.values, preds)
    assert set(metrics) == {"RMSE", "MAE", "R2"}


def test_set_seed_reproducibility():
    df = _fake_frame()
    X, y = split_xy(df)

    set_seed(42)
    pipe1 = build_full_pipeline(LinearRegression())
    pipe1.fit(X, y)
    p1 = pipe1.predict(X)

    set_seed(42)
    pipe2 = build_full_pipeline(LinearRegression())
    pipe2.fit(X, y)
    p2 = pipe2.predict(X)

    np.testing.assert_allclose(p1, p2)


def test_prepare_features_baseline_skips_calendar_fe():
    df = _fake_frame()
    out = prepare_features_baseline(df)
    assert "date_month" not in out.columns and "start_date" not in out.columns
    for col in LEAKAGE_COLS:
        assert col not in out.columns


def test_baseline_pipeline_fit_predict():
    df = _fake_frame()
    train_df, val_df, test_df = time_or_random_split(df, val_size=0.2, test_size=0.2)
    pipe = build_baseline_full_pipeline(LinearRegression())
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert preds.shape == y_test.shape


def test_time_split_preserves_order_when_date_present():
    df = _fake_frame()
    train_df, val_df, test_df = time_or_random_split(df, val_size=0.2, test_size=0.2)
    max_train = train_df["start_date"].max()
    min_val = val_df["start_date"].min()
    min_test = test_df["start_date"].min()
    max_val = val_df["start_date"].max()
    assert max_train <= min_val
    assert max_val <= min_test


def test_all_cat_low_cols_declared_in_schema():
    assert TARGET not in CAT_LOW_COLS
    assert set(CAT_LOW_COLS).isdisjoint(set(NUM_COLS))
    assert set(CAT_LOW_COLS).isdisjoint(set(LEAKAGE_COLS))


def test_numeric_shift_table_smoke():
    df = _fake_frame(120)
    train_df, _, test_df = time_or_random_split(df, val_size=0.15, test_size=0.15)
    tbl = numeric_shift_table(train_df, test_df)
    assert not tbl.empty
    assert "mean_train" in tbl.columns
    assert "median_train" in tbl.columns and "delta_median" in tbl.columns
    assert "abs_delta_mean_over_std_train" in tbl.columns


def test_time_order_false_preserves_start_date_and_shuffles():
    df = _fake_frame(200)
    t_train, _, _ = time_or_random_split(df.copy())
    r_train, _, _ = time_or_random_split(df.copy(), time_order=False, seed=0)
    assert "start_date" in r_train.columns
    assert t_train["start_date"].is_monotonic_increasing
    assert not r_train["start_date"].is_monotonic_increasing
