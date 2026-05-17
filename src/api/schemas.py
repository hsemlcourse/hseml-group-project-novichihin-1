"""Pydantic request / response models for the prediction API.

Fields mirror the feature schema from ``src.config`` (CAT_LOW_COLS + BOOL_COLS + NUM_COLS +
start_date).  LEAKAGE_COLS are intentionally excluded — they reconstruct the target directly.
"""

from __future__ import annotations

import datetime

from pydantic import BaseModel, Field


class CampaignFeatures(BaseModel):
    """Input features for a single ad-campaign profit prediction."""

    # --- date ---
    start_date: datetime.date = Field(..., description="Campaign start date (ISO format, e.g. 2024-06-15)")

    # --- categorical (17 low-cardinality) ---
    campaign_objective: str = Field(..., examples=["Lead Generation"])
    platform: str = Field(..., examples=["Facebook"])
    ad_placement: str = Field(..., examples=["Feed"])
    device_type: str = Field(..., examples=["Mobile"])
    operating_system: str = Field(..., examples=["Android"])
    creative_format: str = Field(..., examples=["Image"])
    creative_size: str = Field(..., examples=["300x250"])
    ad_copy_length: str = Field(..., examples=["Medium"])
    creative_emotion: str = Field(..., examples=["Curiosity"])
    target_audience_age: str = Field(..., examples=["25-34"])
    target_audience_gender: str = Field(..., examples=["Male"])
    audience_interest_category: str = Field(..., examples=["Shoppers"])
    income_bracket: str = Field(..., examples=["$50K-$100K"])
    purchase_intent_score: str = Field(..., examples=["Medium"])
    day_of_week: str = Field(..., examples=["Monday"])
    industry_vertical: str = Field(..., examples=["E-commerce"])
    budget_tier: str = Field(..., examples=["Medium"])

    # --- boolean ---
    has_call_to_action: bool = Field(..., examples=[True])
    retargeting_flag: bool = Field(..., examples=[False])

    # --- numeric ---
    creative_age_days: float = Field(..., examples=[30])
    quarter: float = Field(..., examples=[2])
    hour_of_day: float = Field(..., examples=[14])
    campaign_day: float = Field(..., examples=[10])
    quality_score: float = Field(..., examples=[7])
    impressions: float = Field(..., examples=[50000])
    clicks: float = Field(..., examples=[500])
    conversions: float = Field(..., examples=[20])
    bounce_rate: float = Field(..., examples=[45.0])
    avg_session_duration_seconds: float = Field(..., examples=[120.0])
    pages_per_session: float = Field(..., examples=[3.5])
    CTR: float = Field(..., examples=[1.0])
    conversion_rate: float = Field(..., examples=[4.0])


class PredictResponse(BaseModel):
    profit_prediction: float


class BatchPredictResponse(BaseModel):
    predictions: list[float]


class ModelInfo(BaseModel):
    model_name: str
    sklearn_version: str
    seed: int
    val_rmse: float | None = None
    test_rmse: float | None = None
    test_r2: float | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
