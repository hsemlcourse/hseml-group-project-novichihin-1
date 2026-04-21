"""Project configuration: paths, seed, dataset schema.

All schema edits should happen here — preprocessing/modeling modules import from this file.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORT_DIR: Path = PROJECT_ROOT / "report"
REPORT_IMAGES_DIR: Path = REPORT_DIR / "images"

SEED: int = 42

KAGGLE_DATASET: str = "juniornsa/digital-advertising-campaign-performance-dataset"
RAW_CSV_NAME: str = "tech_advertising_campaigns_dataset.csv"

TARGET: str = "profit"

ID_COLS: list[str] = ["campaign_id"]
DATE_COL: str = "start_date"

# Features that trivially reconstruct the target and must NOT be used.
# profit = revenue - ad_spend
# ROAS = revenue / ad_spend
# CPA = ad_spend / conversions
# CPC / actual_cpc = ad_spend / clicks
LEAKAGE_COLS: list[str] = [
    "revenue",
    "ad_spend",
    "ROAS",
    "CPA",
    "CPC",
    "actual_cpc",
]

# Low-cardinality categorical features suitable for one-hot encoding.
CAT_LOW_COLS: list[str] = [
    "campaign_objective",
    "platform",
    "ad_placement",
    "device_type",
    "operating_system",
    "creative_format",
    "creative_size",
    "ad_copy_length",
    "creative_emotion",
    "target_audience_age",
    "target_audience_gender",
    "audience_interest_category",
    "income_bracket",
    "purchase_intent_score",
    "day_of_week",
    "industry_vertical",
    "budget_tier",
]

BOOL_COLS: list[str] = [
    "has_call_to_action",
    "retargeting_flag",
]

# Numeric features: campaign metadata + auction metric + observed traffic/engagement,
# without cost or revenue components.
NUM_COLS: list[str] = [
    "creative_age_days",
    "quarter",
    "hour_of_day",
    "campaign_day",
    "quality_score",
    "impressions",
    "clicks",
    "conversions",
    "bounce_rate",
    "avg_session_duration_seconds",
    "pages_per_session",
    "CTR",
    "conversion_rate",
]

# Date-derived numeric features produced during preprocessing.
DATE_DERIVED_COLS: list[str] = [
    "date_month",
    "date_dayofweek",
    "date_weekofyear",
    "date_is_weekend",
    "date_dayofyear",
]
