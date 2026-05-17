"""FastAPI application — profit prediction for digital ad campaigns.

Endpoints
---------
GET  /health         — liveness check
GET  /model          — model metadata (name, sklearn version, metrics)
POST /predict        — single campaign prediction
POST /predict_batch  — batch prediction
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BatchPredictResponse,
    CampaignFeatures,
    HealthResponse,
    ModelInfo,
    PredictResponse,
)
from src.api.service import load_model, predict_batch, predict_one
from src.config import SEED


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Ad Campaign Profit Predictor",
    description="Predict profit of a digital advertising campaign based on targeting, creative and auction features.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — open: the API is a public demo, the Streamlit UI on Hugging Face Spaces
# (and any curl / Postman / browser fetch) must be able to call it without a preflight block.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


@app.get("/model", response_model=ModelInfo)
def model_info():
    return ModelInfo(
        model_name="GradientBoostingRegressor (exp03_tree_tuned, CP2)",
        sklearn_version=sklearn.__version__,
        seed=SEED,
        val_rmse=26_778.0,
        test_rmse=64_019.0,
        test_r2=0.733,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: CampaignFeatures):
    try:
        value = predict_one(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PredictResponse(profit_prediction=value)


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_many(payloads: list[CampaignFeatures]):
    if not payloads:
        raise HTTPException(status_code=422, detail="Payload list must not be empty")
    try:
        values = predict_batch(payloads)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return BatchPredictResponse(predictions=values)
