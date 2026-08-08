"""Lahore-only inference using registered models and real feature-store history."""
from __future__ import annotations
from datetime import timedelta
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.feature_contract import FEATURE_COLUMNS, LAHORE, SCHEMA_VERSION, assert_feature_schema, latest_serving_features
from src.feature_store import load_observations
from src.model_registry import load_latest

AQI_THRESHOLDS = ((50, "Good"), (100, "Moderate"), (150, "Unhealthy for Sensitive Groups"), (200, "Unhealthy"), (300, "Very Unhealthy"), (float("inf"), "Hazardous"))

def get_category(aqi: float) -> str:
    return next(label for limit, label in AQI_THRESHOLDS if aqi <= limit)

def forecast_aqi(live_observation: dict) -> dict:
    if live_observation.get("city", "Lahore") != "Lahore":
        raise ValueError("Pearls AQI Predictor supports Lahore only.")
    models, metadata = load_latest()
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Registered model schema is incompatible with the serving feature contract.")
    assert_feature_schema(metadata.get("feature_columns", []))
    features = latest_serving_features(load_observations(), live_observation)
    assert_feature_schema(features.columns)
    observed_at = pd.to_datetime(live_observation["timestamp"], utc=True)
    forecasts = []
    for index, horizon in enumerate(("day1", "day2", "day3"), 1):
        aqi = round(float(models[horizon].predict(features)[0]), 1)
        forecasts.append({"date": (observed_at + timedelta(days=index)).date().isoformat(), "aqi": aqi, "category": get_category(aqi), "hazardous": aqi > 300})
    return {"city": "Lahore", "model_version": metadata["version"], "observed_at": observed_at.isoformat(), "forecasts": forecasts, "metrics": metadata.get("metrics", {}), "features": features.iloc[0].to_dict()}
