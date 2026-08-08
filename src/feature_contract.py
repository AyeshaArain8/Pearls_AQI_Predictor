"""Canonical, Lahore-only feature contract shared by training and serving."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

LAHORE = {"name": "Lahore", "latitude": 31.5204, "longitude": 74.3587}
SCHEMA_VERSION = "lahore-aqi-v2"
RAW_FEATURES = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
FEATURE_COLUMNS = RAW_FEATURES + [
    "month", "day", "day_of_week", "aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_rolling_mean",
]
TARGET_COLUMNS = ["day1_target", "day2_target", "day3_target"]


class DataQualityError(ValueError):
    """Raised when data cannot safely be used for forecasting."""


def validate_observations(frame: pd.DataFrame, *, require_aqi: bool = True) -> pd.DataFrame:
    required = ["timestamp", *RAW_FEATURES] + (["aqi"] if require_aqi else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise DataQualityError(f"Missing required columns: {missing}")
    checked = frame.copy()
    checked["timestamp"] = pd.to_datetime(checked["timestamp"], utc=True, errors="coerce")
    if checked["timestamp"].isna().any() or checked["timestamp"].duplicated().any():
        raise DataQualityError("Timestamps must be valid, unique, and ordered.")
    if not checked["timestamp"].is_monotonic_increasing:
        raise DataQualityError("Timestamps are not in chronological order.")
    for column in RAW_FEATURES + (["aqi"] if require_aqi else []):
        checked[column] = pd.to_numeric(checked[column], errors="coerce")
        if checked[column].isna().any() or (checked[column] < 0).any():
            raise DataQualityError(f"Invalid or missing values in {column}.")
    return checked


def make_feature_rows(observations: pd.DataFrame, *, include_targets: bool) -> pd.DataFrame:
    """Build the exact same model inputs from chronological real observations."""
    data = validate_observations(observations).copy()
    data["month"] = data.timestamp.dt.month
    data["day"] = data.timestamp.dt.day
    data["day_of_week"] = data.timestamp.dt.dayofweek
    data["aqi_lag1"] = data.aqi.shift(1)
    data["aqi_lag2"] = data.aqi.shift(2)
    data["aqi_lag3"] = data.aqi.shift(3)
    data["aqi_rolling_mean"] = data.aqi.shift(1).rolling(3).mean()
    if include_targets:
        for horizon, target in enumerate(TARGET_COLUMNS, 1):
            data[target] = data.aqi.shift(-horizon)
    columns = ["timestamp", "aqi", *FEATURE_COLUMNS] + (TARGET_COLUMNS if include_targets else [])
    return data[columns].dropna().reset_index(drop=True)


def latest_serving_features(observations: pd.DataFrame, live_observation: dict) -> pd.DataFrame:
    """Create a one-row serving frame using three actual prior observations only."""
    history = validate_observations(observations)
    if len(history) < 3:
        raise DataQualityError("Insufficient historical AQI: at least 3 prior observations are required.")
    timestamp = pd.to_datetime(live_observation["timestamp"], utc=True)
    if timestamp <= history.timestamp.iloc[-1]:
        raise DataQualityError("Live observation timestamp must be newer than feature-store history.")
    row = {"timestamp": timestamp, "aqi": float(live_observation["aqi"])}
    row.update({key: float(live_observation[key]) for key in RAW_FEATURES})
    combined = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    features = make_feature_rows(combined, include_targets=False)
    if features.empty:
        raise DataQualityError("Unable to calculate real lag features from historical observations.")
    return features.loc[:, FEATURE_COLUMNS].tail(1)


def assert_feature_schema(columns: Iterable[str]) -> None:
    actual = list(columns)
    if actual != FEATURE_COLUMNS:
        raise DataQualityError(f"Feature schema mismatch. Expected {FEATURE_COLUMNS}, got {actual}")
