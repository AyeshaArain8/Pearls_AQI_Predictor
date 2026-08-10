"""Canonical, Lahore-only feature contract shared by training and serving."""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


# ============================================================
# LAHORE CONTRACT
# ============================================================

LAHORE = {
    "name": "Lahore",
    "latitude": 31.5204,
    "longitude": 74.3587,
}

SCHEMA_VERSION = "lahore-aqi-v2"

RAW_FEATURES = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]

FEATURE_COLUMNS = RAW_FEATURES + [
    "hour",
    "month",
    "day",
    "day_of_week",
    "aqi_lag1",
    "aqi_lag2",
    "aqi_lag3",
    "aqi_rolling_mean",
]

TARGET_COLUMNS = (
    "day1_target",
    "day2_target",
    "day3_target",
)

# SCHEMA_VERSION = "lahore-aqi-v2"

# RAW_FEATURES = (
#     "pm10",
#     "pm2_5",
#     "carbon_monoxide",
#     "nitrogen_dioxide",
#     "sulphur_dioxide",
#     "ozone",
# )
# FEATURE_COLUMNS = RAW_FEATURES + [
#     "hour",
#     "month",
#     "day",
#     "day_of_week",
#     "aqi_lag1",
#     "aqi_lag2",
#     "aqi_lag3",
#     "aqi_rolling_mean",
# ]

# Existing approved models were trained with exactly these
# 13 features. Keep this order identical for:
# - training
# - serving
# - model registry
# - tests
#
# "hour" exists in cloud/Feast, but is intentionally NOT part
# of the existing approved model inference schema.
# FEATURE_COLUMNS = RAW_FEATURES + (
#     "month",
#     "day",
#     "day_of_week",
#     "aqi_lag1",
#     "aqi_lag2",
#     "aqi_lag3",
#     "aqi_rolling_mean",
# )

# TARGET_COLUMNS = (
#     "day1_target",
#     "day2_target",
#     "day3_target",
# )


# ============================================================
# PM2.5 -> US AQI
# ============================================================

# EPA PM2.5 AQI breakpoints.
PM25_AQI_BREAKPOINTS = (
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.4, 301, 500),
)


class DataQualityError(ValueError):
    """Raised when data cannot safely be used for forecasting."""


def pm25_to_us_aqi(value: float) -> float:
    """Deterministically calculate US AQI from PM2.5."""

    concentration = float(value)

    if not math.isfinite(concentration) or concentration < 0:
        raise DataQualityError(
            "PM2.5 must be a finite non-negative value."
        )

    for low, high, aqi_low, aqi_high in PM25_AQI_BREAKPOINTS:
        if concentration <= high:
            return round(
                (
                    (aqi_high - aqi_low)
                    * (concentration - low)
                    / (high - low)
                )
                + aqi_low,
                1,
            )

    return 500.0


# ============================================================
# TIMESTAMP NORMALIZATION
# ============================================================

def to_utc_naive_datetime(value):
    """
    Normalize timestamps for PostgreSQL
    timestamp without time zone columns.
    """

    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.tz_localize(None).to_pydatetime()


def to_utc_datetime(value):
    """
    Normalize timestamps for PostgreSQL
    timestamp with time zone columns.
    """

    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.to_pydatetime()


# ============================================================
# OBSERVATION VALIDATION
# ============================================================

def validate_observations(
    frame: pd.DataFrame,
    *,
    require_aqi: bool = True,
) -> pd.DataFrame:
    """Validate genuine chronological Lahore observations."""

    required = [
        "timestamp",
        *RAW_FEATURES,
    ]

    if require_aqi:
        required.append("aqi")

    missing = sorted(
        set(required) - set(frame.columns)
    )

    if missing:
        raise DataQualityError(
            f"Missing required columns: {missing}"
        )

    checked = frame.copy()

    checked["timestamp"] = pd.to_datetime(
        checked["timestamp"],
        utc=True,
        errors="coerce",
    )

    if checked["timestamp"].isna().any():
        raise DataQualityError(
            "Timestamps must be valid, unique, and ordered."
        )

    if checked["timestamp"].duplicated().any():
        raise DataQualityError(
            "Timestamps must be valid, unique, and ordered."
        )

    if not checked["timestamp"].is_monotonic_increasing:
        raise DataQualityError(
            "Timestamps are not in chronological order."
        )

    numeric_columns = list(RAW_FEATURES)

    if require_aqi:
        numeric_columns.append("aqi")

    for column in numeric_columns:
        checked[column] = pd.to_numeric(
            checked[column],
            errors="coerce",
        )

        if checked[column].isna().any():
            raise DataQualityError(
                f"Invalid or missing values in {column}."
            )

        if (checked[column] < 0).any():
            raise DataQualityError(
                f"Invalid or missing values in {column}."
            )

    return checked


# ============================================================
# TRAINING FEATURES
# ============================================================

def make_feature_rows(
    observations: pd.DataFrame,
    *,
    include_targets: bool,
) -> pd.DataFrame:
    """
    Build the exact same model inputs used by training and serving.

    Lag features are calculated only from genuine previous AQI
    observations. No synthetic/fake lag values are created.
    """

    data = validate_observations(
        observations
    ).copy()

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    # Keep hour available in the feature dataframe because it is
    # present in the cloud/Feast source, but do not include it in
    # FEATURE_COLUMNS because the approved models are 13-feature
    # models and were not trained with hour.
    data["hour"] = data["timestamp"].dt.hour
    data["month"] = data["timestamp"].dt.month
    data["day"] = data["timestamp"].dt.day
    data["day_of_week"] = (
        data["timestamp"].dt.dayofweek
    )

    # --------------------------------------------------------
    # Real chronological AQI lag features
    # --------------------------------------------------------

    data["aqi_lag1"] = data["aqi"].shift(1)
    data["aqi_lag2"] = data["aqi"].shift(2)
    data["aqi_lag3"] = data["aqi"].shift(3)

    # Rolling mean uses the three previous real AQI values.
    data["aqi_rolling_mean"] = (
        data["aqi"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    # --------------------------------------------------------
    # Forecast targets
    # --------------------------------------------------------

    if include_targets:
        for horizon, target in enumerate(
            TARGET_COLUMNS,
            1,
        ):
            data[target] = data["aqi"].shift(
                -horizon
            )

    columns = [
        "timestamp",
        "aqi",
        *FEATURE_COLUMNS,
    ]

    if include_targets:
        columns.extend(TARGET_COLUMNS)

    return (
        data[columns]
        .dropna()
        .reset_index(drop=True)
    )


# ============================================================
# LIVE SERVING FEATURES
# ============================================================

def latest_serving_features(
    observations: pd.DataFrame,
    live_observation: dict,
) -> pd.DataFrame:
    """
    Create one serving row using only three actual
    prior observations.
    """

    history = validate_observations(
        observations
    )

    if len(history) < 3:
        raise DataQualityError(
            "Insufficient historical AQI: "
            "at least 3 prior observations are required."
        )

    timestamp = pd.to_datetime(
        live_observation["timestamp"],
        utc=True,
    )

    if timestamp <= history.timestamp.iloc[-1]:
        raise DataQualityError(
            "Live observation timestamp must be "
            "newer than feature-store history."
        )

    row = {
        "timestamp": timestamp,
        "aqi": float(
            live_observation["aqi"]
        ),
    }

    row.update(
        {
            key: float(
                live_observation[key]
            )
            for key in RAW_FEATURES
        }
    )

    combined = pd.concat(
        [
            history,
            pd.DataFrame([row]),
        ],
        ignore_index=True,
    )

    features = make_feature_rows(
        combined,
        include_targets=False,
    )

    if features.empty:
        raise DataQualityError(
            "Unable to calculate real lag features "
            "from historical observations."
        )

    return features.loc[
        :,
        FEATURE_COLUMNS,
    ].tail(1)


# ============================================================
# SCHEMA VALIDATION
# ============================================================

def assert_feature_schema(
    columns: Iterable[str],
) -> None:
    """Ensure training and serving use exactly the same schema."""

    actual = list(columns)

    if actual != list(FEATURE_COLUMNS):
        raise DataQualityError(
            "Feature schema mismatch. "
            f"Expected {list(FEATURE_COLUMNS)}, "
            f"got {actual}"
        )