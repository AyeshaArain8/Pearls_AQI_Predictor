"""Lahore inference from cloud history and the versioned Model Registry.

"""

from datetime import datetime, timezone
import pandas as pd
from src.feature_contract import (
    FEATURE_COLUMNS,
    LAHORE,
    SCHEMA_VERSION,
    assert_feature_schema,
    resample_daily,
    make_daily_feature_rows,
)
from src.feature_store import historical_features
from src.model_registry import load_latest


AQI_THRESHOLDS = (
    (50, "Good"),
    (100, "Moderate"),
    (150, "Unhealthy for Sensitive Groups"),
    (200, "Unhealthy"),
    (300, "Very Unhealthy"),
    (float("inf"), "Hazardous"),
)


def get_category(aqi: float) -> str:
    """Return the centralized AQI category."""
    return next(
        label
        for limit, label in AQI_THRESHOLDS
        if aqi <= limit
    )


def forecast_aqi(city: str = LAHORE["name"]) -> dict:
    """Generate a three-day (24h/48h/72h) Lahore AQI forecast."""

    if city != LAHORE["name"]:
        raise ValueError(
            "Only Lahore is supported for AQI forecasting."
        )

    # Models are cached by the Model Registry.
    models, metadata = load_latest()

    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Registered model schema is incompatible "
            "with the current feature contract."
        )

    assert_feature_schema(
        metadata.get("feature_columns", [])
    )

    # Build today's real daily feature vector fresh from cloud history,
    # the same way training built its rows.
    history = historical_features()
    daily_history = resample_daily(history)
    daily_rows = make_daily_feature_rows(daily_history, include_targets=False)

    if daily_rows.empty:
        raise ValueError(
            "Not enough consecutive daily Lahore history yet to build "
            "a real daily feature vector."
        )

    features = daily_rows.iloc[[-1]][list(FEATURE_COLUMNS)]
    assert_feature_schema(features.columns)

    observed_at = datetime.now(timezone.utc)

    forecasts = []

    for index, horizon in enumerate(
        ("day1", "day2", "day3"),
        start=1,
    ):
        prediction = models[horizon].predict(features)[0]
        aqi = round(float(prediction), 1)

        forecasts.append(
            {
                "date": (
                    daily_rows["timestamp"].iloc[-1]
                    + pd.Timedelta(days=index)
                ).date().isoformat(),
                "aqi": aqi,
                "category": get_category(aqi),
                "hazardous": aqi > 300,
            }
        )

    return {
        "city": city,
        "model_version": metadata["version"],
        "observed_at": observed_at.isoformat(),
        "last_daily_observation": daily_rows["timestamp"].iloc[-1].isoformat(),
        "forecasts": forecasts,
        "metrics": metadata.get("metrics", {}),
        "features": features.iloc[0].to_dict(),
    }

