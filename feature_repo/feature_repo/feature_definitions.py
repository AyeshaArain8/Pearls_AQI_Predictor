"""Production Feast definitions for Lahore AQI, backed by cloud PostgreSQL."""

from datetime import timedelta
import sys
from pathlib import Path

from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource
from feast.types import Float32, Int64
from feast.value_type import ValueType


# ---------------------------------------------------------
# Make project root importable when Feast loads this file
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Shared feature contract
# ---------------------------------------------------------
from src.feature_contract import FEATURE_COLUMNS, RAW_FEATURES


# ---------------------------------------------------------
# Lahore entity
# ---------------------------------------------------------
lahore = Entity(
    name="city",
    join_keys=["city"],
    value_type=ValueType.STRING,
)


# ---------------------------------------------------------
# Cloud PostgreSQL source
# ---------------------------------------------------------
lahore_source = PostgreSQLSource(
    name="lahore_aqi_observations_source",
    table="aqi_observations",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)


# ---------------------------------------------------------
# Feature data types
# ---------------------------------------------------------
FEATURE_TYPES = {
    name: Float32
    for name in RAW_FEATURES
    + [
        "aqi_lag1",
        "aqi_lag2",
        "aqi_lag3",
        "aqi_rolling_mean",
    ]
}

FEATURE_TYPES.update(
    {
        name: Int64
        for name in [
            "month",
            "day",
            "day_of_week",
        ]
    }
)


# ---------------------------------------------------------
# Feast Feature View
# ---------------------------------------------------------
lahore_aqi_features = FeatureView(
    name="lahore_aqi_features",
    entities=[lahore],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="aqi", dtype=Float32)
    ]
    + [
        Field(
            name=name,
            dtype=FEATURE_TYPES[name],
        )
        for name in FEATURE_COLUMNS
    ],
    source=lahore_source,
    online=True,
)