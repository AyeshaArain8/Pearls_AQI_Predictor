"""Production Feast definitions for Lahore AQI, backed by cloud PostgreSQL."""
from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource
from feast.types import Float32, Int64
from feast.value_type import ValueType

from src.feature_contract import FEATURE_COLUMNS, RAW_FEATURES

lahore = Entity(name="city", join_keys=["city"], value_type=ValueType.STRING)

lahore_source = PostgreSQLSource(
    name="lahore_aqi_observations_source",
    table="aqi_observations",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

FEATURE_TYPES = {name: Float32 for name in RAW_FEATURES + ["aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_rolling_mean"]}
FEATURE_TYPES.update({name: Int64 for name in ["month", "day", "day_of_week"]})

lahore_aqi_features = FeatureView(
    name="lahore_aqi_features",
    entities=[lahore],
    ttl=timedelta(days=3650),
    schema=[Field(name="aqi", dtype=Float32)] + [Field(name=name, dtype=FEATURE_TYPES[name]) for name in FEATURE_COLUMNS],
    source=lahore_source,
    online=True,
)
