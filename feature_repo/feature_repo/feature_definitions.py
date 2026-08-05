from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from feast.value_type import ValueType


# Entity
aqi = Entity(
    name="aqi_id",
    join_keys=["aqi_id"],
    value_type=ValueType.INT64,
)


# Data Source
aqi_source = FileSource(
    path="data/forecast_dataset.parquet",
    timestamp_field="timestamp",
)


# Feature View
aqi_features = FeatureView(
    name="aqi_features",
    entities=[aqi],
    ttl=timedelta(days=30),
    schema=[
        Field(name="city", dtype=String),
        Field(name="latitude", dtype=Float32),
        Field(name="longitude", dtype=Float32),

        Field(name="pm10", dtype=Float32),
        Field(name="pm2_5", dtype=Float32),
        Field(name="carbon_monoxide", dtype=Float32),
        Field(name="nitrogen_dioxide", dtype=Float32),
        Field(name="sulphur_dioxide", dtype=Float32),
        Field(name="ozone", dtype=Float32),
        Field(name="aerosol_optical_depth", dtype=Float32),
        Field(name="dust", dtype=Float32),
        Field(name="uv_index", dtype=Float32),

        Field(name="temperature", dtype=Float32),
        Field(name="humidity", dtype=Float32),
        Field(name="wind_speed", dtype=Float32),

        Field(name="hour", dtype=Int64),
        Field(name="year", dtype=Int64),
        Field(name="month", dtype=Int64),
        Field(name="day", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),

        Field(name="aqi_lag1", dtype=Float32),
        Field(name="aqi_lag2", dtype=Float32),
        Field(name="aqi_lag3", dtype=Float32),
        Field(name="aqi_rolling_mean", dtype=Float32),
    ],
    source=aqi_source,
    online=True,
)