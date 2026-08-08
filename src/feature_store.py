"""Cloud-only Feast access layer. No local feature data fallback exists."""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

from src.feature_contract import FEATURE_COLUMNS, LAHORE, RAW_FEATURES, make_feature_rows, validate_observations

ROOT = Path(__file__).resolve().parents[1]
FEAST_REPO = ROOT / "feature_repo" / "feature_repo"
TABLE = "aqi_observations"
VIEW = "lahore_aqi_features"

def require_cloud_configuration() -> None:
    required = ["FEAST_POSTGRES_URL", "FEAST_POSTGRES_HOST", "FEAST_POSTGRES_PORT", "FEAST_POSTGRES_DATABASE", "FEAST_POSTGRES_SCHEMA", "FEAST_POSTGRES_USER", "FEAST_POSTGRES_PASSWORD"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Cloud Feast configuration missing: {', '.join(missing)}")
    if "supabase" in os.environ["FEAST_POSTGRES_HOST"].lower():
        raise RuntimeError("Supabase is not an approved Feature Store backend for this project.")

def cloud_engine():
    require_cloud_configuration()
    return create_engine(os.environ["FEAST_POSTGRES_URL"], pool_pre_ping=True)

def feast_store():
    require_cloud_configuration()
    from feast import FeatureStore
    return FeatureStore(repo_path=str(FEAST_REPO))

def _raw_cloud_observations() -> pd.DataFrame:
    schema = os.getenv("FEAST_POSTGRES_SCHEMA", "public")
    query = text(f'SELECT event_timestamp AS timestamp, aqi, {", ".join(RAW_FEATURES)} FROM {schema}.{TABLE} WHERE city = :city ORDER BY event_timestamp')
    with cloud_engine().connect() as connection:
        frame = pd.read_sql(query, connection, params={"city": LAHORE["name"]})
    return validate_observations(frame) if not frame.empty else frame

def ingest_observation(observation: dict) -> pd.DataFrame:
    """Persist a Lahore observation remotely, then materialize it into Feast online serving."""
    if observation.get("city") != LAHORE["name"]:
        raise ValueError("Only Lahore observations may enter the Feature Store.")
    history = _raw_cloud_observations()
    candidate = pd.DataFrame([{key: observation[key] for key in ["timestamp", "aqi", *RAW_FEATURES]}])
    combined = pd.concat([history, candidate], ignore_index=True).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    combined = validate_observations(combined).reset_index(drop=True)
    derived = make_feature_rows(combined, include_targets=False)
    row = candidate.copy().rename(columns={"timestamp": "event_timestamp"})
    row["created_timestamp"] = pd.Timestamp.now(tz="UTC")
    row["city"] = LAHORE["name"]
    # Keep a stable remote table schema even before enough history exists for lags.
    for column in FEATURE_COLUMNS:
        if column not in row:
            row[column] = None
    if not derived.empty:
        latest = derived.iloc[-1]
        if pd.to_datetime(latest["timestamp"], utc=True) == pd.to_datetime(row.event_timestamp.iloc[0], utc=True):
            for column in FEATURE_COLUMNS:
                row[column] = latest[column]
    with cloud_engine().begin() as connection:
        row.to_sql(TABLE, connection, schema=os.environ["FEAST_POSTGRES_SCHEMA"], if_exists="append", index=False, method="multi")
    store = feast_store()
    store.materialize_incremental(end_date=pd.Timestamp.now(tz="UTC").to_pydatetime())
    return row

def historical_features() -> pd.DataFrame:
    """Retrieve cloud historical values through Feast's point-in-time API for training."""
    timestamps = _raw_cloud_observations()[["timestamp"]]
    if timestamps.empty:
        return pd.DataFrame()
    entities = timestamps.rename(columns={"timestamp": "event_timestamp"})
    entities["city"] = LAHORE["name"]
    fields = [f"{VIEW}:aqi"] + [f"{VIEW}:{name}" for name in RAW_FEATURES]
    output = feast_store().get_historical_features(entity_df=entities, features=fields).to_df()
    output = output.rename(columns={"event_timestamp": "timestamp"})
    return validate_observations(output[["timestamp", "aqi", *RAW_FEATURES]].sort_values("timestamp"))

def latest_online_features() -> pd.DataFrame:
    """Retrieve the latest serving vector from the remote Feast online store."""
    fields = [f"{VIEW}:{name}" for name in FEATURE_COLUMNS]
    values = feast_store().get_online_features(features=fields, entity_rows=[{"city": LAHORE["name"]}]).to_dict()
    frame = pd.DataFrame({name: values.get(name, [None])[0] for name in FEATURE_COLUMNS}, index=[0])
    if frame.isna().any(axis=None):
        raise ValueError("Feast online store has insufficient real history to serve all lag features.")
    return frame.loc[:, FEATURE_COLUMNS]
