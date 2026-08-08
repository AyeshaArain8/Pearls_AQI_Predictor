"""Small local feature store; portable, free, and deliberately explicit."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from src.feature_contract import LAHORE, RAW_FEATURES, validate_observations

ROOT = Path(__file__).resolve().parents[1]
STORE_PATH = ROOT / "data" / "feature_store" / "lahore_observations.csv"
METADATA_PATH = ROOT / "data" / "feature_store" / "schema.json"
HISTORICAL_METADATA_PATH = ROOT / "data" / "historical" / "metadata.json"


def initialise_from_historical(historical_csv: Path | None = None) -> pd.DataFrame:
    source = historical_csv or ROOT / "data" / "historical" / "air_quality_historical.csv"
    provenance = _historical_provenance()
    if provenance.get("geographic_scope") != "Lahore, Pakistan" or provenance.get("lahore_verified") is not True or not provenance.get("provenance"):
        raise ValueError(
            "Historical dataset provenance is not verified for Lahore. "
            "Update data/historical/metadata.json with an auditable Lahore source before training."
        )
    raw = pd.read_csv(source)
    raw = raw.rename(columns={"date": "timestamp", "us_aqi": "aqi"})
    raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
    rows = raw[["timestamp", "aqi", *RAW_FEATURES]].dropna().sort_values("timestamp")
    rows = rows.drop_duplicates("timestamp").reset_index(drop=True)
    return _write(rows)


def _write(observations: pd.DataFrame, *, source: str = "verified historical Lahore dataset") -> pd.DataFrame:
    checked = validate_observations(observations)
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save = checked.copy()
    save["timestamp"] = save.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    save.to_csv(STORE_PATH, index=False)
    metadata = {"city": LAHORE, "observation_columns": ["timestamp", "aqi", *RAW_FEATURES], "source": source}
    if source == "verified historical Lahore dataset":
        metadata["historical_provenance"] = _historical_provenance()
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return checked


def load_observations() -> pd.DataFrame:
    if not STORE_PATH.exists():
        raise FileNotFoundError("Feature Store is empty. Run the hourly feature pipeline until it has enough real Lahore observations.")
    if not METADATA_PATH.exists():
        raise ValueError("Feature Store metadata is missing; refusing to use data of unknown city/provenance.")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("city") != LAHORE or metadata.get("source") not in {"verified historical Lahore dataset", "OpenWeather Lahore live observations"}:
        raise ValueError("Feature Store city/provenance is not approved for Lahore serving.")
    rows = pd.read_csv(STORE_PATH)
    return validate_observations(rows)


def assert_lahore_training_data() -> None:
    """Training must not use historical data whose Lahore provenance is unknown."""
    provenance = _historical_provenance()
    if provenance.get("geographic_scope") != "Lahore, Pakistan" or provenance.get("lahore_verified") is not True or not provenance.get("provenance"):
        raise ValueError("Training blocked: historical AQI geography is unverified. See data/historical/metadata.json.")


def _historical_provenance() -> dict:
    return json.loads(HISTORICAL_METADATA_PATH.read_text(encoding="utf-8"))


def append_observation(observation: dict) -> pd.DataFrame:
    candidate = pd.DataFrame([observation])
    if not STORE_PATH.exists():
        return _write(candidate, source="OpenWeather Lahore live observations")
    existing = load_observations()
    combined = pd.concat([existing, candidate], ignore_index=True).sort_values("timestamp")
    combined = combined.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return _write(combined, source="OpenWeather Lahore live observations")
