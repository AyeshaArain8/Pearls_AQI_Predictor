"""Controlled Lahore historical backfill. Defaults to dry-run and never uses AQI categories."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.feature_contract import FEATURE_COLUMNS, LAHORE, RAW_FEATURES, make_feature_rows, pm25_to_us_aqi, to_utc_datetime, validate_observations

DEFAULT_DATASET = Path("data/historical/Backup/old_pakistan_air_quality_final_clean.csv")


def prepare_backfill(dataset: Path = DEFAULT_DATASET) -> pd.DataFrame:
    """Prepare a valid, chronological Lahore-only payload without writing to Neon."""
    source = pd.read_csv(dataset)
    lahore = source.loc[source["city"].eq(LAHORE["name"]), ["timestamp", *RAW_FEATURES]].copy()
    lahore["timestamp"] = pd.to_datetime(lahore["timestamp"], utc=True, errors="coerce")
    # Validate source pollutants before deriving the only permitted numeric AQI target.
    validate_observations(lahore.assign(aqi=0.0))
    lahore["aqi"] = lahore["pm2_5"].map(pm25_to_us_aqi)
    lahore = lahore.sort_values("timestamp").reset_index(drop=True)
    validated = validate_observations(lahore[["timestamp", "aqi", *RAW_FEATURES]])
    if validated.duplicated("timestamp").any():
        raise ValueError("Historical Lahore data contains duplicate timestamps.")
    derived = make_feature_rows(validated, include_targets=False).set_index("timestamp")
    payload = validated.rename(columns={"timestamp": "event_timestamp"}).copy()
    payload["event_timestamp"] = payload["event_timestamp"].map(to_utc_datetime)
    payload["created_timestamp"] = to_utc_datetime(pd.Timestamp.now(tz="UTC"))
    payload["city"] = LAHORE["name"]
    for column in [name for name in FEATURE_COLUMNS if name not in RAW_FEATURES]:
        payload[column] = derived[column].reindex(pd.to_datetime(validated["timestamp"], utc=True)).to_numpy()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write the prepared rows to Neon after review.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    payload = prepare_backfill(args.dataset)
    print(f"Dry-run: {len(payload)} Lahore rows; {payload.event_timestamp.nunique()} unique city/timestamp keys.")
    print(f"Rows with complete real lag features: {payload[FEATURE_COLUMNS].dropna().shape[0]}.")
    if not args.apply:
        print("No Neon write performed. Re-run with --apply only after approving this dry-run.")
        return
    from src.feature_store import insert_observations_conflict_safe
    inserted = insert_observations_conflict_safe(payload)
    print(f"Inserted {inserted} rows; duplicates were skipped by the Neon unique index.")


if __name__ == "__main__":
    main()
