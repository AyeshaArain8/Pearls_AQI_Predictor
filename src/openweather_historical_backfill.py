"""Controlled OpenWeather Lahore historical backfill.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from src.feature_contract import (
    LAHORE,
    RAW_FEATURES,
    DataQualityError,
    make_daily_feature_rows,
    pm25_to_us_aqi,
    resample_daily,
    validate_observations,
)


load_dotenv()


# ============================================================
# CONSTANTS
# ============================================================

HISTORY_URL = (
    "https://api.openweathermap.org/data/2.5/air_pollution/history"
)

OPENWEATHER_FIELD_MAP = {
    "pm10": "pm10",
    "pm2_5": "pm2_5",
    "co": "carbon_monoxide",
    "no2": "nitrogen_dioxide",
    "so2": "sulphur_dioxide",
    "o3": "ozone",
}

DEFAULT_GAP_START = "2026-02-05T00:00:00Z"
DEFAULT_GAP_END = "2026-08-08T13:00:00Z"

DEFAULT_CHUNK_HOURS = 24 * 7

MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 30

FEAST_ROW_COLUMNS = [
    "event_timestamp",
    "created_timestamp",
    "city",
    "aqi",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "hour",
    "month",
    "day",
    "day_of_week",
    "aqi_lag1",
    "aqi_lag2",
    "aqi_lag3",
    "aqi_rolling_mean",
]


# ============================================================
# FETCHING
# ============================================================

@dataclass
class ChunkResult:
    start: pd.Timestamp
    end: pd.Timestamp
    records: list = field(default_factory=list)
    error: Optional[str] = None


def _parse_utc(value: str) -> pd.Timestamp:
    """Parse an ISO timestamp and normalize it to UTC."""

    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp


def _chunk_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_hours: int,
):
    """Yield non-overlapping UTC request windows."""

    if end < start:
        raise ValueError("end must not be before start.")

    if chunk_hours <= 0:
        raise ValueError(
            "chunk_hours must be greater than zero."
        )

    step = pd.Timedelta(hours=chunk_hours)

    cursor = start

    while cursor <= end:
        chunk_end = min(
            cursor + step - pd.Timedelta(hours=1),
            end,
        )

        yield cursor, chunk_end

        cursor = (
            chunk_end
            + pd.Timedelta(hours=1)
        )


def _get_api_key() -> str:
    """Read the OpenWeather API key from the environment."""

    key = (
        os.getenv("OPENWEATHER_API_KEY")
        or os.getenv("API_KEY")
    )

    if not key:
        raise ValueError(
            "OPENWEATHER_API_KEY is required. "
            "Set it locally in .env."
        )

    return key


def _fetch_chunk(
    session: requests.Session,
    api_key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> ChunkResult:
    """Fetch one OpenWeather historical air-pollution chunk."""

    start_epoch = int(start.timestamp())
    end_epoch = int(end.timestamp())

    params = {
        "lat": LAHORE["latitude"],
        "lon": LAHORE["longitude"],
        "start": start_epoch,
        "end": end_epoch,
        "appid": api_key,
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                HISTORY_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            response.raise_for_status()

            payload = response.json()

            records = payload.get("list", [])

            return ChunkResult(
                start=start,
                end=end,
                records=records,
            )

        except Exception as exc:
            last_error = str(exc)

            if attempt < MAX_RETRIES:
                time.sleep(
                    RETRY_BACKOFF_SECONDS * attempt
                )

    return ChunkResult(
        start=start,
        end=end,
        records=[],
        error=last_error,
    )


def _extract_records(
    chunk: ChunkResult,
) -> list[dict]:
    """Convert OpenWeather response records to project schema."""

    output = []

    for item in chunk.records:
        timestamp_value = item.get("dt")

        if timestamp_value is None:
            continue

        components = item.get("components") or {}

        row = {
            "timestamp": pd.to_datetime(
                timestamp_value,
                unit="s",
                utc=True,
            )
        }

        for source_name, target_name in (
            OPENWEATHER_FIELD_MAP.items()
        ):
            row[target_name] = components.get(
                source_name
            )

        output.append(row)

    return output


def fetch_openweather_history(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    chunk_hours: int = DEFAULT_CHUNK_HOURS,
    api_key: Optional[str] = None,
) -> tuple[list[dict], list[dict]]:
    """Fetch historical OpenWeather observations."""

    api_key = api_key or _get_api_key()

    all_records = []
    failed_chunks = []

    session = requests.Session()

    windows = list(
        _chunk_windows(
            start,
            end,
            chunk_hours,
        )
    )

    for index, (
        chunk_start,
        chunk_end,
    ) in enumerate(windows):

        print(
            "Fetching OpenWeather historical chunk "
            f"{index + 1}/{len(windows)}: "
            f"{chunk_start.isoformat()} -> "
            f"{chunk_end.isoformat()}"
        )

        result = _fetch_chunk(
            session,
            api_key,
            chunk_start,
            chunk_end,
        )

        if result.error:
            failed_chunks.append(
                {
                    "start": chunk_start,
                    "end": chunk_end,
                    "error": result.error,
                }
            )
            continue

        all_records.extend(
            _extract_records(result)
        )

    return all_records, failed_chunks


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _row_is_valid(row: pd.Series) -> bool:
    """
    Validate one OpenWeather observation.

    Timestamp is validated separately.
    Only pollutant values are converted to float.
    """

    if "timestamp" not in row.index:
        return False

    timestamp = row["timestamp"]

    if pd.isna(timestamp):
        return False

    try:
        parsed_timestamp = pd.Timestamp(timestamp)

        if parsed_timestamp.tzinfo is None:
            parsed_timestamp = (
                parsed_timestamp.tz_localize("UTC")
            )
        else:
            parsed_timestamp = (
                parsed_timestamp.tz_convert("UTC")
            )

    except Exception:
        return False

    for column in RAW_FEATURES:

        if column not in row.index:
            return False

        value = row[column]

        if pd.isna(value):
            return False

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return False

        if numeric_value < 0:
            return False

    return True


def _missing_hour_blocks(
    expected_grid: pd.DatetimeIndex,
    actual_timestamps: set,
) -> list[dict]:
    """Return consecutive blocks of missing hourly observations."""

    missing = [
        timestamp
        for timestamp in expected_grid
        if timestamp not in actual_timestamps
    ]

    if not missing:
        return []

    blocks = []

    block_start = missing[0]
    previous = missing[0]

    for timestamp in missing[1:]:

        if timestamp == (
            previous + pd.Timedelta(hours=1)
        ):
            previous = timestamp
            continue

        blocks.append(
            {
                "start": block_start,
                "end": (
                    previous
                    + pd.Timedelta(hours=1)
                ),
                "hours": int(
                    (
                        previous - block_start
                    ).total_seconds()
                    / 3600
                    + 1
                ),
            }
        )

        block_start = timestamp
        previous = timestamp

    blocks.append(
        {
            "start": block_start,
            "end": (
                previous
                + pd.Timedelta(hours=1)
            ),
            "hours": int(
                (
                    previous - block_start
                ).total_seconds()
                / 3600
                + 1
            ),
        }
    )

    return blocks


def compute_gap_safe_hourly_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute lag/rolling features only inside consecutive
    hourly observation blocks.
    """

    if frame.empty:
        return frame.copy()

    working = (
        frame.copy()
        .sort_values("timestamp")
        .drop_duplicates(
            subset="timestamp",
            keep="first",
        )
        .reset_index(drop=True)
    )

    working["timestamp"] = pd.to_datetime(
        working["timestamp"],
        utc=True,
    )

    result_parts = []

    group_start = 0

    for index in range(1, len(working)):

        previous = working.loc[
            index - 1,
            "timestamp",
        ]

        current = working.loc[
            index,
            "timestamp",
        ]

        if current != (
            previous
            + pd.Timedelta(hours=1)
        ):

            result_parts.append(
                working.iloc[
                    group_start:index
                ].copy()
            )

            group_start = index

    result_parts.append(
        working.iloc[group_start:].copy()
    )

    processed_parts = []

    for part in result_parts:

        part = part.copy()

        part["aqi_lag1"] = (
            part["aqi"].shift(1)
        )

        part["aqi_lag2"] = (
            part["aqi"].shift(2)
        )

        part["aqi_lag3"] = (
            part["aqi"].shift(3)
        )

        part["aqi_rolling_mean"] = (
            part["aqi"]
            .shift(1)
            .rolling(
                window=3,
                min_periods=3,
            )
            .mean()
        )

        part["hour"] = (
            part["timestamp"].dt.hour
        )

        part["month"] = (
            part["timestamp"].dt.month
        )

        part["day"] = (
            part["timestamp"].dt.day
        )

        part["day_of_week"] = (
            part["timestamp"].dt.dayofweek
        )

        processed_parts.append(part)

    return (
        pd.concat(
            processed_parts,
            ignore_index=True,
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# ============================================================
# CLOUD HISTORY
# ============================================================

def _read_cloud_history() -> tuple[pd.DataFrame, bool]:
    """
    Read existing cloud history using the project's
    read-only historical retrieval helper.
    """

    try:
        from src.feature_store import (
            historical_features
        )
    except Exception:
        return pd.DataFrame(), False

    try:
        history = historical_features()
    except Exception:
        return pd.DataFrame(), False

    if history is None:
        return pd.DataFrame(), True

    return history.copy(), True


def _normalise_history_timestamp(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return a copy with a normalized timestamp column."""

    if frame is None or frame.empty:
        return pd.DataFrame()

    working = frame.copy()

    if "timestamp" not in working.columns:

        if "event_timestamp" in working.columns:

            working["timestamp"] = pd.to_datetime(
                working["event_timestamp"],
                utc=True,
                errors="coerce",
            )

        else:
            return pd.DataFrame()

    else:

        working["timestamp"] = pd.to_datetime(
            working["timestamp"],
            utc=True,
            errors="coerce",
        )

    working = working.dropna(
        subset=["timestamp"]
    )

    return working


def _usable_daily_row_count(
    frame: pd.DataFrame,
) -> int:
    """Count usable daily rows through the existing project pipeline."""

    if frame is None or frame.empty:
        return 0

    working = _normalise_history_timestamp(
        frame
    )

    if working.empty:
        return 0

    required = [
        "timestamp",
        "aqi",
        *RAW_FEATURES,
    ]

    if not all(
        column in working.columns
        for column in required
    ):
        return 0

    try:
        validated = validate_observations(
            working[required]
        )
    except Exception:
        return 0

    if validated.empty:
        return 0

    try:
        daily = make_daily_feature_rows(
            resample_daily(validated)
        )
        return len(daily)

    except Exception:

        try:
            daily = resample_daily(
                validated
            )
            return len(daily)

        except Exception:
            return 0


# ============================================================
# REPORT
# ============================================================

@dataclass
class BackfillDryRunReport:

    total_fetched: int
    valid_records: int
    invalid_records: int
    duplicate_timestamps: int
    missing_expected_timestamps: int

    earliest_timestamp: Optional[str]
    latest_timestamp: Optional[str]

    expected_hourly_coverage: int
    actual_coverage: int
    coverage_pct: float

    overlap_with_cloud: Optional[int]
    new_rows_to_insert: int
    overlap_checked: bool

    failed_chunks: list
    missing_hour_blocks: list

    current_daily_rows: Optional[int]
    combined_daily_rows: Optional[int]
    daily_rows_gain: Optional[int]

    prepared_feast_rows: int
    rows_removed_for_unsafe_lags: int

    safe_to_backfill: bool

    notes: list[str] = field(
        default_factory=list
    )

    def print_report(self) -> None:

        print("=" * 72)
        print(
            "OpenWeather Lahore Historical Backfill"
        )
        print(
            "VALIDATION REPORT"
        )
        print("=" * 72)

        print(
            "Lahore coordinates: "
            f"{LAHORE['latitude']}, "
            f"{LAHORE['longitude']}"
        )

        print(
            f"Total records fetched:          "
            f"{self.total_fetched}"
        )

        print(
            f"Valid records:                  "
            f"{self.valid_records}"
        )

        print(
            f"Invalid records:                "
            f"{self.invalid_records}"
        )

        print(
            f"Duplicate timestamps:           "
            f"{self.duplicate_timestamps}"
        )

        print(
            f"Missing expected timestamps:    "
            f"{self.missing_expected_timestamps}"
        )

        print(
            f"Earliest timestamp:             "
            f"{self.earliest_timestamp}"
        )

        print(
            f"Latest timestamp:               "
            f"{self.latest_timestamp}"
        )

        print(
            f"Expected hourly coverage:       "
            f"{self.expected_hourly_coverage} hours"
        )

        print(
            f"Actual coverage:                "
            f"{self.actual_coverage} hours "
            f"({self.coverage_pct:.1f}%)"
        )

        if self.overlap_checked:

            print(
                "Overlap with existing cloud data: "
                f"{self.overlap_with_cloud}"
            )

            print(
                "New rows to insert:               "
                f"{self.new_rows_to_insert}"
            )

        else:

            print(
                "Overlap with existing cloud data: "
                "NOT CHECKED"
            )

            print(
                "New rows to insert:               "
                "NOT AVAILABLE"
            )

        print("-" * 72)

        print("Missing-hour blocks:")

        if self.missing_hour_blocks:

            for block in self.missing_hour_blocks:

                print(
                    f"  - "
                    f"{block['start'].isoformat()} "
                    f"-> "
                    f"{block['end'].isoformat()} "
                    f"({block['hours']} hour(s))"
                )

        else:

            print("  None")

        print("-" * 72)

        print("Daily training-row impact:")

        print(
            f"  Current daily rows:           "
            f"{self.current_daily_rows}"
        )

        print(
            f"  Combined daily rows:          "
            f"{self.combined_daily_rows}"
        )

        if self.daily_rows_gain is not None:

            print(
                f"  Daily row change:             "
                f"{self.daily_rows_gain:+d}"
            )

        else:

            print(
                "  Daily row change:             N/A"
            )

        print("-" * 72)

        print(
            f"Prepared Feast-compatible rows: "
            f"{self.prepared_feast_rows}"
        )

        print(
            f"Rows removed for unsafe lags:   "
            f"{self.rows_removed_for_unsafe_lags}"
        )

        print("-" * 72)

        for note in self.notes:

            print(
                f"NOTE: {note}"
            )

        print("-" * 72)

        print(
            f"SAFE TO BACKFILL: "
            f"{self.safe_to_backfill}"
        )


# ============================================================
# PREPARATION
# ============================================================

def _empty_payload() -> pd.DataFrame:
    """Return an empty Feast-compatible payload."""

    return pd.DataFrame(
        columns=FEAST_ROW_COLUMNS
    )


def prepare_backfill_dry_run(
    start: str = DEFAULT_GAP_START,
    end: str = DEFAULT_GAP_END,
    *,
    chunk_hours: int = DEFAULT_CHUNK_HOURS,
    api_key: Optional[str] = None,
) -> tuple[pd.DataFrame, BackfillDryRunReport]:

    start_ts = _parse_utc(start)
    end_ts = _parse_utc(end)

    if end_ts < start_ts:
        raise ValueError(
            "Backfill end timestamp must not be before start timestamp."
        )

    fetch_start = start_ts.floor("h")
    fetch_end = end_ts.floor("h")

    raw_records, failed_chunks = (
        fetch_openweather_history(
            fetch_start,
            fetch_end,
            chunk_hours=chunk_hours,
            api_key=api_key,
        )
    )

    total_fetched = len(raw_records)

    frame = pd.DataFrame(raw_records)

    if frame.empty:

        report = BackfillDryRunReport(
            total_fetched=total_fetched,
            valid_records=0,
            invalid_records=0,
            duplicate_timestamps=0,
            missing_expected_timestamps=0,
            earliest_timestamp=None,
            latest_timestamp=None,
            expected_hourly_coverage=0,
            actual_coverage=0,
            coverage_pct=0.0,
            overlap_with_cloud=None,
            new_rows_to_insert=0,
            overlap_checked=False,
            failed_chunks=failed_chunks,
            missing_hour_blocks=[],
            current_daily_rows=None,
            combined_daily_rows=None,
            daily_rows_gain=None,
            prepared_feast_rows=0,
            rows_removed_for_unsafe_lags=0,
            safe_to_backfill=False,
            notes=[
                "No OpenWeather records were returned."
            ],
        )

        return _empty_payload(), report

    # --------------------------------------------------------
    # Normalize timestamp
    # --------------------------------------------------------

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Remove invalid timestamp rows
    # --------------------------------------------------------

    invalid_timestamp_mask = (
        frame["timestamp"].isna()
    )

    invalid_timestamp_count = int(
        invalid_timestamp_mask.sum()
    )

    frame = frame.loc[
        ~invalid_timestamp_mask
    ].copy()

    frame = (
        frame
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Keep only timestamps inside the requested range.

    frame = frame.loc[
        frame["timestamp"].between(
            fetch_start,
            fetch_end,
            inclusive="both",
        )
    ].copy()

    # --------------------------------------------------------
    # Duplicate detection
    # --------------------------------------------------------

    duplicate_mask = (
        frame["timestamp"]
        .duplicated(keep="first")
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    frame = frame.loc[
        ~duplicate_mask
    ].copy()

    # --------------------------------------------------------
    # Pollutant validation
    # --------------------------------------------------------

    valid_mask = frame.apply(
        _row_is_valid,
        axis=1,
    )

    valid_frame = frame.loc[
        valid_mask
    ].copy()

    invalid_count = (
        invalid_timestamp_count
        + int((~valid_mask).sum())
    )

    if valid_frame.empty:

        report = BackfillDryRunReport(
            total_fetched=total_fetched,
            valid_records=0,
            invalid_records=invalid_count,
            duplicate_timestamps=duplicate_count,
            missing_expected_timestamps=0,
            earliest_timestamp=None,
            latest_timestamp=None,
            expected_hourly_coverage=0,
            actual_coverage=0,
            coverage_pct=0.0,
            overlap_with_cloud=None,
            new_rows_to_insert=0,
            overlap_checked=False,
            failed_chunks=failed_chunks,
            missing_hour_blocks=[],
            current_daily_rows=None,
            combined_daily_rows=None,
            daily_rows_gain=None,
            prepared_feast_rows=0,
            rows_removed_for_unsafe_lags=0,
            safe_to_backfill=False,
            notes=[
                "No valid records survived pollutant validation."
            ],
        )

        return _empty_payload(), report

    # --------------------------------------------------------
    # PM2.5 -> US AQI
    # --------------------------------------------------------

    aqi_values = []
    aqi_valid_mask = []

    for value in valid_frame["pm2_5"]:

        try:

            aqi_values.append(
                pm25_to_us_aqi(value)
            )

            aqi_valid_mask.append(True)

        except DataQualityError:

            aqi_values.append(None)
            aqi_valid_mask.append(False)

    valid_frame["aqi"] = aqi_values

    aqi_failures = int(
        sum(
            1
            for ok in aqi_valid_mask
            if not ok
        )
    )

    invalid_count += aqi_failures

    valid_frame = valid_frame.loc[
        aqi_valid_mask
    ].reset_index(drop=True)

    if valid_frame.empty:

        report = BackfillDryRunReport(
            total_fetched=total_fetched,
            valid_records=0,
            invalid_records=invalid_count,
            duplicate_timestamps=duplicate_count,
            missing_expected_timestamps=0,
            earliest_timestamp=None,
            latest_timestamp=None,
            expected_hourly_coverage=0,
            actual_coverage=0,
            coverage_pct=0.0,
            overlap_with_cloud=None,
            new_rows_to_insert=0,
            overlap_checked=False,
            failed_chunks=failed_chunks,
            missing_hour_blocks=[],
            current_daily_rows=None,
            combined_daily_rows=None,
            daily_rows_gain=None,
            prepared_feast_rows=0,
            rows_removed_for_unsafe_lags=0,
            safe_to_backfill=False,
            notes=[
                "No valid records survived AQI conversion."
            ],
        )

        return _empty_payload(), report

    # --------------------------------------------------------
    # Existing project validation
    # --------------------------------------------------------

    validated_backfill = validate_observations(
        valid_frame[
            [
                "timestamp",
                "aqi",
                *RAW_FEATURES,
            ]
        ]
    )

    validated_backfill = (
        validated_backfill
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Expected hourly grid
    # --------------------------------------------------------

    expected_grid = pd.date_range(
        fetch_start,
        fetch_end,
        freq="h",
        tz="UTC",
    )

    expected_hourly_coverage = len(
        expected_grid
    )

    actual_timestamps = set(
        validated_backfill["timestamp"]
    )

    actual_coverage = len(
        actual_timestamps
    )

    missing_expected = len(
        set(expected_grid)
        - actual_timestamps
    )

    coverage_pct = (
        actual_coverage
        / expected_hourly_coverage
        * 100
        if expected_hourly_coverage
        else 0.0
    )

    missing_blocks = _missing_hour_blocks(
        expected_grid,
        actual_timestamps,
    )

    # --------------------------------------------------------
    # Existing cloud history
    # --------------------------------------------------------

    cloud_history, cloud_checked = (
        _read_cloud_history()
    )

    cloud_history = _normalise_history_timestamp(
        cloud_history
    )

    overlap_count = None

    if cloud_checked:

        if cloud_history.empty:

            overlap_count = 0

        else:

            cloud_timestamps = set(
                cloud_history["timestamp"]
            )

            overlap_count = int(
                validated_backfill["timestamp"]
                .isin(cloud_timestamps)
                .sum()
            )

    # --------------------------------------------------------
    # Merge in memory only
    # --------------------------------------------------------

    if cloud_history.empty:

        merged = validated_backfill.copy()

    else:

        required_merge_columns = [
            "timestamp",
            "aqi",
            *RAW_FEATURES,
        ]

        available_cloud_columns = [
            column
            for column in required_merge_columns
            if column in cloud_history.columns
        ]

        cloud_for_merge = cloud_history[
            available_cloud_columns
        ].copy()

        if (
            "timestamp" not in cloud_for_merge.columns
            or "aqi" not in cloud_for_merge.columns
        ):

            merged = validated_backfill.copy()

        else:

            merged = pd.concat(
                [
                    cloud_for_merge,
                    validated_backfill,
                ],
                ignore_index=True,
            )

            merged["timestamp"] = pd.to_datetime(
                merged["timestamp"],
                utc=True,
            )

            # Cloud history comes first deliberately:
            # an already-existing cloud observation wins over
            # a duplicate candidate from OpenWeather.

            merged = (
                merged
                .sort_values("timestamp")
                .drop_duplicates(
                    subset="timestamp",
                    keep="first",
                )
                .reset_index(drop=True)
            )

    # --------------------------------------------------------
    # Gap-safe lag calculation
    # --------------------------------------------------------

    gap_safe = (
        compute_gap_safe_hourly_features(
            merged
        )
    )

    backfill_timestamp_set = set(
        validated_backfill["timestamp"]
    )

    prepared = gap_safe[
        gap_safe["timestamp"].isin(
            backfill_timestamp_set
        )
    ].copy()

    lag_columns = [
        "aqi_lag1",
        "aqi_lag2",
        "aqi_lag3",
        "aqi_rolling_mean",
    ]

    complete_lag_mask = (
        prepared[lag_columns]
        .notna()
        .all(axis=1)
    )

    rows_removed_for_unsafe_lags = int(
        (~complete_lag_mask).sum()
    )

    prepared = prepared.loc[
        complete_lag_mask
    ].copy()

    # --------------------------------------------------------
    # Build Feast-compatible payload
    # --------------------------------------------------------

    if prepared.empty:

        payload = _empty_payload()

    else:

        prepared["event_timestamp"] = (
            prepared["timestamp"]
        )

        prepared["created_timestamp"] = (
            pd.Timestamp.now(tz="UTC")
        )

        prepared["city"] = LAHORE["name"]

        payload = prepared[
            FEAST_ROW_COLUMNS
        ].copy()

        payload = (
            payload
            .sort_values("event_timestamp")
            .drop_duplicates(
                subset=[
                    "city",
                    "event_timestamp",
                ],
                keep="first",
            )
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # Exclude existing cloud rows from the apply payload
    # --------------------------------------------------------

    new_rows_to_insert = len(payload)

    if cloud_checked and not cloud_history.empty:

        existing_timestamps = set(
            cloud_history["timestamp"]
        )

        if not payload.empty:

            new_mask = ~(
                payload["event_timestamp"]
                .isin(existing_timestamps)
            )

            payload = payload.loc[
                new_mask
            ].reset_index(drop=True)

    new_rows_to_insert = len(payload)

    # --------------------------------------------------------
    # Daily impact
    # --------------------------------------------------------

    current_daily_rows = None
    combined_daily_rows = None
    daily_rows_gain = None

    if cloud_checked:

        current_daily_rows = (
            _usable_daily_row_count(
                cloud_history
            )
        )

        combined_for_daily = merged.copy()

        combined_daily_rows = (
            _usable_daily_row_count(
                combined_for_daily
            )
        )

        daily_rows_gain = (
            combined_daily_rows
            - current_daily_rows
        )

    # --------------------------------------------------------
    # Notes
    # --------------------------------------------------------

    notes = []

    if duplicate_count:

        notes.append(
            f"{duplicate_count} duplicate timestamp(s) "
            "were found and collapsed."
        )

    if invalid_count:

        notes.append(
            f"{invalid_count} record(s) were excluded "
            "because of invalid/missing pollutant values, "
            "timestamps, or invalid PM2.5 AQI conversion."
        )

    if missing_expected:

        notes.append(
            f"{missing_expected} expected hourly timestamp(s) "
            "have no OpenWeather observation. "
            "They remain missing and were not interpolated."
        )

    if missing_blocks:

        largest_block = max(
            missing_blocks,
            key=lambda item: item["hours"],
        )

        notes.append(
            "Largest missing block is "
            f"{largest_block['hours']} hour(s): "
            f"{largest_block['start'].isoformat()} "
            "-> "
            f"{largest_block['end'].isoformat()}."
        )

    if failed_chunks:

        notes.append(
            f"{len(failed_chunks)} fetch chunk(s) failed "
            "after retries."
        )

    if not cloud_checked:

        notes.append(
            "Cloud history could not be read. "
            "Overlap and daily-impact checks "
            "were not available."
        )

    if overlap_count is not None:

        notes.append(
            f"{overlap_count} candidate timestamp(s) "
            "already exist in cloud history."
        )

    if rows_removed_for_unsafe_lags:

        notes.append(
            f"{rows_removed_for_unsafe_lags} backfill row(s) "
            "were excluded because lag/rolling features "
            "could not be proven to come from genuinely "
            "consecutive hourly observations."
        )

    if daily_rows_gain is not None:

        notes.append(
            "Combined daily pipeline changes usable rows by "
            f"{daily_rows_gain:+d}."
        )

    if not payload.empty:

        notes.append(
            f"{len(payload)} new Feast-compatible row(s) "
            "remain after excluding existing cloud timestamps."
        )

    # --------------------------------------------------------
    # Safety decision
    # --------------------------------------------------------

    safe_to_backfill = bool(
        cloud_checked
        and not failed_chunks
        and actual_coverage > 0
        and not payload.empty
    )

    if not cloud_checked:

        safe_to_backfill = False

        notes.append(
            "Marked NOT safe because existing cloud history "
            "could not be verified."
        )

    elif failed_chunks:

        safe_to_backfill = False

        notes.append(
            "Marked NOT safe because one or more "
            "OpenWeather fetch chunks failed."
        )

    elif payload.empty:

        safe_to_backfill = False

        notes.append(
            "Marked NOT safe because no new complete "
            "Feast-compatible rows were prepared."
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    report = BackfillDryRunReport(
        total_fetched=total_fetched,
        valid_records=actual_coverage,
        invalid_records=invalid_count,
        duplicate_timestamps=duplicate_count,
        missing_expected_timestamps=missing_expected,
        earliest_timestamp=(
            validated_backfill["timestamp"]
            .min()
            .isoformat()
            if not validated_backfill.empty
            else None
        ),
        latest_timestamp=(
            validated_backfill["timestamp"]
            .max()
            .isoformat()
            if not validated_backfill.empty
            else None
        ),
        expected_hourly_coverage=(
            expected_hourly_coverage
        ),
        actual_coverage=actual_coverage,
        coverage_pct=coverage_pct,
        overlap_with_cloud=overlap_count,
        new_rows_to_insert=new_rows_to_insert,
        overlap_checked=cloud_checked,
        failed_chunks=failed_chunks,
        missing_hour_blocks=missing_blocks,
        current_daily_rows=current_daily_rows,
        combined_daily_rows=combined_daily_rows,
        daily_rows_gain=daily_rows_gain,
        prepared_feast_rows=len(payload),
        rows_removed_for_unsafe_lags=(
            rows_removed_for_unsafe_lags
        ),
        safe_to_backfill=safe_to_backfill,
        notes=notes,
    )

    return payload, report


# ============================================================
# APPLY
# ============================================================

def apply_backfill(
    payload: pd.DataFrame,
) -> int:
    """
    Insert the reviewed payload using the existing
    conflict-safe Neon insertion helper.

    This function does not train or register a model.
    """

    if payload.empty:
        return 0

    from src.feature_store import (
        insert_observations_conflict_safe
    )

    inserted = insert_observations_conflict_safe(
        payload.copy()
    )

    return int(inserted)


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Prepare or apply the OpenWeather Lahore "
            "historical gap."
        )
    )

    parser.add_argument(
        "--start",
        default=DEFAULT_GAP_START,
        help="Gap start, ISO-8601 UTC.",
    )

    parser.add_argument(
        "--end",
        default=DEFAULT_GAP_END,
        help="Gap end, ISO-8601 UTC.",
    )

    parser.add_argument(
        "--chunk-hours",
        type=int,
        default=DEFAULT_CHUNK_HOURS,
        help="Hours per OpenWeather request.",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Write the reviewed safe payload to Neon. "
            "Default is dry-run only."
        ),
    )

    args = parser.parse_args()

    payload, report = (
        prepare_backfill_dry_run(
            args.start,
            args.end,
            chunk_hours=args.chunk_hours,
        )
    )

    report.print_report()

    print()

    print(
        f"Prepared Feast-compatible payload: "
        f"{len(payload)} rows."
    )

    if not args.apply:

        print(
            "DRY RUN: No file was written."
        )

        print(
            "DRY RUN: No database write was performed."
        )

        print(
            "DRY RUN: No Feast materialization was performed."
        )

        print(
            "No model training or registration was performed."
        )

        print(
            "To apply only after reviewing the report, run:"
        )

        print(
            "  python -m src.openweather_historical_backfill --apply"
        )

        return

    # --------------------------------------------------------
    # APPLY MODE
    # --------------------------------------------------------

    if not report.safe_to_backfill:

        print(
            "APPLY BLOCKED: The validation report did not "
            "mark this backfill as safe."
        )

        print(
            "No database write was performed."
        )

        return

    print(
        "APPLY MODE: Writing the reviewed payload to Neon..."
    )

    inserted = apply_backfill(payload)

    print(
        f"Inserted {inserted} new row(s) into Neon."
    )

    print(
        "Existing duplicate city/timestamp rows were "
        "protected by the conflict-safe insertion path."
    )

    print(
        "No model training or registration was performed."
    )

    print(
        "No models/registry/ files were modified."
    )

    print(
        "No reports/ files were modified."
    )


if __name__ == "__main__":
    main()