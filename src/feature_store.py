"""Cloud-only Feast access layer. No local feature data fallback exists."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import pandas as pd
from dotenv import load_dotenv
from psycopg import OperationalError as PsycopgOperationalError
from sqlalchemy import create_engine, text

from src.feature_contract import (
    FEATURE_COLUMNS,
    LAHORE,
    RAW_FEATURES,
    latest_serving_features,
    to_utc_datetime,
    validate_observations,
)

load_dotenv()

# ============================================================
# PATHS / CONSTANTS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEAST_REPO = (
    ROOT
    / "feature_repo"
    / "feature_repo"
)

TABLE = "aqi_observations"
VIEW = "lahore_aqi_features"

DEFAULT_HISTORICAL_BATCH_SIZE = 250

# ============================================================
# CLOUD CONFIGURATION
# ============================================================

def require_cloud_configuration() -> None:

    required = [
        "FEAST_POSTGRES_URL",
        "FEAST_POSTGRES_HOST",
        "FEAST_POSTGRES_PORT",
        "FEAST_POSTGRES_DATABASE",
        "FEAST_POSTGRES_SCHEMA",
        "FEAST_POSTGRES_USER",
        "FEAST_POSTGRES_PASSWORD",
    ]

    missing = [
        key
        for key in required
        if not os.getenv(key)
    ]

    if missing:
        raise RuntimeError(
            f"Cloud Feast configuration missing: "
            f"{', '.join(missing)}"
        )

    if (
        "supabase"
        in os.environ["FEAST_POSTGRES_HOST"].lower()
    ):
        raise RuntimeError(
            "Supabase is not an approved "
            "Feature Store backend for this project."
        )


# ============================================================
# POSTGRES URL
# ============================================================

def psycopg_url(database_url: str) -> str:

    if database_url.startswith(
        "postgresql://"
    ):
        return (
            "postgresql+psycopg://"
            + database_url.removeprefix(
                "postgresql://"
            )
        )

    return database_url


# ============================================================
# CLOUD DATABASE ENGINE
# ============================================================

def cloud_engine():

    require_cloud_configuration()

    return create_engine(
        psycopg_url(
            os.environ["FEAST_POSTGRES_URL"]
        ),
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=2,
    )


# ============================================================
# FEAST STORE
#
# Cached so Feast is initialized only once per Python process.
# This does NOT create local storage or change the cloud backend.
# ============================================================

@lru_cache(maxsize=1)
def feast_store():

    require_cloud_configuration()

    from feast import FeatureStore

    os.environ["FEAST_POSTGRES_URL"] = (
        psycopg_url(
            os.environ["FEAST_POSTGRES_URL"]
        )
    )

    return FeatureStore(
        repo_path=str(FEAST_REPO)
    )


# ============================================================
# FEAST RECONNECT-ON-STALE-CONNECTION
#
# feast_store() is cached for the life of the process (see above),
# which is correct for normal operation but means a long-lived
# Streamlit process can hold a Feast client whose underlying cloud
# PostgreSQL connection (registry and/or online store) has since
# been closed server-side after being idle between reruns
# (observed as psycopg.OperationalError: "SSL connection has been
# closed unexpectedly"). Unlike cloud_engine() above, Feast's own
# client has no pool_pre_ping equivalent, so a dead connection is
# only discovered when actually used. This helper runs a Feast
# operation against the cached store and, ONLY on that specific
# stale-connection error, clears the cache and retries once against
# a freshly constructed FeatureStore. It does not change Feast
# initialization otherwise and does not retry on unrelated errors.
# ============================================================

def _with_feast_reconnect(operation):
    try:
        return operation(feast_store())
    except PsycopgOperationalError:
        feast_store.cache_clear()
        return operation(feast_store())


# ============================================================
# HISTORICAL BATCH SIZE
# ============================================================

def historical_batch_size() -> int:

    value = int(
        os.getenv(
            "FEAST_HISTORICAL_BATCH_SIZE",
            DEFAULT_HISTORICAL_BATCH_SIZE,
        )
    )

    if value < 1:
        raise ValueError(
            "FEAST_HISTORICAL_BATCH_SIZE "
            "must be a positive integer."
        )

    return value


# ============================================================
# FULL CLOUD HISTORY
# Used for training/status where required
# ============================================================

def _raw_cloud_observations() -> pd.DataFrame:

    schema = os.getenv(
        "FEAST_POSTGRES_SCHEMA",
        "public",
    )

    query = text(
        f"""
        SELECT
            event_timestamp AS timestamp,
            aqi,
            {", ".join(RAW_FEATURES)}
        FROM {schema}.{TABLE}
        WHERE city = :city
        ORDER BY event_timestamp
        """
    )

    with cloud_engine().connect() as connection:

        frame = pd.read_sql(
            query,
            connection,
            params={
                "city": LAHORE["name"]
            },
        )

    if frame.empty:
        return frame

    return validate_observations(frame)


# ============================================================
# RECENT HISTORY
# Only the latest real observations are required for
# aqi_lag1, aqi_lag2, aqi_lag3 and rolling mean.
# ============================================================

def _recent_cloud_observations(
    limit: int = 3,
) -> pd.DataFrame:

    schema = os.getenv(
        "FEAST_POSTGRES_SCHEMA",
        "public",
    )

    query = text(
        f"""
        SELECT
            event_timestamp AS timestamp,
            aqi,
            {", ".join(RAW_FEATURES)}
        FROM {schema}.{TABLE}
        WHERE city = :city
        ORDER BY event_timestamp DESC
        LIMIT :limit
        """
    )

    with cloud_engine().connect() as connection:

        frame = pd.read_sql(
            query,
            connection,
            params={
                "city": LAHORE["name"],
                "limit": limit,
            },
        )

    if frame.empty:
        return frame

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
    )

    return (
        frame
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# ============================================================
# CHECK DUPLICATE TIMESTAMP
# ============================================================

def _observation_exists(
    timestamp,
) -> bool:

    schema = os.getenv(
        "FEAST_POSTGRES_SCHEMA",
        "public",
    )

    query = text(
        f"""
        SELECT 1
        FROM {schema}.{TABLE}
        WHERE city = :city
          AND event_timestamp = :timestamp
        LIMIT 1
        """
    )

    with cloud_engine().connect() as connection:

        result = connection.execute(
            query,
            {
                "city": LAHORE["name"],
                "timestamp": timestamp,
            },
        )

        return result.first() is not None


# ============================================================
# CLOUD STATUS
# ============================================================

def cloud_observation_status() -> dict:
    """
    Return the number and latest timestamp of genuine
    Lahore observations stored in cloud PostgreSQL.
    """

    schema = os.getenv(
        "FEAST_POSTGRES_SCHEMA",
        "public",
    )

    query = text(
        f"""
        SELECT
            COUNT(*) AS count,
            MAX(event_timestamp) AS latest_timestamp
        FROM {schema}.{TABLE}
        WHERE city = :city
        """
    )

    with cloud_engine().connect() as connection:

        result = connection.execute(
            query,
            {
                "city": LAHORE["name"]
            },
        ).mappings().one()

    latest = result["latest_timestamp"]

    return {
        "count": int(
            result["count"] or 0
        ),
        "chronological": True,
        "latest_timestamp": (
            None
            if latest is None
            else pd.Timestamp(
                latest
            ).isoformat()
        ),
    }


# ============================================================
# UNIQUE INDEX
# ============================================================

def ensure_observation_uniqueness(
    connection,
) -> None:

    schema = os.environ[
        "FEAST_POSTGRES_SCHEMA"
    ]

    connection.execute(
        text(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS
            {TABLE}_city_event_timestamp_key
            ON {schema}.{TABLE}
            (city, event_timestamp)
            """
        )
    )


# ============================================================
# CONFLICT-SAFE INSERT
# ============================================================

def insert_observations_conflict_safe(
    rows: pd.DataFrame,
) -> int:

    if rows.empty:
        return 0

    schema = os.environ[
        "FEAST_POSTGRES_SCHEMA"
    ]

    columns = list(rows.columns)

    placeholders = ", ".join(
        f":{column}"
        for column in columns
    )

    statement = text(
        f"""
        INSERT INTO {schema}.{TABLE}
        ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT
        (city, event_timestamp)
        DO NOTHING
        """
    )

    records = (
        rows.astype(object)
        .where(
            pd.notna(rows),
            None,
        )
        .to_dict(
            orient="records"
        )
    )

    with cloud_engine().begin() as connection:

        ensure_observation_uniqueness(
            connection
        )

        result = connection.execute(
            statement,
            records,
        )

    return result.rowcount or 0


# ============================================================
# BUILD LIVE OBSERVATION
# ============================================================

def build_live_observation_row(
    history: pd.DataFrame,
    observation: dict,
) -> pd.DataFrame:

    """
    Build one Lahore observation using only genuine
    prior cloud observations.

    No fake lag values are generated.
    """

    candidate = pd.DataFrame(
        [
            {
                key: observation[key]
                for key in [
                    "timestamp",
                    "aqi",
                    *RAW_FEATURES,
                ]
            }
        ]
    )

    candidate["timestamp"] = pd.to_datetime(
        candidate["timestamp"],
        utc=True,
    )

    serving = latest_serving_features(
        history,
        candidate.iloc[0].to_dict(),
    )

    lag_columns = [
        "aqi_lag1",
        "aqi_lag2",
        "aqi_lag3",
        "aqi_rolling_mean",
    ]

    if (
        serving.loc[
            :,
            lag_columns,
        ]
        .isna()
        .any(axis=None)
    ):

        raise ValueError(
            "Live observation has insufficient "
            "real prior AQI history for lag features."
        )

    row = candidate.rename(
        columns={
            "timestamp": "event_timestamp"
        }
    ).copy()

    row["event_timestamp"] = (
        row["event_timestamp"]
        .map(to_utc_datetime)
    )

    row["created_timestamp"] = (
        to_utc_datetime(
            pd.Timestamp.now(
                tz="UTC"
            )
        )
    )

    row["city"] = LAHORE["name"]

    for column in FEATURE_COLUMNS:

        row[column] = (
            serving.iloc[0][column]
        )

    return row


# ============================================================
# LIVE INGESTION
# ============================================================

def ingest_observation(
    observation: dict,
) -> pd.DataFrame:

    """
    Persist one Lahore observation remotely,
    calculate real lag features and update Feast
    online serving.

    Only recent cloud history is loaded.
    """

    if (
        observation.get("city")
        != LAHORE["name"]
    ):

        raise ValueError(
            "Only Lahore observations "
            "may enter the Feature Store."
        )

    # --------------------------------------------------------
    # Prepare timestamp
    # --------------------------------------------------------

    candidate_timestamp = pd.to_datetime(
        observation["timestamp"],
        utc=True,
    )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    if _observation_exists(
        candidate_timestamp
    ):

        return pd.DataFrame(
            [
                {
                    "event_timestamp":
                        candidate_timestamp,
                    "city":
                        LAHORE["name"],
                    "aqi":
                        observation["aqi"],
                }
            ]
        )

    # --------------------------------------------------------
    # Load only the latest real history
    # --------------------------------------------------------

    history_started = perf_counter()

    history = _recent_cloud_observations(
        limit=3
    )

    print(
        "Cloud ingestion: loaded "
        f"{len(history)} recent Lahore "
        "observations in "
        f"{perf_counter() - history_started:.2f}s.",
        flush=True,
    )

    # --------------------------------------------------------
    # Build real lag features
    # --------------------------------------------------------

    row = build_live_observation_row(
        history,
        observation,
    )

    # --------------------------------------------------------
    # Insert into cloud PostgreSQL
    # --------------------------------------------------------

    inserted = (
        insert_observations_conflict_safe(
            row
        )
    )

    if not inserted:
        return row

    # --------------------------------------------------------
    # Feast online materialization
    # --------------------------------------------------------

    materialize_started = perf_counter()

    def _materialize(store):
        store.materialize(
            start_date=start_date.to_pydatetime(),
            end_date=end_date.to_pydatetime(),
        )

    event_timestamp = pd.to_datetime(
        row["event_timestamp"].iloc[0],
        utc=True,
    )

    start_date = (
        event_timestamp
        - pd.Timedelta(minutes=1)
    )

    end_date = (
        event_timestamp
        + pd.Timedelta(minutes=1)
    )

    _with_feast_reconnect(_materialize)

    print(
        "Cloud ingestion: Feast online "
        "materialization completed in "
        f"{perf_counter() - materialize_started:.2f}s.",
        flush=True,
    )

    return row


# ============================================================
# HISTORICAL FEATURES FOR TRAINING
# ============================================================

def historical_features() -> pd.DataFrame:

    """
    Load genuine chronological Lahore history directly
    from indexed cloud PostgreSQL.

    Training derives lag and rolling features afterward
    through the shared feature contract.
    """

    retrieval_started = perf_counter()

    print(
        "Cloud historical retrieval: loading "
        "Lahore observations directly from "
        "indexed PostgreSQL.",
        flush=True,
    )

    schema = os.getenv(
        "FEAST_POSTGRES_SCHEMA",
        "public",
    )

    query = text(
        f"""
        SELECT
            event_timestamp AS timestamp,
            aqi,
            {", ".join(RAW_FEATURES)}
        FROM {schema}.{TABLE}
        WHERE city = :city
        ORDER BY event_timestamp
        """
    )

    with cloud_engine().connect() as connection:

        output = pd.read_sql(
            query,
            connection,
            params={
                "city": LAHORE["name"]
            },
        )

    if output.empty:
        return output

    output["timestamp"] = pd.to_datetime(
        output["timestamp"],
        utc=True,
    )

    output = (
        output
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    output = validate_observations(
        output
    )

    print(
        "Cloud historical retrieval: "
        f"returned {len(output)} Lahore rows "
        f"in {perf_counter() - retrieval_started:.2f}s.",
        flush=True,
    )

    return output[
        [
            "timestamp",
            "aqi",
            *RAW_FEATURES,
        ]
    ]


# ============================================================
# FEAST ONLINE FEATURES
# ============================================================

def latest_online_features() -> pd.DataFrame:

    """
    Retrieve the latest real serving vector
    from the remote Feast PostgreSQL online store.
    """

    fields = [
        f"{VIEW}:{name}"
        for name in FEATURE_COLUMNS
    ]

    def _fetch(store):
        return (
            store
            .get_online_features(
                features=fields,
                entity_rows=[
                    {
                        "city":
                            LAHORE["name"]
                    }
                ],
            )
            .to_dict()
        )

    values = _with_feast_reconnect(_fetch)

    frame = pd.DataFrame(
        {
            name: values.get(
                name,
                [None],
            )[0]
            for name in FEATURE_COLUMNS
        },
        index=[0],
    )

    if frame.isna().any(axis=None):

        raise ValueError(
            "Feast online store has insufficient "
            "real history to serve all lag features."
        )

    return frame.loc[
        :,
        FEATURE_COLUMNS,
    ]