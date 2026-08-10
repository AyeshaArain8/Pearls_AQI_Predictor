"""Cloud-only Feast access layer. No local feature data fallback exists."""
from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import pandas as pd
from dotenv import load_dotenv
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

ROOT = Path(__file__).resolve().parents[1]
FEAST_REPO = ROOT / "feature_repo" / "feature_repo"
TABLE = "aqi_observations"
VIEW = "lahore_aqi_features"

DEFAULT_HISTORICAL_BATCH_SIZE = 250


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

    missing = [key for key in required if not os.getenv(key)]

    if missing:
        raise RuntimeError(
            f"Cloud Feast configuration missing: {', '.join(missing)}"
        )

    if "supabase" in os.environ["FEAST_POSTGRES_HOST"].lower():
        raise RuntimeError(
            "Supabase is not an approved Feature Store backend for this project."
        )


def psycopg_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return "postgresql+psycopg://" + database_url.removeprefix(
            "postgresql://"
        )

    return database_url


def cloud_engine():
    require_cloud_configuration()

    return create_engine(
        psycopg_url(os.environ["FEAST_POSTGRES_URL"]),
        pool_pre_ping=True,
    )


def feast_store():
    require_cloud_configuration()

    from feast import FeatureStore

    os.environ["FEAST_POSTGRES_URL"] = psycopg_url(
        os.environ["FEAST_POSTGRES_URL"]
    )

    return FeatureStore(repo_path=str(FEAST_REPO))


def historical_batch_size() -> int:
    value = int(
        os.getenv(
            "FEAST_HISTORICAL_BATCH_SIZE",
            DEFAULT_HISTORICAL_BATCH_SIZE,
        )
    )

    if value < 1:
        raise ValueError(
            "FEAST_HISTORICAL_BATCH_SIZE must be a positive integer."
        )

    return value


def _raw_cloud_observations() -> pd.DataFrame:
    schema = os.getenv("FEAST_POSTGRES_SCHEMA", "public")

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
            params={"city": LAHORE["name"]},
        )

    return validate_observations(frame) if not frame.empty else frame


def cloud_observation_status() -> dict:
    """Return only genuine Lahore observations currently stored in Neon."""
    observations = _raw_cloud_observations()

    return {
        "count": len(observations),
        "chronological": (
            observations.empty
            or observations["timestamp"].is_monotonic_increasing
        ),
        "latest_timestamp": (
            None
            if observations.empty
            else observations["timestamp"].iloc[-1].isoformat()
        ),
    }


def ensure_observation_uniqueness(connection) -> None:
    """Create the idempotency guard used by live collection and approved backfills."""
    schema = os.environ["FEAST_POSTGRES_SCHEMA"]

    connection.execute(
        text(
            f"CREATE UNIQUE INDEX IF NOT EXISTS "
            f"{TABLE}_city_event_timestamp_key "
            f"ON {schema}.{TABLE} (city, event_timestamp)"
        )
    )


def insert_observations_conflict_safe(rows: pd.DataFrame) -> int:
    """Insert rows into Neon once; duplicate city/timestamp rows are skipped."""
    if rows.empty:
        return 0

    schema = os.environ["FEAST_POSTGRES_SCHEMA"]
    columns = list(rows.columns)

    placeholders = ", ".join(
        f":{column}" for column in columns
    )

    statement = text(
        f"INSERT INTO {schema}.{TABLE} "
        f"({', '.join(columns)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (city, event_timestamp) DO NOTHING"
    )

    records = (
        rows.astype(object)
        .where(pd.notna(rows), None)
        .to_dict(orient="records")
    )

    with cloud_engine().begin() as connection:
        ensure_observation_uniqueness(connection)

        result = connection.execute(
            statement,
            records,
        )

    return result.rowcount or 0


def build_live_observation_row(
    history: pd.DataFrame,
    observation: dict,
) -> pd.DataFrame:
    """Build one insert-ready row using only prior chronological Lahore AQI values.

    This is deliberately fail-closed: a live event is never inserted with
    placeholder lag values when there are fewer than three real predecessors.
    """
    candidate = pd.DataFrame(
        [
            {
                key: observation[key]
                for key in ["timestamp", "aqi", *RAW_FEATURES]
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

    if serving.loc[
        :,
        [
            "aqi_lag1",
            "aqi_lag2",
            "aqi_lag3",
            "aqi_rolling_mean",
        ],
    ].isna().any(axis=None):
        raise ValueError(
            "Live observation has insufficient real prior AQI history "
            "for lag features."
        )

    row = candidate.rename(
        columns={"timestamp": "event_timestamp"}
    ).copy()

    row["event_timestamp"] = row["event_timestamp"].map(
        to_utc_datetime
    )

    row["created_timestamp"] = to_utc_datetime(
        pd.Timestamp.now(tz="UTC")
    )

    row["city"] = LAHORE["name"]

    for column in FEATURE_COLUMNS:
        row[column] = serving.iloc[0][column]

    return row


def ingest_observation(observation: dict) -> pd.DataFrame:
    """Persist a Lahore observation remotely, then materialize it into Feast online serving."""
    if observation.get("city") != LAHORE["name"]:
        raise ValueError(
            "Only Lahore observations may enter the Feature Store."
        )

    history = _raw_cloud_observations()

    candidate = pd.DataFrame(
        [
            {
                key: observation[key]
                for key in ["timestamp", "aqi", *RAW_FEATURES]
            }
        ]
    )

    candidate["timestamp"] = pd.to_datetime(
        candidate["timestamp"],
        utc=True,
    )

    if (
        not history.empty
        and candidate["timestamp"].iloc[0]
        in set(history["timestamp"])
    ):
        return candidate.rename(
            columns={"timestamp": "event_timestamp"}
        )

    row = build_live_observation_row(
        history,
        observation,
    )

    inserted = insert_observations_conflict_safe(row)

    if not inserted:
        return row

    store = feast_store()

    store.materialize_incremental(
        end_date=pd.Timestamp.now(tz="UTC").to_pydatetime()
    )

    return row


def historical_features() -> pd.DataFrame:
    """Load genuine chronological Lahore history from cloud PostgreSQL.

    Training uses the indexed cloud source directly instead of Feast's
    expensive point-in-time historical join. Real lag and rolling features
    are derived afterward by the shared feature contract.
    """
    retrieval_started = perf_counter()

    print(
        "Cloud historical retrieval: loading Lahore observations "
        "directly from indexed PostgreSQL.",
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
            params={"city": LAHORE["name"]},
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

    output = validate_observations(output)

    print(
        f"Cloud historical retrieval: returned "
        f"{len(output)} Lahore rows in "
        f"{perf_counter() - retrieval_started:.2f}s.",
        flush=True,
    )

    return output[
        ["timestamp", "aqi", *RAW_FEATURES]
    ]


def latest_online_features() -> pd.DataFrame:
    """Retrieve the latest serving vector from the remote Feast online store."""
    fields = [
        f"{VIEW}:{name}"
        for name in FEATURE_COLUMNS
    ]

    values = feast_store().get_online_features(
        features=fields,
        entity_rows=[
            {
                "city": LAHORE["name"]
            }
        ],
    ).to_dict()

    frame = pd.DataFrame(
        {
            name: values.get(name, [None])[0]
            for name in FEATURE_COLUMNS
        },
        index=[0],
    )

    if frame.isna().any(axis=None):
        raise ValueError(
            "Feast online store has insufficient real history "
            "to serve all lag features."
        )

    return frame.loc[:, FEATURE_COLUMNS]