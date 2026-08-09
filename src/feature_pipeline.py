"""Hourly Lahore collector that writes to cloud PostgreSQL and Feast online serving."""
from src.api_fetch import get_lahore_data
from src.feature_store import cloud_observation_status, ingest_observation

def main():
    row = ingest_observation(get_lahore_data())
    status = cloud_observation_status()
    print(f"Stored and materialized Lahore feature event at {row.event_timestamp.iloc[0]}.")
    print(f"Cloud Feast history: {status['count']} chronological Lahore observations; latest={status['latest_timestamp']}.")
    if status["count"] < 30:
        print(f"Training is not ready: {30 - status['count']} more genuine hourly observations are required.")

if __name__ == "__main__":
    main()
