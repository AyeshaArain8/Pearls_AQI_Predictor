"""Hourly Lahore collector that writes to cloud PostgreSQL and Feast online serving."""
from src.api_fetch import get_lahore_data
from src.feature_store import ingest_observation

def main():
    row = ingest_observation(get_lahore_data())
    print(f"Stored and materialized Lahore feature event at {row.event_timestamp.iloc[0]}.")

if __name__ == "__main__":
    main()
