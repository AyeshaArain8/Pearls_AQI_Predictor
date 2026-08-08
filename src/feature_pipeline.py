"""Hourly collection and local Feature Store update."""
from __future__ import annotations
from src.api_fetch import get_lahore_data
from src.feature_store import append_observation

def main():
    observation = get_lahore_data()
    store = append_observation({key: observation[key] for key in ["timestamp", "aqi", "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]})
    print(f"Stored Lahore observation at {store.timestamp.iloc[-1].isoformat()} ({len(store)} records).")
if __name__ == "__main__": main()
