"""Retired: the production pipeline uses the free local Feature Store.

Kept only to give prior command users a clear migration path rather than silently
uploading data to a paid/external service.
"""
from src.feature_store import initialise_from_historical

if __name__ == "__main__":
    store = initialise_from_historical()
    print(f"Supabase migration retired. Initialized local Lahore Feature Store with {len(store)} rows.")
