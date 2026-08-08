"""Manual cloud connectivity check; requires real Feast PostgreSQL credentials."""
from src.feature_store import feast_store, latest_online_features

if __name__ == "__main__":
    store = feast_store()
    print("Feast project:", store.config.project)
    print(latest_online_features())
