"""Versioned local model registry with cached model loading."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "models" / "registry"


def register(models: dict, metadata: dict) -> str:
    """Register a new approved version of the forecast models."""

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = REGISTRY / version
    directory.mkdir(parents=True, exist_ok=False)

    for horizon, model in models.items():
        joblib.dump(
            model,
            directory / f"{horizon}_model.joblib",
        )

    final_metadata = metadata | {
        "version": version,
        "approved": True,
    }

    (directory / "metadata.json").write_text(
        json.dumps(final_metadata, indent=2),
        encoding="utf-8",
    )

    (REGISTRY / "latest.json").write_text(
        json.dumps({"version": version}, indent=2),
        encoding="utf-8",
    )

    # New registration must never reuse an old cached model.
    load_latest.cache_clear()

    return version


@lru_cache(maxsize=1)
def load_latest() -> tuple[dict, dict]:
    """
    Load the latest approved model registry version.

    Cached so Streamlit does not reload three ~20 MB models
    every time the application reruns.
    """

    pointer = REGISTRY / "latest.json"

    if not pointer.exists():
        raise FileNotFoundError(
            "No registered model. Run "
            "`python -m src.train_forecast` first."
        )

    version = json.loads(
        pointer.read_text(encoding="utf-8")
    )["version"]

    directory = REGISTRY / version

    metadata = json.loads(
        (directory / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    if not metadata.get("approved"):
        raise ValueError(
            f"Model {version} is not approved for serving."
        )

    models = {
        horizon: joblib.load(
            directory / f"{horizon}_model.joblib"
        )
        for horizon in ("day1", "day2", "day3")
    }

    return models, metadata