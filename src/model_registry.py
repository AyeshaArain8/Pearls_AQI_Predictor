"""Versioned local model registry, with metadata required for safe serving."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "models" / "registry"


def register(models: dict, metadata: dict) -> str:
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = REGISTRY / version
    directory.mkdir(parents=True, exist_ok=False)
    for horizon, model in models.items():
        joblib.dump(model, directory / f"{horizon}_model.joblib")
    (directory / "metadata.json").write_text(json.dumps(metadata | {"version": version, "approved": True}, indent=2), encoding="utf-8")
    (REGISTRY / "latest.json").write_text(json.dumps({"version": version}, indent=2), encoding="utf-8")
    return version


def load_latest() -> tuple[dict, dict]:
    pointer = REGISTRY / "latest.json"
    if not pointer.exists():
        raise FileNotFoundError("No registered model. Run `python src/train_forecast.py` first.")
    version = json.loads(pointer.read_text(encoding="utf-8"))["version"]
    directory = REGISTRY / version
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("approved"):
        raise ValueError(f"Model {version} is not approved for serving.")
    return ({h: joblib.load(directory / f"{h}_model.joblib") for h in ("day1", "day2", "day3")}, metadata)
