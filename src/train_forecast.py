"""Train from Feast historical retrieval and register the best versioned models."""
from __future__ import annotations

import json
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_contract import (
    FEATURE_COLUMNS,
    SCHEMA_VERSION,
    TARGET_COLUMNS,
    assert_feature_schema,
    resample_daily,
    make_daily_feature_rows,
)
from src.feature_store import historical_features
from src.model_registry import register
from src.explainability import save_global_feature_report


def _evaluate_model(model, x_test, y_test) -> dict:
    """Fit-independent evaluation helper."""
    predicted = model.predict(x_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, predicted)))
    mae = float(mean_absolute_error(y_test, predicted))
    r2 = float(r2_score(y_test, predicted))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def _build_candidates() -> dict:
    """Return the candidate regressors required for model comparison."""
    return {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=10.0)),
            ]
        ),
    }


def _is_better(candidate_metrics: dict, best_metrics: dict | None) -> bool:
    """
    Select the genuinely better model.

    Primary criterion:
        lower RMSE

    Tie-breakers:
        lower MAE, then higher R².
    """
    if best_metrics is None:
        return True

    candidate_key = (
        candidate_metrics["rmse"],
        candidate_metrics["mae"],
        -candidate_metrics["r2"],
    )
    best_key = (
        best_metrics["rmse"],
        best_metrics["mae"],
        -best_metrics["r2"],
    )

    return candidate_key < best_key


def train() -> tuple[str, dict]:
    retrieval_started = perf_counter()

    print("Training: starting Feast historical retrieval.")
    history = historical_features()

    retrieval_seconds = perf_counter() - retrieval_started

    print(
        f"Training: Feast historical retrieval finished with "
        f"{len(history)} rows in {retrieval_seconds:.1f}s."
    )

    if len(history) < 30:
        raise ValueError(
            f"Feast contains only {len(history)} genuine chronological "
            "Lahore observations; at least 30 are required before training."
        )

    print("Training: resampling hourly history to one row per calendar day.")

    daily_history = resample_daily(history)

    print(
        f"Training: daily resample produced {len(daily_history)} "
        "calendar days (including any gap days as NaN)."
    )

    dataset = make_daily_feature_rows(
        daily_history,
        include_targets=True,
    )

    if len(dataset) < 15:
        raise ValueError(
            f"Only {len(dataset)} usable daily Lahore rows after resampling; "
            "need more consecutive days of history before a real 3-day "
            "model can be trained."
        )

    split = int(len(dataset) * 0.8)

    train_rows = dataset.iloc[:split].copy()
    test_rows = dataset.iloc[split:].copy()

    print(
        f"Training: {len(dataset)} usable daily rows; "
        f"{len(train_rows)} train rows; "
        f"{len(test_rows)} test rows."
    )

    assert_feature_schema(train_rows[FEATURE_COLUMNS].columns)

    x_train = train_rows[FEATURE_COLUMNS]
    x_test = test_rows[FEATURE_COLUMNS]

    models = {}
    metrics = {}
    selected_model_names = {}
    candidate_metrics = {}

    horizons = ("day1", "day2", "day3")

    for horizon, target in zip(horizons, TARGET_COLUMNS):
        print()
        print(f"Training: comparing models for {horizon}.")

        y_train = train_rows[target]
        y_test = test_rows[target]

        best_model = None
        best_model_name = None
        best_model_metrics = None

        candidate_metrics[horizon] = {}

        for model_name, model in _build_candidates().items():
            model_started = perf_counter()

            print(f"Training: fitting {horizon} with {model_name}.")

            model.fit(x_train, y_train)

            current_metrics = _evaluate_model(
                model,
                x_test,
                y_test,
            )

            elapsed = perf_counter() - model_started

            candidate_metrics[horizon][model_name] = current_metrics

            print(
                f"Training: {horizon} {model_name} finished in "
                f"{elapsed:.1f}s."
            )
            print(
                f"  RMSE: {current_metrics['rmse']:.4f}"
            )
            print(
                f"  MAE:  {current_metrics['mae']:.4f}"
            )
            print(
                f"  R²:   {current_metrics['r2']:.4f}"
            )

            if _is_better(
                current_metrics,
                best_model_metrics,
            ):
                best_model = model
                best_model_name = model_name
                best_model_metrics = current_metrics

        if best_model is None or best_model_metrics is None:
            raise RuntimeError(
                f"No valid model was produced for {horizon}."
            )

        models[horizon] = best_model
        metrics[horizon] = best_model_metrics
        selected_model_names[horizon] = best_model_name

        print(
            f"Training: selected {best_model_name} for {horizon} "
            f"(RMSE={best_model_metrics['rmse']:.4f}, "
            f"MAE={best_model_metrics['mae']:.4f}, "
            f"R²={best_model_metrics['r2']:.4f})."
        )

    metadata = {
        "city": "Lahore",
        "schema_version": SCHEMA_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "training_rows": len(train_rows),
        "test_rows": len(test_rows),
        "training_end": train_rows.timestamp.iloc[-1].isoformat(),
        "metrics": metrics,
        "candidate_metrics": candidate_metrics,
        "selected_models": selected_model_names,
        "data_source": (
            "Feast cloud PostgreSQL historical retrieval, "
            "resampled to one row per calendar day"
        ),
        "forecast_granularity": (
            "daily (day1=+24h, day2=+48h, day3=+72h)"
        ),
        "model_type": "per-horizon best model from RandomForestRegressor and Ridge",
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "horizons": list(horizons),
    }

    print()
    print("Training: model comparison complete.")
    print(
        "Training: selected models: "
        + ", ".join(
            f"{horizon}={selected_model_names[horizon]}"
            for horizon in horizons
        )
    )

    print("Training: registering day1/day2/day3 models.")

    version = register(
        models,
        metadata,
    )

    print(
        f"Training: registered model version {version}."
    )

    save_global_feature_report(
        models["day1"],
        train_rows[FEATURE_COLUMNS],
        version,
        "day1",
    )

    print(
        f"Training: saved SHAP report for model version {version}."
    )

    result = {
        "version": version,
        "selected_models": selected_model_names,
        "metrics": metrics,
        "candidate_metrics": candidate_metrics,
    }

    print(json.dumps(result, indent=2))

    return version, metadata


if __name__ == "__main__":
    train()