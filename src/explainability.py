"""Reusable SHAP explanations for the registered AQI forecasting models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_contract import FEATURE_COLUMNS, assert_feature_schema

REPORTS_DIR = Path("reports")


def _shap_values(model, features: pd.DataFrame) -> np.ndarray:
    """
    Return feature contributions as a 2-D array: rows x features.

    Supports:
    - Tree-based models such as RandomForestRegressor via SHAP TreeExplainer.
    - sklearn Ridge pipelines containing StandardScaler + Ridge.

    For Ridge, the contribution of each feature is calculated from the
    fitted standardized feature value multiplied by the fitted Ridge
    coefficient. This gives the exact linear contribution used by the
    model in standardized feature space.
    """

    assert_feature_schema(features.columns)

    import shap

    X = features.loc[:, FEATURE_COLUMNS].copy()

    # ------------------------------------------------------------
    # Ridge / linear Pipeline support
    # ------------------------------------------------------------
    if hasattr(model, "named_steps") and "ridge" in model.named_steps:
        scaler = model.named_steps.get("scaler")
        ridge = model.named_steps["ridge"]

        if scaler is None:
            raise ValueError(
                "Ridge model pipeline is missing its StandardScaler step."
            )

        transformed = scaler.transform(X)

        coefficients = np.asarray(ridge.coef_, dtype=float)

        # Ridge is expected to be a single-output regressor.
        if coefficients.ndim != 1:
            coefficients = coefficients.reshape(-1)

        if len(coefficients) != len(FEATURE_COLUMNS):
            raise ValueError(
                "Ridge coefficient count does not match the production "
                f"feature schema: {len(coefficients)} vs "
                f"{len(FEATURE_COLUMNS)} features."
            )

        values = transformed * coefficients

        values = np.asarray(values, dtype=float)

        if values.ndim != 2:
            raise ValueError(
                f"Unexpected Ridge contribution shape: {values.shape}"
            )

        return values

    # ------------------------------------------------------------
    # Tree model support
    # ------------------------------------------------------------
    try:
        explanation = shap.TreeExplainer(model)(X)

        values = explanation.values

        if isinstance(values, list):
            values = values[0]

        values = np.asarray(values)

        if values.ndim == 3:
            values = values[:, :, 0]

        if values.ndim == 1:
            values = values.reshape(1, -1)

        if values.ndim != 2:
            raise ValueError(
                f"Unexpected SHAP output shape: {values.shape}"
            )

        if values.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(
                "SHAP feature count does not match the production "
                f"feature schema: {values.shape} vs "
                f"{len(FEATURE_COLUMNS)} features."
            )

        return values

    except Exception as exc:
        raise TypeError(
            "The registered model is not supported by the explainability "
            "implementation. Supported models are RandomForestRegressor "
            "and Ridge pipelines containing StandardScaler + Ridge."
        ) from exc


def local_feature_importance(
    model,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Return a local feature-contribution explanation."""

    assert_feature_schema(features.columns)

    values = _shap_values(model, features)

    row = features.iloc[0]

    return (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "shap_value": values[0],
                "feature_value": [
                    row[column] for column in FEATURE_COLUMNS
                ],
            }
        )
        .sort_values(
            "shap_value",
            key=lambda column: column.abs(),
            ascending=False,
        )
        .reset_index(drop=True)
    )


def save_global_feature_report(
    model,
    features: pd.DataFrame,
    model_version: str,
    horizon: str,
) -> Path:
    """Persist genuine mean-absolute feature contribution importance."""

    assert_feature_schema(features.columns)

    values = _shap_values(model, features)

    report = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "mean_abs_shap": np.abs(values).mean(axis=0),
                "model_version": model_version,
                "horizon": horizon,
            }
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    REPORTS_DIR.mkdir(exist_ok=True)

    destination = (
        REPORTS_DIR
        / f"shap_{horizon}_{model_version}.csv"
    )

    report.to_csv(
        destination,
        index=False,
    )

    return destination