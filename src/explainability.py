"""Reusable SHAP explanations for the registered AQI forecasting models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_contract import FEATURE_COLUMNS, assert_feature_schema

REPORTS_DIR = Path("reports")


def _shap_values(model, features: pd.DataFrame) -> np.ndarray:
    """Return SHAP values as a 2-D array: rows x features."""

    assert_feature_schema(features.columns)

    import shap

    X = features.loc[:, FEATURE_COLUMNS].copy()

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


def local_feature_importance(
    model,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Return a local SHAP explanation for one production model vector."""

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
    """Persist genuine mean-absolute SHAP feature importance."""

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
