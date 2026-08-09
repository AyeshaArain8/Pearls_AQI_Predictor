"""Reusable SHAP explanations for the registered AQI forecasting models."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.feature_contract import FEATURE_COLUMNS, assert_feature_schema

REPORTS_DIR = Path("reports")


def local_feature_importance(model, features: pd.DataFrame) -> pd.DataFrame:
    """Return a local SHAP explanation for one production model feature vector."""
    assert_feature_schema(features.columns)
    import shap

    explanation = shap.TreeExplainer(model)(features.loc[:, FEATURE_COLUMNS])
    return pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "shap_value": explanation.values[0],
        "feature_value": features.iloc[0].loc[FEATURE_COLUMNS].to_numpy(),
    }).sort_values("shap_value", key=abs, ascending=False)


def save_global_feature_report(model, features: pd.DataFrame, model_version: str, horizon: str) -> Path:
    """Persist genuine SHAP mean-absolute importance after a successful training run."""
    assert_feature_schema(features.columns)
    import shap

    explanation = shap.TreeExplainer(model)(features.loc[:, FEATURE_COLUMNS])
    report = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "mean_abs_shap": abs(explanation.values).mean(axis=0),
        "model_version": model_version,
        "horizon": horizon,
    }).sort_values("mean_abs_shap", ascending=False)
    REPORTS_DIR.mkdir(exist_ok=True)
    destination = REPORTS_DIR / f"shap_{horizon}_{model_version}.csv"
    report.to_csv(destination, index=False)
    return destination
