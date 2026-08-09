"""Train from Feast historical retrieval and register the versioned models separately."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.feature_contract import FEATURE_COLUMNS, SCHEMA_VERSION, TARGET_COLUMNS, assert_feature_schema, make_feature_rows
from src.feature_store import historical_features
from src.model_registry import register
from src.explainability import save_global_feature_report

def train() -> tuple[str, dict]:
    history = historical_features()
    if len(history) < 30:
        raise ValueError(f"Feast contains only {len(history)} genuine chronological Lahore observations; at least 30 are required before training.")
    dataset = make_feature_rows(history, include_targets=True)
    if len(dataset) < 30: raise ValueError("Feast returned fewer than 30 usable chronological Lahore records.")
    split = int(len(dataset) * .8); train_rows, test_rows = dataset.iloc[:split], dataset.iloc[split:]
    assert_feature_schema(train_rows[FEATURE_COLUMNS].columns)
    models, metrics = {}, {}
    for horizon, target in zip(("day1", "day2", "day3"), TARGET_COLUMNS):
        model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
        model.fit(train_rows[FEATURE_COLUMNS], train_rows[target]); predicted = model.predict(test_rows[FEATURE_COLUMNS])
        metrics[horizon] = {"rmse":float(np.sqrt(mean_squared_error(test_rows[target],predicted))),"mae":float(mean_absolute_error(test_rows[target],predicted)),"r2":float(r2_score(test_rows[target],predicted))}
        models[horizon] = model
    metadata = {"city":"Lahore","schema_version":SCHEMA_VERSION,"feature_columns":FEATURE_COLUMNS,"target_columns":TARGET_COLUMNS,"training_rows":len(train_rows),"test_rows":len(test_rows),"training_end":train_rows.timestamp.iloc[-1].isoformat(),"metrics":metrics,"data_source":"Feast cloud PostgreSQL historical retrieval","model_type":"RandomForestRegressor","trained_at":pd.Timestamp.now(tz="UTC").isoformat(),"horizons":["day1","day2","day3"]}
    version = register(models, metadata)
    save_global_feature_report(models["day1"], train_rows[FEATURE_COLUMNS], version, "day1")
    print(json.dumps({"version":version,"metrics":metrics}, indent=2)); return version, metadata
if __name__ == "__main__": train()
