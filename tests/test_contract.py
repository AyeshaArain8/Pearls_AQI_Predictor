import ast
from pathlib import Path
import pandas as pd
import pytest
from src.feature_contract import FEATURE_COLUMNS, LAHORE, RAW_FEATURES, DataQualityError, latest_serving_features, make_feature_rows

ROOT = Path(__file__).resolve().parents[1]
def observations(): return pd.DataFrame({"timestamp":pd.date_range("2026-01-01",periods=8,freq="D",tz="UTC"),"aqi":[10,20,30,40,50,60,70,80],**{x:[1]*8 for x in RAW_FEATURES}})
def test_train_serve_schema_and_real_lags():
    train=make_feature_rows(observations(),include_targets=True); serve=latest_serving_features(observations().iloc[:5],{"timestamp":"2026-01-06T00:00:00Z","aqi":60,**{x:1 for x in RAW_FEATURES}})
    assert list(train[FEATURE_COLUMNS])==FEATURE_COLUMNS==list(serve.columns)
    assert train.iloc[0].aqi_lag1==30 and train.iloc[0].aqi_rolling_mean==20 and serve.iloc[0].aqi_lag1==50
def test_insufficient_history_fails_closed():
    with pytest.raises(DataQualityError): latest_serving_features(observations().iloc[:2],{"timestamp":"2026-02-01T00:00:00Z","aqi":5,**{x:1 for x in RAW_FEATURES}})
def test_lahore_only_contract(): assert LAHORE["name"]=="Lahore"
def test_feast_definition_uses_canonical_schema():
    definition=(ROOT/"feature_repo/feature_repo/feature_definitions.py").read_text(encoding="utf-8")
    assert "FEATURE_COLUMNS" in definition and "PostgreSQLSource" in definition and "aerosol_optical_depth" not in definition
def test_no_active_supabase_imports():
    active=[ROOT/"src/feature_pipeline.py",ROOT/"src/feature_store.py",ROOT/"src/train_forecast.py",ROOT/"src/predict.py",ROOT/"app.py"]
    assert all("supabase" not in path.read_text(encoding="utf-8").lower() for path in active)
def test_cloud_feature_store_has_no_local_fallback():
    source=(ROOT/"src/feature_store.py").read_text(encoding="utf-8")
    assert "get_historical_features" in source and "get_online_features" in source and "to_csv" not in source
def test_model_registry_remains_separate():
    source=(ROOT/"src/model_registry.py").read_text(encoding="utf-8")
    assert "models" in source and "feast" not in source.lower()
def test_model_registry_version_roundtrip(tmp_path, monkeypatch):
    from src import model_registry
    monkeypatch.setattr(model_registry, "REGISTRY", tmp_path)
    models={"day1":{"v":1},"day2":{"v":2},"day3":{"v":3}}
    version=model_registry.register(models,{"city":"Lahore","feature_columns":FEATURE_COLUMNS})
    loaded, metadata=model_registry.load_latest()
    assert metadata["version"]==version and metadata["approved"] is True and loaded==models
