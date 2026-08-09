import ast
from pathlib import Path
import pandas as pd
import pytest
from src.feature_contract import FEATURE_COLUMNS, LAHORE, RAW_FEATURES, DataQualityError, latest_serving_features, make_feature_rows, pm25_to_us_aqi, to_utc_naive_datetime, validate_observations
from src.feature_store import build_live_observation_row

ROOT = Path(__file__).resolve().parents[1]
def observations(): return pd.DataFrame({"timestamp":pd.date_range("2026-01-01",periods=8,freq="D",tz="UTC"),"aqi":[10,20,30,40,50,60,70,80],**{x:[1]*8 for x in RAW_FEATURES}})
def test_train_serve_schema_and_real_lags():
    train=make_feature_rows(observations(),include_targets=True); serve=latest_serving_features(observations().iloc[:5],{"timestamp":"2026-01-06T00:00:00Z","aqi":60,**{x:1 for x in RAW_FEATURES}})
    assert list(train[FEATURE_COLUMNS])==FEATURE_COLUMNS==list(serve.columns)
    assert train.iloc[0].aqi_lag1==30 and train.iloc[0].aqi_rolling_mean==20 and serve.iloc[0].aqi_lag1==50
    assert train.iloc[0].day1_target == 50 and train.iloc[0].day3_target == 70

def test_aqi_calculation_and_timestamp_normalization():
    assert pm25_to_us_aqi(12.0) == 50.0
    assert pm25_to_us_aqi(35.4) == 100.0
    assert to_utc_naive_datetime("2026-08-08T10:00:00+05:00").isoformat() == "2026-08-08T05:00:00"

def test_duplicate_and_out_of_order_timestamps_are_rejected():
    duplicate = observations().iloc[:2].copy()
    duplicate.loc[1, "timestamp"] = duplicate.loc[0, "timestamp"]
    with pytest.raises(DataQualityError): validate_observations(duplicate)
    unordered = observations().iloc[:3].iloc[[1, 0, 2]]
    with pytest.raises(DataQualityError): validate_observations(unordered)

def test_missing_feature_is_rejected():
    missing = observations().drop(columns=["ozone"])
    with pytest.raises(DataQualityError, match="ozone"):
        validate_observations(missing)
def test_insufficient_history_fails_closed():
    with pytest.raises(DataQualityError): latest_serving_features(observations().iloc[:2],{"timestamp":"2026-02-01T00:00:00Z","aqi":5,**{x:1 for x in RAW_FEATURES}})


def test_live_insert_row_uses_three_real_prior_aqi_values():
    history = observations().iloc[:4]
    row = build_live_observation_row(
        history,
        {"city": "Lahore", "timestamp": "2026-01-05T00:00:00Z", "aqi": 50, **{x: 1 for x in RAW_FEATURES}},
    )
    assert row.loc[0, ["aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_rolling_mean"]].tolist() == [40, 30, 20, 30]
def test_lahore_only_contract(): assert LAHORE["name"]=="Lahore"


def test_dashboard_and_inference_are_lahore_only():
    from src.predict import forecast_aqi

    dashboard = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "Supported City: Lahore" in dashboard
    assert "Karachi" not in dashboard
    assert "selectbox" not in dashboard and "text_input" not in dashboard
    with pytest.raises(ValueError, match="Only Lahore"):
        forecast_aqi("Karachi")


def test_lahore_schema_is_the_only_inference_schema():
    assert FEATURE_COLUMNS == RAW_FEATURES + [
        "month", "day", "day_of_week", "aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_rolling_mean",
    ]
def test_feast_definition_uses_canonical_schema():
    definition=(ROOT/"feature_repo/feature_repo/feature_definitions.py").read_text(encoding="utf-8")
    assert "FEATURE_COLUMNS" in definition and "PostgreSQLSource" in definition and "aerosol_optical_depth" not in definition
def test_no_active_supabase_imports():
    active=[ROOT/"src/feature_pipeline.py",ROOT/"src/feature_store.py",ROOT/"src/train_forecast.py",ROOT/"src/predict.py",ROOT/"app.py"]
    assert all("from supabase" not in path.read_text(encoding="utf-8").lower() for path in active)
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

def test_historical_backfill_is_lahore_only_and_dry_run_ready():
    from src.historical_backfill import prepare_backfill
    payload = prepare_backfill(ROOT / "data/historical/Backup/old_pakistan_air_quality_final_clean.csv")
    assert len(payload) == 2184
    assert payload.city.eq("Lahore").all()
    assert payload.duplicated(["city", "event_timestamp"]).sum() == 0
    assert payload.aqi.notna().all()
