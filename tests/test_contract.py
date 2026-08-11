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
        "hour", "month", "day", "day_of_week", "aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_rolling_mean",
    ]


def test_hour_is_derived_from_timestamp():
    frame = observations().copy()
    frame["timestamp"] = pd.date_range("2026-01-01T05:00:00", periods=8, freq="h", tz="UTC")
    train = make_feature_rows(frame, include_targets=True)
    assert train["hour"].tolist() == [(5 + i) % 24 for i in range(3, 5)]

    serve = latest_serving_features(
        frame.iloc[:5],
        {"timestamp": "2026-01-01T14:00:00Z", "aqi": 60, **{x: 1 for x in RAW_FEATURES}},
    )
    assert serve.iloc[0]["hour"] == 14


def test_train_and_serve_hour_schema_match():
    frame = observations()
    train = make_feature_rows(frame, include_targets=True)
    serve = latest_serving_features(
        frame.iloc[:5],
        {"timestamp": "2026-01-06T00:00:00Z", "aqi": 60, **{x: 1 for x in RAW_FEATURES}},
    )
    assert "hour" in train.columns and "hour" in serve.columns
    assert list(train[FEATURE_COLUMNS].columns) == list(serve[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS


def test_active_paths_have_no_fake_features_or_other_city_support():
    active = [
        ROOT / "app.py", ROOT / "src/predict.py",
        ROOT / "src/feature_pipeline.py", ROOT / "src/feature_store.py", ROOT / "src/train_forecast.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8").lower() for path in active)
    for unsupported in ("karachi", "aerosol_optical_depth", "dust", "uv_index", "current_aqi -"):
        assert unsupported not in source


def test_historical_retrieval_is_batched_and_hourly_workflow_skips_apply():
    store_source = (ROOT / "src/feature_store.py").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/data_pipeline.yml").read_text(encoding="utf-8")
    assert "HISTORICAL_BATCH_SIZE" in store_source and "Retrieving Feast historical batch" in store_source
    assert "feast -c feature_repo/feature_repo apply" not in workflow
def test_feast_definition_uses_canonical_schema():
    definition=(ROOT/"feature_repo/feature_repo/feature_definitions.py").read_text(encoding="utf-8")
    assert "FEATURE_COLUMNS" in definition and "PostgreSQLSource" in definition and "aerosol_optical_depth" not in definition
def test_no_active_supabase_imports():
    active=[ROOT/"src/feature_pipeline.py",ROOT/"src/feature_store.py",ROOT/"src/train_forecast.py",ROOT/"src/predict.py",ROOT/"app.py"]
    assert all("from supabase" not in path.read_text(encoding="utf-8").lower() for path in active)
def test_cloud_feature_store_has_no_local_fallback():
    # Note: this intentionally does NOT require Feast's own
    # get_historical_features() point-in-time join. historical_features()
    # deliberately queries the indexed cloud PostgreSQL table directly
    # (see its docstring) because that join is expensive at this scale.
    # get_online_features() is still required for live serving, and no
    # local CSV fallback is allowed either way.
    source=(ROOT/"src/feature_store.py").read_text(encoding="utf-8")
    assert "get_online_features" in source and "to_csv" not in source
def test_model_registry_remains_separate():
    source=(ROOT/"src/model_registry.py").read_text(encoding="utf-8")
    assert "models" in source and "feast" not in source.lower()


def test_feast_reconnects_once_on_stale_ssl_connection_then_raises_other_errors(monkeypatch):
    from psycopg import OperationalError
    from src import feature_store

    calls = {"n": 0}

    def flaky(store):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OperationalError("consuming input failed: SSL connection has been closed unexpectedly")
        return "ok"

    monkeypatch.setattr(feature_store, "feast_store", lambda: "store")
    cache_cleared = {"called": False}
    feature_store.feast_store.cache_clear = lambda: cache_cleared.__setitem__("called", True)

    assert feature_store._with_feast_reconnect(flaky) == "ok"
    assert calls["n"] == 2
    assert cache_cleared["called"] is True

    def always_broken(store):
        raise ValueError("unrelated bug")

    with pytest.raises(ValueError):
        feature_store._with_feast_reconnect(always_broken)

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