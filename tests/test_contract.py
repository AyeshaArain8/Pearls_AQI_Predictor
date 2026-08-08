import pandas as pd
import pytest
from src.feature_contract import FEATURE_COLUMNS, LAHORE, DataQualityError, latest_serving_features, make_feature_rows
from src import predict

def observations():
    return pd.DataFrame({"timestamp":pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC"), "aqi":[10,20,30,40,50,60,70,80], "pm10":[1]*8, "pm2_5":[1]*8, "carbon_monoxide":[1]*8, "nitrogen_dioxide":[1]*8, "sulphur_dioxide":[1]*8, "ozone":[1]*8})

def test_training_features_are_canonical():
    rows = make_feature_rows(observations(), include_targets=True)
    assert list(rows[FEATURE_COLUMNS]) == FEATURE_COLUMNS
    assert rows.iloc[0].aqi_lag1 == 30
    assert rows.iloc[0].aqi_lag2 == 20
    assert rows.iloc[0].aqi_lag3 == 10
    assert rows.iloc[0].aqi_rolling_mean == 20

def test_serving_parity_and_real_lags():
    history = observations().iloc[:5]
    live = {"timestamp":"2026-01-06T00:00:00Z", "aqi":60, "pm10":1, "pm2_5":1, "carbon_monoxide":1, "nitrogen_dioxide":1, "sulphur_dioxide":1, "ozone":1}
    frame = latest_serving_features(history, live)
    assert list(frame.columns) == FEATURE_COLUMNS
    assert frame.iloc[0].aqi_lag1 == 50 and frame.iloc[0].aqi_rolling_mean == 40

def test_insufficient_history_is_not_fabricated():
    with pytest.raises(DataQualityError): latest_serving_features(observations().iloc[:2], {"timestamp":"2026-02-01T00:00:00Z", "aqi":5, "pm10":1, "pm2_5":1, "carbon_monoxide":1, "nitrogen_dioxide":1, "sulphur_dioxide":1, "ozone":1})

def test_lahore_configuration():
    assert LAHORE["name"] == "Lahore" and LAHORE["latitude"] == 31.5204

def test_unverified_historical_provenance_blocks_training(monkeypatch):
    from src import feature_store
    monkeypatch.setattr(feature_store, "HISTORICAL_METADATA_PATH", __import__("pathlib").Path("data/historical/metadata.json"))
    with pytest.raises(ValueError, match="unverified"):
        feature_store.assert_lahore_training_data()

def test_unknown_feature_store_metadata_is_rejected(tmp_path, monkeypatch):
    from src import feature_store
    path = tmp_path / "observations.csv"
    observations().to_csv(path, index=False)
    monkeypatch.setattr(feature_store, "STORE_PATH", path)
    monkeypatch.setattr(feature_store, "METADATA_PATH", tmp_path / "schema.json")
    with pytest.raises(ValueError, match="metadata"):
        feature_store.load_observations()

def test_prediction_shape_and_lahore_only(monkeypatch):
    class Model:
        def predict(self, frame):
            assert list(frame.columns) == FEATURE_COLUMNS
            return [123.4]
    monkeypatch.setattr(predict, "load_latest", lambda: ({"day1": Model(), "day2": Model(), "day3": Model()}, {"version":"test", "schema_version":"lahore-aqi-v2", "feature_columns":FEATURE_COLUMNS}))
    monkeypatch.setattr(predict, "load_observations", lambda: observations().iloc[:5])
    result = predict.forecast_aqi({"city":"Lahore", "timestamp":"2026-01-06T00:00:00Z", "aqi":60, "pm10":1, "pm2_5":1, "carbon_monoxide":1, "nitrogen_dioxide":1, "sulphur_dioxide":1, "ozone":1})
    assert len(result["forecasts"]) == 3 and all(item["aqi"] == 123.4 for item in result["forecasts"])
    with pytest.raises(ValueError):
        predict.forecast_aqi({"city":"Karachi"})
