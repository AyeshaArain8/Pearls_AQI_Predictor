"""Lahore inference from Feast online serving and the separate Model Registry."""
from datetime import datetime, timedelta, timezone
from src.feature_contract import FEATURE_COLUMNS, LAHORE, SCHEMA_VERSION, assert_feature_schema
from src.feature_store import latest_online_features
from src.model_registry import load_latest

AQI_THRESHOLDS=((50,"Good"),(100,"Moderate"),(150,"Unhealthy for Sensitive Groups"),(200,"Unhealthy"),(300,"Very Unhealthy"),(float("inf"),"Hazardous"))
def get_category(aqi): return next(label for limit,label in AQI_THRESHOLDS if aqi <= limit)
def forecast_aqi(city: str = LAHORE["name"]) -> dict:
    if city != LAHORE["name"]:
        raise ValueError("Only Lahore is supported for AQI forecasting.")
    models, metadata = load_latest()
    if metadata.get("schema_version") != SCHEMA_VERSION: raise ValueError("Registered model schema is incompatible with Feast serving schema.")
    assert_feature_schema(metadata.get("feature_columns", [])); features = latest_online_features(); assert_feature_schema(features.columns)
    observed_at = datetime.now(timezone.utc); forecasts=[]
    for index,horizon in enumerate(("day1","day2","day3"),1):
        aqi=round(float(models[horizon].predict(features)[0]),1); forecasts.append({"date":(observed_at+timedelta(days=index)).date().isoformat(),"aqi":aqi,"category":get_category(aqi),"hazardous":aqi>300})
    return {"city":city,"model_version":metadata["version"],"observed_at":observed_at.isoformat(),"forecasts":forecasts,"metrics":metadata.get("metrics",{}),"features":features.iloc[0].to_dict()}
