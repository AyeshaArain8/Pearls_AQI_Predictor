import json
from pathlib import Path
import streamlit as st
import pandas as pd
import altair as alt

from src.api_fetch import get_lahore_data
from src.predict import forecast_aqi
from src.feature_store import cloud_observation_status, ingest_observation

st.set_page_config(page_title="Pearls Lahore AQI Predictor", page_icon="🌍", layout="wide")
st.title("Lahore AQI Predictor")
st.caption("Supported City: Lahore")
st.caption("OpenWeather, cloud Feast Feature Store, and a separate versioned Model Registry.")

st.caption("Cloud Feast Feature Store: managed PostgreSQL offline source + Feast PostgreSQL online serving.")

try:
    store_status = cloud_observation_status()
    st.caption(
        f"Feast cloud history: {store_status['count']} chronological Lahore observations"
        + (f" (latest: {store_status['latest_timestamp']})" if store_status["latest_timestamp"] else "")
    )
    if store_status["count"] < 30:
        st.warning(f"Training needs {30 - store_status['count']} more genuine hourly observations.")
except Exception as error:
    st.warning(f"Feast status is unavailable: {error}")

if st.button("Fetch current Lahore observation and forecast", type="primary"):
    try:
        observation = get_lahore_data()
        ingest_observation(observation)
        result = forecast_aqi()
        st.success(f"Forecast generated from model {result['model_version']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current PM2.5", f"{observation['pm2_5']:.1f} µg/m³")
        c2.metric("Current PM10", f"{observation['pm10']:.1f} µg/m³")
        c3.metric("Derived US AQI", observation["aqi"])
        c4.metric("Weather", f"{observation['temperature']:.1f} °C")
        forecast = pd.DataFrame(result["forecasts"])
        st.subheader("Next 3 days")
        st.dataframe(forecast[["date", "aqi", "category"]], hide_index=True, use_container_width=True)
        st.altair_chart(alt.Chart(forecast).mark_line(point=True).encode(x="date:T", y="aqi:Q", tooltip=["date", "aqi", "category"]), use_container_width=True)
        if forecast.hazardous.any(): st.error("Hazardous AQI alert: at least one forecast exceeds the centralized hazardous threshold (300).")
        st.subheader("Model metrics")
        st.json(result["metrics"])
        st.subheader("Explainability")
        try:
            from src.model_registry import load_latest
            from src.explainability import local_feature_importance
            models, _ = load_latest()
            values = local_feature_importance(models["day1"], pd.DataFrame([result["features"]])).rename(columns={"shap_value": "SHAP value"})
            st.caption("Local explanation for tomorrow's forecast (positive values increase predicted AQI).")
            st.bar_chart(values.set_index("feature")[["SHAP value"]])
        except Exception as error:
            st.info(f"SHAP explanation is unavailable ({error}). Forecasting remains available.")
    except Exception as error:
        st.error(f"Could not forecast safely: {error}")

st.subheader("Data source and safety")
st.write("Pollutants come from OpenWeather. The dashboard writes the latest Lahore observation to the cloud Feast Feature Store, then reads the serving vector from Feast. Lag and rolling AQI features are calculated from actual prior cloud observations; it refuses to forecast when history is insufficient.")
