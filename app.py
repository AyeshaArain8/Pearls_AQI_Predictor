import json
from pathlib import Path
import streamlit as st
import pandas as pd
import altair as alt

from src.api_fetch import get_lahore_data
from src.predict import forecast_aqi
from src.feature_store import load_observations

st.set_page_config(page_title="Pearls Lahore AQI Predictor", page_icon="🌍", layout="wide")
st.title("Lahore AQI Predictor")
st.caption("Lahore, Pakistan only - OpenWeather observations, local feature store, and a versioned model registry.")

try:
    history = load_observations()
    st.caption(f"Feature store updated: {history.timestamp.iloc[-1].strftime('%Y-%m-%d %H:%M UTC')}")
except Exception as error:
    st.error(f"Feature store unavailable: {error}")
    st.stop()

if st.button("Fetch current Lahore observation and forecast", type="primary"):
    try:
        observation = get_lahore_data()
        result = forecast_aqi(observation)
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
            import shap
            from src.model_registry import load_latest
            models, _ = load_latest()
            explanation = shap.TreeExplainer(models["day1"])(pd.DataFrame([result["features"]]))
            values = pd.DataFrame({"feature": explanation.feature_names, "SHAP value": explanation.values[0]}).sort_values("SHAP value", key=abs, ascending=False)
            st.caption("Local explanation for tomorrow's forecast (positive values increase predicted AQI).")
            st.bar_chart(values.set_index("feature"))
        except Exception as error:
            st.info(f"SHAP explanation is unavailable ({error}). Forecasting remains available.")
    except Exception as error:
        st.error(f"Could not forecast safely: {error}")

st.subheader("Data source and safety")
st.write("Pollutants come from OpenWeather. The model uses only pollutants available in both historical and live data. Lag and rolling AQI features are calculated from actual prior observations in the local feature store; it will refuse to forecast when sufficient history is absent.")
