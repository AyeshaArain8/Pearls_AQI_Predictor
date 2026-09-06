import streamlit as st
import pandas as pd
import altair as alt
from time import perf_counter

from src.api_fetch import get_lahore_data
from src.predict import forecast_aqi
from src.feature_store import cloud_observation_status, ingest_observation


st.set_page_config(
    page_title="Pearls Lahore AQI Predictor",
    page_icon="🌍",
    layout="wide",
)

st.title("Lahore AQI Predictor")
st.caption("Supported City: Lahore")
st.caption(
    "OpenWeather, cloud Feast Feature Store, and a separate versioned Model Registry."
)
st.caption(
    "Cloud Feast Feature Store: managed PostgreSQL offline source + "
    "Feast PostgreSQL online serving."
)


# ---------------------------------------------------------
# Cached cloud status
# This is intentionally NOT called during initial page load.
# It runs only after the user clicks the forecast button.
# ---------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def get_cached_cloud_status():
    return cloud_observation_status()


# ---------------------------------------------------------
# Main action
# ---------------------------------------------------------
if st.button(
    "Fetch current Lahore observation and forecast",
    type="primary",
):

    total_started = perf_counter()

    try:
        # -------------------------------------------------
        # 1. OpenWeather
        # -------------------------------------------------
        started = perf_counter()

        observation = get_lahore_data()

        st.write(
            f"OpenWeather fetch: {perf_counter() - started:.2f}s"
        )

        # -------------------------------------------------
        # 2. Cloud PostgreSQL + Feast
        # -------------------------------------------------
        started = perf_counter()

        ingest_observation(observation)

        st.write(
            f"Cloud/Feast ingestion: {perf_counter() - started:.2f}s"
        )

        # New observation was inserted, so refresh cached status.
        get_cached_cloud_status.clear()

        # -------------------------------------------------
        # 3. Forecast
        # -------------------------------------------------
        started = perf_counter()

        result = forecast_aqi()

        st.write(
            f"Forecast/model/online features: "
            f"{perf_counter() - started:.2f}s"
        )

        st.success(
            f"Forecast generated from model "
            f"{result['model_version']}"
        )

        # -------------------------------------------------
        # Current observation metrics
        # -------------------------------------------------
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Current PM2.5",
            f"{observation['pm2_5']:.1f} µg/m³",
        )

        c2.metric(
            "Current PM10",
            f"{observation['pm10']:.1f} µg/m³",
        )

        c3.metric(
            "Derived US AQI",
            observation["aqi"],
        )

        c4.metric(
            "Weather",
            f"{observation['temperature']:.1f} °C",
        )

        # -------------------------------------------------
        # Forecast table
        # -------------------------------------------------
        forecast = pd.DataFrame(result["forecasts"])

        st.subheader("Next 3 days")

        st.dataframe(
            forecast[["date", "aqi", "category"]],
            hide_index=True,
            width="stretch",
        )

        # -------------------------------------------------
        # Forecast chart
        # -------------------------------------------------
        st.altair_chart(
            alt.Chart(forecast)
            .mark_line(point=True)
            .encode(
                x="date:T",
                y="aqi:Q",
                tooltip=["date", "aqi", "category"],
            ),
            width="stretch",
        )

        # -------------------------------------------------
        # Hazardous alert
        # -------------------------------------------------
        if forecast.hazardous.any():
            st.error(
                "Hazardous AQI alert: at least one forecast "
                "exceeds the centralized hazardous threshold (300)."
            )

        # -------------------------------------------------
        # Model metrics
        # -------------------------------------------------
        st.subheader("Model metrics")
        st.json(result["metrics"])

        # -------------------------------------------------
        # SHAP explainability
        # -------------------------------------------------
        st.subheader("Explainability")

        try:
            started = perf_counter()

            from src.model_registry import load_latest
            from src.explainability import local_feature_importance

            models, _ = load_latest()

            values = local_feature_importance(
                models["day1"],
                pd.DataFrame([result["features"]]),
            ).rename(
                columns={"shap_value": "SHAP value"}
            )

            st.caption(
                "Local explanation for tomorrow's forecast "
                "(positive values increase predicted AQI)."
            )

            st.bar_chart(
                values.set_index("feature")[["SHAP value"]]
            )

            st.write(
                f"SHAP generation: "
                f"{perf_counter() - started:.2f}s"
            )

        except Exception as error:
            st.info(
                f"SHAP explanation is unavailable ({error}). "
                "Forecasting remains available."
            )

        # -------------------------------------------------
        # Cloud status
        # Loaded only AFTER forecast processing.
        # This prevents it from delaying the initial UI.
        # -------------------------------------------------
        try:
            store_status = get_cached_cloud_status()

            st.caption(
                f"Feast cloud history: "
                f"{store_status['count']} chronological Lahore observations"
                + (
                    f" (latest: {store_status['latest_timestamp']})"
                    if store_status["latest_timestamp"]
                    else ""
                )
            )

            if store_status["count"] < 30:
                st.warning(
                    f"Training needs {30 - store_status['count']} "
                    "more genuine hourly observations."
                )

        except Exception as error:
            st.warning(
                f"Feast status is unavailable: {error}"
            )

        # -------------------------------------------------
        # Total time
        # -------------------------------------------------
        st.success(
            f"Total forecast time: "
            f"{perf_counter() - total_started:.2f}s"
        )

    except Exception as error:
        st.error(
            f"Could not forecast safely: {error}"
        )


# ---------------------------------------------------------
# Data source and safety
# ---------------------------------------------------------
st.subheader("Data source and safety")

st.write(
    "Pollutants come from OpenWeather. The dashboard writes the latest "
    "Lahore observation to the cloud Feast Feature Store, then reads the "
    "serving vector from Feast. Lag and rolling AQI features are calculated "
    "from actual prior cloud observations; it refuses to forecast when "
    "history is insufficient."
)