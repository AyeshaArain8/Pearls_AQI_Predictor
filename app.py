import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

from src.predict import forecast_aqi
from src.api_fetch import get_city_data


# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="AQI 3-Day Forecast Dashboard",
    page_icon="🌍",
    layout="wide"
)


# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #F8F9FA;
    }

    .block-container {
        padding-top: 2rem;
    }

    h1, h2, h3 {
        color: #1B4332;
    }

    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #E5E5E5;
    }

    .footer {
        text-align: center;
        color: gray;
        padding-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ======================================================
# HEALTH ADVICE
# ======================================================

def health_advice(category):

    if "Good" in category:
        return (
            "😊 Air quality is good. "
            "Outdoor activities are safe."
        )

    elif "Moderate" in category:
        return (
            "🙂 Air quality is acceptable "
            "for most people."
        )

    elif "Sensitive" in category:
        return (
            "😷 Sensitive people should reduce "
            "prolonged outdoor exposure."
        )

    elif "Unhealthy" in category:
        return (
            "⚠️ Limit outdoor activities "
            "and wear a mask if needed."
        )

    elif "Very" in category:
        return (
            "🚨 Avoid outdoor activities."
        )

    else:
        return (
            "☠️ Stay indoors and avoid "
            "outdoor exposure."
        )


# ======================================================
# CITY COORDINATES
# ======================================================

cities = {

    "Karachi": {
        "latitude": 24.8607,
        "longitude": 67.0011
    },

    "Lahore": {
        "latitude": 31.5204,
        "longitude": 74.3587
    }

}


# ======================================================
# HEADER
# ======================================================

st.title(
    "🌍 Air Quality Index - 3 Day Forecast"
)

st.write(
    """
    This dashboard predicts the **next 3-day AQI**
    using a trained **Random Forest Machine Learning model**
    and live pollution/weather data from OpenWeather API.
    """
)

st.divider()


# ======================================================
# CITY SELECTION
# ======================================================

st.subheader("📍 Select Location")

selected_city = st.selectbox(
    "Choose City",
    list(cities.keys())
)

location = cities[selected_city]


# ======================================================
# FETCH LIVE DATA
# ======================================================

try:

    city_data = get_city_data(
        selected_city,
        location["latitude"],
        location["longitude"]
    )

except Exception as e:

    st.error(
        f"API Error: {e}"
    )

    st.stop()


# ======================================================
# LOCATION CARD
# ======================================================

st.success(
    f"""
    📍 **City:** {city_data['city']}

    🌐 **Latitude:** {city_data['latitude']}

    🌐 **Longitude:** {city_data['longitude']}
    """
)


# ======================================================
# CURRENT AIR QUALITY
# ======================================================

st.subheader(
    "🌫 Current Air Quality"
)

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "PM2.5",
        f"{city_data['pm2_5']:.2f}"
    )


with c2:

    st.metric(
        "PM10",
        f"{city_data['pm10']:.2f}"
    )


with c3:

    st.metric(
        "CO",
        f"{city_data['carbon_monoxide']:.2f}"
    )


with c4:

    st.metric(
        "OpenWeather AQI",
        city_data["openweather_aqi"]
    )


st.divider()


# ======================================================
# CURRENT WEATHER
# ======================================================

st.subheader(
    "🌦 Current Weather"
)

w1, w2, w3, w4 = st.columns(4)


with w1:

    st.metric(
        "🌡 Temperature",
        f"{city_data['temperature']:.1f} °C"
    )


with w2:

    st.metric(
        "💧 Humidity",
        f"{city_data['humidity']} %"
    )


with w3:

    st.metric(
        "💨 Wind Speed",
        f"{city_data['wind_speed']:.2f} m/s"
    )


with w4:

    st.metric(
        "☁ Weather",
        city_data["weather"].title()
    )


st.divider()


# ======================================================
# DATE FEATURES
# ======================================================

today = datetime.now()

month = today.month

day = today.day

day_of_week = today.weekday()


# ======================================================
# ESTIMATE CURRENT AQI FOR MODEL
# ======================================================
# OpenWeather AQI is a 1-5 level.
# The trained ML model expects numeric AQI values.
#
# We estimate a numeric AQI from PM2.5 so that the
# scale is compatible with the historical training data.
# ======================================================

def estimate_current_aqi(pm2_5):

    if pm2_5 <= 12:
        return pm2_5 * (50 / 12)

    elif pm2_5 <= 35.4:
        return (
            50
            + (pm2_5 - 12)
            * (50 / (35.4 - 12))
        )

    elif pm2_5 <= 55.4:
        return (
            100
            + (pm2_5 - 35.4)
            * (50 / (55.4 - 35.4))
        )

    elif pm2_5 <= 150.4:
        return (
            150
            + (pm2_5 - 55.4)
            * (50 / (150.4 - 55.4))
        )

    elif pm2_5 <= 250.4:
        return (
            200
            + (pm2_5 - 150.4)
            * (100 / (250.4 - 150.4))
        )

    else:
        return 300


current_aqi_for_model = estimate_current_aqi(
    city_data["pm2_5"]
)


# ======================================================
# PREDICT BUTTON
# ======================================================

predict = st.button(
    "🚀 Predict 3-Day AQI Forecast",
    use_container_width=True
)


# ======================================================
# PREDICTION
# ======================================================

if predict:

    result = forecast_aqi(

        pm10=city_data["pm10"],

        pm2_5=city_data["pm2_5"],

        carbon_monoxide=city_data[
            "carbon_monoxide"
        ],

        nitrogen_dioxide=city_data[
            "nitrogen_dioxide"
        ],

        sulphur_dioxide=city_data[
            "sulphur_dioxide"
        ],

        ozone=city_data["ozone"],

        aerosol_optical_depth=city_data[
            "aerosol_optical_depth"
        ],

        dust=city_data["dust"],

        uv_index=city_data["uv_index"],

        month=month,

        day=day,

        day_of_week=day_of_week,

        current_aqi=current_aqi_for_model

    )


    st.success(
        "✅ Forecast Generated Successfully"
    )

    st.divider()


    # ==================================================
    # AQI CARDS
    # ==================================================

    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "🌤 Tomorrow AQI",
            result["Tomorrow"]["AQI"]
        )

        st.info(
            result["Tomorrow"]["Category"]
        )


    with c2:

        st.metric(
            "🌦 Day 2 AQI",
            result["Day 2"]["AQI"]
        )

        st.warning(
            result["Day 2"]["Category"]
        )


    with c3:

        st.metric(
            "🌧 Day 3 AQI",
            result["Day 3"]["AQI"]
        )

        st.error(
            result["Day 3"]["Category"]
        )


    # ==================================================
    # AQI TREND
    # ==================================================

    st.divider()

    st.subheader(
        "📈 AQI Trend"
    )


    chart_data = pd.DataFrame({

        "Day": [
            "Tomorrow",
            "Day 2",
            "Day 3"
        ],

        "AQI": [
            result["Tomorrow"]["AQI"],
            result["Day 2"]["AQI"],
            result["Day 3"]["AQI"]
        ]

    })


    chart = alt.Chart(
        chart_data
    ).mark_line(
        point=True
    ).encode(

        x=alt.X(
            "Day",
            title="Forecast Day"
        ),

        y=alt.Y(
            "AQI",
            title="Predicted AQI",
            scale=alt.Scale(
                domain=[0, 300]
            )
        ),

        tooltip=[
            "Day",
            "AQI"
        ]

    ).properties(
        height=350
    )


    st.altair_chart(
        chart,
        use_container_width=True
    )


    # ==================================================
    # FORECAST SUMMARY
    # ==================================================

    st.divider()

    st.subheader(
        "📋 Forecast Summary"
    )


    table = pd.DataFrame({

        "Day": [
            "Tomorrow",
            "Day 2",
            "Day 3"
        ],

        "Predicted AQI": [
            result["Tomorrow"]["AQI"],
            result["Day 2"]["AQI"],
            result["Day 3"]["AQI"]
        ],

        "Category": [
            result["Tomorrow"]["Category"],
            result["Day 2"]["Category"],
            result["Day 3"]["Category"]
        ]

    })


    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True
    )


    # ==================================================
    # HEALTH RECOMMENDATION
    # ==================================================

    st.divider()

    st.subheader(
        "❤️ Health Recommendation"
    )


    categories = [

        result["Tomorrow"]["Category"],

        result["Day 2"]["Category"],

        result["Day 3"]["Category"]

    ]


    priority = [

        "Hazardous",

        "Very Unhealthy",

        "Unhealthy",

        "Sensitive",

        "Moderate",

        "Good"

    ]


    selected = "Good"


    for priority_level in priority:

        if any(
            priority_level in category
            for category in categories
        ):

            selected = priority_level

            break


    st.info(
        health_advice(selected)
    )


# ======================================================
# FOOTER
# ======================================================

st.divider()

st.markdown(
    """
    <div class="footer">

    <h4>🌍 Air Quality Index Forecast System</h4>

    Developed by <b>Ayesha Shahid</b>

    <br><br>

    Pearls Technologies Internship Project

    </div>
    """,
    unsafe_allow_html=True
)