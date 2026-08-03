import joblib
import pandas as pd

# =====================================================
# Load Models
# =====================================================

day1_model = joblib.load("models/day1_model.pkl")
day2_model = joblib.load("models/day2_model.pkl")
day3_model = joblib.load("models/day3_model.pkl")


# =====================================================
# AQI Category
# =====================================================

def get_category(aqi):

    if aqi <= 50:
        return "🟢 Good"

    elif aqi <= 100:
        return "🟡 Moderate"

    elif aqi <= 150:
        return "🟠 Unhealthy for Sensitive Groups"

    elif aqi <= 200:
        return "🔴 Unhealthy"

    elif aqi <= 300:
        return "🟣 Very Unhealthy"

    else:
        return "⚫ Hazardous"


# =====================================================
# Forecast Function
# =====================================================

def forecast_aqi(
    pm10,
    pm2_5,
    carbon_monoxide,
    nitrogen_dioxide,
    sulphur_dioxide,
    ozone,
    aerosol_optical_depth,
    dust,
    uv_index,
    month,
    day,
    day_of_week,
    current_aqi
):

    # ---------- Estimated Previous AQI ----------
    # (Used because historical AQI is not available from user)

    aqi_lag1 = current_aqi

    aqi_lag2 = max(current_aqi - 8, 0)

    aqi_lag3 = max(current_aqi - 15, 0)

    aqi_rolling_mean = (
        aqi_lag1 +
        aqi_lag2 +
        aqi_lag3
    ) / 3

    # ---------- Model Features ----------

    features = pd.DataFrame([{

        "pm10": pm10,
        "pm2_5": pm2_5,
        "carbon_monoxide": carbon_monoxide,
        "nitrogen_dioxide": nitrogen_dioxide,
        "sulphur_dioxide": sulphur_dioxide,
        "ozone": ozone,
        "aerosol_optical_depth": aerosol_optical_depth,
        "dust": dust,
        "uv_index": uv_index,

        "month": month,
        "day": day,
        "day_of_week": day_of_week,

        "aqi_lag1": aqi_lag1,
        "aqi_lag2": aqi_lag2,
        "aqi_lag3": aqi_lag3,
        "aqi_rolling_mean": aqi_rolling_mean

    }])

    # ---------- Predictions ----------

    tomorrow = round(float(day1_model.predict(features)[0]), 1)
    day2 = round(float(day2_model.predict(features)[0]), 1)
    day3 = round(float(day3_model.predict(features)[0]), 1)

    return {

        "Tomorrow": {
            "AQI": tomorrow,
            "Category": get_category(tomorrow)
        },

        "Day 2": {
            "AQI": day2,
            "Category": get_category(day2)
        },

        "Day 3": {
            "AQI": day3,
            "Category": get_category(day3)
        }

    }


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    prediction = forecast_aqi(

        pm10=40,
        pm2_5=20,
        carbon_monoxide=500,
        nitrogen_dioxide=18,
        sulphur_dioxide=6,
        ozone=50,
        aerosol_optical_depth=0.25,
        dust=4,
        uv_index=3,

        month=8,
        day=1,
        day_of_week=5,

        current_aqi=75

    )

    print(prediction)