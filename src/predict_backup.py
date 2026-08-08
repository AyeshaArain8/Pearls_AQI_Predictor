import joblib
import pandas as pd



# LOAD TRAINED MODELS


day1_model = joblib.load("models/day1_model.pkl")
day2_model = joblib.load("models/day2_model.pkl")
day3_model = joblib.load("models/day3_model.pkl")

# AQI CATEGORY

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
# FORECAST FUNCTION
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

    
    # AQI LAG FEATURES
   
    aqi_lag1 = current_aqi

    aqi_lag2 = max(current_aqi - 8, 0)

    aqi_lag3 = max(current_aqi - 15, 0)

   
    # ROLLING AQI MEAN
   
    aqi_rolling_mean = (
        aqi_lag1
        + aqi_lag2
        + aqi_lag3
    ) / 3

    
    # MODEL FEATURES
    
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


    
    # EXACT FEATURE ORDER USED DURING TRAINING
    
    feature_order = [

        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "aerosol_optical_depth",
        "dust",
        "uv_index",
        "month",
        "day",
        "day_of_week",
        "aqi_lag1",
        "aqi_lag2",
        "aqi_lag3",
        "aqi_rolling_mean"

    ]

    
    # APPLY FEATURE ORDER
    
    features = features[feature_order]


    
    # PREDICTIONS
    
    tomorrow = round(
        float(day1_model.predict(features)[0]),
        1
    )

    day2 = round(
        float(day2_model.predict(features)[0]),
        1
    )

    day3 = round(
        float(day3_model.predict(features)[0]),
        1
    )

    
    # RETURN FORECAST

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



# TESTING

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
        day=4,
        day_of_week=1,

        current_aqi=75

    )


    print("\n" + "=" * 60)
    print("AQI FORECAST TEST")
    print("=" * 60)


    print("\nTomorrow:")

    print(
        "AQI:",
        prediction["Tomorrow"]["AQI"]
    )

    print(
        "Category:",
        prediction["Tomorrow"]["Category"]
    )


    print("\nDay 2:")

    print(
        "AQI:",
        prediction["Day 2"]["AQI"]
    )

    print(
        "Category:",
        prediction["Day 2"]["Category"]
    )


    print("\nDay 3:")

    print(
        "AQI:",
        prediction["Day 3"]["AQI"]
    )

    print(
        "Category:",
        prediction["Day 3"]["Category"]
    )


    print("\n" + "=" * 60)
    print("Prediction Test Completed")
    print("=" * 60)