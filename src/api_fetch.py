import os
import requests

from dotenv import load_dotenv


# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError(
        "API_KEY is missing from .env file."
    )


# =====================================================
# OPENWEATHER API URLS
# =====================================================

AIR_POLLUTION_URL = (
    "https://api.openweathermap.org/data/2.5/air_pollution"
)

WEATHER_URL = (
    "https://api.openweathermap.org/data/2.5/weather"
)


# =====================================================
# GET CITY DATA
# =====================================================

def get_city_data(
    city,
    latitude,
    longitude
):

    # =================================================
    # AIR POLLUTION DATA
    # =================================================

    pollution_params = {

        "lat": latitude,

        "lon": longitude,

        "appid": API_KEY

    }


    pollution_response = requests.get(
        AIR_POLLUTION_URL,
        params=pollution_params,
        timeout=15
    )


    pollution_response.raise_for_status()


    pollution_data = pollution_response.json()


    if not pollution_data.get("list"):

        raise ValueError(
            "No air pollution data returned from OpenWeather."
        )


    pollution = pollution_data["list"][0]


    components = pollution.get(
        "components",
        {}
    )


    openweather_aqi = pollution.get(
        "main",
        {}
    ).get(
        "aqi",
        1
    )


    # =================================================
    # WEATHER DATA
    # =================================================

    weather_params = {

        "lat": latitude,

        "lon": longitude,

        "appid": API_KEY,

        "units": "metric"

    }


    weather_response = requests.get(
        WEATHER_URL,
        params=weather_params,
        timeout=15
    )


    weather_response.raise_for_status()


    weather_data = weather_response.json()


    # =================================================
    # WEATHER VALUES
    # =================================================

    main_data = weather_data.get(
        "main",
        {}
    )


    wind_data = weather_data.get(
        "wind",
        {}
    )


    weather_list = weather_data.get(
        "weather",
        []
    )


    if weather_list:

        weather_description = weather_list[0].get(
            "description",
            "Unknown"
        )

    else:

        weather_description = "Unknown"


    # =================================================
    # RETURN DATA
    # =================================================

    return {

        # ---------------------------------------------
        # LOCATION
        # ---------------------------------------------

        "city": city,

        "latitude": latitude,

        "longitude": longitude,


        # ---------------------------------------------
        # AIR POLLUTION
        # ---------------------------------------------

        "pm10": float(
            components.get(
                "pm10",
                0
            )
        ),

        "pm2_5": float(
            components.get(
                "pm2_5",
                0
            )
        ),

        "carbon_monoxide": float(
            components.get(
                "co",
                0
            )
        ),

        "nitrogen_dioxide": float(
            components.get(
                "no2",
                0
            )
        ),

        "sulphur_dioxide": float(
            components.get(
                "so2",
                0
            )
        ),

        "ozone": float(
            components.get(
                "o3",
                0
            )
        ),


        # ---------------------------------------------
        # ADDITIONAL MODEL FEATURES
        # ---------------------------------------------
        # OpenWeather air-pollution endpoint does not
        # provide these values in the same response.
        # Safe defaults are therefore used.
        # ---------------------------------------------

        "aerosol_optical_depth": 0.30,

        "dust": 5.0,

        "uv_index": 4.0,


        # ---------------------------------------------
        # OPENWEATHER AQI
        # ---------------------------------------------

        "openweather_aqi": int(
            openweather_aqi
        ),


        # ---------------------------------------------
        # WEATHER
        # ---------------------------------------------

        "temperature": float(
            main_data.get(
                "temp",
                0
            )
        ),

        "humidity": int(
            main_data.get(
                "humidity",
                0
            )
        ),

        "wind_speed": float(
            wind_data.get(
                "speed",
                0
            )
        ),

        "weather": weather_description

    }