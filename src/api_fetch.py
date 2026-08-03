import os
import requests
from dotenv import load_dotenv


# Load .env file
load_dotenv()


API_KEY = os.getenv("API_KEY")


# ======================================================
# Fetch Air Pollution + Weather Data
# ======================================================

def get_city_data(city, latitude, longitude):

    if not API_KEY:
        raise ValueError("API_KEY not found in .env file")


    # Air Pollution API

    pollution_url = (
        "https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={latitude}&lon={longitude}&appid={API_KEY}"
    )


    pollution_response = requests.get(
        pollution_url
    )

    pollution_data = pollution_response.json()



    # Weather API

    weather_url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={latitude}&lon={longitude}&appid={API_KEY}&units=metric"
    )


    weather_response = requests.get(
        weather_url
    )

    weather_data = weather_response.json()



    pollution = pollution_data["list"][0]


    return {

        "city": city,

        "latitude": latitude,

        "longitude": longitude,


        "pm10": pollution["components"]["pm10"],

        "pm2_5": pollution["components"]["pm2_5"],

        "carbon_monoxide": pollution["components"]["co"],

        "nitrogen_dioxide": pollution["components"]["no2"],

        "sulphur_dioxide": pollution["components"]["so2"],

        "ozone": pollution["components"]["o3"],


        "temperature": weather_data["main"]["temp"],

        "humidity": weather_data["main"]["humidity"],

        "wind_speed": weather_data["wind"]["speed"],

        "weather": weather_data["weather"][0]["description"],

        "openweather_aqi": pollution["main"]["aqi"]

    }

if __name__ == "__main__":

    data = get_city_data(
        "Karachi",
        24.8607,
        67.0011
    )

    print(data)