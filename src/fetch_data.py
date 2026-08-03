import os
import json
import csv
from datetime import datetime

import requests

from config import API_KEY


AIR_POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


def fetch_air_pollution(latitude, longitude):

    params = {
        "lat": latitude,
        "lon": longitude,
        "appid": API_KEY
    }

    response = requests.get(AIR_POLLUTION_URL, params=params)

    if response.status_code != 200:
        raise Exception(f"Air Pollution API Error: {response.status_code}")

    return response.json()


def fetch_weather(latitude, longitude):

    params = {
        "lat": latitude,
        "lon": longitude,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(WEATHER_URL, params=params)

    if response.status_code != 200:
        raise Exception(f"Weather API Error: {response.status_code}")

    return response.json()


def read_cities():

    cities = []

    with open("data/cities.csv", "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            cities.append(row)

    return cities


def save_raw_data(data):

    os.makedirs("data/raw", exist_ok=True)

    filename = datetime.now().strftime("%Y%m%d_%H%M%S.json")
    filepath = os.path.join("data", "raw", filename)

    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Data saved: {filepath}")


def main():

    cities = read_cities()

    for city in cities:

        print("\n====================")
        print("Fetching:", city["city"])
        print("====================")

        air_data = fetch_air_pollution(
            city["latitude"],
            city["longitude"]
        )

        weather_data = fetch_weather(
            city["latitude"],
            city["longitude"]
        )


        combined_data = {

            "timestamp": datetime.now().isoformat(),

            "city": city["city"],

            "latitude": city["latitude"],

            "longitude": city["longitude"],

            "air_pollution": air_data,

            "weather": weather_data

        }


        save_raw_data(combined_data)


        pollution = air_data["list"][0]


        print("AQI:", pollution["main"]["aqi"])
        print("PM2.5:", pollution["components"]["pm2_5"])
        print("PM10:", pollution["components"]["pm10"])

        print(
            "Temperature:",
            weather_data["main"]["temp"],
            "°C"
        )


if __name__ == "__main__":
    main()