import os
import json
import glob
import pandas as pd
from datetime import datetime


# Fixed city encoding (same as training dataset)
city_mapping = {
    "Faisalabad": 0,
    "Islamabad": 1,
    "Karachi": 2,
    "Lahore": 3,
    "Multan": 4,
    "Peshawar": 5,
    "Quetta": 6,
    "Rahim Yar Khan": 7,
    "Rawalpindi": 8,
    "Sialkot": 9
}


def get_raw_files():

    files = glob.glob("data/raw/*.json")

    if not files:
        raise FileNotFoundError("No raw data found!")

    return files


def extract_features():

    all_features = []

    files = get_raw_files()

    for file_path in files:

        with open(file_path, "r") as file:
            data = json.load(file)

        pollution = data["air_pollution"]["list"][0]
        weather = data["weather"]

        timestamp = datetime.fromisoformat(data["timestamp"])

        features = {

            "timestamp": timestamp,

            "city": city_mapping[data["city"]],

            "latitude": float(data["latitude"]),

            "longitude": float(data["longitude"]),

            "aqi": pollution["main"]["aqi"],

            "pm10": pollution["components"]["pm10"],

            "pm2_5": pollution["components"]["pm2_5"],

            "carbon_monoxide": pollution["components"]["co"],

            "nitrogen_dioxide": pollution["components"]["no2"],

            "ozone": pollution["components"]["o3"],

            "temperature": weather["main"]["temp"],

            "humidity": weather["main"]["humidity"],

            "wind_speed": weather["wind"]["speed"],

            "hour": timestamp.hour,

            "month": timestamp.month

        }

        all_features.append(features)

    return all_features


def save_features(features):

    os.makedirs("data/processed", exist_ok=True)

    df = pd.DataFrame(features)

    output_file = "data/processed/features.csv"

    df.to_csv(output_file, index=False)

    print(f"Features saved to {output_file}")

    print("\nExtracted Features:")
    print(df)


def main():

    features = extract_features()

    save_features(features)


if __name__ == "__main__":
    main()