import os
import json
import glob
from datetime import datetime

from dotenv import load_dotenv
from supabase import create_client


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_URL or SUPABASE_KEY is missing from .env file"
    )

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------
# Fixed city encoding
# Same encoding used in training dataset
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Get raw JSON files
# ---------------------------------------------------------
def get_raw_files():

    files = glob.glob("data/raw/*.json")

    if not files:
        raise FileNotFoundError("No raw data found in data/raw/")

    return files


# ---------------------------------------------------------
# Extract features from JSON files
# ---------------------------------------------------------
def extract_features():

    all_features = []

    files = get_raw_files()

    for file_path in files:

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        pollution = data["air_pollution"]["list"][0]
        weather = data["weather"]

        timestamp = datetime.fromisoformat(data["timestamp"])

        city_name = data["city"]

        features = {

            "timestamp": timestamp.isoformat(),

            "city": city_mapping[city_name],

            "city_code": city_mapping[city_name],

            "latitude": float(data["latitude"]),

            "longitude": float(data["longitude"]),

            "aqi": float(pollution["main"]["aqi"]),

            "pm10": float(pollution["components"]["pm10"]),

            "pm2_5": float(pollution["components"]["pm2_5"]),

            "carbon_monoxide": float(
                pollution["components"]["co"]
            ),

            "nitrogen_dioxide": float(
                pollution["components"]["no2"]
            ),

            "ozone": float(
                pollution["components"]["o3"]
            ),

            "temperature": float(
                weather["main"]["temp"]
            ),

            "humidity": float(
                weather["main"]["humidity"]
            ),

            "wind_speed": float(
                weather["wind"]["speed"]
            ),

            "hour": int(timestamp.hour),

            "month": int(timestamp.month)
        }

        all_features.append(features)

    return all_features


# ---------------------------------------------------------
# Upload features to Supabase
# ---------------------------------------------------------
def save_features_to_supabase(features):

    if not features:
        print("No features to upload.")
        return

    try:

        response = (
            supabase
            .table("aqi_features")
            .insert(features)
            .execute()
        )

        print("\n===================================")
        print("Features uploaded to Supabase!")
        print("===================================")

        print(f"Rows uploaded: {len(features)}")

        if response.data:
            print("\nUploaded data:")
            for row in response.data:
                print(row)

    except Exception as e:

        print("\n===================================")
        print("ERROR uploading features!")
        print("===================================")

        print(e)
        raise


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    print("\nStarting feature pipeline...")

    features = extract_features()

    print(f"Found {len(features)} feature records.")

    save_features_to_supabase(features)

    print("\nFeature pipeline completed successfully!")


if __name__ == "__main__":
    main()