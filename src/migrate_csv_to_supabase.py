import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client


# =========================================================
# SUPABASE CONNECTION
# =========================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_URL or SUPABASE_KEY is missing from .env"
    )

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# =========================================================
# DATASET
# =========================================================

CSV_FILE = "data/processed/forecast_dataset.csv"


# =========================================================
# EXACT COLUMNS FROM YOUR LAHORE DATASET
# =========================================================

CSV_COLUMNS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "aerosol_optical_depth",
    "dust",
    "uv_index",
    "year",
    "month",
    "day",
    "day_of_week",
    "aqi_lag1",
    "aqi_lag2",
    "aqi_lag3",
    "aqi_rolling_mean",
    "day1_target",
    "day2_target",
    "day3_target"
]


# =========================================================
# MAIN MIGRATION
# =========================================================

def main():

    print("\n===================================")
    print("Lahore CSV → Supabase Migration")
    print("===================================\n")

    # -----------------------------------------------------
    # Read CSV
    # -----------------------------------------------------

    df = pd.read_csv(CSV_FILE)

    print(f"CSV rows found: {len(df)}")

    # -----------------------------------------------------
    # Validate columns
    # -----------------------------------------------------

    missing_columns = [
        column
        for column in CSV_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns in CSV: {missing_columns}"
        )

    # -----------------------------------------------------
    # Lahore metadata
    # -----------------------------------------------------

    df["city"] = "3"
    df["city_code"] = 3

    df["latitude"] = 31.5204
    df["longitude"] = 74.3587

    # -----------------------------------------------------
    # Create timestamp from historical date columns
    # -----------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        {
            "year": df["year"],
            "month": df["month"],
            "day": df["day"]
        },
        errors="coerce"
    )

    # -----------------------------------------------------
    # Check for invalid dates
    # -----------------------------------------------------

    invalid_dates = df["timestamp"].isna().sum()

    if invalid_dates > 0:

        raise ValueError(
            f"{invalid_dates} rows have invalid "
            "year/month/day values."
        )

    # -----------------------------------------------------
    # Convert timestamp to ISO format
    # -----------------------------------------------------

    df["timestamp"] = df["timestamp"].apply(
        lambda x: x.isoformat()
        if pd.notna(x)
        else None
    )

    # -----------------------------------------------------
    # Historical CSV does not contain these live fields.
    # Do NOT invent values.
    # -----------------------------------------------------

    df["aqi"] = None
    df["temperature"] = None
    df["humidity"] = None
    df["wind_speed"] = None
    df["hour"] = None

    # -----------------------------------------------------
    # Supabase table columns
    # -----------------------------------------------------

    upload_columns = [
        "timestamp",
        "city",
        "city_code",
        "latitude",
        "longitude",
        "aqi",

        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",

        "aerosol_optical_depth",
        "dust",
        "uv_index",

        "temperature",
        "humidity",
        "wind_speed",

        "hour",
        "month",
        "year",
        "day",
        "day_of_week",

        "aqi_lag1",
        "aqi_lag2",
        "aqi_lag3",
        "aqi_rolling_mean",

        "day1_target",
        "day2_target",
        "day3_target"
    ]

    # -----------------------------------------------------
    # Convert NaN → None
    # -----------------------------------------------------

    df = df.where(pd.notna(df), None)

    # -----------------------------------------------------
    # Prepare records
    # -----------------------------------------------------

    records = df[upload_columns].to_dict(
        orient="records"
    )

    # -----------------------------------------------------
    # Upload in batches
    # -----------------------------------------------------

    batch_size = 500

    total = len(records)
    uploaded = 0

    for start in range(0, total, batch_size):

        batch = records[
            start:start + batch_size
        ]

        supabase \
            .table("aqi_features") \
            .insert(batch) \
            .execute()

        uploaded += len(batch)

        print(
            f"Uploaded {uploaded}/{total} rows"
        )

    # -----------------------------------------------------
    # Finished
    # -----------------------------------------------------

    print("\n===================================")
    print("MIGRATION SUCCESSFUL")
    print("===================================")

    print(
        f"Lahore historical rows uploaded: {uploaded}"
    )


if __name__ == "__main__":
    main()