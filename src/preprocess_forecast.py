import os
import pandas as pd

# =====================================================
# Load Historical Dataset
# =====================================================

DATA_PATH = "data/historical/air_quality_historical.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Historical Dataset Loaded Successfully")
print("=" * 60)
print(df.head())

# =====================================================
# Remove Missing Values
# =====================================================

df = df.dropna().reset_index(drop=True)

print("\nMissing Values Removed")
print("Dataset Shape:", df.shape)

# =====================================================
# Convert Date
# =====================================================

df["date"] = pd.to_datetime(df["date"])

# =====================================================
# Date Features
# =====================================================

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek

# =====================================================
# Lag Features (NEW)
# =====================================================

df["aqi_lag1"] = df["us_aqi"].shift(1)
df["aqi_lag2"] = df["us_aqi"].shift(2)
df["aqi_lag3"] = df["us_aqi"].shift(3)

df["aqi_rolling_mean"] = (
    df["us_aqi"]
    .rolling(window=3)
    .mean()
)

# Remove NaN rows created by lag features
df = df.dropna().reset_index(drop=True)

# =====================================================
# Forecast Targets
# =====================================================

df["day1_target"] = df["us_aqi"].shift(-1)
df["day2_target"] = df["us_aqi"].shift(-2)
df["day3_target"] = df["us_aqi"].shift(-3)

# Remove last rows
df = df.dropna().reset_index(drop=True)

# =====================================================
# Final Dataset
# =====================================================

forecast_df = df[
    [
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

        # Lag Features
        "aqi_lag1",
        "aqi_lag2",
        "aqi_lag3",
        "aqi_rolling_mean",

        # Targets
        "day1_target",
        "day2_target",
        "day3_target"
    ]
]

# =====================================================
# Save Dataset
# =====================================================

os.makedirs("data/processed", exist_ok=True)

OUTPUT_PATH = "data/processed/forecast_dataset.csv"

forecast_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nForecast Dataset Created Successfully")
print(forecast_df.head())
print("\nDataset Shape:", forecast_df.shape)
print(f"\nSaved To : {OUTPUT_PATH}")