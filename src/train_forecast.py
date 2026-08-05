import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from dotenv import load_dotenv
from supabase import create_client

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =====================================================
# MLFLOW CONFIGURATION
# =====================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "Pearls_AQI_Lahore_Forecasting"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

print("=" * 60)
print("MLflow Tracking Configured")
print("Tracking URI:", MLFLOW_TRACKING_URI)
print("Experiment:", MLFLOW_EXPERIMENT_NAME)
print("=" * 60)


# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_URL or SUPABASE_KEY is missing from .env file."
    )


# =====================================================
# CONNECT TO SUPABASE
# =====================================================

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

print("=" * 60)
print("Supabase Connection Successful")
print("=" * 60)


# =====================================================
# LOAD DATA FROM SUPABASE
# =====================================================

TABLE_NAME = "aqi_features"

all_rows = []
start = 0
batch_size = 1000

while True:

    response = (
        supabase
        .table(TABLE_NAME)
        .select("*")
        .range(start, start + batch_size - 1)
        .execute()
    )

    rows = response.data

    if not rows:
        break

    all_rows.extend(rows)

    print(
        f"Loaded {len(all_rows)} rows from Supabase..."
    )

    if len(rows) < batch_size:
        break

    start += batch_size


if not all_rows:
    raise ValueError(
        "No data found in Supabase table: aqi_features"
    )


df = pd.DataFrame(all_rows)


print("\n" + "=" * 60)
print("Supabase Dataset Loaded Successfully")
print("=" * 60)

print("Total rows:", len(df))

print("Columns found:")
print(df.columns.tolist())


# =====================================================
# SELECT LAHORE DATA
# =====================================================
# Lahore city_code = 3
# Karachi city_code = 2

if "city_code" not in df.columns:

    raise ValueError(
        "city_code column not found in aqi_features."
    )


df["city_code"] = pd.to_numeric(
    df["city_code"],
    errors="coerce"
)


df = df[df["city_code"] == 3].copy()


print("\nLahore rows found:", len(df))


if df.empty:

    raise ValueError(
        "No Lahore training data found in aqi_features."
    )


# =====================================================
# TARGET COLUMNS
# =====================================================

targets = {
    "day1": "day1_target",
    "day2": "day2_target",
    "day3": "day3_target"
}


# =====================================================
# FEATURE COLUMNS
# =====================================================

feature_columns = [
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


# =====================================================
# CHECK REQUIRED COLUMNS
# =====================================================

required_columns = (
    feature_columns
    + list(targets.values())
)


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing columns in Supabase: "
        + str(missing_columns)
    )


# =====================================================
# CONVERT FEATURES TO NUMERIC
# =====================================================

for column in feature_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# =====================================================
# CONVERT TARGETS TO NUMERIC
# =====================================================

for target in targets.values():

    df[target] = pd.to_numeric(
        df[target],
        errors="coerce"
    )


# =====================================================
# REMOVE INVALID TRAINING ROWS
# =====================================================

before_cleaning = len(df)

df = df.dropna(
    subset=required_columns
).copy()

after_cleaning = len(df)


print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

print(
    "Rows before cleaning:",
    before_cleaning
)

print(
    "Rows after cleaning:",
    after_cleaning
)

print(
    "Rows removed:",
    before_cleaning - after_cleaning
)


if after_cleaning < 10:

    raise ValueError(
        "Not enough valid training rows after cleaning."
    )


# =====================================================
# CREATE FEATURE DATASET
# =====================================================

X = df[feature_columns].copy()


# =====================================================
# SHOW FEATURES
# =====================================================

print("\n" + "=" * 60)
print("FEATURES USED FOR TRAINING")
print("=" * 60)

print(
    "Number of features:",
    len(feature_columns)
)

print("\nFeature List:")

for number, column in enumerate(
    feature_columns,
    start=1
):

    print(
        f"{number:02d}. {column}"
    )


# =====================================================
# CREATE MODELS FOLDER
# =====================================================

os.makedirs(
    "models",
    exist_ok=True
)


# =====================================================
# TRAIN THREE FORECAST MODELS
# =====================================================

for model_name, target in targets.items():

    print("\n" + "=" * 60)

    print(
        "Training",
        model_name.upper(),
        "Model"
    )

    print("=" * 60)

    # -------------------------------------------------
    # START MLFLOW RUN
    # -------------------------------------------------

    with mlflow.start_run(
        run_name=f"Lahore_{model_name}_RandomForest"
    ):

        # -------------------------------------------------
        # TARGET
        # -------------------------------------------------

        y = df[target].copy()

        print(
            "Training rows:",
            len(X)
        )


        # -------------------------------------------------
        # TRAIN / TEST SPLIT
        # -------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        print(
            "Training set:",
            len(X_train)
        )

        print(
            "Testing set:",
            len(X_test)
        )


        # -------------------------------------------------
        # RANDOM FOREST PARAMETERS
        # -------------------------------------------------

        n_estimators = 500
        max_depth = 20
        min_samples_split = 5
        min_samples_leaf = 2
        random_state = 42


        # -------------------------------------------------
        # LOG PARAMETERS TO MLFLOW
        # -------------------------------------------------

        mlflow.log_params({

            "city": "Lahore",

            "city_code": 3,

            "forecast_day": model_name,

            "target": target,

            "n_features": len(feature_columns),

            "n_estimators": n_estimators,

            "max_depth": max_depth,

            "min_samples_split": min_samples_split,

            "min_samples_leaf": min_samples_leaf,

            "random_state": random_state,

            "test_size": 0.20,

            "training_rows": len(X_train),

            "testing_rows": len(X_test)

        })


        # -------------------------------------------------
        # RANDOM FOREST MODEL
        # -------------------------------------------------

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )


        # -------------------------------------------------
        # TRAIN MODEL
        # -------------------------------------------------

        model.fit(
            X_train,
            y_train
        )


        # -------------------------------------------------
        # PREDICTIONS
        # -------------------------------------------------

        predictions = model.predict(
            X_test
        )


        # -------------------------------------------------
        # EVALUATION
        # -------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )


        print(
            f"MAE  : {mae:.3f}"
        )

        print(
            f"RMSE : {rmse:.3f}"
        )

        print(
            f"R2   : {r2:.3f}"
        )


        # -------------------------------------------------
        # LOG METRICS TO MLFLOW
        # -------------------------------------------------

        mlflow.log_metrics({

            "MAE": mae,

            "RMSE": rmse,

            "R2": r2

        })


        # -------------------------------------------------
        # SAVE MODEL LOCALLY
        # -------------------------------------------------

        model_path = (
            f"models/{model_name}_model.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        print(
            "\nModel Saved:",
            model_path
        )


        # -------------------------------------------------
        # LOG MODEL TO MLFLOW
        # -------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )

        print(
            "Model logged to MLflow successfully."
        )


        # -------------------------------------------------
        # FEATURE IMPORTANCE
        # -------------------------------------------------

        importance = pd.DataFrame({

            "Feature": feature_columns,

            "Importance": model.feature_importances_

        })


        importance = importance.sort_values(
            by="Importance",
            ascending=False
        )


        print("\nFeature Importance:")

        print(
            importance.to_string(index=False)
        )


        # -------------------------------------------------
        # LOG FEATURE IMPORTANCE TO MLFLOW
        # -------------------------------------------------

        mlflow.log_text(
            importance.to_csv(index=False),
            "feature_importance.csv"
        )


        # -------------------------------------------------
        # LOG TAGS TO MLFLOW
        # -------------------------------------------------

        mlflow.set_tags({

            "project": "Pearls_AQI_Predictor",

            "city": "Lahore",

            "model_type": "RandomForestRegressor",

            "feature_store": "Supabase",

            "data_source": "Supabase aqi_features",

            "forecast_horizon": model_name

        })


        print(
            "Metrics, parameters, feature importance, "
            "and model logged to MLflow."
        )


# =====================================================
# TRAINING COMPLETE
# =====================================================

print("\n" + "=" * 60)

print(
    "ALL FORECAST MODELS TRAINED SUCCESSFULLY"
)

print("=" * 60)

print("\nModels created:")

print(
    "models/day1_model.pkl"
)

print(
    "models/day2_model.pkl"
)

print(
    "models/day3_model.pkl"
)

print("\nMLflow experiment:")

print(
    MLFLOW_EXPERIMENT_NAME
)

print(
    "\nOpen MLflow Dashboard:"
)

print(
    MLFLOW_TRACKING_URI
)