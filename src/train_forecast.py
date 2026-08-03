import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# Load Forecast Dataset
# =====================================================

DATA_PATH = "data/processed/forecast_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Forecast Dataset Loaded Successfully")
print("=" * 60)

print(df.head())

# =====================================================
# Features & Targets
# =====================================================

X = df.drop(
    columns=[
        "day1_target",
        "day2_target",
        "day3_target",
        "year"          # remove year feature
    ],
    errors="ignore"
)

targets = {
    "day1": "day1_target",
    "day2": "day2_target",
    "day3": "day3_target"
}

print("\nFeatures Used:")
print(X.columns.tolist())

# =====================================================
# Create Models Folder
# =====================================================

os.makedirs("models", exist_ok=True)

# =====================================================
# Train Models
# =====================================================

for model_name, target in targets.items():

    print("\n" + "=" * 60)
    print(f"Training {model_name.upper()} Model")
    print("=" * 60)

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # model = RandomForestRegressor(
    #     n_estimators=300,
    #     random_state=42,
    #     n_jobs=-1
    # )
    model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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

    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

    model_path = f"models/{model_name}_model.pkl"

    joblib.dump(model, model_path)

    print(f"\nModel Saved : {model_path}")

    # ----------------------------
    # Feature Importance
    # ----------------------------

    importance = pd.DataFrame({

        "Feature": X.columns,
        "Importance": model.feature_importances_

    }).sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop Important Features")

    # print(
    #     importance.head(10)
    # )
    print(importance)

print("\n" + "=" * 60)
print("All Forecast Models Trained Successfully")
print("=" * 60)