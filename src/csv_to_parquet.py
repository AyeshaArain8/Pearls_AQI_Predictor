import pandas as pd

df = pd.read_csv("data/processed/forecast_dataset.csv")

df.to_parquet(
    "feature_repo/feature_repo/data/forecast_dataset.parquet",
    index=False
)

print("Done")
import pandas as pd

df = pd.read_csv("data/processed/forecast_dataset.csv")

df.to_parquet(
    "feature_repo/feature_repo/data/forecast_dataset.parquet",
    index=False
)

print("Done")