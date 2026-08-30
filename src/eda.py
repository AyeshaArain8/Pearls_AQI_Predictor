"""Lightweight, reproducible EDA for the Lahore AQI project.

Uses the same real Lahore historical dataset that
`src.historical_backfill` prepares (city-filtered rows from
`data/historical/Backup/old_pakistan_air_quality_final_clean.csv`),
and the same shared feature contract (`src.feature_contract`) so
numbers here match what training/backfill actually use.

No fabricated statistics: everything is computed directly from the
CSV on disk. Run with:

    python -m src.eda

Outputs a Markdown summary and a few PNG charts under `reports/eda/`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.feature_contract import LAHORE, RAW_FEATURES, pm25_to_us_aqi
from src.historical_backfill import DEFAULT_DATASET

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "eda"


def load_lahore_history(dataset: Path = DEFAULT_DATASET) -> pd.DataFrame:
    """Load the real Lahore-only historical rows used for backfill/training."""
    source = pd.read_csv(dataset)
    lahore = source.loc[source["city"].eq(LAHORE["name"])].copy()
    lahore["timestamp"] = pd.to_datetime(lahore["timestamp"], utc=True, errors="coerce")
    lahore = lahore.sort_values("timestamp").reset_index(drop=True)
    lahore["aqi"] = lahore["pm2_5"].map(pm25_to_us_aqi)
    return lahore


def category_counts(aqi: pd.Series) -> pd.Series:
    bins = [-0.1, 50, 100, 150, 200, 300, 10_000]
    labels = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]
    return pd.cut(aqi, bins=bins, labels=labels).value_counts().reindex(labels)


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_lahore_history()

    n_obs = len(df)
    date_min, date_max = df["timestamp"].min(), df["timestamp"].max()
    missing = df[RAW_FEATURES + ["aqi"]].isna().sum()
    missing_pct = (missing / n_obs * 100).round(2)

    pollutant_summary = df[RAW_FEATURES].describe().T
    aqi_summary = df["aqi"].describe()
    cat_counts = category_counts(df["aqi"])

    corr = df[RAW_FEATURES + ["aqi"]].corr(numeric_only=True)["aqi"].drop("aqi").sort_values(
        ascending=False
    )

    daily = df.set_index("timestamp")["aqi"].resample("1D").mean()

    # ---- charts -------------------------------------------------
    plt.figure(figsize=(6, 4))
    df["aqi"].plot(kind="hist", bins=40, color="#4C72B0")
    plt.title("Lahore AQI distribution (hourly, PM2.5-derived)")
    plt.xlabel("US AQI")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "aqi_distribution.png")
    plt.close()

    plt.figure(figsize=(9, 4))
    daily.plot(color="#C44E52")
    plt.title("Lahore daily mean AQI over time")
    plt.ylabel("US AQI")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "aqi_over_time.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    corr.plot(kind="barh", color="#55A868")
    plt.title("Correlation of pollutants with AQI")
    plt.xlabel("Pearson correlation")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pollutant_aqi_correlation.png")
    plt.close()

    # ---- markdown summary ----------------------------------------
    lines = [
        "# Lahore AQI - Exploratory Data Analysis",
        "",
        f"Source: `{DEFAULT_DATASET.as_posix()}` (filtered to `city == \"Lahore\"`), "
        "the same file `src.historical_backfill` prepares for backfill.",
        "",
        "## Overview",
        f"- Observations: **{n_obs}**",
        f"- Date range: **{date_min.date()}** to **{date_max.date()}**",
        f"- Pollutant/AQI missing values:\n"
        + "\n".join(f"  - {col}: {missing[col]} ({missing_pct[col]}%)" for col in RAW_FEATURES + ["aqi"]),
        "",
        "## AQI distribution",
        "```",
        aqi_summary.to_string(),
        "```",
        "",
        "AQI category breakdown (US AQI, PM2.5-derived):",
        "",
        "```",
        cat_counts.to_string(),
        "```",
        "",
        "## Pollutant distributions",
        "```",
        pollutant_summary.to_string(),
        "```",
        "",
        "## Correlation with AQI",
        "```",
        corr.to_string(),
        "```",
        "",
        "## Charts",
        "- `aqi_distribution.png` - histogram of hourly AQI values",
        "- `aqi_over_time.png` - daily mean AQI trend across the full history",
        "- `pollutant_aqi_correlation.png` - which pollutants track AQI most closely",
        "",
        "## Observations",
        f"- PM2.5 and PM10 are the strongest correlates of AQI here "
        f"(AQI is derived directly from PM2.5, so this is expected), "
        f"with corr(PM2.5, AQI) = {corr['pm2_5']:.3f} and corr(PM10, AQI) = {corr['pm10']:.3f}.",
        f"- The most common AQI category in this history is "
        f"**{cat_counts.idxmax()}** ({int(cat_counts.max())} of {n_obs} hourly rows).",
        f"- Mean AQI over the full period is {aqi_summary['mean']:.1f}, "
        f"with a max of {aqi_summary['max']:.1f}, showing Lahore regularly experiences "
        "unhealthy-or-worse air quality in this dataset.",
    ]

    report_path = OUT_DIR / "eda_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "n_obs": n_obs,
        "date_min": str(date_min.date()),
        "date_max": str(date_max.date()),
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    result = run()
    print(f"EDA complete: {result['n_obs']} Lahore rows, "
          f"{result['date_min']} to {result['date_max']}.")
    print(f"Report written to {result['report_path']}")
