# Pearls Lahore AQI Predictor - Project Report

## Problem and scope

The deployed Feast source currently contains Lahore observations and forecasts Lahore AQI for days 1, 2, and 3. The local historical backup includes Karachi observations, but Karachi is not yet registered in the active Feast/Neon source or live collector; therefore no Karachi prediction is claimed. Supporting Karachi requires an approved, real Karachi ingestion and Feast materialization run before the same shared contract can be trained and served for that city.

## Data and AQI methodology

Live pollutant readings come from OpenWeather for Lahore coordinates. Numeric AQI is calculated from PM2.5 using the EPA breakpoint function in `src.feature_contract.pm25_to_us_aqi`. The historical backup file has Lahore hourly pollutant data but only categorical AQI labels; its backfill derives numeric AQI only through that same shared PM2.5 function, never through category-to-number mapping.

## Feature engineering and parity

The authoritative feature contract is `FEATURE_COLUMNS`: six shared pollutants, calendar fields, AQI lags 1-3, and a 3-observation rolling AQI mean. The cloud collector computes lag features from chronological Lahore history, Feast stores/serves the resulting feature vector, training reconstructs chronological targets with the same contract, and inference retrieves the Feast online vector. Missing lag history is an error, not a default value.

## Feature Store and backend

Feast project `pearls_lahore_aqi` uses Neon PostgreSQL for the source table, SQL registry, and online feature serving. `src.feature_store` is the only cloud access layer. The live collector is hourly; `src.historical_backfill` is separate, defaults to dry-run, and uses a `(city, event_timestamp)` unique index plus conflict-safe insertion when explicitly approved.

## Forecasting, registry, and explainability

Training retrieves historical observations through Feast, makes chronological day1/day2/day3 targets, evaluates RMSE/MAE/R2, and writes versioned RandomForest artifacts plus metadata into the separate Model Registry. SHAP uses the registered production day-1 model and the exact feature contract; training writes a global CSV report and Streamlit shows a local explanation.

## Automation and dashboard

GitHub Actions runs hourly data collection and daily training with Python 3.11 and cloud secrets. Scheduled jobs use the already-registered Feast definitions and run the project modules directly; Feast definition deployment is a separate controlled operation because a remote `feast apply` can block a scheduled collection run. Streamlit shows Feast readiness, the dated three-day forecast, metrics, hazardous alerts, and SHAP output. All cloud operations require configured OpenWeather and `FEAST_POSTGRES_*` secrets.

## Limitations

No backfill, training, or production registry should be claimed until the historical dry-run is explicitly approved and the Neon/Feast write, materialization, historical retrieval, online retrieval, and training steps have been run with real credentials.
