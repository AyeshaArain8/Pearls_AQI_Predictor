# Pearls Lahore AQI Predictor

Lahore-only, 3-day AQI forecasting with OpenWeather, Feast, managed cloud PostgreSQL, scikit-learn, a separate versioned Model Registry, Streamlit, SHAP, and GitHub Actions.

## Production architecture

`OpenWeather -> cloud PostgreSQL aqi_observations -> Feast PostgreSQL offline source + online store -> training/dashboard -> separate Model Registry`

Feast is selected because Hopsworks and Vertex AI are excluded and it provides a standard feature contract plus point-in-time historical and online retrieval. The intended backend is a managed cloud PostgreSQL project (for example, Neon PostgreSQL), not Supabase. The same remote database holds the `aqi_observations` offline source, Feast's SQL registry, and Feast online-serving tables. No local CSV, DuckDB, or local filesystem is used as the production feature source of truth.

The data collector writes a real Lahore OpenWeather observation into the cloud source table, then calls Feast materialization. The dashboard inserts its freshly fetched Lahore observation and reads the complete serving vector through Feast's `get_online_features`. Training obtains chronological raw observations through Feast's `get_historical_features`, then uses the shared feature contract to derive point-in-time lags, rolling values, and the three targets.

## Canonical feature schema

`src/feature_contract.py` owns the only `FEATURE_COLUMNS`: PM10, PM2.5, CO, NO2, SO2, O3, month/day/day-of-week, AQI lags 1-3, and a 3-observation rolling AQI mean. The Feast FeatureView imports this schema. AOD, dust, UV, city encoding, Karachi, and fabricated lag values are not used. The system refuses online prediction where the Feature Store lacks genuine lag history.

## Model Registry

`src/model_registry.py` is separate from Feast. Training creates versioned `day1`, `day2`, and `day3` model artifacts and version metadata under `models/registry/`; `latest.json` points to the approved model. Metadata includes the schema version, model metrics (RMSE, MAE, R2), training date, and source. The daily workflow commits approved registry artifacts to GitHub so deployed applications can load the intended version independently of the cloud Feature Store.

## SHAP and dashboard

The Streamlit dashboard loads the registered production day-1 model and feeds it the exact Feast online vector for a local SHAP explanation. It displays dated three-day predictions and hazardous AQI alerts. If SHAP is unavailable, the forecast remains available and the explanation error is shown clearly.

## Cloud setup and secrets

Create a managed PostgreSQL database and database/schema with permission to create Feast tables and the `aqi_observations` source table. Configure these locally in `.env`, in Streamlit deployment secrets, and as GitHub Secrets:

- `OPENWEATHER_API_KEY`
- `FEAST_POSTGRES_URL` (SQLAlchemy/psycopg URL)
- `FEAST_POSTGRES_HOST`, `FEAST_POSTGRES_PORT`, `FEAST_POSTGRES_DATABASE`
- `FEAST_POSTGRES_SCHEMA`, `FEAST_POSTGRES_USER`, `FEAST_POSTGRES_PASSWORD`

Run:

```bash
pip install -r requirements.txt
feast -c feature_repo/feature_repo apply
python src/feature_pipeline.py
python src/train_forecast.py
streamlit run app.py
pytest -q
```

The hourly workflow runs at `0 * * * *`; it validates secrets, applies Feast definitions, stores an observation remotely, and materializes it. The daily workflow runs at `0 0 * * *`; it retrieves history from the same Feast configuration, trains models, and persists the separate registry. Both have manual dispatch.

## Historical CSV limitation

`data/historical/air_quality_historical.csv` has no repository evidence proving Lahore provenance. It is deliberately not part of the active production pipeline and is not uploaded to Feast. Do not backfill it unless an auditable Lahore source is established first.
