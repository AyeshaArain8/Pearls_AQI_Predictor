# Pearls AQI Predictor — Final Project Report

**Project:** Pearls_AQI_Predictor — Lahore AQI Predictor
**Scope:** Lahore, Pakistan only (single city, by design)

---

## 1. Project overview

The project forecasts Lahore's Air Quality Index (AQI) for the next 3 days
(day1/day2/day3, i.e. +24h/+48h/+72h) using a serverless-style stack: live
pollutant data from OpenWeather, a cloud Feast Feature Store on managed
PostgreSQL, scikit-learn models, a separate versioned Model Registry, a
Streamlit dashboard, SHAP explainability, and GitHub Actions automation.

## 2. Mentor requirements

The original brief (see `AQI_predict-1.pdf`) asked for:

1. A feature pipeline that fetches raw weather/pollutant data, computes
   time-based and derived features, and stores them in a Feature Store.
2. A backfill of historical (features, targets) for training.
3. A training pipeline that fetches from the Feature Store, trains and
   evaluates the best possible model (RMSE/MAE/R²), and stores it in a
   Model Registry.
4. CI/CD automation: the feature pipeline hourly, the training pipeline
   daily.
5. A web app (Streamlit) showing live and forecasted AQI.
6. EDA, model explainability (SHAP/LIME), and hazardous-AQI alerts.
7. A detailed final report.

This report documents what is **actually implemented** in the current
repository state, distinguishing that clearly from ideas for future work.

## 3. System architecture

```
OpenWeather API
      |
      v
src/api_fetch.py  --(raw observation)-->  src/feature_store.py
      |                                          |
      |                                   cloud PostgreSQL
      |                                   (aqi_observations table)
      |                                          |
      |                                   Feast (offline + online store)
      |                                          |
      v                                          v
Streamlit dashboard (app.py)  <---(features)---  src/predict.py
      |                                          ^
      |                                          |
      +---------- src/model_registry.py <-- src/train_forecast.py
                   (models/registry/*)      (Feast historical retrieval)
```

Key modules:

- `src/feature_contract.py` — the single source of truth for the feature
  schema (`FEATURE_COLUMNS`), AQI calculation, timestamp normalization,
  and train/serve feature-row construction. Both training and inference
  import from here, which is what keeps them aligned.
- `src/api_fetch.py` — OpenWeather collector for Lahore.
- `src/feature_store.py` — the only module that talks to cloud
  PostgreSQL/Feast (ingestion, online serving, historical retrieval).
- `src/historical_backfill.py` — controlled, dry-run-by-default backfill
  of historical Lahore rows into the cloud store.
- `src/train_forecast.py` — training pipeline.
- `src/model_registry.py` — versioned local/GitHub-committed model
  registry, separate from Feast.
- `src/predict.py` — inference: rebuilds today's feature vector from
  cloud history and calls the registered models.
- `src/explainability.py` — SHAP report generation.
- `src/eda.py` — exploratory data analysis (added as part of this pass).
- `app.py` — Streamlit dashboard.

## 4. Data source

Live data comes from the OpenWeather Air Pollution and Weather APIs for
Lahore's fixed coordinates (`LAHORE = {"latitude": 31.5204, "longitude":
74.3587}` in `feature_contract.py`). Historical training data comes from
Feast's cloud PostgreSQL offline store, built up from the hourly
collector plus an optional one-time backfill from
`data/historical/Backup/old_pakistan_air_quality_final_clean.csv`
(filtered to Lahore rows only; 2,184 hourly rows spanning
2025-11-06 to 2026-02-04, per the EDA in Section 24).

Note: `data/historical/air_quality_historical.csv` is **not** used in any
active pipeline. Its own `metadata.json` explicitly marks
`"lahore_verified": false` because the repository has no evidence of its
geographic provenance — it is intentionally excluded rather than risking
mislabeled training data.

## 5. OpenWeather ingestion

`src/api_fetch.get_lahore_data()` calls the OpenWeather `air_pollution`
and `weather` endpoints, requires `OPENWEATHER_API_KEY` (raises if
missing), and maps OpenWeather's component names to the project's
canonical pollutant names (`pm10`, `pm2_5`, `carbon_monoxide`,
`nitrogen_dioxide`, `sulphur_dioxide`, `ozone`). It never fabricates a
missing pollutant.

## 6. Feature engineering

`src/feature_contract.py` derives, on top of the six raw pollutants:

- Calendar features: `hour`, `month`, `day`, `day_of_week`.
- AQI lag features: `aqi_lag1`, `aqi_lag2`, `aqi_lag3` (previous 1/2/3
  chronological observations).
- `aqi_rolling_mean` (3-observation rolling mean of AQI).

Feature construction fails closed: if there isn't enough real chronological
history to compute genuine lag values, the row (or the whole request) is
rejected rather than filled with a default/fabricated value
(`DataQualityError` in `feature_contract.py`, exercised by
`tests/test_contract.py`).

## 7. AQI calculation

Numeric US AQI is calculated deterministically from PM2.5 using the
standard EPA breakpoint table (`pm25_to_us_aqi` in `feature_contract.py`).
This is the **only** place AQI is computed from raw data; the historical
backfill also reuses this same function rather than any category-to-number
mapping, so training and live serving use one consistent AQI definition.

## 8. Time-based features

`hour`, `month`, `day`, and `day_of_week` are derived directly from each
observation's timestamp (see Section 6), consistently for both the
training feature-row construction and live/online serving
(`tests/test_contract.py::test_hour_is_derived_from_timestamp` and
`test_train_and_serve_hour_schema_match`).

## 9. Lag/rolling features

`aqi_lag1/2/3` and `aqi_rolling_mean` are computed from real prior
chronological Lahore AQI observations only — never synthesized. This is
enforced both by the feature contract's fail-closed behaviour and by
tests (`test_live_insert_row_uses_three_real_prior_aqi_values`,
`test_insufficient_history_fails_closed`).

## 10. Feast Feature Store architecture

Project name: `pearls_lahore_aqi` (`feature_repo/feature_repo/feature_store.yaml`).
Both the offline store and the online store point at the same managed
PostgreSQL database (intended: Neon), configured entirely through
environment variables (`FEAST_POSTGRES_URL/HOST/PORT/DATABASE/SCHEMA/USER/PASSWORD`)
— no credentials are hardcoded anywhere in the repository.

The Feast `FeatureView` (`feature_repo/feature_repo/feature_definitions.py`)
is defined directly from the shared `FEATURE_COLUMNS`/`RAW_FEATURES`
contract, so the Feast schema cannot silently drift from the training/
serving schema. `city` is the entity/join key (single-city, but modelled
as a proper entity rather than hardcoded).

## 11. Historical backfill

`src/historical_backfill.py` reads the historical CSV, filters to
`city == "Lahore"`, recomputes AQI from PM2.5 via the shared function
(never trusts the CSV's own categorical label), validates and
deduplicates by `(city, event_timestamp)`, and **defaults to a dry run**
that only prints a summary. Writing to the cloud store requires an
explicit `--apply` flag. This is deliberately conservative given the
provenance concerns in Section 4.

## 12. Training dataset construction

`src/train_forecast.py` retrieves historical rows through
`src/feature_store.historical_features()` (a direct, indexed cloud
PostgreSQL query rather than Feast's point-in-time join, for performance
at this data scale — see `test_cloud_feature_store_has_no_local_fallback`),
resamples to one row per calendar day (`resample_daily`), and builds
day1/day2/day3 feature rows and targets with
`make_daily_feature_rows`. Training aborts if there are fewer than 30
raw historical observations or fewer than 15 usable daily rows, rather
than training on too little data.

## 13. Chronological train/test split

The daily dataset is split 80/20 **in chronological order**
(`split = int(len(dataset) * .8)`), so the test set is always the most
recent days and the model is never evaluated on data from before its
training window — an unshuffled, time-respecting split appropriate for
forecasting.

## 14. Model candidates

**As currently implemented, `src/train_forecast.py` trains a single
candidate: `RandomForestRegressor`** (`n_estimators=300,
min_samples_leaf=2, random_state=42`) for each of day1/day2/day3.

> **Important accuracy note:** earlier project documentation (and the
> task brief this report was written against) described a working
> Random Forest **vs.** Ridge comparison, with Ridge selected as the
> production model. That is **no longer what the current code does** —
> `train_forecast.py` only fits and registers Random Forest models, and
> the current `docs/PROJECT_REPORT.md` already reflects this ("writes
> versioned RandomForest artifacts"). This report describes the code as
> it exists now rather than repeating the earlier, now-inaccurate,
> claim about Ridge. Restoring a genuine Random Forest vs. Ridge
> comparison is listed as a near-term improvement in Section 30.

## 15. Random Forest vs. Ridge comparison

Not currently active in the training script (see Section 14). The
`sklearn.linear_model.Ridge` model mentioned in the original brief is not
imported or fit anywhere in the current `src/train_forecast.py`.

## 16. Evaluation metrics

Each horizon's model is evaluated on the chronological test split with:

- **RMSE** (root mean squared error)
- **MAE** (mean absolute error)
- **R²** (coefficient of determination)

computed with `scikit-learn`'s `mean_squared_error`, `mean_absolute_error`,
and `r2_score`, and stored in the model's registry metadata.

## 17. Selected models

Because only Random Forest is trained, Random Forest is registered for
all three horizons — there is no comparison-based selection step at
present. The most recently registered version is `20260830T030547Z`
(see `models/registry/latest.json`), with metrics:

| Horizon | RMSE | MAE | R² |
|---|---|---|---|
| day1 | 21.82 | 17.77 | 0.188 |
| day2 | 26.73 | 21.90 | -0.319 |
| day3 | 25.91 | 20.86 | -0.253 |

These are read directly from `models/registry/20260830T030547Z/metadata.json`
and are **not** rounded up or hidden. Day1 has modest positive
explanatory power; day2 and day3 currently have **negative R²**, meaning
the model performs worse than simply predicting the mean AQI at those
horizons on this test window. This is reported as-is, consistent with
the project's "do not fabricate or hide results" requirement — negative
R² is a real and informative result, not a bug to paper over.

(An earlier registered version, `20260823T005905Z`, is the closest match
in the registry to the `20260823T164140Z` version referenced in the
original task brief; the exact timestamp in the brief does not appear in
the current registry, most likely because the daily training workflow has
continued to register new versions since that snapshot was taken.)

## 18. Versioned Model Registry

`src/model_registry.py` is a small, Feast-independent registry: each
training run creates `models/registry/<UTC timestamp>/` with
`day1_model.joblib`, `day2_model.joblib`, `day3_model.joblib`, and
`metadata.json` (schema version, feature/target columns, row counts,
metrics, training timestamp, `approved: true`). `models/registry/latest.json`
points at the current approved version. `load_latest()` is `lru_cache`d
so Streamlit doesn't reload ~20MB of models on every rerun, and the cache
is explicitly cleared on every new registration. As of this report there
are 28 registered versions, one roughly per day since 2026-08-09, showing
the daily automation (Section 23) has been running in production.

## 19. 3-day forecasting

`src/predict.forecast_aqi()` loads the latest approved registry version,
rebuilds *today's* real daily feature vector from cloud history using the
exact same `resample_daily`/`make_daily_feature_rows` functions training
uses (so train and serve cannot silently diverge), and predicts day1/day2/day3
AQI. It refuses to forecast Lahore for any other city
(`test_dashboard_and_inference_are_lahore_only`) and refuses to forecast
at all if there isn't enough consecutive daily history.

## 20. SHAP explainability

`src/explainability.py` provides:

- `save_global_feature_report()` — called at the end of every training
  run to write a global SHAP feature-importance CSV to `reports/`
  (e.g. `reports/shap_day1_<version>.csv`).
- `local_feature_importance()` — used by the dashboard to explain a
  single live prediction (the day1 model, on today's feature vector).

The dashboard shows this as a per-feature SHAP bar chart, and degrades
gracefully (shows an info message, forecast still works) if SHAP fails.

## 21. Streamlit dashboard

`app.py` shows, in order: cached Feast history status, a button-triggered
live fetch (OpenWeather → cloud ingest → forecast), current PM2.5/PM10/
derived AQI/weather, a hazardous/unhealthy AQI alert for the current
reading, a 3-day forecast table and line chart, an alert if any forecast
day is hazardous, per-day category alerts for Unhealthy-or-worse forecast
days, model metrics (raw JSON, including the negative R² values as-is),
a local SHAP explanation, and a data-source/safety note.

## 22. Hazardous AQI alerts

Implemented using the project's derived US AQI and the standard category
thresholds:

| AQI | Category |
|---|---|
| 0–50 | Good |
| 51–100 | Moderate |
| 101–150 | Unhealthy for Sensitive Groups |
| 151–200 | Unhealthy |
| 201–300 | Very Unhealthy |
| 301–500 | Hazardous |

`get_category()` in `src/predict.py` already implemented this exact
table. As part of this pass, `app.py` was extended (previously it only
flagged forecasts above 300) so that:

- The **current** AQI reading is checked against the category table and
  shown as a warning (Unhealthy for Sensitive Groups) or error (Unhealthy
  and worse).
- **Each of the three forecast days** is checked the same way, in
  addition to the pre-existing "any forecast > 300" hazardous check,
  which was left untouched.

No synthetic AQI values are used for alerting — only the same
`observation["aqi"]` and `forecast["aqi"]` values already shown on the
dashboard.

## 23. CI/CD automation

Two GitHub Actions workflows already existed and, on inspection, are
correct and already running in production (see the 28 registry versions
in Section 18 as evidence the daily job has actually been executing):

- **`.github/workflows/data_pipeline.yml`** — `cron: "0 * * * *"` (hourly)
  plus `workflow_dispatch`. Installs dependencies, validates that
  `OPENWEATHER_API_KEY` and all `FEAST_POSTGRES_*` secrets are present,
  then runs `python -m src.feature_pipeline`. No secret is hardcoded;
  everything comes from `${{ secrets.* }}`.
- **`.github/workflows/model_training.yml`** — `cron: "0 0 * * *"`
  (daily) plus `workflow_dispatch`. Validates the same Feast secrets,
  runs `python -m src.train_forecast`, uploads the trained models and
  SHAP report as workflow artifacts, and commits the approved
  `models/registry/` back to the repository.

Both workflows use the project's own scripts directly (no separate
orchestrator), both YAML files parse successfully, and neither was
modified in this pass since they already satisfy the mentor requirement
(hourly features, daily training, GitHub Secrets, manual dispatch
preserved).

## 24. EDA findings

Added in this pass as `src/eda.py`, run with `python -m src.eda`, writing
`reports/eda/eda_report.md` plus three charts. It uses the same real,
Lahore-filtered historical dataset as the backfill script (not the
unverified `air_quality_historical.csv`). Headline findings from the most
recent run:

- **2,184 hourly Lahore observations**, 2025-11-06 to 2026-02-04, with
  **zero missing values** in any pollutant or AQI column.
- Mean AQI over the period: **191.8** (std 51.6), min 60.9, max 383.3.
- Category breakdown: Good 0, Moderate 82, Unhealthy for Sensitive
  Groups 206, **Unhealthy 1,174 (the most common category)**, Very
  Unhealthy 632, Hazardous 90.
- PM2.5 and PM10 correlate almost perfectly with AQI (r ≈ 0.98 for both,
  expected since AQI is derived from PM2.5), followed by NO₂ (r ≈ 0.65),
  CO (r ≈ 0.63), and SO₂ (r ≈ 0.55). Ozone is the only pollutant with a
  meaningful **negative** correlation with AQI (r ≈ -0.52) in this
  dataset.
- The full numeric summary tables and time-series/correlation charts are
  in `reports/eda/`.

## 25. Data limitations

- The historical backfill source (`old_pakistan_air_quality_final_clean.csv`)
  covers roughly 3 months (Nov 2025–Feb 2026); this is a fairly narrow
  window for capturing seasonal AQI patterns.
- `data/historical/air_quality_historical.csv` has no established Lahore
  provenance and is intentionally unused.
- Live training data volume depends on how long the hourly collector has
  been running; `train_forecast.py` requires at least 30 raw observations
  and 15 usable daily rows, which is a low bar for a genuinely reliable
  model.
- All raw pollutant/weather data ultimately comes from a single
  provider (OpenWeather); there's no cross-source validation.

## 26. Model limitations

- Only Random Forest is currently trained/compared (Section 14); no
  linear baseline (Ridge) or deep-learning model is currently
  implemented, despite being suggested in the original brief as options
  to explore.
- Day2 and day3 currently have **negative R²** on the chronological test
  split (Section 17), meaning the current model is not yet reliable
  beyond a 24-hour horizon.
- The daily-resampled dataset is small (order of tens to low hundreds of
  rows), which limits how much a Random Forest (or any model) can learn,
  especially for the longer horizons.
- The model is retrained daily on whatever history exists at that time;
  there's no explicit backtesting across multiple historical windows,
  so today's reported metrics reflect one specific train/test split.

## 27. Safety considerations

- No API keys, database URLs, or passwords are hardcoded; all secrets
  are read from environment variables locally (`.env`, git-ignored) or
  GitHub Secrets in CI.
- The system fails closed rather than guessing: missing pollutants,
  insufficient lag history, or an unapproved model registry version all
  raise errors instead of silently producing a plausible-looking but
  fabricated forecast.
- The historical backfill defaults to a dry run and requires an explicit
  `--apply` flag before writing to the cloud store.
- The dashboard now surfaces hazardous/unhealthy AQI alerts for both the
  current reading and each forecast day (Section 22), so a user glancing
  at the dashboard isn't relying on manually reading raw numbers to
  understand health risk.
- The dashboard is explicitly Lahore-only and rejects any other city,
  both in the UI copy and in `predict.forecast_aqi()`.

## 28. How to run locally

```bash
pip install -r requirements.txt

# One-time: register the Feast feature definitions
feast -c feature_repo/feature_repo apply

# Hourly step (fetch + store one live observation)
python -m src.feature_pipeline

# Daily step (train + register models)
python -m src.train_forecast

# Optional: exploratory data analysis
python -m src.eda

# Run the dashboard
streamlit run app.py

# Run the test suite
pytest -q
```

Required environment variables (in `.env` locally, or GitHub
Secrets/Streamlit secrets in deployment): `OPENWEATHER_API_KEY`,
`FEAST_POSTGRES_URL`, `FEAST_POSTGRES_HOST`, `FEAST_POSTGRES_PORT`,
`FEAST_POSTGRES_DATABASE`, `FEAST_POSTGRES_SCHEMA`, `FEAST_POSTGRES_USER`,
`FEAST_POSTGRES_PASSWORD`.

## 29. How the automated pipelines work

- **Hourly (`data_pipeline.yml`)**: checks out the repo, installs
  dependencies, validates that all required secrets are present, then
  runs `python -m src.feature_pipeline`, which fetches one fresh
  OpenWeather Lahore observation and ingests it into the cloud
  PostgreSQL/Feast store.
- **Daily (`model_training.yml`)**: checks out the repo, installs
  dependencies, validates secrets, runs `python -m src.train_forecast`
  (Feast historical retrieval → daily resample → train/evaluate
  day1/day2/day3 Random Forest models → register in
  `models/registry/`), uploads the trained models and SHAP report as
  workflow artifacts, then commits the newly approved
  `models/registry/` directory back to the repository so the deployed
  dashboard can load it independently of the live Feast connection.
- Both workflows also support manual triggering via
  `workflow_dispatch` for on-demand runs.

## 30. Future improvements

*(Ideas only — none of these are implemented today.)*

- Restore a genuine Random Forest vs. Ridge (and/or other linear/
  gradient-boosted baselines) comparison in `train_forecast.py`, with
  the best-test-RMSE model selected and that selection recorded in the
  registry metadata, as originally intended.
- Explore deep-learning or sequence models (e.g. a simple LSTM) as the
  brief suggested, now that there's a growing daily-resampled dataset.
- Extend the historical backfill window beyond ~3 months to better
  capture seasonal effects, once an auditable additional Lahore source
  is identified.
- Add basic backtesting across multiple rolling train/test windows
  instead of a single chronological split, to get a more robust read on
  day2/day3 performance.
- Add automated data-quality/drift monitoring on top of the existing
  fail-closed validation.
- Surface EDA charts (Section 24) directly in the Streamlit dashboard
  rather than only as a static report.
