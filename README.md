# Pearls Lahore AQI Predictor

🚀 Live Demo: https://ayesha-aqi-predictor.streamlit.app/

👉 Open Lahore AQI Predictor — Live Streamlit Dashboard
An end-to-end machine learning and MLOps system for forecasting **Lahore's Air Quality Index (AQI)** for the next **24, 48, and 72 hours** using real environmental observations, cloud-based feature storage, a versioned model registry, automated pipelines, explainable AI, and an interactive Streamlit dashboard.

## Project Overview

Air pollution is a major environmental and public-health concern in Lahore. This project provides a complete forecasting pipeline that collects real air-quality observations, engineers time-series features, trains multi-horizon forecasting models, evaluates their performance, registers approved model versions, and serves forecasts through a web dashboard.

The system is designed around the following principles:

* Lahore-only production scope
* Real OpenWeather observations
* Cloud PostgreSQL as the production data source
* Feast for feature-store infrastructure and online serving
* A shared feature contract between training and prediction
* A separate versioned Model Registry
* 24/48/72-hour AQI forecasting
* SHAP-based local explainability
* Automated GitHub Actions workflows
* Streamlit dashboard for real-time monitoring
* No fabricated pollutant observations or synthetic lag values

---

## System Architecture

```text
                    ┌─────────────────────┐
                    │     OpenWeather     │
                    │  Air Pollution API  │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ Managed Cloud PostgreSQL │
                 │     aqi_observations      │
                 └────────────┬─────────────┘
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
      ┌──────────────────┐        ┌────────────────────┐
      │  Feast Feature   │        │ Historical feature │
      │  Store / Online  │        │ retrieval & daily  │
      │      Store       │        │ feature generation │
      └────────┬─────────┘        └─────────┬──────────┘
               │                            │
               │                            ▼
               │                  ┌────────────────────┐
               │                  │ Forecasting Models │
               │                  │  + Evaluation      │
               │                  └─────────┬──────────┘
               │                            │
               │                            ▼
               │                  ┌────────────────────┐
               │                  │  Model Registry    │
               │                  │  Versioned Models  │
               │                  └─────────┬──────────┘
               │                            │
               └──────────────┬─────────────┘
                              ▼
                    ┌────────────────────┐
                    │ Streamlit Dashboard│
                    │ AQI + Forecasts    │
                    │ SHAP + Alerts      │
                    └────────────────────┘
```

The production data source is a managed PostgreSQL database. Feast uses the same cloud backend for its registry, offline-source configuration, and online-serving infrastructure. The production pipeline does not use local CSV or DuckDB storage as its source of truth.

---

## Main Features

### 1. Real Lahore Environmental Data

The system retrieves real air-pollution observations from OpenWeather using Lahore's coordinates.

The production city configuration is:

```text
City: Lahore
Latitude: 31.5204
Longitude: 74.3587
```

The raw pollutant observations include:

* PM10
* PM2.5
* Carbon Monoxide (CO)
* Nitrogen Dioxide (NO2)
* Sulphur Dioxide (SO2)
* Ozone (O3)

No artificial pollutant readings are generated to fill missing observations.

---

### 2. Time-Series Feature Engineering

The forecasting pipeline uses a shared feature contract so that training and inference use the same feature schema.

The current feature contract contains:

| Feature            | Description                             |
| ------------------ | --------------------------------------- |
| `pm10`             | PM10 concentration                      |
| `pm2_5`            | PM2.5 concentration                     |
| `carbon_monoxide`  | CO concentration                        |
| `nitrogen_dioxide` | NO2 concentration                       |
| `sulphur_dioxide`  | SO2 concentration                       |
| `ozone`            | O3 concentration                        |
| `aqi_lag1`         | Previous AQI observation                |
| `aqi_lag2`         | AQI two observations earlier            |
| `aqi_lag3`         | AQI three observations earlier          |
| `aqi_rolling_mean` | Rolling mean of recent AQI observations |
| `hour`             | Observation hour                        |
| `month`            | Observation month                       |
| `day`              | Day of month                            |
| `day_of_week`      | Day of week                             |

The shared feature contract is maintained in:

```text
src/feature_contract.py
```

This schema is used consistently by the feature-store definitions, training pipeline, and prediction pipeline.

---

## Forecast Horizons

The system produces three future AQI predictions:

```text
Day 1 → approximately +24 hours
Day 2 → approximately +48 hours
Day 3 → approximately +72 hours
```

Each horizon has its own registered model artifact and evaluation metrics.

The model registry stores the approved model version and metadata separately from Feast.

---

## Feature Store

The project uses **Feast** as the feature-store framework with managed PostgreSQL as the cloud backend.

The feature-store configuration is located at:

```text
feature_repo/feature_repo/feature_store.yaml
```

The FeatureView definition is located at:

```text
feature_repo/feature_repo/feature_definitions.py
```

The feature store provides:

* Cloud-backed feature infrastructure
* Online feature serving
* Feature definitions
* Materialization
* A consistent feature schema
* Integration with the production PostgreSQL source

The active production architecture does not rely on a local feature-store fallback.

---

## Historical Data and Backfilling

Historical environmental data is handled separately from the live ingestion path.

The project includes an OpenWeather historical backfill utility:

```text
src/openweather_historical_backfill.py
```

Historical backfill is designed to use real OpenWeather historical observations.

Important principles:

* No fabricated observations
* No artificial interpolation of missing pollutant readings
* Invalid or unsafe records are excluded rather than invented
* Lahore coordinates are used consistently
* Feature engineering must preserve the same production feature contract

The historical CSV located under `data/historical/` is **not treated as an automatically trusted production source** unless its Lahore provenance can be independently established.

---

## Model Training

The main training pipeline is:

```text
src/train_forecast.py
```

The training process:

1. Retrieves the available cloud history.
2. Builds the daily forecasting dataset.
3. Applies the shared feature contract.
4. Creates the required forecast targets.
5. Trains the forecasting models.
6. Evaluates the models.
7. Records RMSE, MAE, and R² metrics.
8. Registers the resulting model version.

Evaluation is performed separately for the three forecast horizons.

### Evaluation Metrics

The project reports:

* RMSE
* MAE
* R²

A negative R² value is not automatically treated as a pipeline failure. It is reported as an evaluation result and indicates that the model is performing worse than a simple mean-based baseline for that horizon.

---

## Model Registry

The model registry is intentionally separate from Feast.

Registry implementation:

```text
src/model_registry.py
```

Registered models are stored under:

```text
models/registry/
```

Each model version contains versioned artifacts and metadata.

The registry records information such as:

* Model version
* Training date
* Feature schema
* Forecast horizon
* RMSE
* MAE
* R²
* Training/source metadata

The latest approved version is referenced through the registry's latest-model pointer.

This separation allows the application to load an approved forecasting model independently of the Feast feature-store registry.

---

## Prediction Pipeline

Prediction logic is implemented in:

```text
src/predict.py
```

The prediction pipeline:

1. Loads the latest approved model from the Model Registry.
2. Validates the model feature schema.
3. Retrieves the latest cloud history.
4. Builds the current daily feature representation.
5. Generates 24/48/72-hour AQI forecasts.
6. Returns the forecast values together with evaluation metadata and feature information.

The prediction pipeline does not create artificial lag values when genuine historical observations are unavailable.

---

## Streamlit Dashboard

The user-facing application is:

```text
app.py
```

The dashboard provides:

* Current Lahore AQI
* Current PM2.5
* Current PM10
* Current weather information
* Next three forecast days
* Model performance metrics
* SHAP explanation
* AQI safety information
* Hazardous AQI alerts
* Cloud/feature-store status

The application retrieves a fresh OpenWeather observation, writes it to the cloud feature-store backend, and then generates the forecast.

---

## Explainable AI

The project uses **SHAP** to provide local model explanations.

The dashboard can show which input features contributed to the current prediction.

Explainability is designed to complement the prediction rather than replace the underlying model evaluation.

If SHAP cannot be generated, the forecast itself remains available and the explanation failure is reported instead of silently fabricating an explanation.

---

## AQI Alerts

The dashboard includes tiered AQI health alerts so that users can understand the severity of predicted air quality.

The application also retains an explicit hazardous-AQI warning for very high pollution levels.

These alerts are intended as informational health guidance and are not a substitute for official medical or environmental advice.

---

## Exploratory Data Analysis

EDA is included in:

```text
src/eda.py
```

Generated EDA outputs are stored under:

```text
reports/eda/
```

The EDA examines the available AQI/environmental data and supports understanding of:

* Data distributions
* AQI behaviour
* Pollutant relationships
* Time-based patterns
* Data quality

The generated EDA report is:

```text
reports/eda/eda_report.md
```

---

## Automated Workflows

GitHub Actions are used to automate the production pipeline.

Workflow files are located under:

```text
.github/workflows/
```

The hourly workflow is responsible for the feature/data pipeline.

The daily workflow is responsible for model training and registry updates.

Both workflows can also support manual dispatch.

The project therefore follows an automated MLOps lifecycle:

```text
Hourly:
OpenWeather
   ↓
Cloud Observation
   ↓
Feast Materialization

Daily:
Cloud History
   ↓
Feature Preparation
   ↓
Model Training
   ↓
Evaluation
   ↓
Model Registry
```

---

## Repository Structure

```text
Pearls_AQI_Predictor/
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml
│       └── model_training.yml
│
├── data/
│   ├── historical/
│   └── raw/
│
├── docs/
│
├── feature_repo/
│   └── feature_repo/
│       ├── feature_definitions.py
│       └── feature_store.yaml
│
├── models/
│   └── registry/
│       └── <model-version>/
│
├── reports/
│   └── eda/
│
├── src/
│   ├── app-related utilities
│   ├── eda.py
│   ├── feature_contract.py
│   ├── feature_pipeline.py
│   ├── feature_store.py
│   ├── model_registry.py
│   ├── openweather_historical_backfill.py
│   ├── predict.py
│   └── train_forecast.py
│
├── tests/
│
├── app.py
|__ FINAL_PROJECT_REPORT.docx
|__ FINAL_PROJECT_REPORT.pdf
├── requirements.txt
└── README.md
```

---

## Technology Stack

| Component            | Technology                          |
| -------------------- | ----------------------------------- |
| Programming Language | Python                              |
| Data Processing      | pandas, NumPy                       |
| Machine Learning     | scikit-learn                        |
| Feature Store        | Feast                               |
| Cloud Database       | Managed PostgreSQL                  |
| Environmental API    | OpenWeather                         |
| Model Registry       | Project-specific versioned registry |
| Explainability       | SHAP                                |
| Dashboard            | Streamlit                           |
| CI/CD                | GitHub Actions                      |
| Testing              | pytest                              |
| Version Control      | Git / GitHub                        |

---
## Production Deployment

The Streamlit dashboard is deployed using Streamlit Community Cloud and is connected to the GitHub repository.

The production application is available at:

```text
https://ayesha-aqi-predictor.streamlit.app/
```

The live dashboard provides the production AQI forecasting interface for Lahore, including current air-quality conditions, three-day forecasts, model metrics, SHAP-based explainability, and safety alerts.

GitHub updates to the `main` branch can trigger an automatic Streamlit redeployment.

The production deployment requires the appropriate Streamlit secrets for:

* OpenWeather
* Managed PostgreSQL / Feast


## Local Setup

### 1. Clone the Repository

```powershell
git clone https://github.com/AyeshaArain8/Pearls_AQI_Predictor.git
cd Pearls_AQI_Predictor
```

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root.

Required production configuration includes:

```text
OPENWEATHER_API_KEY=your_openweather_api_key

FEAST_POSTGRES_URL=your_postgresql_connection_url
FEAST_POSTGRES_HOST=your_host
FEAST_POSTGRES_PORT=your_port
FEAST_POSTGRES_DATABASE=your_database
FEAST_POSTGRES_SCHEMA=your_schema
FEAST_POSTGRES_USER=your_user
FEAST_POSTGRES_PASSWORD=your_password
```

Never commit real API keys, database passwords, or other credentials to GitHub.

---

## Run the Feature Store

From the project root:

```powershell
feast -c feature_repo/feature_repo apply
```

---

## Run the Feature Pipeline

```powershell
python -m src.feature_pipeline
```

---

## Train a Forecasting Model

```powershell
python -m src.train_forecast
```

Training creates a versioned model entry in:

```text
models/registry/
```

---

## Run the Streamlit Dashboard

```powershell
streamlit run app.py
```

The application will open locally in the browser.

---

## Run Tests

Use:

```powershell
pytest -q
```

The project includes tests covering important areas such as:

* Feature contract
* Model registry
* Cloud feature-store configuration
* Prediction behaviour
* Data/feature safety
* Pipeline contracts

---

---

## Data Integrity and Safety Rules

This project intentionally follows strict data-integrity rules.

### The system does not:

* Fabricate missing pollutant observations
* Invent historical AQI values
* Generate fake lag values when genuine history is unavailable
* Silently switch to another city
* Use Karachi data in the Lahore production pipeline
* Use an unverified historical CSV as an authoritative production source
* Fall back to local CSV storage when the cloud production configuration is required

### The system does:

* Use real Lahore observations
* Preserve chronological ordering
* Validate feature schemas
* Keep model versions separate
* Report model evaluation metrics
* Reject unsafe inference situations
* Keep the production feature source cloud-based

---

## Project Scope

The production system is currently **Lahore-only**.

The active location is:

```text
Lahore, Pakistan
31.5204° N, 74.3587° E
```

Multi-city support is intentionally outside the current production feature contract.

---

## Current Production Model

The application uses the latest approved version stored by the project Model Registry.

The registry maintains separate forecasting artifacts for:

```text
Day 1 / +24h
Day 2 / +48h
Day 3 / +72h
```

Model performance should always be interpreted using the recorded RMSE, MAE, and R² values rather than assuming that every forecast horizon will achieve the same accuracy.

---

## Documentation and Reports

Additional project documentation is available under:

```text
docs/
reports/
```
and also in root
root folder
Final Project Report pdf
Final Project Report docx

The repository also contains the final project report and EDA outputs documenting the development and evaluation process.

---

## Project Requirements Coverage

The system addresses the major project requirements:

* [x] Real external environmental data ingestion
* [x] Historical data/backfill pipeline
* [x] Feature engineering
* [x] Cloud Feature Store architecture
* [x] 24/48/72-hour forecasting
* [x] Model training and evaluation
* [x] Versioned Model Registry
* [x] Automated CI/CD workflows
* [x] Streamlit interactive dashboard
* [x] SHAP explainability
* [x] AQI hazard alerts
* [x] Exploratory Data Analysis
* [x] Automated testing
* [x] Cloud deployment
* [x] Detailed project documentation

---

## Author

**Ayesha Shahid**

**Project:** Pearls Lahore AQI Predictor

An end-to-end AQI forecasting and MLOps project developed to demonstrate real-world data ingestion, feature engineering, cloud feature-store integration, machine-learning forecasting, model versioning, explainability, automation, testing, and deployment.
