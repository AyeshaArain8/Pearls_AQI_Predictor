# 🌍 Air Quality Index Prediction System

## Overview

This project predicts Air Quality Index (AQI) categories using machine learning and real-time air pollution data from the OpenWeather API.

## Features

- Fetches live air pollution data
- Extracts important AQI features
- Preprocesses historical dataset
- Trains a Random Forest model
- Predicts AQI category
- Interactive Streamlit dashboard

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- OpenWeather API

## Project Structure

```
Pearls_AQI_Predictor/
│
├── app.py
├── requirements.txt
├── README.md
├── models/
├── src/
└── data/
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## AQI Categories

- Good
- Moderate
- Unhealthy for Sensitive Groups
- Unhealthy
- Very Unhealthy
- Hazardous