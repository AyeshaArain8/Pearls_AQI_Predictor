"""OpenWeather collector. It never invents unavailable model features."""
from __future__ import annotations
import os
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv
from src.feature_contract import LAHORE, RAW_FEATURES

load_dotenv()
BASE = "https://api.openweathermap.org/data/2.5"

def pm25_to_us_aqi(value: float) -> float:
    # EPA PM2.5 breakpoints; this is a documented, deterministic derived observation.
    for lo, hi, aqi_lo, aqi_hi in ((0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)):
        if value <= hi: return round((aqi_hi-aqi_lo)/(hi-lo)*(value-lo)+aqi_lo, 1)
    return 500.0

def get_lahore_data() -> dict:
    key = os.getenv("OPENWEATHER_API_KEY") or os.getenv("API_KEY")
    if not key: raise ValueError("OPENWEATHER_API_KEY is required; set it locally or as a GitHub Secret.")
    params = {"lat": LAHORE["latitude"], "lon": LAHORE["longitude"], "appid": key}
    pollution = requests.get(f"{BASE}/air_pollution", params=params, timeout=20); pollution.raise_for_status()
    weather = requests.get(f"{BASE}/weather", params=params | {"units":"metric"}, timeout=20); weather.raise_for_status()
    components = pollution.json().get("list", [{}])[0].get("components", {})
    if not components: raise ValueError("OpenWeather response did not contain pollutant components.")
    mapping = {"pm10":"pm10", "pm2_5":"pm2_5", "carbon_monoxide":"co", "nitrogen_dioxide":"no2", "sulphur_dioxide":"so2", "ozone":"o3"}
    result = {"city":"Lahore", "timestamp": datetime.now(timezone.utc).isoformat(), **{out:float(components[source]) for out,source in mapping.items()}}
    result["aqi"] = pm25_to_us_aqi(result["pm2_5"])
    payload = weather.json(); result.update({"temperature": payload.get("main",{}).get("temp"), "humidity":payload.get("main",{}).get("humidity"), "wind_speed":payload.get("wind",{}).get("speed"), "source":"OpenWeather"})
    return result

# Compatibility for existing callers; city/coordinates are intentionally ignored.
def get_city_data(city="Lahore", latitude=None, longitude=None):
    if city != "Lahore": raise ValueError("Only Lahore is supported.")
    return get_lahore_data()
