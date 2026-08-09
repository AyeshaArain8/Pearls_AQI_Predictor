"""Deprecated compatibility entry point for the Lahore-only prediction path."""

from src.predict import forecast_aqi


if __name__ == "__main__":
    print(forecast_aqi())
