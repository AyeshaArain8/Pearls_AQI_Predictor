# Production Feast repository

This directory contains the production Feast definitions for Pearls Lahore AQI Predictor. It uses a managed cloud PostgreSQL backend for the offline source, Feast SQL registry, and online serving. Configure the `FEAST_POSTGRES_*` environment variables before running `feast -c feature_repo/feature_repo apply`.

The definitions import `src.feature_contract.FEATURE_COLUMNS`; do not duplicate or extend the feature list here. The old quickstart/demo configuration is retired.
