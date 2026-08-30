# Lahore AQI - Exploratory Data Analysis

Source: `data/historical/Backup/old_pakistan_air_quality_final_clean.csv` (filtered to `city == "Lahore"`), the same file `src.historical_backfill` prepares for backfill.

## Overview
- Observations: **2184**
- Date range: **2025-11-06** to **2026-02-04**
- Pollutant/AQI missing values:
  - pm10: 0 (0.0%)
  - pm2_5: 0 (0.0%)
  - carbon_monoxide: 0 (0.0%)
  - nitrogen_dioxide: 0 (0.0%)
  - sulphur_dioxide: 0 (0.0%)
  - ozone: 0 (0.0%)
  - aqi: 0 (0.0%)

## AQI distribution
```
count    2184.000000
mean      191.787683
std        51.628431
min        60.900000
25%       162.600000
50%       184.900000
75%       216.600000
max       383.300000
```

AQI category breakdown (US AQI, PM2.5-derived):

```
aqi
Good                                 0
Moderate                            82
Unhealthy for Sensitive Groups     206
Unhealthy                         1174
Very Unhealthy                     632
Hazardous                           90
```

## Pollutant distributions
```
                   count         mean          std    min     25%     50%     75%      max
pm10              2184.0   129.252473    63.113182   17.2    81.0   123.9   168.1    357.6
pm2_5             2184.0   126.761264    63.113241   16.8    78.0   121.1   166.2    353.9
carbon_monoxide   2184.0  2930.477564  1733.142466  349.0  1623.0  2578.5  3833.0  11482.0
nitrogen_dioxide  2184.0    60.598489    34.695727    3.7    32.5    59.4    84.4    185.1
sulphur_dioxide   2184.0    27.843681    11.684303    0.1    19.2    26.8    34.4     75.1
ozone             2184.0    67.250000    62.686656    0.0    18.0    45.0   109.0    265.0
```

## Correlation with AQI
```
pm2_5               0.983140
pm10                0.982539
nitrogen_dioxide    0.653210
carbon_monoxide     0.634808
sulphur_dioxide     0.548003
ozone              -0.519409
```

## Charts
- `aqi_distribution.png` - histogram of hourly AQI values
- `aqi_over_time.png` - daily mean AQI trend across the full history
- `pollutant_aqi_correlation.png` - which pollutants track AQI most closely

## Observations
- PM2.5 and PM10 are the strongest correlates of AQI here (AQI is derived directly from PM2.5, so this is expected), with corr(PM2.5, AQI) = 0.983 and corr(PM10, AQI) = 0.983.
- The most common AQI category in this history is **Unhealthy** (1174 of 2184 hourly rows).
- Mean AQI over the full period is 191.8, with a max of 383.3, showing Lahore regularly experiences unhealthy-or-worse air quality in this dataset.