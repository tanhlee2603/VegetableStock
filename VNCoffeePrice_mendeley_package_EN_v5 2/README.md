# VNCoffeePrice: A Multi-Source Dataset for Robusta Coffee Price Forecasting in Vietnam (2020–2026)

## Overview

This dataset integrates six data sources to support machine-learning research on Robusta coffee
price forecasting in Vietnam: (1) daily domestic Robusta coffee prices at five provinces in the
Central Highlands, (2) local weather observations, (3) ICE London Robusta coffee futures prices,
(4) USD/VND exchange rates, (5) world crude oil prices (Brent, WTI), and (6) Vietnamese retail
fuel prices. Coverage spans **1 January 2020 to 22 May 2026** (5,445 daily records, 25 variables,
zero missing values after preprocessing).

This dataset accompanies the manuscript "Benchmarking Machine Learning Models for Multivariate
Robusta Coffee Price Forecasting: A Multi-Source Dataset and Feature Ablation Study" (submitted to
ISDS 2026). Please cite the manuscript (once published) if you use this dataset.

## Files

```
vietnam_coffee_price_dataset.csv   Main processed/merged dataset (USE THIS FILE for modeling)
raw_data/                          Original, unprocessed source files (for provenance/transparency)
  coffee_price_train.csv             Domestic coffee price + weather, 2020-2024 (train split)
  coffee_price_val.csv               Domestic coffee price + weather, 2025 (validation split)
  coffee_price_test.csv              Domestic coffee price + weather, 2026 (test split)
  london_robusta_futures_historical.csv   ICE London futures, source: investing.com
  usd_vnd_exchange_rate_historical.csv    USD/VND exchange rate, source: investing.com
  world_crude_oil_prices_2020_2026.csv    World crude oil prices (Brent, WTI)
  vietnam_fuel_prices_2020_2026.csv       Vietnam retail fuel prices (4 fuel types)
code/
  build_dataset.py            Script used to merge raw_data/ into vietnam_coffee_price_dataset.csv
  rq1_experiment.py           Benchmark + feature ablation experiment script (reproduces paper results)
```

## Main file: vietnam_coffee_price_dataset.csv — Data Dictionary

| Column | Description | Unit |
|---|---|---|
| date | Observation date | date (YYYY-MM-DD) |
| split | Time-based split: train (2020-2024) / val (2025) / test (2026) | category |
| province | Province (5 values: Dak Lak, Dak Nong, Gia Lai, Kon Tum, Lam Dong) | category |
| commodity | Commodity name (Robusta coffee, raw beans) | text |
| coffee_price_vnd_kg | Domestic Robusta coffee price | VND/kg |
| london_price_usd_ton | ICE London Robusta futures price (original) | USD/ton |
| usd_vnd_rate | USD/VND exchange rate | VND/USD |
| london_price_vnd_kg | London futures price converted: (USD/ton × VND/USD) / 1000 | VND/kg |
| domestic_london_price_gap_vnd_kg | Domestic price minus converted London price | VND/kg |
| brent_price_usd_barrel, wti_price_usd_barrel | World crude oil prices (Brent, WTI) | USD/barrel |
| brent_price_vnd_liter_converted, wti_price_vnd_liter_converted | Crude oil price converted: (USD/barrel × VND/USD) / 158.987 | VND/liter (converted; excludes refining/tax) |
| ron95_price_vnd_liter, e5ron92_price_vnd_liter, diesel_price_vnd_liter, kerosene_price_vnd_liter | Vietnam retail fuel prices (RON95-III, E5 RON92, Diesel 0.05S, Kerosene) | VND/liter |
| temperature_2m_mean / _min / _max | Mean/min/max air temperature at 2m | °C |
| precipitation_sum, rain_sum | Total precipitation / rainfall | mm |
| lat, lon | Province representative coordinates | decimal degrees |
| coord_source | Provenance of coordinates (manual_fallback = province centroid, not exact farm-gate location) | category |

## Data construction summary

1. Daily domestic coffee price (per province) is used as the primary time axis.
2. Lower-frequency exogenous series (London futures, USD/VND — traded on business days only;
   world oil prices — occasional gaps; Vietnamese retail fuel prices — updated only at official
   price-adjustment dates, ~7-10 days apart) are aligned to daily frequency by forward-filling the
   most recent known value. Backward-fill is applied only to the few leading observations that
   precede the first available value in a source series.
3. Unit conversions applied: London futures USD/ton → VND/kg; world crude oil USD/barrel → VND/liter
   (1 barrel = 158.987 liters). See `code/build_dataset.py` for the exact implementation.

## Known data-quality checks performed

- Zero missing values after preprocessing (5,445 × 25).
- Augmented Dickey-Fuller test: coffee price level is non-stationary (p=0.723); first difference is
  stationary (p<0.001).
- Cross-validated against two independently documented market events: (1) WTI crude oil price went
  negative (-36.98 USD/barrel) on 2020-04-20, matching the well-known COVID-19 storage crisis event;
  (2) a sustained, simultaneous surge across all four Vietnamese fuel types in March-April 2026
  (RON95 peaking at 33,840 VND/liter), consistent across multiple consecutive price-adjustment
  cycles rather than an isolated data-entry artifact.

## Known limitations

- Three provinces (Gia Lai, Kon Tum, Dak Nong) have shorter history, starting March 2021, versus
  Dak Lak and Lam Dong which start January 2020.
- Weather variables are recorded at a single representative coordinate per province (centroid), not
  at the exact farm-gate/collection-point location.
- Provenance of the original domestic coffee price and Vietnam retail fuel price series (prior to
  being provided for this project) should be independently verified by users requiring
  official-statistics-grade traceability.

## Reproducing the benchmark

Run from the package root directory:

```
pip install -r code/requirements.txt   # exact pinned versions used to generate the paper's results
python code/build_dataset.py     # rebuilds vietnam_coffee_price_dataset.csv from raw_data/
python code/rq1_experiment.py    # reproduces the ADF test, correlation analysis, 8-group feature
                                  # ablation (incl. world oil and VN retail fuel), and the
                                  # price-level-vs-differenced-target benchmark (Tables 6, 8-9 of the paper)
```

**Reproducibility note.** Linear Regression results, the full feature-group ablation study
(Table 9), and all differenced-target rows of the benchmark (Table 8) are exactly reproducible
byte-for-byte with any recent `scikit-learn` version. The price-level Gradient Boosting row in
Table 8 is the sole exception: because it uses 300 sequential boosting stages (vs. 200 for the
differenced-target model), tiny floating-point differences across `scikit-learn`/BLAS versions can
compound and shift its MAE/MAPE by up to ~2%, even with a fixed `random_state`. This does not
affect any conclusion in the paper. Use `code/requirements.txt` to reproduce the exact numbers
reported in the manuscript.

The ablation study covers eight feature-group configurations: own-lag only; own-lag + London
futures; own-lag + exchange rate; own-lag + weather; own-lag + world crude oil (Brent, WTI);
own-lag + Vietnamese retail fuel (RON95, E5RON92, Diesel, Kerosene); own-lag + London + FX +
weather; and the full combination of all exogenous sources. Only the London-futures-inclusive
configurations achieve a statistically significant (Diebold-Mariano test) improvement over naive
persistence; world oil and Vietnamese retail fuel prices do not help in isolation and degrade
performance when added to the best-performing combination, likely due to the four retail fuel
series being highly collinear step functions (see the manuscript, Section 5.3, for details).

## License

Recommended: CC BY 4.0 (allows reuse with attribution). [Confirm/adjust upon deposit.]

## Contact

Duc-Tan Nguyen (corresponding author) — Yersin University of Da Lat, Lam Dong, Viet Nam — ductan8190@gmail.com
Chi-Hai Nguyen — Yersin University of Da Lat, Lam Dong, Viet Nam — yu2301010067@yersin.edu.vn

## Citation / DOI

This dataset is deposited on Mendeley Data: DOI [10.17632/p8fhcw59b7.1](https://doi.org/10.17632/p8fhcw59b7.1).
