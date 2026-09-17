# Data-Driven Revenue Optimisation in Large-Scale Event Management
**BSc Thesis | Digital Business & Data Science**

## Overview
A full-stack revenue management research system combining machine learning demand forecasting, Monte Carlo dynamic pricing simulation, customer segmentation, CLV modelling, and a novel composite measurement framework (ROI-E).

## Key Results
| Model | MAPE | R² |
|---|---|---|
| OLS Baseline | 66.6% | 0.168 |
| XGBoost | 10.0% | 0.890 |
| Random Forest | 13.6% | 0.864 |
| Hybrid Ensemble | **9.9%** | **0.894** |

**Dynamic Pricing Uplift over Static Baseline:**
- S2 Rule-Based: +0.0%
- S3 Demand-Responsive: +20.7%
- S4 ML-Optimised: +23.9%

## Scripts
| File | Description |
|---|---|
| `01_dataset_generation.py` | Synthetic-empirical dataset (5,200 events, 42 variables) |
| `02_eda_feature_engineering.py` | Exploratory analysis and feature engineering |
| `03_demand_forecasting_FIXED_v3.py` | 5-model ML demand forecasting pipeline |
| `04_dynamic_pricing_simulation_FIXED_v2.py` | Monte Carlo pricing simulation (S1–S4) |
| `05_customer_segmentation_clv_FINAL.py` | k-means segmentation + CLV estimation |
| `06_roie_scoring.py` | ROI-E composite index scoring |

## Tech Stack
Python · pandas · scikit-learn · XGBoost · TensorFlow/Keras · matplotlib · scipy · numpy

## Dataset
Synthetic-empirical dataset calibrated against Pollstar, Deloitte, PwC, UFI, and MPI Foundation benchmarks. Not included in this repository due to file size.
