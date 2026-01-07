Link to the cars.com Kaggle dataset: https://www.kaggle.com/datasets/andreinovikov/used-cars-dataset 

# Used Car Price Prediction (HistGradientBoostingRegressor)

This project trains a machine learning model to predict used car prices from a large CSV dataset of vehicle listings.  
It is implemented as a single Python script that performs data cleaning, feature engineering, preprocessing, model training, evaluation, and artifact saving.

Main script:
- used_car_price_project.py

---

## Overview

The script implements an end-to-end regression pipeline for used car pricing:

- Loads a CSV dataset of vehicle listings
- Cleans and filters unrealistic or extreme records
- Engineers structured features from raw numeric and text fields
- Builds a preprocessing pipeline for numeric and categorical data
- Trains a HistGradientBoostingRegressor on log-transformed prices
- Evaluates predictions on a held-out test set
- Saves the trained model, preprocessing pipeline, feature list, and metrics

The focus of this project is robustness and realism rather than aggressive hyperparameter optimization.

---

## Data filtering

The dataset is filtered to remove extreme outliers and invalid records:

- Rows missing `price` or `year` are dropped
- Year range: 1995–2024
- Price range: $2,000–$120,000
- Mileage range: 0–300,000 (if mileage is present)

These constraints reduce noise from rare or invalid listings.

---

## Feature engineering

The script derives and processes the following features:

- Vehicle age: `age = 2025 - year`
- Engine parsing from free-text `engine` column:
  - engine displacement (liters)
  - cylinder count (when available)
  - turbo/supercharger flag
  - hybrid flag
  - electric/EV flag
- High-cardinality categorical reduction:
  - Keeps only the most frequent values for model, manufacturer, and colors
  - All other categories are mapped to `"Other"`

---

## Preprocessing

Preprocessing is implemented using a `ColumnTransformer`:

- Numeric features:
  - Median imputation
  - Standard scaling
- Categorical features:
  - Missing-value imputation
  - One-hot encoding

All preprocessing is fit on the training set and saved for reuse during inference.

---

## Model

- Model: HistGradientBoostingRegressor
- Target transformation: `log1p(price)`
- Predictions are converted back using `expm1`
- Early stopping is enabled to reduce overfitting

Evaluation metrics are computed on the original price scale:
- RMSE
- MAE
- R²

---

## Requirements

Python 3.10 or newer is recommended.

Install dependencies:

pip install requirements.txt
