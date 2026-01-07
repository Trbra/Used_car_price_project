#!/usr/bin/env python3
"""
used_car_price_project_hgbr_only.py

Pipeline (Option A filtering) using only HistGradientBoostingRegressor:
 - Filters: year 1995-2024, price 2000-120000, mileage 0-300000
 - Log-transform target (log1p)
 - Engine text parsing (displacement, cylinders, turbo/hybrid/electric)
 - Cardinality reduction for categorical fields (top N -> Other)
 - OneHot encoding and numeric scaling
 - Trains HGBR only
 - Saves model, preprocessor, and metrics JSON
"""

import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Configuration
# -----------------------------
FILTER_YEAR_MIN = 1995
FILTER_YEAR_MAX = 2024
FILTER_PRICE_MIN = 2000
FILTER_PRICE_MAX = 120000
FILTER_MILEAGE_MIN = 0
FILTER_MILEAGE_MAX = 300000

TOP_N_MODELS = 60
TOP_N_MANUFACTURERS = 60
TOP_N_COLORS = 50

RANDOM_STATE = 42

# -----------------------------
# Utilities & feature helpers
# -----------------------------
def reduce_categories(series: pd.Series, top_n: int, other_name="Other"):
    top = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top), other_name).fillna("Unknown")

_engine_disp_re = re.compile(r'(\d+(?:\.\d+)?)\s*L', flags=re.IGNORECASE)

def parse_engine_text(s: str):
    if not isinstance(s, str):
        return (np.nan, np.nan, 0, 0, 0)
    text = s.lower()
    # displacement
    disp = None
    m = _engine_disp_re.search(text)
    if m:
        try:
            disp = float(m.group(1))
        except:
            disp = None
    # cylinders
    cyl = None
    m_cyl = re.search(r'(\d{1,2})\s*[-]?\s*(?:cyl|cylinder|cylinders)', text)
    if m_cyl:
        try:
            cyl = int(m_cyl.group(1))
        except:
            cyl = None
    if cyl is None:
        m2 = re.search(r'\b([iv])\s?(\d)\b', text)
        if m2:
            try:
                cyl = int(m2.group(2))
            except:
                cyl = None
    # flags
    is_turbo = 1 if 'turbo' in text or 'turb' in text else 0
    is_super = 1 if 'supercharger' in text or 'supercharged' in text else 0
    is_hybrid = 1 if 'hybrid' in text else 0
    is_electric = 1 if ('electric' in text or ('ev' in text and 'engine' not in text)) else 0
    return (disp if disp is not None else np.nan,
            cyl if cyl is not None else np.nan,
            1 if (is_turbo or is_super) else 0,
            is_hybrid,
            is_electric)

def apply_engine_parser(df, engine_col='engine'):
    disp_list, cyl_list, turbo_list, hybrid_list, ev_list = [], [], [], [], []
    for s in df.get(engine_col, pd.Series(dtype="object")):
        d, c, t, h, e = parse_engine_text(s)
        disp_list.append(d); cyl_list.append(c); turbo_list.append(t)
        hybrid_list.append(h); ev_list.append(e)
    df['engine_disp_l'] = pd.to_numeric(disp_list, errors='coerce')
    df['engine_cyl'] = pd.to_numeric(cyl_list, errors='coerce')
    df['engine_turbo_or_super'] = turbo_list
    df['engine_is_hybrid'] = hybrid_list
    df['engine_is_electric'] = ev_list
    return df

# -----------------------------
# Preprocessing pipeline builder (HGBR)
# -----------------------------
def build_preprocessor(df):
    numeric_feats = [c for c in ['age', 'mileage', 'engine_disp_l', 'engine_cyl', 'driver_rating', 'seller_rating', 'price_drop'] if c in df.columns]
    cat_candidates = [c for c in ['manufacturer', 'model_reduced', 'transmission', 'drivetrain', 'fuel_type', 'exterior_color_reduced', 'interior_color_reduced'] if c in df.columns]

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_feats),
            ('cat', categorical_transformer, cat_candidates)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    features = {'numeric': numeric_feats, 'categorical': cat_candidates}
    return preprocessor, features

# -----------------------------
# Evaluation utility
# -----------------------------
def evaluate_preds(original_y, predicted_y):
    y_true = np.asarray(original_y).astype(float)
    y_pred = np.asarray(predicted_y).astype(float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='path to cars.csv')
    parser.add_argument('--outdir', default='results', help='output folder')
    parser.add_argument('--mode', choices=['dryrun','train','eval'], default='dryrun')
    parser.add_argument('--nrows', type=int, default=None)
    parser.add_argument('--topn_models', type=int, default=TOP_N_MODELS)
    args = parser.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path} (nrows={args.nrows}) ...")
    df = pd.read_csv(data_path, nrows=args.nrows)
    print(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.")

    df.columns = [c.strip() for c in df.columns]

    # Ensure numeric columns
    for col in ['price','year','mileage']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace('[\\$,]', '', regex=True), errors='coerce')

    df = df[df['price'].notna() & df['year'].notna()]
    before_rows = len(df)
    df = df[(df['year'] >= FILTER_YEAR_MIN) & (df['year'] <= FILTER_YEAR_MAX)]
    df = df[(df['price'] >= FILTER_PRICE_MIN) & (df['price'] <= FILTER_PRICE_MAX)]
    if 'mileage' in df.columns:
        df = df[(df['mileage'] >= FILTER_MILEAGE_MIN) & (df['mileage'] <= FILTER_MILEAGE_MAX)]
    after_rows = len(df)
    print(f"Filtered dataset: {before_rows:,} -> {after_rows:,} rows (Option A)")

    # Features
    now_year = 2025
    df['age'] = now_year - df['year']
    df['miles_per_year'] = df['mileage'] / df['age'].replace({0:1}).clip(lower=1)

    if 'engine' in df.columns:
        df = apply_engine_parser(df, engine_col='engine')

    # Reduce categories
    if 'model' in df.columns:
        df['model_reduced'] = reduce_categories(df['model'].fillna('Unknown'), args.topn_models, other_name='Other')
    if 'manufacturer' in df.columns:
        df['manufacturer'] = df['manufacturer'].fillna('Unknown')
        df['manufacturer_reduced'] = reduce_categories(df['manufacturer'], TOP_N_MANUFACTURERS, other_name='Other')
    if 'exterior_color' in df.columns:
        df['exterior_color_reduced'] = reduce_categories(df['exterior_color'].fillna('Unknown'), TOP_N_COLORS, other_name='Other')
    if 'interior_color' in df.columns:
        df['interior_color_reduced'] = reduce_categories(df['interior_color'].fillna('Unknown'), TOP_N_COLORS, other_name='Other')

    # Preprocessor
    preprocessor, features = build_preprocessor(df)
    all_feature_cols = features['numeric'] + features['categorical']

    X = df[all_feature_cols].copy()
    y = df['price'].copy()

    if args.mode == 'dryrun':
        print("Dryrun mode: showing sizes and sample processed features.")
        print(X.head(5).to_string(index=False))
        print("y sample:", y.head(10).tolist())
        return

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Training rows: {len(X_train):,}, Test rows: {len(X_test):,}")

    # Log transform target
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # -----------------------------
    # HGBR pipeline
    # -----------------------------
    preprocessor.fit(X_train)
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    hgb = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=12,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1
    )

    print("Training HistGradientBoostingRegressor on log-price...")
    hgb.fit(X_train_trans, y_train_log)
    print("HGBR training complete.")

    preds_log = hgb.predict(X_test_trans)
    preds = np.expm1(preds_log)
    metrics_hgb = evaluate_preds(y_test.values, preds)
    print("HGBR metrics (original scale):", metrics_hgb)

    # Save artifacts
    joblib.dump(hgb, outdir / "model_hgb.joblib")
    joblib.dump(preprocessor, outdir / "preprocessor.joblib")
    with open(outdir / "feature_columns.json", "w") as f:
        json.dump(all_feature_cols, f, indent=2)
    with open(outdir / "metrics_summary.json", "w") as f:
        json.dump({'hgb': metrics_hgb, 'n_train': len(X_train), 'n_test': len(X_test)}, f, indent=2)

    print(f"Saved HGBR model and preprocessor to {outdir.resolve()}")
    print("Metrics summary saved to metrics_summary.json")

    print("Done.")

if __name__ == "__main__":
    main()
