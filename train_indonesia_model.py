#!/usr/bin/env python3
"""
AOL AI — FINAL OPTIMIZED TRAINING FILE (LGBM Focused)
- FULL temporal + lag + rolling + EMA + aggregates.
- Training for two core goals: Daily prediction and Stable Horizon (30/90/180 days).
- FIXED: Robust RMSE initialization for large target variables (Revenue/Supply).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# FEATURE LIST (DAILY MODEL)
# ---------------------------------------------------------

BASE_FEATURES = [
    'Province_encoded', 'Category_encoded',
    'month', 'year', 'day_of_week',
    'price_idr', 'promotion_flag', 'holiday_flag',
    'temp_c', 'precipitation_mm', 'humidity_pct',
    'cpi_index', 'mobility_index',
    'month_sin', 'month_cos',
    'day_of_week_sin', 'day_of_week_cos',
    'days_since_start',
    'day_of_month', 'week_of_year',
    'quarter', 'is_weekend',
    'is_month_start', 'is_month_end'
]

LAG_WINDOWS = [1, 7, 14, 30]
ROLL_WINDOWS = [7, 14, 30]
STD_WINDOWS = [7, 14]
EMA_WINDOWS = [7, 14, 30]

TEMPORAL_FEATURES = []

# Lag Features
for w in LAG_WINDOWS:
    TEMPORAL_FEATURES += [
        f'units_sold_lag_{w}',
        f'revenue_lag_{w}'
    ]

# Rolling Means
for w in ROLL_WINDOWS:
    TEMPORAL_FEATURES += [
        f'units_sold_rolling_{w}',
        f'revenue_rolling_{w}',
    ]

# Rolling STD
for w in STD_WINDOWS:
    TEMPORAL_FEATURES += [
        f'units_sold_rolling_std_{w}',
    ]

# EMA
for w in EMA_WINDOWS:
    TEMPORAL_FEATURES += [
        f'units_sold_ema_{w}',
    ]

# Growth features
TEMPORAL_FEATURES += [
    'units_sold_growth_7',
    'units_sold_growth_30'
]

# Aggregates
TEMPORAL_FEATURES += [
    'province_category_mean', 'province_category_std',
    'province_mean_units', 'category_mean_units'
]

FEATURES = BASE_FEATURES + TEMPORAL_FEATURES


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------
def load_indonesia_data(csv_file='indonesia_supermarket_5yr_synthetic.csv'):
    print("Loading CSV...")
    df = pd.read_csv(csv_file)
    print(f"Rows: {len(df)}  Date range: {df['date'].min()} - {df['date'].max()}")
    
    df['Date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['province', 'category', 'Date']).reset_index(drop=True)
    return df


# ---------------------------------------------------------
# PREPROCESSING + BASIC FEATURES
# ---------------------------------------------------------
def create_temporal_features(df):
    print("Creating base time features + cyclical encodings...")

    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['day_of_week'] = df['Date'].dt.dayofweek

    df['day_of_month'] = df['Date'].dt.day
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['Date'].dt.quarter

    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = (df['Date'].dt.day <= 7).astype(int)
    df['is_month_end'] = (df['Date'].dt.day >= 25).astype(int)

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    min_date = df['Date'].min()
    df['days_since_start'] = (df['Date'] - min_date).dt.days

    print("Encoding province & category...")
    le_province = LabelEncoder()
    df['Province_encoded'] = le_province.fit_transform(df['province'])

    le_category = LabelEncoder()
    df['Category_encoded'] = le_category.fit_transform(df['category'])

    os.makedirs("models", exist_ok=True)
    dump(le_province, "models/province_encoder.joblib")
    dump(le_category, "models/category_encoder.joblib")
    print("Encoders saved.")

    return df


# ---------------------------------------------------------
# LAG, ROLLING, EMA, GROWTH FEATURES
# ---------------------------------------------------------
def create_lag_rolling_features(df):
    print("Creating lag / rolling / ema / growth features grouped by province+category...")

    df = df.copy()
    g = df.groupby(['province', 'category'])

    # LAG FEATURES
    for w in LAG_WINDOWS:
        df[f'units_sold_lag_{w}'] = g['units_sold'].shift(w)
        df[f'revenue_lag_{w}'] = g['revenue_idr'].shift(w)

    # ROLLING MEANS (shifted/closed='left' for real-world prediction)
    for w in ROLL_WINDOWS:
        df[f'units_sold_rolling_{w}'] = g['units_sold'].transform(lambda x: x.rolling(w, min_periods=1, closed='left').mean().shift(1))
        df[f'revenue_rolling_{w}'] = g['revenue_idr'].transform(lambda x: x.rolling(w, min_periods=1, closed='left').mean().shift(1))

    # ROLLING STD
    for w in STD_WINDOWS:
        df[f'units_sold_rolling_std_{w}'] = g['units_sold'].transform(lambda x: x.rolling(w, min_periods=1, closed='left').std().shift(1))

    # EMA FEATURES (shifted/lagged)
    for w in EMA_WINDOWS:
        df[f'units_sold_ema_{w}'] = g['units_sold'].transform(lambda x: x.ewm(span=w, adjust=False).mean()).shift(1)

    # GROWTH FEATURES
    df['units_sold_growth_7'] = g['units_sold'].pct_change(7)
    df['units_sold_growth_30'] = g['units_sold'].pct_change(30)

    df = df.replace([np.inf, -np.inf], np.nan)

    # AGGREGATES
    stats = df.groupby(['province', 'category'])['units_sold'].agg(['mean', 'std']).reset_index()
    stats.columns = ['province', 'category', 'province_category_mean', 'province_category_std']
    df = df.merge(stats, on=['province', 'category'], how='left')

    prov_stats = df.groupby('province')['units_sold'].mean().reset_index()
    prov_stats.columns = ['province', 'province_mean_units']
    df = df.merge(prov_stats, on='province', how='left')

    cat_stats = df.groupby('category')['units_sold'].mean().reset_index()
    cat_stats.columns = ['category', 'category_mean_units']
    df = df.merge(cat_stats, on='category', how='left')

    df = df.fillna(0)
    return df

# ---------------------------------------------------------
# MODEL TRAINING (DAILY MODELS)
# ---------------------------------------------------------
def train_daily_models(df):
    df_train = df[df['days_since_start'] >= 30].copy()
    X = df_train[FEATURES]
    y_demand = df_train['units_sold']
    y_supply = df_train['revenue_idr']
    tscv = TimeSeriesSplit(n_splits=3)
    
    # ------------------------ DEMAND MODEL ------------------------
    print("\nTraining DAILY DEMAND model (LGBM)...")
    best_rmse_d = np.inf
    best_demand = None
    for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_demand.iloc[train_idx], y_demand.iloc[val_idx]
        model = lgb.LGBMRegressor(objective="regression", n_estimators=3000, learning_rate=0.03, max_depth=12, num_leaves=63, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        if rmse < best_rmse_d:
            best_rmse_d = rmse
            best_demand = model
    dump(best_demand, "models/demand_model.joblib")
    dump(FEATURES, "models/demand_features.joblib")
    print(f"Best DAILY DEMAND RMSE = {best_rmse_d:.2f}")

    # ------------------------ SUPPLY MODEL ------------------------
    print("\nTraining DAILY SUPPLY model (LGBM)...")
    best_rmse_s = np.inf
    best_supply = None
    for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_supply.iloc[train_idx], y_supply.iloc[val_idx]
        model = lgb.LGBMRegressor(objective="regression", n_estimators=3000, learning_rate=0.03, max_depth=12, num_leaves=63, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        if rmse < best_rmse_s:
            best_rmse_s = rmse
            best_supply = model
    dump(best_supply, "models/supply_model.joblib")
    dump(FEATURES, "models/supply_features.joblib")
    print(f"Best DAILY SUPPLY RMSE = {best_rmse_s:.2f}")

# ---------------------------------------------------------
# MODEL TRAINING (HORIZON AGGREGATE MODELS)
# ---------------------------------------------------------
def train_horizon_models(df):
    HORIZONS = [30, 90, 180]
    # Simplified Horizon Features: Exclude specific daily/weekly features that reduce stability for long-term sums
    horizon_features = [f for f in FEATURES if f not in ['day_of_month', 'week_of_year', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end', 'day_of_week_sin', 'day_of_week_cos']]
    horizon_features.append('horizon_days')

    print(f"\nTraining HORIZON AGGREGATE models for {HORIZONS} days...")

    df_horizon = []
    # Create the aggregated dataset for each horizon
    for H in HORIZONS:
        g = df.groupby(['province', 'category'])
        # Target: Sum of units/revenue for the next H days (shifted/closed='left')
        df[f'units_sold_sum_{H}'] = g['units_sold'].transform(lambda x: x.rolling(H, min_periods=H, closed='left').sum().shift(-H))
        df[f'revenue_idr_sum_{H}'] = g['revenue_idr'].transform(lambda x: x.rolling(H, min_periods=H, closed='left').sum().shift(-H))
        
        df_H = df[df[f'units_sold_sum_{H}'].notna()].copy()
        df_H['horizon_days'] = H
        df_H['units_sold_sum'] = df_H[f'units_sold_sum_{H}']
        df_H['revenue_idr_sum'] = df_H[f'revenue_idr_sum_{H}']
        df_horizon.append(df_H)
    
    df_train_h = pd.concat(df_horizon)
    # Ensure enough history for features (30 days) AND the longest target horizon (180 days)
    df_train_h = df_train_h[df_train_h['days_since_start'] >= 30 + 180] 

    X_h = df_train_h[horizon_features]
    y_demand_h = df_train_h['units_sold_sum']
    y_supply_h = df_train_h['revenue_idr_sum']
    tscv = TimeSeriesSplit(n_splits=3)

    for H in HORIZONS:
        print(f"\n--- Training for Horizon H={H} ---")
        X_H = X_h[X_h['horizon_days'] == H]
        y_demand_H = y_demand_h.loc[X_H.index]
        y_supply_H = y_supply_h.loc[X_H.index]

        # ------------------------ HORIZON DEMAND MODEL ------------------------
        best_rmse_d = np.inf
        best_demand_h = None
        for i, (train_idx, val_idx) in enumerate(tscv.split(X_H)):
            X_train, X_val = X_H.iloc[train_idx], X_H.iloc[val_idx]
            y_train, y_val = y_demand_H.iloc[train_idx], y_demand_H.iloc[val_idx]
            model = lgb.LGBMRegressor(objective="regression_l1", n_estimators=3000, learning_rate=0.03, max_depth=8, num_leaves=31, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=0.5, random_state=42)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            if rmse < best_rmse_d:
                best_rmse_d = rmse
                best_demand_h = model
        dump(best_demand_h, f"models/demand_{H}_model.joblib")
        print(f"Best HORIZON DEMAND H={H} RMSE = {best_rmse_d:.2f}")

        # ------------------------ HORIZON SUPPLY MODEL (REVENUE) ------------------------
        best_rmse_s = np.inf
        best_supply_h = None
        for i, (train_idx, val_idx) in enumerate(tscv.split(X_H)):
            X_train, X_val = X_H.iloc[train_idx], X_H.iloc[val_idx]
            y_train, y_val = y_supply_H.iloc[train_idx], y_supply_H.iloc[val_idx]
            model = lgb.LGBMRegressor(objective="regression_l1", n_estimators=3000, learning_rate=0.03, max_depth=8, num_leaves=31, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=0.5, random_state=42)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            if rmse < best_rmse_s:
                best_rmse_s = rmse
                best_supply_h = model
        
        # Fallback: save the last trained model if cross-validation didn't find a 'best' one (due to large error)
        if best_supply_h is None and 'model' in locals(): 
            best_supply_h = model
            
        dump(best_supply_h, f"models/supply_{H}_model.joblib")
        print(f"Best HORIZON SUPPLY H={H} RMSE = {best_rmse_s:.2f}")

    dump(horizon_features, "models/horizon_features.joblib")


# ---------------------------------------------------------
# INSIGHTS
# ---------------------------------------------------------
def save_insights(df):
    print("Saving basic insights...")
    insights = {
        "rows": int(len(df)),
        "provinces": int(df['province'].nunique()),
        "categories": int(df['category'].nunique()),
        "date_range": f"{df['Date'].min().date()} to {df['Date'].max().date()}"
    }
    with open("models/insights.json", "w") as f:
        json.dump(insights, f, indent=2)
    print("Insights saved.")


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def main():
    df = load_indonesia_data()
    df = create_temporal_features(df)
    df = create_lag_rolling_features(df)
    save_insights(df)
    train_daily_models(df)
    train_horizon_models(df)

    print("\nALL DONE — FULL FEATURE MODEL TRAINED SUCCESSFULLY!")


if __name__ == '__main__':
    main()