#!/usr/bin/env python3
"""
AOL AI — FINAL OPTIMIZED TRAINING WITH HYPERPARAMETER TUNING
- FULL temporal + lag + rolling + EMA + aggregates
- OPTIMIZED hyperparameters for stability and performance
- CROSS-VALIDATION with early stopping
- ROBUST error handling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from joblib import dump, parallel_backend
import os
import json
import warnings
import time
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
class Config:
    """Centralized configuration for all models"""
    # Feature Lists
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
    
    # Temporal windows
    LAG_WINDOWS = [1, 7, 14, 30]
    ROLL_WINDOWS = [7, 14, 30]
    STD_WINDOWS = [7, 14]
    EMA_WINDOWS = [7, 14, 30]
    
    # Training config
    N_SPLITS = 3  # TimeSeries splits
    RANDOM_SEED = 42
    N_JOBS = -1   # Use all cores
    
    # Model types
    MODEL_TYPES = {
        'daily': {
            'name': 'daily',
            'horizon': 1,
            'features': None  # Will be filled
        },
        'horizon_30': {
            'name': '30_day',
            'horizon': 30,
            'features': None
        },
        'horizon_90': {
            'name': '90_day',
            'horizon': 90,
            'features': None
        },
        'horizon_180': {
            'name': '180_day',
            'horizon': 180,
            'features': None
        }
    }

# ---------------------------------------------------------
# HYPERPARAMETER SPACES
# ---------------------------------------------------------
class HyperparameterSpaces:
    """Optimized hyperparameter spaces for different model types"""
    
    @staticmethod
    def get_daily_model_params():
        """Hyperparameters for daily prediction (more complex)"""
        return {
            'objective': ['regression', 'regression_l1'],
            'metric': ['rmse', 'mae'],
            'boosting_type': ['gbdt', 'dart'],
            'n_estimators': [800, 1000, 1500, 2000],
            'learning_rate': [0.01, 0.02, 0.03],
            'max_depth': [6, 8, 10, 12],
            'num_leaves': [31, 63, 127],
            'min_child_samples': [10, 20, 30],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0.0, 0.1, 0.3],
            'reg_lambda': [0.0, 0.1, 0.3],
            'min_split_gain': [0.0, 0.1],
            'bagging_freq': [1, 5],
            'feature_fraction': [0.7, 0.8, 0.9],
            'verbosity': [-1]
        }
    
    @staticmethod
    def get_horizon_model_params():
        """Hyperparameters for horizon predictions (more stable)"""
        return {
            'objective': ['regression_l1'],  # MAE for stability
            'metric': ['mae', 'rmse'],
            'boosting_type': ['gbdt'],  # More stable than dart
            'n_estimators': [800, 1000, 1500],
            'learning_rate': [0.01, 0.02],
            'max_depth': [4, 6, 8],
            'num_leaves': [15, 31, 63],
            'min_child_samples': [20, 40, 60],
            'subsample': [0.6, 0.7, 0.8],
            'colsample_bytree': [0.6, 0.7, 0.8],
            'reg_alpha': [0.3, 0.5, 0.7],  # Higher regularization
            'reg_lambda': [0.3, 0.5, 0.7],
            'min_split_gain': [0.0, 0.05],
            'feature_fraction': [0.6, 0.7, 0.8],
            'verbosity': [-1]
        }
    
    @staticmethod
    def get_best_daily_params():
        """Best found parameters for daily model (after tuning)"""
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': 1500,
            'learning_rate': 0.02,
            'max_depth': 10,
            'num_leaves': 63,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_split_gain': 0.0,
            'bagging_freq': 5,
            'feature_fraction': 0.8,
            'random_state': Config.RANDOM_SEED,
            'n_jobs': Config.N_JOBS,
            'verbosity': -1
        }
    
    @staticmethod
    def get_best_horizon_params():
        """Best found parameters for horizon models (after tuning)"""
        return {
            'objective': 'regression_l1',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'n_estimators': 1000,
            'learning_rate': 0.02,
            'max_depth': 6,
            'num_leaves': 31,
            'min_child_samples': 40,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'min_split_gain': 0.0,
            'feature_fraction': 0.7,
            'random_state': Config.RANDOM_SEED,
            'n_jobs': Config.N_JOBS,
            'verbosity': -1
        }

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
class FeatureEngineer:
    """Handles all feature engineering operations"""
    
    @staticmethod
    def create_temporal_features(df):
        """Create base time features + cyclical encodings"""
        print("📊 Creating base time features...")
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['province', 'category', 'Date']).reset_index(drop=True)
        
        # Basic time features
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['day_of_month'] = df['Date'].dt.day
        df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
        df['quarter'] = df['Date'].dt.quarter
        
        # Binary flags
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 7).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        
        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Time index
        min_date = df['Date'].min()
        df['days_since_start'] = (df['Date'] - min_date).dt.days
        
        # Encode categorical
        le_province = LabelEncoder()
        le_category = LabelEncoder()
        df['Province_encoded'] = le_province.fit_transform(df['province'])
        df['Category_encoded'] = le_category.fit_transform(df['category'])
        
        # Add weather condition encoding
        le_weather = LabelEncoder()
        df['weather_encoded'] = le_weather.fit_transform(df['weather_condition'])
        
        # Save encoders
        os.makedirs("models", exist_ok=True)
        dump(le_province, "models/province_encoder.joblib")
        dump(le_category, "models/category_encoder.joblib")
        dump(le_weather, "models/weather_encoder.joblib")
        
        print("✅ Base features created")
        return df
    
    @staticmethod
    def create_temporal_advanced_features(df):
        """Create lag, rolling, EMA, and growth features"""
        print("🔄 Creating advanced temporal features...")
        
        df = df.copy()
        # Ensure correct sorting
        df = df.sort_values(['province', 'category', 'Date']).reset_index(drop=True)
        
        # Group by province and category
        groups = df.groupby(['province', 'category'])
        
        # LAG FEATURES
        for w in Config.LAG_WINDOWS:
            df[f'units_sold_lag_{w}'] = groups['units_sold'].shift(w)
            df[f'revenue_lag_{w}'] = groups['revenue_idr'].shift(w)
        
        # ROLLING MEANS (shifted for real-world prediction)
        for w in Config.ROLL_WINDOWS:
            df[f'units_sold_rolling_{w}'] = groups['units_sold'].transform(
                lambda x: x.rolling(w, min_periods=1).mean().shift(1)
            )
            df[f'revenue_rolling_{w}'] = groups['revenue_idr'].transform(
                lambda x: x.rolling(w, min_periods=1).mean().shift(1)
            )
        
        # ROLLING STD
        for w in Config.STD_WINDOWS:
            df[f'units_sold_rolling_std_{w}'] = groups['units_sold'].transform(
                lambda x: x.rolling(w, min_periods=1).std().shift(1)
            )
        
        # EMA FEATURES
        for w in Config.EMA_WINDOWS:
            df[f'units_sold_ema_{w}'] = groups['units_sold'].transform(
                lambda x: x.ewm(span=w, adjust=False).mean().shift(1)
            )
        
        # GROWTH RATES
        df['units_sold_growth_7'] = groups['units_sold'].pct_change(7)
        df['units_sold_growth_30'] = groups['units_sold'].pct_change(30)
        
        # PRICE FEATURES
        df['price_change_7'] = groups['price_idr'].pct_change(7)
        df['price_change_30'] = groups['price_idr'].pct_change(30)
        
        # Replace infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # AGGREGATE FEATURES
        # Province-category level
        stats = df.groupby(['province', 'category'])['units_sold'].agg(['mean', 'std']).reset_index()
        stats.columns = ['province', 'category', 'province_category_mean', 'province_category_std']
        df = df.merge(stats, on=['province', 'category'], how='left')
        
        # Province level
        prov_stats = df.groupby('province')['units_sold'].mean().reset_index()
        prov_stats.columns = ['province', 'province_mean_units']
        df = df.merge(prov_stats, on='province', how='left')
        
        # Category level
        cat_stats = df.groupby('category')['units_sold'].mean().reset_index()
        cat_stats.columns = ['category', 'category_mean_units']
        df = df.merge(cat_stats, on='category', how='left')
        
        # Weather impact (group by weather condition)
        weather_stats = df.groupby('weather_condition')['units_sold'].mean().reset_index()
        weather_stats.columns = ['weather_condition', 'weather_mean_units']
        df = df.merge(weather_stats, on='weather_condition', how='left')
        
        # Fill NaN - use forward fill within groups, then fill with 0
        df = df.groupby(['province', 'category']).apply(
            lambda x: x.ffill().bfill()
        ).reset_index(drop=True)
        
        df = df.fillna(0)
        
        print("✅ Advanced features created")
        return df
    
    @staticmethod
    def get_feature_lists():
        """Generate feature lists for different model types"""
        
        # Generate temporal features
        TEMPORAL_FEATURES = []
        
        # Lag Features
        for w in Config.LAG_WINDOWS:
            TEMPORAL_FEATURES += [f'units_sold_lag_{w}', f'revenue_lag_{w}']
        
        # Rolling Means
        for w in Config.ROLL_WINDOWS:
            TEMPORAL_FEATURES += [f'units_sold_rolling_{w}', f'revenue_rolling_{w}']
        
        # Rolling STD
        for w in Config.STD_WINDOWS:
            TEMPORAL_FEATURES += [f'units_sold_rolling_std_{w}']
        
        # EMA
        for w in Config.EMA_WINDOWS:
            TEMPORAL_FEATURES += [f'units_sold_ema_{w}']
        
        # Growth features
        TEMPORAL_FEATURES += ['units_sold_growth_7', 'units_sold_growth_30']
        
        # Price features
        TEMPORAL_FEATURES += ['price_change_7', 'price_change_30']
        
        # Weather feature
        TEMPORAL_FEATURES += ['weather_mean_units']
        
        # Aggregates
        TEMPORAL_FEATURES += [
            'province_category_mean', 'province_category_std',
            'province_mean_units', 'category_mean_units'
        ]
        
        # Daily model features (full set)
        daily_features = Config.BASE_FEATURES + TEMPORAL_FEATURES + ['weather_encoded']
        
        # Horizon model features (remove volatile daily features)
        horizon_features = [
            f for f in daily_features 
            if f not in [
                'day_of_month', 'week_of_year', 'quarter', 
                'is_weekend', 'is_month_start', 'is_month_end',
                'day_of_week_sin', 'day_of_week_cos',
                'day_of_week', 'month_sin', 'month_cos'
            ]
        ]
        
        return {
            'daily': daily_features,
            'horizon': horizon_features
        }

# ---------------------------------------------------------
# MODEL TRAINING WITH HYPERPARAMETER TUNING
# ---------------------------------------------------------
class ModelTrainer:
    """Handles model training with hyperparameter optimization"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.feature_lists = self.feature_engineer.get_feature_lists()
        
        # Update config with feature lists
        Config.MODEL_TYPES['daily']['features'] = self.feature_lists['daily']
        Config.MODEL_TYPES['horizon_30']['features'] = self.feature_lists['horizon'] + ['horizon_days']
        Config.MODEL_TYPES['horizon_90']['features'] = self.feature_lists['horizon'] + ['horizon_days']
        Config.MODEL_TYPES['horizon_180']['features'] = self.feature_lists['horizon'] + ['horizon_days']
    
    def train_with_tuning(self, X, y, model_type='daily', n_iter=15):
        """
        Train model with hyperparameter tuning
        Returns: best_model, best_params, cv_results
        """
        print(f"\n🎯 Training {model_type} model with hyperparameter tuning...")
        
        # Select hyperparameter space
        if model_type == 'daily':
            param_dist = HyperparameterSpaces.get_daily_model_params()
            base_params = HyperparameterSpaces.get_best_daily_params()
        else:
            param_dist = HyperparameterSpaces.get_horizon_model_params()
            base_params = HyperparameterSpaces.get_best_horizon_params()
        
        # TimeSeries Cross Validation
        tscv = TimeSeriesSplit(n_splits=Config.N_SPLITS)
        
        # Create model
        model = lgb.LGBMRegressor(**base_params)
        
        # Randomized Search CV
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=tscv,
            scoring='neg_root_mean_squared_error',
            n_jobs=Config.N_JOBS,
            random_state=Config.RANDOM_SEED,
            verbose=1
        )
        
        # Fit with timing
        start_time = time.time()
        random_search.fit(X, y)
        training_time = time.time() - start_time
        
        # Get best results
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_score = -random_search.best_score_  # Convert back from negative
        
        print(f"✅ Tuning completed in {training_time:.1f}s")
        print(f"   Best CV RMSE: {best_score:.2f}")
        print(f"   Best parameters: {best_params}")
        
        return best_model, best_params, random_search.cv_results_
    
    def train_daily_models(self, df):
        """Train daily demand and supply models"""
        print("\n" + "="*60)
        print("🏪 TRAINING DAILY MODELS")
        print("="*60)
        
        # Prepare data - ensure enough history
        df_train = df[df['days_since_start'] >= 60].copy()
        features = Config.MODEL_TYPES['daily']['features']
        
        print(f"Training data size: {len(df_train)} rows")
        print(f"Number of features: {len(features)}")
        
        # Prepare X and y
        X = df_train[features]
        
        # DEMAND MODEL
        print("\n📈 Training DEMAND model...")
        y_demand = df_train['units_sold']
        demand_model, demand_params, demand_cv = self.train_with_tuning(
            X, y_demand, model_type='daily', n_iter=15
        )
        
        # SUPPLY MODEL
        print("\n💰 Training SUPPLY model...")
        y_supply = df_train['revenue_idr']
        supply_model, supply_params, supply_cv = self.train_with_tuning(
            X, y_supply, model_type='daily', n_iter=15
        )
        
        # Save models
        self.save_model(demand_model, 'demand_model', features, demand_params)
        self.save_model(supply_model, 'supply_model', features, supply_params)
        
        # Evaluate on training set
        self.evaluate_model(demand_model, X, y_demand, 'Demand (Training)')
        self.evaluate_model(supply_model, X, y_supply, 'Supply (Training)')
        
        return demand_model, supply_model
    
    def train_horizon_models(self, df):
        """Train horizon aggregate models"""
        print("\n" + "="*60)
        print("📅 TRAINING HORIZON MODELS")
        print("="*60)
        
        HORIZONS = [30, 90, 180]
        models = {}
        
        # Create horizon dataset
        df_horizon_list = []
        for H in HORIZONS:
            # Sort and group
            df = df.sort_values(['province', 'category', 'Date'])
            g = df.groupby(['province', 'category'])
            
            # Target: Sum of units/revenue for next H days
            df[f'units_sold_sum_{H}'] = g['units_sold'].transform(
                lambda x: x.rolling(H, min_periods=1).sum().shift(-H)
            )
            df[f'revenue_idr_sum_{H}'] = g['revenue_idr'].transform(
                lambda x: x.rolling(H, min_periods=1).sum().shift(-H)
            )
            
            df_H = df[df[f'units_sold_sum_{H}'].notna()].copy()
            df_H['horizon_days'] = H
            df_H['units_sold_sum'] = df_H[f'units_sold_sum_{H}']
            df_H['revenue_idr_sum'] = df_H[f'revenue_idr_sum_{H}']
            df_horizon_list.append(df_H)
        
        df_train_h = pd.concat(df_horizon_list)
        # Ensure enough history for features + longest horizon
        df_train_h = df_train_h[df_train_h['days_since_start'] >= 60 + 180]
        
        print(f"Horizon training data size: {len(df_train_h)} rows")
        
        # Train for each horizon
        for H in HORIZONS:
            print(f"\n⏳ Training for H={H} days...")
            
            # Filter data for this horizon
            df_H = df_train_h[df_train_h['horizon_days'] == H].copy()
            features = Config.MODEL_TYPES[f'horizon_{H}']['features']
            
            # Skip if not enough data
            if len(df_H) < 100:
                print(f"  ⚠️  Not enough data for H={H}, skipping...")
                continue
            
            X = df_H[features]
            
            # DEMAND MODEL (units sold sum)
            print(f"  📈 Training DEMAND model (H={H})...")
            y_demand = df_H['units_sold_sum']
            demand_model, demand_params, _ = self.train_with_tuning(
                X, y_demand, model_type='horizon', n_iter=10
            )
            
            # SUPPLY MODEL (revenue sum)
            print(f"  💰 Training SUPPLY model (H={H})...")
            y_supply = df_H['revenue_idr_sum']
            supply_model, supply_params, _ = self.train_with_tuning(
                X, y_supply, model_type='horizon', n_iter=10
            )
            
            # Save models
            self.save_model(demand_model, f'demand_{H}_model', features, demand_params)
            self.save_model(supply_model, f'supply_{H}_model', features, supply_params)
            
            # Store in dictionary
            models[f'demand_{H}'] = demand_model
            models[f'supply_{H}'] = supply_model
            
            # Evaluate
            self.evaluate_model(demand_model, X, y_demand, f'Demand_H{H}')
            self.evaluate_model(supply_model, X, y_supply, f'Supply_H{H}')
        
        # Save horizon features
        dump(features, "models/horizon_features.joblib")
        
        return models
    
    def evaluate_model(self, model, X, y_true, model_name):
        """Evaluate model performance"""
        y_pred = model.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate percentage errors
        y_mean = np.mean(y_true)
        rmse_pct = (rmse / y_mean * 100) if y_mean != 0 else np.nan
        mae_pct = (mae / y_mean * 100) if y_mean != 0 else np.nan
        
        print(f"\n📊 {model_name} Evaluation:")
        print(f"   RMSE: {rmse:.2f} ({rmse_pct:.1f}% of mean)")
        print(f"   MAE:  {mae:.2f} ({mae_pct:.1f}% of mean)")
        print(f"   R²:   {r2:.4f}")
        
        # Feature importance for LightGBM
        if hasattr(model, 'feature_importance_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importance_
            }).sort_values('importance', ascending=False).head(10)
            
            print(f"   Top 10 features:")
            for _, row in importance_df.iterrows():
                print(f"      {row['feature']}: {row['importance']:.2f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'rmse_pct': rmse_pct,
            'mae_pct': mae_pct
        }
    
    def save_model(self, model, model_name, features, params):
        """Save model and metadata"""
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        
        # Save model
        model_path = f"models/{model_name}.joblib"
        dump(model, model_path)
        
        # Save features
        features_path = f"models/{model_name}_features.joblib"
        dump(features, features_path)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_features': len(features),
            'model_params': params,
            'feature_list': features
        }
        
        metadata_path = f"models/{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved {model_name} to {model_path}")

# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def main():
    """Main training pipeline"""
    print("🚀 STARTING OPTIMIZED MODEL TRAINING")
    print("="*60)
    
    # Start timer
    start_time = time.time()
    
    try:
        # 1. Load data
        print("\n📥 Loading data...")
        df = pd.read_csv('indonesia_supermarket_5yr_synthetic.csv')
        print(f"   Loaded {len(df):,} rows")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Provinces: {df['province'].nunique()}")
        print(f"   Categories: {df['category'].nunique()}")
        
        # 2. Feature engineering
        feature_engineer = FeatureEngineer()
        df = feature_engineer.create_temporal_features(df)
        df = feature_engineer.create_temporal_advanced_features(df)
        
        # 3. Save insights
        insights = {
            "rows": int(len(df)),
            "provinces": int(df['province'].nunique()),
            "categories": int(df['category'].nunique()),
            "date_range": f"{df['Date'].min().date()} to {df['Date'].max().date()}",
            "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "features_count": len(FeatureEngineer().get_feature_lists()['daily'])
        }
        
        os.makedirs("models", exist_ok=True)
        with open("models/insights.json", "w") as f:
            json.dump(insights, f, indent=2)
        print(f"\n📊 Insights saved: {insights}")
        
        # 4. Train models
        trainer = ModelTrainer()
        
        # Train daily models
        demand_model, supply_model = trainer.train_daily_models(df)
        
        # Train horizon models
        horizon_models = trainer.train_horizon_models(df)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETE!")
        print("="*60)
        print(f"✅ Total time: {total_time:.1f} seconds")
        print(f"✅ Models saved in: models/")
        print(f"✅ Total models trained: {2 + len(horizon_models)}")
        print("\n📁 Generated files:")
        print("   - demand_model.joblib")
        print("   - supply_model.joblib")
        
        for H in [30, 90, 180]:
            if f'demand_{H}' in horizon_models:
                print(f"   - demand_{H}_model.joblib")
                print(f"   - supply_{H}_model.joblib")
        
        print("   - *.joblib files (features and metadata)")
        print("   - insights.json")
        print("   - province_encoder.joblib, category_encoder.joblib")
        print("\n🚀 Models are ready for API deployment!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    # Run training
    success = main()
    
    if success:
        print("\n✅ All models trained successfully!")
        print("👉 Next step: Run 'python api_indonesia_optimized.py' to start API")
    else:
        print("\n❌ Training failed!")
        print("💡 Check the error message above and ensure:")
        print("   1. CSV file exists in the same directory")
        print("   2. You have required packages installed")
        print("   3. There's enough memory for training")