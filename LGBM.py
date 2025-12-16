#!/usr/bin/env python3
"""
AOL AI — OPTIMIZED PREDICTION API
- Real-time predictions with full feature engineering
- Dynamic feature creation matching training pipeline
- Support for daily and horizon predictions
- Robust error handling
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import timedelta, datetime
import logging
import os
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================
app = Flask(__name__)
CORS(app)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

DATA_FILE = 'indonesia_supermarket_5yr_synthetic.csv'
MODEL_PATH = 'models/'
os.makedirs(MODEL_PATH, exist_ok=True)

# Global variables
models = {}
encoders = {}
data_df = None
feature_config = None

# ==============================================================================
# 2. MODEL LOADER
# ==============================================================================
class ModelLoader:
    """Loads and manages all trained models"""
    
    @staticmethod
    def load_all_models():
        """Load all trained models from disk"""
        model_files = {
            'daily_demand': 'demand_model.joblib',
            'daily_supply': 'supply_model.joblib',
            'horizon_30_demand': 'demand_30_model.joblib',
            'horizon_30_supply': 'supply_30_model.joblib',
            'horizon_90_demand': 'demand_90_model.joblib',
            'horizon_90_supply': 'supply_90_model.joblib',
            'horizon_180_demand': 'demand_180_model.joblib',
            'horizon_180_supply': 'supply_180_model.joblib'
        }
        
        loaded_models = {}
        
        for model_name, model_file in model_files.items():
            model_path = os.path.join(MODEL_PATH, model_file)
            features_path = os.path.join(MODEL_PATH, model_file.replace('model.joblib', 'model_features.joblib'))
            metadata_path = os.path.join(MODEL_PATH, model_file.replace('model.joblib', 'model_metadata.json'))
            
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    features = joblib.load(features_path) if os.path.exists(features_path) else None
                    
                    # Load metadata if exists
                    metadata = {}
                    if os.path.exists(metadata_path):
                        import json
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    
                    loaded_models[model_name] = {
                        'model': model,
                        'features': features,
                        'metadata': metadata,
                        'loaded_at': datetime.now()
                    }
                    
                    logging.info(f"✅ Loaded model: {model_name} with {len(features) if features else 'unknown'} features")
                except Exception as e:
                    logging.warning(f"⚠️ Failed to load {model_name}: {e}")
            else:
                logging.warning(f"⚠️ Model file not found: {model_path}")
        
        return loaded_models
    
    @staticmethod
    def load_encoders():
        """Load all encoders"""
        encoder_files = {
            'province': 'province_encoder.joblib',
            'category': 'category_encoder.joblib',
            'weather': 'weather_encoder.joblib'
        }
        
        loaded_encoders = {}
        
        for encoder_name, encoder_file in encoder_files.items():
            encoder_path = os.path.join(MODEL_PATH, encoder_file)
            
            if os.path.exists(encoder_path):
                try:
                    encoder = joblib.load(encoder_path)
                    loaded_encoders[encoder_name] = encoder
                    logging.info(f"✅ Loaded encoder: {encoder_name}")
                except Exception as e:
                    logging.warning(f"⚠️ Failed to load {encoder_name} encoder: {e}")
        
        return loaded_encoders

# ==============================================================================
# 3. FEATURE ENGINEERING FOR PREDICTION
# ==============================================================================
class PredictionFeatureEngineer:
    """Creates features for prediction matching training pipeline"""
    
    def __init__(self, historical_data, encoders):
        self.historical_data = historical_data
        self.encoders = encoders
        self.feature_cache = {}
    
    def create_base_features(self, province, category, prediction_date):
        """Create base time and categorical features"""
        features = {}
        
        # 1. Time features (must match training)
        features['day_of_week'] = prediction_date.dayofweek
        features['month'] = prediction_date.month
        features['year'] = prediction_date.year
        features['day_of_month'] = prediction_date.day
        features['week_of_year'] = prediction_date.isocalendar().week
        features['quarter'] = (prediction_date.month - 1) // 3 + 1
        features['is_weekend'] = 1 if prediction_date.dayofweek >= 5 else 0
        features['is_month_start'] = 1 if prediction_date.day <= 7 else 0
        features['is_month_end'] = 1 if prediction_date.day >= 25 else 0
        
        # Cyclical features
        features['month_sin'] = np.sin(2 * np.pi * prediction_date.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * prediction_date.month / 12)
        features['day_of_week_sin'] = np.sin(2 * np.pi * prediction_date.dayofweek / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * prediction_date.dayofweek / 7)
        
        # Days since start (from training data)
        if 'Date' in self.historical_data.columns:
            min_date = self.historical_data['Date'].min()
            features['days_since_start'] = (prediction_date - min_date).days
        else:
            features['days_since_start'] = 1000
        
        # 2. Encoded categorical features
        if 'province' in self.encoders:
            try:
                features['Province_encoded'] = self.encoders['province'].transform([province])[0]
            except:
                features['Province_encoded'] = 0
        
        if 'category' in self.encoders:
            try:
                features['Category_encoded'] = self.encoders['category'].transform([category])[0]
            except:
                features['Category_encoded'] = 0
        
        # Weather encoding (use most common weather for prediction)
        if 'weather' in self.encoders:
            try:
                # Get most recent weather for this province
                province_data = self.historical_data[self.historical_data['province'] == province]
                if not province_data.empty:
                    recent_weather = province_data['weather_condition'].mode()
                    if not recent_weather.empty:
                        features['weather_encoded'] = self.encoders['weather'].transform([recent_weather.iloc[0]])[0]
            except:
                features['weather_encoded'] = 0
        
        return features
    
    def create_historical_features(self, province, category, prediction_date):
        """Create features based on historical data"""
        features = {}
        
        # Filter historical data for this province and category
        filtered_data = self.historical_data[
            (self.historical_data['province'] == province) & 
            (self.historical_data['category'] == category)
        ].copy()
        
        # If no exact match, try partial
        if filtered_data.empty:
            filtered_data = self.historical_data[
                (self.historical_data['province'].str.contains(province, case=False, na=False))
            ].copy()
        
        if filtered_data.empty:
            # Use global data
            filtered_data = self.historical_data.copy()
        
        # Sort by date
        filtered_data = filtered_data.sort_values('Date')
        
        # Get recent data (last 60 days)
        recent_cutoff = prediction_date - timedelta(days=60)
        recent_data = filtered_data[filtered_data['Date'] <= prediction_date].tail(60)
        
        if not recent_data.empty:
            # Basic statistics
            features['price_idr'] = recent_data['price_idr'].mean()
            features['temp_c'] = recent_data['temp_c'].mean()
            features['precipitation_mm'] = recent_data['precipitation_mm'].mean()
            features['humidity_pct'] = recent_data['humidity_pct'].mean()
            features['cpi_index'] = recent_data['cpi_index'].mean()
            features['mobility_index'] = recent_data['mobility_index'].mean()
            features['promotion_flag'] = recent_data['promotion_flag'].mean()
            features['holiday_flag'] = 0  # Assume no holiday in future
            
            # Lag features (matching training LAG_WINDOWS = [1, 7, 14, 30])
            if len(recent_data) >= 1:
                features['units_sold_lag_1'] = recent_data['units_sold'].iloc[-1]
                features['revenue_lag_1'] = recent_data['revenue_idr'].iloc[-1]
            
            if len(recent_data) >= 7:
                features['units_sold_lag_7'] = recent_data['units_sold'].iloc[-7]
                features['revenue_lag_7'] = recent_data['revenue_idr'].iloc[-7]
            
            if len(recent_data) >= 14:
                features['units_sold_lag_14'] = recent_data['units_sold'].iloc[-14]
                features['revenue_lag_14'] = recent_data['revenue_idr'].iloc[-14]
            
            if len(recent_data) >= 30:
                features['units_sold_lag_30'] = recent_data['units_sold'].iloc[-30]
                features['revenue_lag_30'] = recent_data['revenue_idr'].iloc[-30]
            
            # Rolling means (matching training ROLL_WINDOWS = [7, 14, 30])
            for window in [7, 14, 30]:
                if len(recent_data) >= window:
                    features[f'units_sold_rolling_{window}'] = recent_data['units_sold'].tail(window).mean()
                    features[f'revenue_rolling_{window}'] = recent_data['revenue_idr'].tail(window).mean()
                else:
                    features[f'units_sold_rolling_{window}'] = recent_data['units_sold'].mean()
                    features[f'revenue_rolling_{window}'] = recent_data['revenue_idr'].mean()
            
            # Rolling STD (matching training STD_WINDOWS = [7, 14])
            for window in [7, 14]:
                if len(recent_data) >= window:
                    features[f'units_sold_rolling_std_{window}'] = recent_data['units_sold'].tail(window).std()
                else:
                    features[f'units_sold_rolling_std_{window}'] = 10
            
            # EMA features (matching training EMA_WINDOWS = [7, 14, 30])
            for window in [7, 14, 30]:
                if len(recent_data) >= window:
                    ema = recent_data['units_sold'].ewm(span=window, adjust=False).mean()
                    features[f'units_sold_ema_{window}'] = ema.iloc[-1]
                else:
                    features[f'units_sold_ema_{window}'] = recent_data['units_sold'].mean()
            
            # Growth rates
            if len(recent_data) >= 8:
                current = recent_data['units_sold'].iloc[-1]
                week_ago = recent_data['units_sold'].iloc[-8]
                features['units_sold_growth_7'] = (current - week_ago) / week_ago if week_ago != 0 else 0
            else:
                features['units_sold_growth_7'] = 0
            
            if len(recent_data) >= 31:
                current = recent_data['units_sold'].iloc[-1]
                month_ago = recent_data['units_sold'].iloc[-31]
                features['units_sold_growth_30'] = (current - month_ago) / month_ago if month_ago != 0 else 0
            else:
                features['units_sold_growth_30'] = 0
            
            # Price changes
            if len(recent_data) >= 8:
                current_price = recent_data['price_idr'].iloc[-1]
                week_ago_price = recent_data['price_idr'].iloc[-8]
                features['price_change_7'] = (current_price - week_ago_price) / week_ago_price if week_ago_price != 0 else 0
            else:
                features['price_change_7'] = 0
            
            if len(recent_data) >= 31:
                current_price = recent_data['price_idr'].iloc[-1]
                month_ago_price = recent_data['price_idr'].iloc[-31]
                features['price_change_30'] = (current_price - month_ago_price) / month_ago_price if month_ago_price != 0 else 0
            else:
                features['price_change_30'] = 0
            
            # Aggregate features
            # Province-category level
            province_category_data = self.historical_data[
                (self.historical_data['province'] == province) & 
                (self.historical_data['category'] == category)
            ]
            
            if not province_category_data.empty:
                features['province_category_mean'] = province_category_data['units_sold'].mean()
                features['province_category_std'] = province_category_data['units_sold'].std()
            else:
                features['province_category_mean'] = recent_data['units_sold'].mean()
                features['province_category_std'] = recent_data['units_sold'].std()
            
            # Province level
            province_data = self.historical_data[self.historical_data['province'] == province]
            if not province_data.empty:
                features['province_mean_units'] = province_data['units_sold'].mean()
            else:
                features['province_mean_units'] = recent_data['units_sold'].mean()
            
            # Category level
            category_data = self.historical_data[self.historical_data['category'] == category]
            if not category_data.empty:
                features['category_mean_units'] = category_data['units_sold'].mean()
            else:
                features['category_mean_units'] = recent_data['units_sold'].mean()
            
            # Weather mean units
            if 'weather_condition' in recent_data.columns:
                weather_mean = recent_data.groupby('weather_condition')['units_sold'].mean()
                if not weather_mean.empty:
                    features['weather_mean_units'] = weather_mean.mean()
            
        else:
            # Default values if no historical data
            defaults = {
                'price_idr': 50000,
                'temp_c': 27,
                'precipitation_mm': 100,
                'humidity_pct': 75,
                'cpi_index': 100,
                'mobility_index': 100,
                'promotion_flag': 0,
                'holiday_flag': 0,
                'units_sold_lag_1': 100,
                'units_sold_lag_7': 100,
                'units_sold_lag_14': 100,
                'units_sold_lag_30': 100,
                'revenue_lag_1': 10000,
                'revenue_lag_7': 10000,
                'revenue_lag_14': 10000,
                'revenue_lag_30': 10000,
                'units_sold_rolling_7': 100,
                'units_sold_rolling_14': 100,
                'units_sold_rolling_30': 100,
                'revenue_rolling_7': 10000,
                'revenue_rolling_14': 10000,
                'revenue_rolling_30': 10000,
                'units_sold_rolling_std_7': 10,
                'units_sold_rolling_std_14': 10,
                'units_sold_ema_7': 100,
                'units_sold_ema_14': 100,
                'units_sold_ema_30': 100,
                'units_sold_growth_7': 0,
                'units_sold_growth_30': 0,
                'price_change_7': 0,
                'price_change_30': 0,
                'province_category_mean': 100,
                'province_category_std': 10,
                'province_mean_units': 100,
                'category_mean_units': 100,
                'weather_mean_units': 100,
                'weather_encoded': 0
            }
            features.update(defaults)
        
        # Ensure all features are present
        required_features = [
            'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_14', 'units_sold_lag_30',
            'revenue_lag_1', 'revenue_lag_7', 'revenue_lag_14', 'revenue_lag_30',
            'units_sold_rolling_7', 'units_sold_rolling_14', 'units_sold_rolling_30',
            'revenue_rolling_7', 'revenue_rolling_14', 'revenue_rolling_30',
            'units_sold_rolling_std_7', 'units_sold_rolling_std_14',
            'units_sold_ema_7', 'units_sold_ema_14', 'units_sold_ema_30',
            'units_sold_growth_7', 'units_sold_growth_30',
            'price_change_7', 'price_change_30',
            'province_category_mean', 'province_category_std',
            'province_mean_units', 'category_mean_units',
            'weather_mean_units', 'weather_encoded'
        ]
        
        for feat in required_features:
            if feat not in features:
                features[feat] = 0
        
        return features
    
    def prepare_features(self, province, category, prediction_date, model_type='daily'):
        """Prepare all features for prediction"""
        # Create cache key
        cache_key = f"{province}_{category}_{prediction_date.strftime('%Y%m%d')}_{model_type}"
        
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # Get base features
        features = self.create_base_features(province, category, prediction_date)
        
        # Add historical features
        historical_features = self.create_historical_features(province, category, prediction_date)
        features.update(historical_features)
        
        # Add horizon days for horizon models
        if model_type == 'horizon':
            features['horizon_days'] = 30  # Default, will be overridden
        
        # Cache the result
        self.feature_cache[cache_key] = features
        
        return features

# ==============================================================================
# 4. PREDICTION ENGINE
# ==============================================================================
class PredictionEngine:
    """Main prediction engine"""
    
    def __init__(self, models, historical_data, encoders):
        self.models = models
        self.historical_data = historical_data
        self.encoders = encoders
        self.feature_engineer = PredictionFeatureEngineer(historical_data, encoders)
    
    def predict_daily(self, province, category, days):
        """Make daily predictions"""
        predictions = []
        
        # Get last date from historical data
        last_date = self.historical_data['Date'].max()
        
        # Get appropriate model
        model_key = 'daily_demand'
        if model_key not in self.models:
            model_key = 'daily_supply'  # Fallback
            if model_key not in self.models:
                raise ValueError("Daily models not loaded")
        
        model_info = self.models.get('daily_demand', self.models.get('daily_supply'))
        model = model_info['model']
        expected_features = model_info['features']
        
        # Also get supply model if available
        supply_model_info = self.models.get('daily_supply')
        supply_model = supply_model_info['model'] if supply_model_info else None
        supply_features = supply_model_info['features'] if supply_model_info else expected_features
        
        for i in range(1, days + 1):
            prediction_date = last_date + timedelta(days=i)
            
            # Prepare features
            features = self.feature_engineer.prepare_features(
                province, category, prediction_date, 'daily'
            )
            
            # Prepare feature DataFrame for demand model
            if expected_features:
                feature_values = [features.get(f, 0) for f in expected_features]
                feature_df = pd.DataFrame([feature_values], columns=expected_features)
            else:
                feature_df = pd.DataFrame([features])
            
            # Make demand prediction
            try:
                demand_pred = model.predict(feature_df)[0]
                demand_pred = max(0, float(demand_pred))
            except Exception as e:
                logging.error(f"Demand prediction error: {e}")
                demand_pred = 100  # Default fallback
            
            # Make supply prediction if model available
            if supply_model and supply_features:
                try:
                    supply_feature_values = [features.get(f, 0) for f in supply_features]
                    supply_feature_df = pd.DataFrame([supply_feature_values], columns=supply_features)
                    supply_pred = supply_model.predict(supply_feature_df)[0]
                    supply_pred = max(0, float(supply_pred))
                except:
                    supply_pred = demand_pred * 0.9  # Fallback: supply is 90% of demand
            else:
                supply_pred = demand_pred * 0.9
            
            # Calculate confidence based on data availability
            confidence = 0.85
            province_category_data = self.historical_data[
                (self.historical_data['province'] == province) & 
                (self.historical_data['category'] == category)
            ]
            
            if len(province_category_data) < 10:
                confidence = 0.7
            elif len(province_category_data) < 30:
                confidence = 0.8
            
            predictions.append({
                'date': prediction_date.strftime('%Y-%m-%d'),
                'demand': round(demand_pred, 2),
                'supply': round(supply_pred, 2),
                'confidence': round(confidence, 2),
                'features_used': len(expected_features) if expected_features else len(features)
            })
        
        return predictions
    
    def predict_horizon(self, province, category, horizon_days):
        """Make horizon predictions (aggregated)"""
        # Check if horizon model exists
        model_key = f'horizon_{horizon_days}_demand'
        if model_key not in self.models:
            # Fallback to daily prediction
            logging.warning(f"Horizon {horizon_days} model not found, using daily predictions")
            return self.predict_daily(province, category, horizon_days), None
        
        model_info = self.models[model_key]
        model = model_info['model']
        expected_features = model_info['features']
        
        # Get last date
        last_date = self.historical_data['Date'].max()
        
        # Prepare features (use the prediction date as last_date + 1)
        prediction_date = last_date + timedelta(days=1)
        features = self.feature_engineer.prepare_features(
            province, category, prediction_date, 'horizon'
        )
        
        # Add horizon days
        features['horizon_days'] = horizon_days
        
        # Convert to DataFrame
        if expected_features:
            feature_values = [features.get(f, 0) for f in expected_features]
            feature_df = pd.DataFrame([feature_values], columns=expected_features)
        else:
            feature_df = pd.DataFrame([features])
        
        # Make prediction for total demand
        try:
            total_demand = model.predict(feature_df)[0]
            total_demand = max(0, float(total_demand))
            
            # Try to get supply prediction
            supply_model_key = f'horizon_{horizon_days}_supply'
            if supply_model_key in self.models:
                supply_model = self.models[supply_model_key]['model']
                total_supply = supply_model.predict(feature_df)[0]
                total_supply = max(0, float(total_supply))
            else:
                total_supply = total_demand * 0.9
            
        except Exception as e:
            logging.error(f"Horizon prediction error: {e}")
            total_demand = 3000 * horizon_days / 30  # Fallback
            total_supply = total_demand * 0.9
        
        # Calculate daily averages
        avg_daily_demand = total_demand / horizon_days
        avg_daily_supply = total_supply / horizon_days
        
        # Generate daily breakdown with seasonality
        predictions = []
        
        # Get seasonality factor
        seasonal_factor = self.get_seasonality_factor(category, last_date.month)
        
        for i in range(1, horizon_days + 1):
            date = last_date + timedelta(days=i)
            
            # Add daily variation (weekly pattern)
            day_of_week = date.weekday()
            if day_of_week in [5, 6]:  # Weekend
                day_factor = 1.2
            else:
                day_factor = 1.0
            
            # Add small random variation
            random_factor = 1 + 0.05 * np.random.randn()
            
            daily_demand = avg_daily_demand * day_factor * random_factor * seasonal_factor
            daily_supply = avg_daily_supply * day_factor * random_factor * seasonal_factor
            
            predictions.append({
                'date': date.strftime('%Y-%m-%d'),
                'demand': round(max(0, daily_demand), 2),
                'supply': round(max(0, daily_supply), 2),
                'confidence': 0.8,
                'seasonal_factor': round(seasonal_factor, 2),
                'day_factor': round(day_factor, 2)
            })
        
        return predictions, total_demand
    
    def get_seasonality_factor(self, category, month):
        """Calculate seasonality factor based on category and month"""
        # Seasonality patterns based on Indonesian context
        season_factors = {
            'fruit': {
                1: 1.1, 2: 1.0, 3: 1.0, 4: 1.2, 5: 1.3, 6: 1.4,
                7: 1.4, 8: 1.3, 9: 1.2, 10: 1.1, 11: 1.0, 12: 1.1
            },
            'vegetable': {
                1: 1.0, 2: 1.0, 3: 1.1, 4: 1.2, 5: 1.3, 6: 1.3,
                7: 1.2, 8: 1.1, 9: 1.1, 10: 1.0, 11: 1.0, 12: 1.0
            },
            'meat': {
                1: 1.3, 2: 1.1, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.1,
                7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.1, 12: 1.4  # High in Dec for holidays
            },
            'dairy': {
                1: 1.1, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0,
                7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.2
            },
            'staple': {
                1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0,
                7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0  # Stable
            }
        }
        
        category_lower = category.lower()
        for cat_key, factors in season_factors.items():
            if cat_key in category_lower:
                return factors.get(month, 1.0)
        
        return 1.0

# ==============================================================================
# 5. FLASK API ENDPOINTS
# ==============================================================================
@app.route('/predict', methods=['GET'])
def predict_api():
    """Main prediction endpoint"""
    
    # Get parameters
    province = request.args.get('province', '').strip()
    category = request.args.get('category', '').strip()
    days = request.args.get('days', '30')
    prediction_type = request.args.get('type', 'daily')  # 'daily' or 'horizon'
    
    # Validate parameters
    if not province or not category:
        return jsonify({"error": "Province and category parameters are required"}), 400
    
    try:
        days = int(days)
        if days <= 0 or days > 365:
            return jsonify({"error": "Days must be between 1 and 365"}), 400
    except ValueError:
        return jsonify({"error": "Days must be a number"}), 400
    
    logging.info(f"Prediction request: {province}, {category}, {days} days, type={prediction_type}")
    
    try:
        # Initialize prediction engine
        prediction_engine = PredictionEngine(models, data_df, encoders)
        
        if prediction_type == 'horizon' and days in [30, 90, 180]:
            # Use horizon model
            predictions, total_demand = prediction_engine.predict_horizon(province, category, days)
            total_demand = total_demand if total_demand is not None else sum(p['demand'] for p in predictions)
        else:
            # Use daily model
            predictions = prediction_engine.predict_daily(province, category, days)
            total_demand = sum(p['demand'] for p in predictions)
        
        # Calculate statistics
        avg_demand = total_demand / days
        
        # Prepare response
        response = {
            "predictions": predictions,
            "summary": {
                "province": province,
                "category": category,
                "prediction_type": prediction_type,
                "period_days": days,
                "average_daily_demand": round(float(avg_demand), 2),
                "total_period_demand": round(float(total_demand), 2),
                "prediction_count": len(predictions),
                "model_used": f"{prediction_type}_{days}" if prediction_type == 'horizon' and days in [30, 90, 180] else "daily",
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        }
        
        logging.info(f"Prediction successful: {len(predictions)} days predicted")
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            "error": f"Prediction failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(models),
        "available_models": list(models.keys()),
        "data_rows": len(data_df) if data_df is not None else 0,
        "encoders_loaded": list(encoders.keys()),
        "api_version": "2.0.0"
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List all available models"""
    model_list = []
    for model_name, model_info in models.items():
        model_list.append({
            "name": model_name,
            "loaded_at": model_info.get('loaded_at').isoformat() if model_info.get('loaded_at') else None,
            "features_count": len(model_info.get('features', [])),
            "has_metadata": 'metadata' in model_info and model_info['metadata'] is not None
        })
    
    return jsonify({
        "models": model_list,
        "total_models": len(model_list)
    })

@app.route('/provinces', methods=['GET'])
def list_provinces():
    """List all available provinces"""
    if data_df is not None:
        provinces = sorted(data_df['province'].unique().tolist())
        return jsonify({
            "provinces": provinces,
            "count": len(provinces)
        })
    return jsonify({"provinces": [], "count": 0})

@app.route('/categories', methods=['GET'])
def list_categories():
    """List all available categories"""
    if data_df is not None:
        categories = sorted(data_df['category'].unique().tolist())
        return jsonify({
            "categories": categories,
            "count": len(categories)
        })
    return jsonify({"categories": [], "count": 0})

@app.route('/features', methods=['GET'])
def list_features():
    """List features for a specific model"""
    model_name = request.args.get('model', 'daily_demand')
    
    if model_name in models:
        model_info = models[model_name]
        features = model_info.get('features', [])
        
        return jsonify({
            "model": model_name,
            "features": features,
            "feature_count": len(features)
        })
    
    return jsonify({
        "error": f"Model {model_name} not found"
    }), 404

# ==============================================================================
# 6. INITIALIZATION
# ==============================================================================
def initialize():
    """Initialize the application"""
    global models, encoders, data_df
    
    try:
        logging.info("🚀 Initializing Supermarket Analytics API...")
        
        # 1. Load data
        logging.info("📥 Loading data...")
        if not os.path.exists(DATA_FILE):
            logging.error(f"Data file not found: {DATA_FILE}")
            return False
        
        data_df = pd.read_csv(DATA_FILE)
        data_df['date'] = pd.to_datetime(data_df['date'])
        data_df['Date'] = data_df['date']  # Alias for compatibility
        
        logging.info(f"✅ Data loaded: {len(data_df)} rows")
        logging.info(f"   Date range: {data_df['date'].min()} to {data_df['date'].max()}")
        logging.info(f"   Provinces: {data_df['province'].nunique()}")
        logging.info(f"   Categories: {data_df['category'].nunique()}")
        
        # 2. Load models
        logging.info("🤖 Loading models...")
        models = ModelLoader.load_all_models()
        
        if not models:
            logging.error("❌ No models loaded!")
            logging.info("💡 Run train_optimized.py first to train models")
            return False
        
        # 3. Load encoders
        logging.info("🔤 Loading encoders...")
        encoders = ModelLoader.load_encoders()
        
        logging.info(f"✅ Initialization complete!")
        logging.info(f"   Models: {len(models)}")
        logging.info(f"   Encoders: {len(encoders)}")
        logging.info(f"   Data: {len(data_df)} rows")
        
        # Test prediction engine
        try:
            prediction_engine = PredictionEngine(models, data_df, encoders)
            test_province = data_df['province'].iloc[0]
            test_category = data_df['category'].iloc[0]
            test_predictions = prediction_engine.predict_daily(test_province, test_category, 1)
            logging.info(f"   Test prediction: {test_province}, {test_category} -> {test_predictions[0]['demand']:.2f}")
        except Exception as e:
            logging.warning(f"Test prediction failed: {e}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Initialization failed: {e}", exc_info=True)
        return False

# ==============================================================================
# 7. MAIN
# ==============================================================================
if __name__ == '__main__':
    # Initialize
    if initialize():
        logging.info("🌐 Starting Flask server on http://0.0.0.0:5000")
        logging.info("🔗 Available endpoints:")
        logging.info("   GET /health - Health check")
        logging.info("   GET /predict - Make predictions")
        logging.info("   GET /models - List loaded models")
        logging.info("   GET /provinces - List provinces")
        logging.info("   GET /categories - List categories")
        logging.info("   GET /features - List model features")
        logging.info("\n📋 Example prediction request:")
        logging.info("   http://localhost:5000/predict?province=Jakarta&category=staple&days=30")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        logging.error("❌ Failed to initialize server")
        logging.error("💡 Make sure to:")
        logging.error("   1. Run train_optimized.py first to train models")
        logging.error("   2. Check that indonesia_supermarket_5yr_synthetic.csv exists")