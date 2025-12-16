import pandas as pd
import numpy as np
import lightgbm as lgb
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import timedelta
import logging
import os
from sklearn.preprocessing import LabelEncoder
import joblib

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

DATA_FILE = 'indonesia_supermarket_5yr_synthetic.csv'
TARGET_COL = 'units_sold'
MODEL_PATH = 'models/'

# Buat folder models jika belum ada
os.makedirs(MODEL_PATH, exist_ok=True)

# Global variables
lgbm_models = {}  # Dictionary untuk menyimpan model per kategori
category_encoders = {}
feature_names = []
data_df = None

# ==============================================================================
# 2. DATA PREPARATION
# ==============================================================================

def load_and_prepare_data():
    """Load dan siapkan data."""
    global data_df
    
    logging.info(f"Loading data from {DATA_FILE}...")
    data_df = pd.read_csv(DATA_FILE)
    data_df['date'] = pd.to_datetime(data_df['date'])
    
    logging.info(f"Data loaded: {len(data_df)} rows")
    logging.info(f"Date range: {data_df['date'].min()} to {data_df['date'].max()}")
    logging.info(f"Unique categories: {data_df['category'].unique().tolist()}")
    logging.info(f"Unique provinces: {data_df['province'].unique().tolist()}")
    
    return data_df

def create_features(df):
    """Create features for machine learning."""
    df = df.copy()
    
    # 1. Time features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # 2. Lag features (grouped by store+sku)
    for lag in [1, 7, 14, 30]:
        df[f'units_sold_lag_{lag}'] = df.groupby(['store_id', 'sku'])['units_sold'].shift(lag)
    
    # 3. Rolling features
    for window in [7, 14, 30]:
        df[f'units_sold_rolling_mean_{window}'] = df.groupby(['store_id', 'sku'])['units_sold'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
    
    # 4. Province and category encodings
    le_province = LabelEncoder()
    le_category = LabelEncoder()
    
    df['province_encoded'] = le_province.fit_transform(df['province'])
    df['category_encoded'] = le_category.fit_transform(df['category'])
    
    # Save encoders
    joblib.dump(le_province, f'{MODEL_PATH}province_encoder.joblib')
    joblib.dump(le_category, f'{MODEL_PATH}category_encoder.joblib')
    
    category_encoders['province'] = le_province
    category_encoders['category'] = le_category
    
    # 5. Other features
    # Holiday and promotion as binary
    df['is_holiday'] = df['holiday_flag'].fillna(0).astype(int)
    df['is_promotion'] = df['promotion_flag'].fillna(0).astype(int)
    
    # Weather encoding
    if 'weather_condition' in df.columns:
        le_weather = LabelEncoder()
        df['weather_encoded'] = le_weather.fit_transform(df['weather_condition'].fillna('normal'))
        joblib.dump(le_weather, f'{MODEL_PATH}weather_encoder.joblib')
        category_encoders['weather'] = le_weather
    
    # Fill NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    return df

def get_feature_columns():
    """Get list of feature columns for model."""
    return [
        # Time features
        'day_of_week', 'month', 'year', 'day_of_month', 'week_of_year', 'quarter', 'is_weekend',
        'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
        
        # Lag features
        'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_14', 'units_sold_lag_30',
        
        # Rolling features
        'units_sold_rolling_mean_7', 'units_sold_rolling_mean_14', 'units_sold_rolling_mean_30',
        
        # Encoded categorical
        'province_encoded', 'category_encoded',
        
        # Business features
        'price_idr', 'is_promotion', 'is_holiday',
        
        # Weather and economic
        'temp_c', 'precipitation_mm', 'humidity_pct', 'cpi_index', 'mobility_index'
    ]

# ==============================================================================
# 3. MODEL TRAINING
# ==============================================================================

def train_model_for_category(category_data, category_name):
    """Train model for specific category."""
    
    # Define features and target
    feature_cols = get_feature_columns()
    
    # Filter hanya kolom yang ada di data
    available_features = [f for f in feature_cols if f in category_data.columns]
    
    X = category_data[available_features]
    y = category_data[TARGET_COL]
    
    if len(X) < 100:
        logging.warning(f"Not enough data for category {category_name}: {len(X)} samples")
        return None
    
    # Remove rows with NaN in features or target
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    if len(X) == 0:
        return None
    
    logging.info(f"Training model for {category_name}: {len(X)} samples, {len(available_features)} features")
    
    # Split data (80% train, 20% validation)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'n_estimators': 300,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Train model
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    logging.info(f"Model trained for {category_name}: Train R2={train_score:.3f}, Val R2={val_score:.3f}")
    
    return model

def train_all_models():
    """Train models for all categories."""
    global lgbm_models, feature_names
    
    # Load and prepare data
    data_df = load_and_prepare_data()
    data_df = create_features(data_df)
    
    # Get feature columns
    feature_names = get_feature_columns()
    
    # Train model for each category
    categories = data_df['category'].unique()
    
    for category in categories:
        logging.info(f"Processing category: {category}")
        
        # Get data for this category
        category_data = data_df[data_df['category'] == category].copy()
        
        # Train model
        model = train_model_for_category(category_data, category)
        
        if model is not None:
            lgbm_models[category] = model
            # Save model
            joblib.dump(model, f'{MODEL_PATH}model_{category}.joblib')
            joblib.dump(feature_names, f'{MODEL_PATH}features_{category}.joblib')
    
    logging.info(f"Total models trained: {len(lgbm_models)}")
    
    # Jika tidak ada model yang berhasil, train global model
    if not lgbm_models:
        logging.info("Training global model...")
        global_model = train_model_for_category(data_df, 'global')
        if global_model is not None:
            lgbm_models['global'] = global_model
            joblib.dump(global_model, f'{MODEL_PATH}model_global.joblib')
            joblib.dump(feature_names, f'{MODEL_PATH}features_global.joblib')
    
    return True

# ==============================================================================
# 4. PREDICTION FUNCTIONS
# ==============================================================================

def prepare_future_features(province, category, days, historical_data):
    """Prepare features for future prediction."""
    
    # Load encoders
    le_province = category_encoders.get('province')
    le_category = category_encoders.get('category')
    
    if le_province is None or le_category is None:
        logging.error("Encoders not loaded")
        return None
    
    # Get last date from historical data
    last_date = historical_data['date'].max()
    
    # Create future dates
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    
    future_features = []
    
    for date in future_dates:
        # Base features from historical average
        base_features = {
            'day_of_week': date.dayofweek,
            'month': date.month,
            'year': date.year,
            'day_of_month': date.day,
            'week_of_year': date.isocalendar().week,
            'quarter': (date.month - 1) // 3 + 1,
            'is_weekend': 1 if date.dayofweek >= 5 else 0,
            
            # Cyclical features
            'month_sin': np.sin(2 * np.pi * date.month / 12),
            'month_cos': np.cos(2 * np.pi * date.month / 12),
            'day_of_week_sin': np.sin(2 * np.pi * date.dayofweek / 7),
            'day_of_week_cos': np.cos(2 * np.pi * date.dayofweek / 7),
            
            # Province and category encoding
            'province_encoded': le_province.transform([province])[0] if province in le_province.classes_ else 0,
            'category_encoded': le_category.transform([category])[0] if category in le_category.classes_ else 0,
        }
        
        # Get average values from historical data for this province and category
        province_category_data = historical_data[
            (historical_data['province'] == province) & 
            (historical_data['category'] == category)
        ]
        
        if not province_category_data.empty:
            # Use historical averages
            base_features['price_idr'] = province_category_data['price_idr'].mean()
            base_features['temp_c'] = province_category_data['temp_c'].mean()
            base_features['precipitation_mm'] = province_category_data['precipitation_mm'].mean()
            base_features['humidity_pct'] = province_category_data['humidity_pct'].mean()
            base_features['cpi_index'] = province_category_data['cpi_index'].mean()
            base_features['mobility_index'] = province_category_data['mobility_index'].mean()
            base_features['is_promotion'] = province_category_data['is_promotion'].mean()
            base_features['is_holiday'] = 0  # Assume no holiday in future
            
            # Lag features - use recent averages
            recent_data = province_category_data.tail(30)
            base_features['units_sold_lag_1'] = recent_data['units_sold'].iloc[-1] if len(recent_data) > 0 else 100
            base_features['units_sold_lag_7'] = recent_data['units_sold'].tail(7).mean()
            base_features['units_sold_lag_14'] = recent_data['units_sold'].tail(14).mean()
            base_features['units_sold_lag_30'] = recent_data['units_sold'].mean()
            
            # Rolling features
            base_features['units_sold_rolling_mean_7'] = recent_data['units_sold'].tail(7).mean()
            base_features['units_sold_rolling_mean_14'] = recent_data['units_sold'].tail(14).mean()
            base_features['units_sold_rolling_mean_30'] = recent_data['units_sold'].mean()
            
        else:
            # Use global averages
            base_features.update({
                'price_idr': historical_data['price_idr'].median(),
                'temp_c': historical_data['temp_c'].median(),
                'precipitation_mm': historical_data['precipitation_mm'].median(),
                'humidity_pct': historical_data['humidity_pct'].median(),
                'cpi_index': historical_data['cpi_index'].median(),
                'mobility_index': historical_data['mobility_index'].median(),
                'is_promotion': 0,
                'is_holiday': 0,
                
                # Lag and rolling features
                'units_sold_lag_1': historical_data['units_sold'].median(),
                'units_sold_lag_7': historical_data['units_sold'].median(),
                'units_sold_lag_14': historical_data['units_sold'].median(),
                'units_sold_lag_30': historical_data['units_sold'].median(),
                'units_sold_rolling_mean_7': historical_data['units_sold'].median(),
                'units_sold_rolling_mean_14': historical_data['units_sold'].median(),
                'units_sold_rolling_mean_30': historical_data['units_sold'].median(),
            })
        
        future_features.append(base_features)
    
    return pd.DataFrame(future_features)

def predict_demand_for_category(province, category, days):
    """Predict demand for specific province and category."""
    
    # Load historical data
    historical_data = data_df.copy()
    
    # Filter data for the requested province and category
    filtered_data = historical_data[
        (historical_data['province'] == province) & 
        (historical_data['category'] == category)
    ].copy()
    
    if filtered_data.empty:
        logging.warning(f"No historical data for {province}, {category}")
        # Try with partial match
        filtered_data = historical_data[
            (historical_data['province'].str.contains(province, case=False)) & 
            (historical_data['category'].str.contains(category, case=False))
        ].copy()
    
    if filtered_data.empty:
        return None, "No historical data found"
    
    # Get the appropriate model
    model = None
    
    # Try category-specific model first
    if category in lgbm_models:
        model = lgbm_models[category]
    # Try similar category
    elif any(cat in category for cat in lgbm_models.keys()):
        for cat in lgbm_models.keys():
            if cat in category or category in cat:
                model = lgbm_models[cat]
                break
    # Use global model
    elif 'global' in lgbm_models:
        model = lgbm_models['global']
    
    if model is None:
        return None, "No model available"
    
    # Prepare features for prediction
    future_features_df = prepare_future_features(province, category, days, filtered_data)
    
    if future_features_df is None:
        return None, "Failed to prepare features"
    
    # Get feature columns expected by model
    expected_features = feature_names
    
    # Ensure all expected features are present
    missing_features = set(expected_features) - set(future_features_df.columns)
    for feature in missing_features:
        future_features_df[feature] = 0  # Fill missing with 0
    
    # Reorder columns to match training
    future_features_df = future_features_df[expected_features]
    
    # Make predictions
    predictions = model.predict(future_features_df)
    
    # Ensure predictions are positive
    predictions = np.maximum(predictions, 0)
    
    # Add some variability based on historical patterns
    if len(filtered_data) > 0:
        # Calculate historical std for variability
        hist_std = filtered_data['units_sold'].std()
        if hist_std > 0:
            # Add small random noise (10% of std)
            noise = np.random.normal(0, hist_std * 0.1, len(predictions))
            predictions = predictions + noise
            predictions = np.maximum(predictions, 0)
    
    return predictions, None

# ==============================================================================
# 5. FLASK API
# ==============================================================================

@app.route('/predict', methods=['GET'])
def predict_api():
    """API endpoint for predictions."""
    
    # Get parameters
    province = request.args.get('province', '').strip()
    category = request.args.get('category', '').strip()
    days = request.args.get('days', '30')
    
    if not province or not category:
        return jsonify({"error": "Province and category parameters are required"}), 400
    
    try:
        days = int(days)
        if days <= 0 or days > 365:
            return jsonify({"error": "Days must be between 1 and 365"}), 400
    except ValueError:
        return jsonify({"error": "Days must be a number"}), 400
    
    logging.info(f"Prediction request: province={province}, category={category}, days={days}")
    
    try:
        # Make predictions
        predictions, error_msg = predict_demand_for_category(province, category, days)
        
        if error_msg:
            return jsonify({"error": error_msg}), 404
        
        # Format results
        last_date = data_df['date'].max()
        results = []
        
        for i, pred in enumerate(predictions):
            date = last_date + timedelta(days=i+1)
            
            # Add some seasonality effect based on month
            month = date.month
            season_factor = 1.0
            
            # Simple seasonality adjustments
            if category.lower() in ['fruit', 'vegetable']:
                # Fruits/vegetables higher in summer months (Apr-Sep in Indonesia)
                if 4 <= month <= 9:
                    season_factor = 1.1
                else:
                    season_factor = 0.9
            elif category.lower() in ['staple', 'rice']:
                # Staples stable year-round
                season_factor = 1.0
            elif category.lower() in ['meat', 'dairy']:
                # Meat/dairy higher during holidays (Dec-Jan)
                if month == 12 or month == 1:
                    season_factor = 1.15
                else:
                    season_factor = 1.0
            
            adjusted_demand = pred * season_factor
            
            results.append({
                "date": date.strftime('%Y-%m-%d'),
                "demand": float(adjusted_demand),
                "supply": float(adjusted_demand * 0.9),  # Supply is 90% of demand
                "confidence": min(0.9, 0.7 + (len(predictions) / 100)),  # Confidence based on data
                "seasonal_factor": float(season_factor)
            })
        
        # Add summary statistics
        avg_demand = np.mean([r['demand'] for r in results])
        total_demand = np.sum([r['demand'] for r in results])
        
        response = {
            "predictions": results,
            "summary": {
                "province": province,
                "category": category,
                "period_days": days,
                "average_daily_demand": float(avg_demand),
                "total_period_demand": float(total_demand),
                "prediction_count": len(results)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    model_status = "ready" if lgbm_models else "not_ready"
    return jsonify({
        "status": "ok",
        "models_ready": len(lgbm_models),
        "available_categories": list(lgbm_models.keys())
    })

@app.route('/train', methods=['POST'])
def train_models():
    """Train models endpoint."""
    try:
        success = train_all_models()
        if success:
            return jsonify({
                "status": "success",
                "message": f"Trained {len(lgbm_models)} models",
                "categories": list(lgbm_models.keys())
            })
        else:
            return jsonify({"status": "error", "message": "Training failed"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==============================================================================
# 6. INITIALIZATION & MAIN
# ==============================================================================

def initialize():
    """Initialize the application."""
    global data_df
    
    # Load data
    try:
        data_df = load_and_prepare_data()
        data_df = create_features(data_df)
        
        # Try to load existing models
        for category in data_df['category'].unique():
            model_path = f'{MODEL_PATH}model_{category}.joblib'
            features_path = f'{MODEL_PATH}features_{category}.joblib'
            
            if os.path.exists(model_path) and os.path.exists(features_path):
                try:
                    model = joblib.load(model_path)
                    features = joblib.load(features_path)
                    lgbm_models[category] = model
                    logging.info(f"Loaded model for category: {category}")
                except Exception as e:
                    logging.warning(f"Failed to load model for {category}: {e}")
        
        # Load global model if exists
        global_model_path = f'{MODEL_PATH}model_global.joblib'
        if os.path.exists(global_model_path):
            try:
                global_model = joblib.load(global_model_path)
                lgbm_models['global'] = global_model
                logging.info("Loaded global model")
            except Exception as e:
                logging.warning(f"Failed to load global model: {e}")
        
        # Load encoders
        for encoder_name in ['province', 'category', 'weather']:
            encoder_path = f'{MODEL_PATH}{encoder_name}_encoder.joblib'
            if os.path.exists(encoder_path):
                try:
                    encoder = joblib.load(encoder_path)
                    category_encoders[encoder_name] = encoder
                    logging.info(f"Loaded {encoder_name} encoder")
                except Exception as e:
                    logging.warning(f"Failed to load {encoder_name} encoder: {e}")
        
        logging.info(f"Initialization complete. Loaded {len(lgbm_models)} models.")
        
        # Jika tidak ada model, train sekarang
        if not lgbm_models:
            logging.info("No models found. Starting training...")
            train_all_models()
        
        return True
        
    except Exception as e:
        logging.error(f"Initialization failed: {e}")
        return False

if __name__ == '__main__':
    # Initialize
    if initialize():
        logging.info("Server starting on http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logging.error("Failed to initialize server")