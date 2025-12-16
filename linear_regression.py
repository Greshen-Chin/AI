#linear_regression.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np # Diperlukan untuk isocalendar().week

def load_and_preprocess_simple(path):
    """
    Memuat data, membuat fitur, dan meng-encode kategori.
    Output: X (fitur), y (target)
    """
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['date'])

    # 1. Feature Engineering (Membuat fitur waktu)
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['day_of_week'] = df['Date'].dt.dayofweek

    # 2. Encoding Kategorikal (Provinsi & Kategori)
    le_province = LabelEncoder()
    le_category = LabelEncoder()
    df['Province_encoded'] = le_province.fit_transform(df['province'])
    df['Category_encoded'] = le_category.fit_transform(df['category'])

    # 3. Definisikan Fitur dan Target
    features = [
        'Province_encoded', 'Category_encoded', 'month', 'year',
        'day_of_week', 'price_idr', 'promotion_flag', 'holiday_flag'
    ]
    
    X = df[features]
    y = df['units_sold']
    
    return X, y

def train_linear_regression(X, y):
    """
    Melatih model Regresi Linier dan menampilkan skor.
    """
    # 1. Bagi Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Latih Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 3. Evaluasi
    test_score = model.score(X_test, y_test)
    print(f"✅ Linear Regression Test Score (R²): {test_score:.3f}")
    
    return model

if __name__ == '__main__':
    # Anggap 'indonesia_supermarket_5yr_synthetic.csv' berada di direktori yang sama
    X_features, y_target = load_and_preprocess_simple("indonesia_supermarket_5yr_synthetic.csv")
    
    print("🚀 Mulai pelatihan Regresi Linier...")
    linear_model = train_linear_regression(X_features, y_target)
    
    # Cetak koefisien untuk interpretasi
    print("\n💡 Koefisien Model (Pengaruh Fitur terhadap units_sold):")
    for name, coef in zip(X_features.columns, linear_model.coef_):
        print(f"   {name}: {coef:.4f}")