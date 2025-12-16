#!/usr/bin/env python3
"""
Create Comprehensive Indonesia Market Dataset
Generate synthetic data for all 34 Indonesian provinces with diverse products
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_comprehensive_indonesia_dataset():
    """Create comprehensive dataset for all Indonesian provinces"""

    # All 34 Indonesian provinces
    provinces = [
        'Aceh', 'North Sumatra', 'West Sumatra', 'Riau', 'Jambi', 'South Sumatra',
        'Bengkulu', 'Lampung', 'Bangka Belitung Islands', 'Riau Islands',
        'DKI Jakarta', 'West Java', 'Central Java', 'DI Yogyakarta', 'East Java',
        'Banten', 'Bali', 'West Nusa Tenggara', 'East Nusa Tenggara',
        'West Kalimantan', 'Central Kalimantan', 'South Kalimantan', 'East Kalimantan', 'North Kalimantan',
        'North Sulawesi', 'Central Sulawesi', 'South Sulawesi', 'South East Sulawesi', 'Gorontalo', 'West Sulawesi',
        'Maluku', 'North Maluku', 'West Papua', 'Papua'
    ]

    # Cities for each province
    province_cities = {
        'DKI Jakarta': ['Jakarta Pusat', 'Jakarta Utara', 'Jakarta Barat', 'Jakarta Selatan', 'Jakarta Timur'],
        'West Java': ['Bandung', 'Bekasi', 'Bogor', 'Depok', 'Cirebon', 'Tasikmalaya'],
        'East Java': ['Surabaya', 'Malang', 'Semarang', 'Yogyakarta', 'Solo'],
        'Central Java': ['Semarang', 'Solo', 'Magelang', 'Pekalongan', 'Tegal'],
        'North Sumatra': ['Medan', 'Binjai', 'Tebing Tinggi', 'Pematangsiantar'],
        'South Sulawesi': ['Makassar', 'Parepare', 'Palopo', 'Pinrang'],
        'Bali': ['Denpasar', 'Badung', 'Gianyar', 'Tabanan'],
        'West Sumatra': ['Padang', 'Bukittinggi', 'Payakumbuh', 'Solok'],
        'Riau': ['Pekanbaru', 'Dumai', 'Duri', 'Bangkinang'],
        'Jambi': ['Jambi', 'Sungai Penuh', 'Muara Bungo'],
        'South Sumatra': ['Palembang', 'Lubuklinggau', 'Prabumulih'],
        'Bengkulu': ['Bengkulu', 'Curup'],
        'Lampung': ['Bandar Lampung', 'Metro', 'Kotabumi'],
        'Bangka Belitung Islands': ['Pangkalpinang', 'Sungailiat'],
        'Riau Islands': ['Tanjungpinang', 'Batam', 'Karimun'],
        'DI Yogyakarta': ['Yogyakarta', 'Sleman', 'Bantul'],
        'Banten': ['Serang', 'Tangerang', 'Cilegon'],
        'West Nusa Tenggara': ['Mataram', 'Praya', 'Selong'],
        'East Nusa Tenggara': ['Kupang', 'Atambua', 'Soe'],
        'West Kalimantan': ['Pontianak', 'Singkawang', 'Sambas'],
        'Central Kalimantan': ['Palangkaraya', 'Pangkalanbun'],
        'South Kalimantan': ['Banjarmasin', 'Banjarbaru', 'Martapura'],
        'East Kalimantan': ['Samarinda', 'Balikpapan', 'Bontang'],
        'North Kalimantan': ['Tarakan', 'Tanjung Selor'],
        'North Sulawesi': ['Manado', 'Bitung', 'Tomohon'],
        'Central Sulawesi': ['Palu', 'Donggala', 'Poso'],
        'South East Sulawesi': ['Kendari', 'Baubau'],
        'Gorontalo': ['Gorontalo', 'Limboto'],
        'West Sulawesi': ['Mamuju', 'Majene'],
        'Maluku': ['Ambon', 'Tual'],
        'North Maluku': ['Ternate', 'Tidore'],
        'West Papua': ['Manokwari', 'Sorong'],
        'Papua': ['Jayapura', 'Merauke', 'Timika'],
        'Aceh': ['Banda Aceh', 'Lhokseumawe', 'Langsa']
    }

    # Comprehensive product categories and items
    product_categories = {
        'fruit': [
            'Pisang Cavendish', 'Pisang Ambon', 'Apel Fuji', 'Apel Granny Smith',
            'Jeruk Sunkist', 'Jeruk Mandarin', 'Mangga Harum Manis', 'Mangga Gedong',
            'Nanas Madu', 'Nanas Queen', 'Salak Bali', 'Salak Pondoh',
            'Rambutan Binjai', 'Rambutan Simacan', 'Durian Musang King', 'Durian Monthong',
            'Alpukat Mentega', 'Alpukat Miki', 'Melon Cantaloupe', 'Melon Honey Dew',
            'Semangka Merah', 'Semangka Kuning', 'Pepaya California', 'Pepaya Bangkok',
            'Manggis', 'Sirsak', 'Jambu Biji', 'Belimbing'
        ],
        'vegetable': [
            'Bayam Hijau', 'Bayam Merah', 'Kangkung', 'Sawi Hijau', 'Sawi Putih',
            'Kol Putih', 'Kol Merah', 'Brokoli', 'Kembang Kol', 'Wortel Import',
            'Wortel Lokal', 'Kentang Granola', 'Kentang Lokal', 'Bawang Merah',
            'Bawang Putih', 'Cabe Merah Besar', 'Cabe Merah Keriting', 'Cabe Rawit Merah',
            'Cabe Rawit Hijau', 'Tomat Merah', 'Tomat Hijau', 'Terong Ungu',
            'Terong Hijau', 'Labu Siam', 'Labu Kuning', 'Timun Jepang', 'Timun Lokal',
            'Kacang Panjang', 'Buncis', 'Jagung Manis', 'Jagung Pipil'
        ],
        'meat': [
            'Ayam Kampung', 'Ayam Broiler', 'Ayam Negeri', 'Daging Sapi Has Dalam',
            'Daging Sapi Sandung Lamur', 'Daging Sapi Tetelan', 'Daging Kambing',
            'Daging Domba', 'Ikan Tuna', 'Ikan Tongkol', 'Ikan Kakap Merah',
            'Ikan Kakap Putih', 'Ikan Gurame', 'Ikan Nila', 'Ikan Lele',
            'Udang Vaname', 'Udang Windu', 'Cumi-Cumi', 'Kepiting'
        ],
        'dairy': [
            'Susu Sapi Full Cream', 'Susu Sapi Low Fat', 'Susu Kambing',
            'Keju Cheddar', 'Keju Mozzarella', 'Keju Parmesan', 'Yogurt Plain',
            'Yogurt Strawberry', 'Yogurt Blueberry', 'Mentega', 'Margarin',
            'Es Krim Vanila', 'Es Krim Coklat', 'Es Krim Stroberi'
        ],
        'staple': [
            'Beras Premium', 'Beras Medium', 'Beras Pandan Wangi', 'Mie Instan Goreng',
            'Mie Instan Kuah', 'Mie Telur', 'Mie Kuning', 'Minyak Goreng Curah',
            'Minyak Goreng Kemasan', 'Gula Pasir', 'Gula Merah', 'Tepung Terigu',
            'Tepung Beras', 'Tepung Tapioka', 'Kecap Manis', 'Kecap Asin',
            'Saos Tomat', 'Saos Sambal', 'Terasi Udang', 'Terasi Ebi'
        ]
    }

    # Generate data
    data = []
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)

    print("🗓️ Generating comprehensive market data...")
    print(f"📊 Provinces: {len(provinces)}")
    print(f"🏙️ Cities: {sum(len(cities) for cities in province_cities.values())}")
    print(f"🥕 Products: {sum(len(products) for products in product_categories.values())}")

    total_days = (end_date - start_date).days + 1
    total_records = len(provinces) * sum(len(cities) for cities in province_cities.values()) * sum(len(products) for products in product_categories.values()) * total_days

    print(f"📈 Total records to generate: {total_records:,}")

    record_count = 0

    for province in provinces:
        cities = province_cities.get(province, [f"{province} City"])

        for city in cities:
            for category, products in product_categories.items():
                for product in products:
                    # Generate time series for each product
                    current_date = start_date
                    while current_date <= end_date:
                        # Base demand patterns
                        base_demand = get_base_demand(category, product, province, current_date)

                        # Add seasonal variations
                        seasonal_factor = get_seasonal_factor(category, current_date.month)

                        # Add regional variations
                        regional_factor = get_regional_factor(province, category)

                        # Add day-of-week variations
                        dow_factor = get_day_of_week_factor(current_date.weekday())

                        # Add random noise
                        noise = np.random.normal(0, 0.1)

                        # Calculate final demand
                        demand = max(0, base_demand * seasonal_factor * regional_factor * dow_factor * (1 + noise))

                        # Calculate supply (usually slightly higher than demand)
                        supply_factor = np.random.uniform(1.05, 1.25)
                        supply = demand * supply_factor

                        # Calculate price based on category and region
                        base_price = get_base_price(category, product)
                        price_variation = np.random.uniform(0.9, 1.1)
                        price = base_price * price_variation

                        # Revenue
                        revenue = demand * price

                        # Promotion and holiday flags
                        promotion_flag = random.random() < 0.1  # 10% chance
                        holiday_flag = is_holiday(current_date)

                        # Weather and mobility (simplified)
                        temp = get_temperature(province, current_date.month)
                        precipitation = get_precipitation(province, current_date.month)
                        humidity = get_humidity(province, current_date.month)
                        weather = get_weather_condition(precipitation)
                        cpi = get_cpi_index(current_date)
                        mobility = get_mobility_index(holiday_flag, promotion_flag)

                        # Store ID and SKU
                        store_id = f"S{province[:3].upper()}{city[:3].upper()}"
                        sku = f"{category}_{product.lower().replace(' ', '_')}"

                        data.append({
                            'date': current_date.strftime('%Y-%m-%d'),
                            'store_id': store_id,
                            'city': city,
                            'province': province,
                            'sku': sku,
                            'category': category,
                            'product_name': product,
                            'units_sold': round(demand, 2),
                            'supply_units': round(supply, 2),
                            'price_idr': round(price, 2),
                            'revenue_idr': round(revenue, 2),
                            'promotion_flag': int(promotion_flag),
                            'holiday_flag': int(holiday_flag),
                            'day_of_week': current_date.weekday(),
                            'month': current_date.month,
                            'year': current_date.year,
                            'temp_c': round(temp, 1),
                            'precipitation_mm': round(precipitation, 1),
                            'humidity_pct': round(humidity, 1),
                            'weather_condition': weather,
                            'cpi_index': round(cpi, 2),
                            'mobility_index': round(mobility, 3)
                        })

                        current_date += timedelta(days=1)
                        record_count += 1

                        if record_count % 10000 == 0:
                            print(f"📊 Generated {record_count:,} records...")

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    output_file = 'indonesia_comprehensive_market_2020_2024.csv'
    df.to_csv(output_file, index=False)

    print("\n✅ Dataset creation completed!")
    print(f"📁 File saved: {output_file}")
    print(f"📊 Total records: {len(df):,}")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"🌍 Provinces: {df['province'].nunique()}")
    print(f"🏙️ Cities: {df['city'].nunique()}")
    print(f"🥭 Products: {df['product_name'].nunique()}")
    print(f"📦 Categories: {df['category'].nunique()}")

    return df

def get_base_demand(category, product, province, date):
    """Get base demand for product based on category and region"""
    base_demands = {
        'fruit': 25,
        'vegetable': 30,
        'meat': 15,
        'dairy': 20,
        'staple': 40
    }

    base = base_demands.get(category, 20)

    # Regional adjustments
    if province in ['DKI Jakarta', 'West Java', 'East Java']:
        base *= 1.5  # Higher demand in Java
    elif province in ['Bali', 'DI Yogyakarta']:
        base *= 1.2  # Tourist areas
    elif province in ['Papua', 'West Papua', 'Maluku', 'North Maluku']:
        base *= 0.7  # Remote areas

    return base

def get_seasonal_factor(category, month):
    """Get seasonal factor for different categories"""
    seasonal_patterns = {
        'fruit': {
            1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.8, 6: 0.9,
            7: 1.0, 8: 1.1, 9: 1.2, 10: 1.3, 11: 1.4, 12: 1.5
        },
        'vegetable': {
            1: 1.1, 2: 1.0, 3: 0.9, 4: 0.8, 5: 0.9, 6: 1.0,
            7: 1.1, 8: 1.2, 9: 1.3, 10: 1.2, 11: 1.1, 12: 1.0
        },
        'meat': {
            1: 1.4, 2: 1.3, 3: 1.2, 4: 1.1, 5: 1.0, 6: 0.9,
            7: 0.9, 8: 1.0, 9: 1.1, 10: 1.2, 11: 1.4, 12: 1.5
        },
        'dairy': {
            1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.9, 6: 1.0,
            7: 1.1, 8: 1.2, 9: 1.1, 10: 1.0, 11: 1.1, 12: 1.2
        },
        'staple': {
            1: 1.3, 2: 1.2, 3: 1.1, 4: 1.0, 5: 0.9, 6: 0.9,
            7: 0.9, 8: 1.0, 9: 1.1, 10: 1.2, 11: 1.3, 12: 1.4
        }
    }

    return seasonal_patterns.get(category, {}).get(month, 1.0)

def get_regional_factor(province, category):
    """Get regional demand factor"""
    # Urban vs rural factors
    urban_provinces = ['DKI Jakarta', 'West Java', 'East Java', 'Central Java', 'DI Yogyakarta', 'Banten', 'Bali']
    if province in urban_provinces:
        return 1.3

    # Island factors
    island_provinces = ['Bali', 'West Nusa Tenggara', 'East Nusa Tenggara', 'Bangka Belitung Islands', 'Riau Islands']
    if province in island_provinces:
        return 0.9

    return 1.0

def get_day_of_week_factor(dow):
    """Get day of week demand factor (0=Monday, 6=Sunday)"""
    dow_factors = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]  # Weekend higher
    return dow_factors[dow]

def get_base_price(category, product):
    """Get base price for different products"""
    price_ranges = {
        'fruit': (15000, 50000),
        'vegetable': (10000, 30000),
        'meat': (80000, 150000),
        'dairy': (25000, 60000),
        'staple': (15000, 40000)
    }

    min_price, max_price = price_ranges.get(category, (10000, 30000))
    return random.uniform(min_price, max_price)

def is_holiday(date):
    """Check if date is a holiday"""
    # Simplified holiday check
    holidays = [
        (1, 1),   # New Year
        (5, 1),   # Labor Day
        (8, 17),  # Independence Day
        (12, 25), # Christmas
        (12, 31), # New Year's Eve
    ]

    return (date.month, date.day) in holidays

def get_temperature(province, month):
    """Get average temperature for province and month"""
    base_temp = 28  # Indonesia average

    # Regional adjustments
    if province in ['DI Yogyakarta', 'Central Java', 'East Java']:
        base_temp -= 2
    elif province in ['West Sumatra', 'North Sumatra']:
        base_temp -= 1
    elif province in ['Papua', 'West Papua']:
        base_temp += 1

    # Seasonal adjustments
    seasonal_temp = [26, 26, 27, 28, 28, 27, 26, 26, 27, 28, 28, 27]
    return base_temp + seasonal_temp[month - 1] - 27

def get_precipitation(province, month):
    """Get average precipitation"""
    # Wet season (Oct-Mar), dry season (Apr-Sep)
    if month in [10, 11, 12, 1, 2, 3]:
        return random.uniform(200, 400)
    else:
        return random.uniform(50, 150)

def get_humidity(province, month):
    """Get average humidity"""
    return random.uniform(70, 90)

def get_weather_condition(precipitation):
    """Get weather condition based on precipitation"""
    if precipitation > 100:
        return random.choice(['rainy', 'cloudy'])
    else:
        return random.choice(['sunny', 'cloudy'])

def get_cpi_index(date):
    """Get CPI index (simplified)"""
    base_cpi = 100
    years_since_2020 = date.year - 2020
    return base_cpi + (years_since_2020 * 2) + random.uniform(-1, 1)

def get_mobility_index(holiday, promotion):
    """Get mobility index"""
    base_mobility = 0.8

    if holiday:
        base_mobility *= 0.7  # Less mobility on holidays
    if promotion:
        base_mobility *= 1.2  # More mobility with promotions

    return min(1.0, max(0.1, base_mobility + random.uniform(-0.1, 0.1)))

if __name__ == '__main__':
    print("🚀 Creating Comprehensive Indonesia Market Dataset")
    print("=" * 60)

    df = create_comprehensive_indonesia_dataset()

    print("\n📊 Dataset Summary:")
    print(f"Total Records: {len(df):,}")
    print(f"Date Range: {df['date'].min()} - {df['date'].max()}")
    print(f"Provinces: {df['province'].nunique()}")
    print(f"Cities: {df['city'].nunique()}")
    print(f"Products: {df['product_name'].nunique()}")
    print(f"Categories: {df['category'].nunique()}")

    print("\n💰 Sample Price Ranges:")
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        print(".0f")

    print("\n📈 Sample Demand by Category:")
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        print(".1f")

    print("\n✅ Dataset ready for AI training!")
    print("📁 File: indonesia_comprehensive_market_2020_2024.csv")