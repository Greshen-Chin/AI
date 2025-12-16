# 🥭 AOL AI - Prediksi Pasar Indonesia Lengkap

Sistem AI canggih untuk membantu supermarket, UMKM, dan pedagang kecil menghindari kerugian dan mengoptimalkan keuntungan di Indonesia.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
py -m pip install pandas scikit-learn prophet flask flask-cors joblib requests pillow
```

### 2. Buat Dataset Lengkap (Opsional)
```bash
py create_comprehensive_dataset.py
```

### 3. Train AI Models
```bash
py train_indonesia_model.py
```

### 4. Start Server
```bash
# Start API prediksi pasar
py api_indonesia.py

# Start API deteksi kualitas (di terminal baru)
py simple_quality_api.py
```

### 5. Buka Website
Buka `index.html` di browser untuk menggunakan semua fitur:
- **Prediksi Pasar**: Scroll ke "Prediksi Pasar Indonesia" → pilih provinsi & periode → klik "Prediksi Pasar"
- **Deteksi Kualitas**: Scroll ke "Deteksi Kualitas" → upload gambar buah → klik "Analisis Kualitas Buah"

## 🎯 Fitur Unggulan

### 📊 **Prediksi Pasar Indonesia**
- **34 Provinsi Lengkap**: Semua wilayah Indonesia tercover
- **100+ Produk**: Dari sembako sampai buah-buahan
- **Prediksi Jangka Waktu**: 1 hari sampai 6 bulan
- **Ranking Produk**: Item paling dicari vs kurang laku
- **Rekomendasi Bisnis**: Strategi khusus untuk UMKM & supermarket

### 🥭 **AI Deteksi Kualitas Buah**
- **Upload Gambar**: Klik atau drag & drop gambar buah
- **Analisis Otomatis**: AI mendeteksi kualitas secara real-time
- **Status Kualitas**: FRESH, GOOD, FAIR, POOR, SPOILED
- **Rekomendasi**: Saran jual/buang berdasarkan kondisi
- **Simple & Cepat**: Hasil dalam hitungan detik

### 🛡️ **Pencegahan Kerugian**
- **Stok Berlebih**: Hindari overstock produk kurang laku
- **Kadaluarsa**: Waktu simpan optimal per produk
- **Quality Control**: Deteksi buah busuk sebelum dijual
- **Diversifikasi**: Minimal 60% produk unggulan

### 💰 **Optimasi Keuntungan**
- **Margin Optimal**: 25-35% untuk produk terlaris
- **Rotasi Cepat**: Target 3-5 hari untuk produk unggulan
- **Promosi Strategis**: Timing promosi yang tepat
- **Target Penjualan**: Prediksi penjualan harian/mingguan

### 📈 **Analisis Tren Musiman**
- **Musim Panas (Apr-Jun)**: Buah naik 25%, sayur turun 15%
- **Musim Hujan (Oct-Dec)**: Sembako naik 30%, buah turun 20%
- **Liburan (Dec-Jan)**: Daging & susu naik 40%

## 🌍 Semua 34 Provinsi Indonesia

### **Pulau Jawa**
- DKI Jakarta, Jawa Barat, Jawa Timur, Jawa Tengah
- DI Yogyakarta, Banten

### **Pulau Sumatera**
- Aceh, Sumatera Utara, Sumatera Barat, Riau, Jambi
- Sumatera Selatan, Bengkulu, Lampung
- Kepulauan Bangka Belitung, Kepulauan Riau

### **Pulau Kalimantan**
- Kalimantan Barat, Kalimantan Tengah, Kalimantan Selatan
- Kalimantan Timur, Kalimantan Utara

### **Pulau Sulawesi**
- Sulawesi Utara, Sulawesi Tengah, Sulawesi Selatan
- Sulawesi Tenggara, Gorontalo, Sulawesi Barat

### **Kepulauan**
- Bali, Nusa Tenggara Barat, Nusa Tenggara Timur

### **Maluku & Papua**
- Maluku, Maluku Utara, Papua Barat, Papua

## 🥭 100+ Produk Lengkap

### **Buah-buahan (28 jenis)**
- Pisang (Cavendish, Ambon), Apel (Fuji, Granny Smith)
- Jeruk (Sunkist, Mandarin), Mangga (Harum Manis, Gedong)
- Nanas (Madu, Queen), Salak (Bali, Pondoh)
- Rambutan (Binjai, Simacan), Durian (Musang King, Monthong)
- Alpukat (Mentega, Miki), Melon (Cantaloupe, Honey Dew)
- Semangka, Pepaya, Manggis, Sirsak, Jambu Biji, Belimbing

### **Sayuran (31 jenis)**
- Bayam, Kangkung, Sawi, Kol, Brokoli, Wortel
- Kentang, Bawang Merah/Putih, Cabe berbagai jenis
- Tomat, Terong, Labu, Timun, Kacang Panjang, Buncis
- Jagung Manis, Jagung Pipil

### **Daging & Seafood (19 jenis)**
- Ayam (Kampung, Broiler, Negeri)
- Daging Sapi (Has Dalam, Sandung Lamur, Tetelan)
- Daging Kambing & Domba
- Ikan (Tuna, Tongkol, Kakap, Gurame, Nila, Lele)
- Udang (Vaname, Windu), Cumi-Cumi, Kepiting

### **Susu & Olahan (14 jenis)**
- Susu Sapi (Full Cream, Low Fat), Susu Kambing
- Keju (Cheddar, Mozzarella, Parmesan)
- Yogurt berbagai rasa, Mentega, Margarin
- Es Krim berbagai rasa

### **Sembako (20 jenis)**
- Beras (Premium, Medium, Pandan Wangi)
- Mie (Instans Goreng/Kuah, Telur, Kuning)
- Minyak Goreng, Gula, Tepung (Terigu, Beras, Tapioka)
- Kecap, Saos, Terasi

## 📸 **AI Deteksi Kualitas Buah & Makanan**

### **Fitur Deteksi Kualitas**
- **Upload Gambar**: Klik atau drag & drop gambar buah/makanan
- **Analisis Otomatis**: AI menganalisis warna, tekstur, dan kondisi
- **Status Kualitas**: FRESH, GOOD, FAIR, POOR, SPOILED
- **Rekomendasi**: Saran untuk jual/buang berdasarkan kondisi
- **Simple & Cepat**: Hasil dalam hitungan detik

### **Cara Menggunakan**
1. **Buka Website**: Buka `index.html` di browser
2. **Scroll ke Deteksi Kualitas**: Klik menu "Deteksi Kualitas" atau scroll ke bagian bawah
3. **Upload Gambar**: Klik area upload atau drag & drop gambar buah
4. **Analisis**: Klik "Analisis Kualitas Buah"
5. **Lihat Hasil**: Status kualitas dan rekomendasi bisnis

### **Teknologi**
- **Computer Vision**: Analisis gambar dengan algoritma AI
- **Color Analysis**: Deteksi perubahan warna yang tidak normal
- **Texture Analysis**: Analisis tekstur permukaan
- **Pattern Recognition**: Deteksi pola pembusukan

### **Kegunaan untuk Pedagang**
- **Cek Stok**: Pastikan buah masih layak jual sebelum dijual
- **Quality Control**: Deteksi buah busuk secara cepat
- **Inventory Management**: Kelola stok berdasarkan kondisi
- **Customer Satisfaction**: Jamin kualitas produk yang dijual

## 📁 File Lengkap

```
├── index.html                           # Website utama dengan semua fitur terintegrasi
├── src/main.js                          # JavaScript dengan prediksi pasar & deteksi kualitas
├── src/style.css                        # Styling modern dan responsive
├── create_comprehensive_dataset.py      # Generator dataset lengkap
├── train_indonesia_model.py             # Training AI models untuk prediksi
├── api_indonesia.py                     # Flask API server utama (port 5000)
├── simple_quality_api.py                # API deteksi kualitas buah (port 5001)
├── quick_start.py                       # Setup cepat untuk pemula
├── test_indonesia_system.py             # Comprehensive testing
├── indonesia_supermarket_5yr_synthetic.csv     # Dataset original
├── indonesia_comprehensive_market_2020_2024.csv # Dataset lengkap (opsional)
└── models/                              # AI models terlatih
    ├── demand_model.joblib
    ├── supply_model.joblib
    ├── province_encoder.joblib
    ├── category_encoder.joblib
    └── market_insights.json
```

## 🎯 Strategi Bisnis Berdasarkan Tipe Usaha

### **🏪 Supermarket Modern**
- **Fokus**: Volume besar dengan margin tipis
- **Strategi**: Private label, delivery service, membership
- **Prediksi**: 1-3 bulan untuk perencanaan inventory besar

### **🏪 UMKM Kecil**
- **Fokus**: Rotasi cepat, modal optimal
- **Strategi**: Jual grosir ke warung, pre-order system
- **Prediksi**: 3 hari - 1 minggu untuk perencanaan harian

### **🏪 Toko Retail**
- **Fokus**: Demografi pelanggan lokal
- **Strategi**: Survey pelanggan, lokasi strategis
- **Prediksi**: 1 minggu untuk perencanaan stok

### **🚛 Grosir**
- **Fokus**: Volume besar, margin minimal
- **Strategi**: Bulk discount, warehouse optimization
- **Prediksi**: 1-3 bulan untuk perencanaan supply chain

## 🔧 Troubleshooting Lengkap

### Dataset Tidak Ada
```bash
# Buat dataset lengkap
py create_comprehensive_dataset.py
```

### Model Training Gagal
```bash
# Pastikan dataset ada
dir indonesia_supermarket_5yr_synthetic.csv

# Train ulang
py train_indonesia_model.py
```

### API Error
```bash
# Cek port 5000 (prediksi pasar)
netstat -an | find "5000"

# Restart server prediksi
taskkill /PID <PID> /F
py api_indonesia.py
```

### Deteksi Kualitas Tidak Bisa Upload
```bash
# Cek port 5001 (deteksi kualitas)
netstat -an | find "5001"

# Start API deteksi kualitas
py simple_quality_api.py

# Buka website utama
index.html

# Scroll ke bagian "Deteksi Kualitas"
```

### Prediksi Tidak Muncul
- Pastikan memilih provinsi yang ada di dataset
- Coba periode yang lebih pendek (1 hari dulu)
- Refresh browser dan coba lagi

## 📊 Data Insights

### **Provinsi dengan Demand Tertinggi**
- DKI Jakarta: 1.5x rata-rata nasional
- Jawa Barat & Jawa Timur: 1.3x rata-rata
- Bali & Yogyakarta: 1.2x (wisatawan)

### **Provinsi dengan Demand Terendah**
- Papua & Papua Barat: 0.7x rata-rata
- Maluku & Maluku Utara: 0.8x rata-rata
- Kalimantan Utara: 0.9x rata-rata

### **Kategori Terlaris**
1. **Sembako**: 40 unit/hari rata-rata
2. **Sayuran**: 30 unit/hari rata-rata
3. **Buah-buahan**: 25 unit/hari rata-rata
4. **Susu & Olahan**: 20 unit/hari rata-rata
5. **Daging**: 15 unit/hari rata-rata

## 💡 Tips Sukses Menggunakan Sistem

1. **Mulai Kecil**: Coba 1 provinsi dulu, lalu expand
2. **Monitor Tren**: Cek prediksi mingguan secara rutin
3. **Quality Control**: Gunakan deteksi kualitas untuk cek stok
4. **Sesuaikan Strategi**: Bedakan taktik supermarket vs UMKM
5. **Update Data**: Model belajar dari pola terbaru setiap hari
6. **Kombinasikan**: Gunakan dengan data penjualan aktual Anda

## 📞 Support & Development

Sistem ini terus dikembangkan untuk membantu UMKM Indonesia:
- ✅ **34 Provinsi Lengkap**: Semua daerah terjangkau
- ✅ **100+ Produk**: Katalog lengkap Indonesia
- ✅ **AI Prediksi Pasar**: Forecasting akurat berdasarkan 5 tahun data
- ✅ **AI Deteksi Kualitas**: Analisis gambar buah secara real-time
- ✅ **Anti-Rugi**: Strategi pencegahan kerugian
- ✅ **Optimasi Profit**: Rekomendasi peningkatan keuntungan
- ✅ **User-Friendly**: Mudah digunakan tanpa teknis
- ✅ **Integrated System**: Semua fitur dalam satu platform

---

**🚀 Sistem AI terlengkap untuk bisnis di Indonesia!**
**🥭 Dari pedagang kaki lima sampai supermarket modern!**
**📸 AI deteksi kualitas + Prediksi pasar dalam satu aplikasi!**
**💰 Maksimalkan keuntungan, minimalkan kerugian!** 📈