import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. LOAD DATA RAW (Total 768 Data)
# Menggunakan link raw dari repositori publik agar mudah diakses
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']

# Membaca data
df_raw = pd.read_csv(url, names=column_names)

print(f"Total Data Raw: {len(df_raw)} baris (Memenuhi syarat > 100)")
print("-" * 50)

# 2. MENGAMBIL 30 DATA ACAK (Untuk Laporan/Tugas)
# Ini adalah bagian "ambil acak 20 sd 30 data" yang diminta dosen untuk ditampilkan
df_sample_display = df_raw.sample(n=30, random_state=42)

print("Berikut adalah 30 Data Acak untuk Referensi Laporan:")
print(df_sample_display)
print("-" * 50)

# 3. PROSES RANDOM FOREST (ML)
# Pisahkan Fitur (X) dan Target (Y)
X = df_raw.drop('Outcome', axis=1)
y = df_raw['Outcome']

# Bagi data menjadi Training dan Testing (80% latih, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi Model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Latih Model
rf_model.fit(X_train, y_train)

# Prediksi
y_pred = rf_model.predict(X_test)

# 4. HASIL EVALUASI
print("Hasil Akurasi Model Random Forest:", accuracy_score(y_test, y_pred))
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))

# Fitur Penting (Insight tambahan untuk analisis)
feature_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFaktor Paling Mempengaruhi Diabetes (Feature Importance):")
print(feature_imp.head(3))