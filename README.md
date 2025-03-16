# Analisis Data Kecelakaan Kendaraan di New York City (NYC)

## **Deskripsi Proyek**

Proyek ini merupakan implementasi tugas Mata Kuliah *Data, Informasi, dan Pengetahuan* (Semester 6, Program Studi Informatika) yang bertujuan untuk menerapkan teknik *Knowledge Discovery in Databases* (KDD) pada dataset dunia nyata. Fokus proyek ini adalah menganalisis dataset kecelakaan kendaraan di New York City (NYC) yang diperoleh dari Kaggle, dengan tujuan mengekstrak *insight* yang dapat digunakan oleh Pemerintah NYC untuk meningkatkan keselamatan lalu lintas. Proyek ini mencakup tahapan *preprocessing* data, analisis multidimensi, visualisasi data, pembuatan dashboard interaktif, dan penyusunan laporan bisnis.

Dataset yang digunakan berisi catatan kecelakaan kendaraan dari tahun 2015 hingga 2017, meliputi informasi seperti lokasi, jenis kendaraan, faktor penyebab, jumlah korban, dan waktu kejadian. Proyek ini dirancang untuk memastikan kualitas data melalui berbagai teknik *preprocessing* dan menyajikan hasil analisis dalam bentuk visualisasi yang informatif serta *insight* yang actionable.

---

## **Tujuan Proyek**

1. **Penerapan Teknik KDD**: Menerapkan tahapan KDD (seleksi data, *preprocessing*, transformasi, *data mining*, evaluasi) untuk mengekstrak pengetahuan dari dataset kecelakaan kendaraan NYC.
2. **Peningkatan Kualitas Data**: Melakukan *preprocessing* untuk menangani *missing values*, duplikat, inkonsistensi, dan *outlier*, sehingga data siap untuk analisis.
3. **Analisis Multidimensi**: Menganalisis data dari berbagai perspektif, seperti distribusi wilayah, jenis kendaraan, faktor penyebab, tren temporal, dan distribusi korban.
4. **Visualisasi Data**: Menyajikan hasil analisis dalam bentuk grafik interaktif menggunakan Plotly dan dashboard multidimensi.
5. **Pengambilan *Insight* Bisnis**: Menyediakan rekomendasi berbasis data untuk Pemerintah NYC guna meningkatkan kebijakan keselamatan lalu lintas.

---

## **Struktur Direktori**

Berikut adalah struktur direktori proyek yang dihasilkan:

```
vehicle_collision_analysis_20250316_151642/
├── vehicle_collisions_processed_20250316_151642.csv  # Dataset yang telah diproses
├── multidimensional_dashboard_20250316_151642.html   # Dashboard multidimensi
├── temporal_dashboard_20250316_151642.html          # Dashboard temporal
├── victims_dashboard_20250316_151642.html           # Dashboard korban
├── missing_values_comparison.png                    # Visualisasi perbandingan missing values
├── total_victims_distribution.png                   # Visualisasi distribusi total korban
├── borough_distribution.png                         # Visualisasi distribusi per borough
├── top_10_vehicle_types.png                         # Visualisasi top 10 jenis kendaraan
├── top_10_vehicle_factors.png                       # Visualisasi top 10 faktor penyebab
├── accident_locations.png                           # Visualisasi lokasi kecelakaan
├── correlation_heatmap.png                          # Visualisasi heatmap korelasi
├── README.md                                        # Dokumen ini
└── laporan_tugas_2.md                               # Laporan lengkap tugas 2
```

Semua file dihasilkan pada tanggal **16 Maret 2025** dengan timestamp **15:16:42** berdasarkan waktu eksekusi kode.

---

## **Persyaratan Sistem**

Untuk menjalankan dan mereplikasi proyek ini, pastikan memiliki environment sebagai berikut:

### **Perangkat Lunak yang Diperlukan**
- **Python 3.7 atau lebih baru**: Bahasa pemrograman utama untuk analisis data.
- **Pustaka Python**:
  - `pandas`: Untuk manipulasi dan analisis data.
  - `numpy`: Untuk operasi numerik.
  - `scikit-learn`: Untuk normalisasi, standarisasi, dan deteksi *outlier*.
  - `matplotlib` dan `seaborn`: Untuk visualisasi statis.
  - `plotly`: Untuk visualisasi interaktif dan dashboard.
  - `plotly.io`: Untuk ekspor visualisasi.
  - `kagglehub`: Untuk mengunduh dataset dari Kaggle.
  - `os` dan `sys`: Untuk operasi sistem.
  - `datetime`: Untuk manipulasi waktu.

### **Instalasi Dependensi**
Jalankan perintah berikut untuk menginstal pustaka yang diperlukan:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly plotly.io kagglehub
```

**Catatan Penting**:
- Ekspor dashboard ke format PNG memerlukan paket tambahan `kaleido`. Jika ingin mendukung ekspor PNG, instal dengan perintah:
  ```bash
  pip install -U kaleido
  ```
  Tanpa `kaleido`, ekspor dashboard hanya akan menghasilkan file HTML.

### **Akses Kaggle**
- Pastikan sudah memiliki akun Kaggle. Unduh dataset dari:
  [NYPD Vehicle Collisions](https://www.kaggle.com/datasets/nypd/vehicle-collisions).

---

## **Cara Menjalankan Proyek**

1. **Kloning atau Unduh Repositori**
   Unduh direktori proyek dari lingkungan tempat kode dijalankan (misalnya, Google Colab atau lokal) atau salin kode dari file utama (lihat bagian "Kode Utama").

2. **Persiapan Lingkungan**
   Pastikan semua dependensi telah diinstal sesuai bagian "Persyaratan Sistem".

3. **Eksekusi Kode**
   Jalankan skrip Python utama dengan perintah:
   ```bash
   python main.py
   ```

4. **Hasil yang Dihasilkan**
   - Dataset yang telah diproses akan disimpan sebagai `vehicle_collisions_processed_20250316_151642.csv`.
   - Visualisasi statis akan disimpan sebagai file PNG di direktori proyek.
   - Dashboard interaktif akan disimpan sebagai file HTML.

5. **Melihat Dashboard**
   Buka file HTML (misalnya, `multidimensional_dashboard_20250316_151642.html`) di browser web untuk melihat visualisasi interaktif.

---

## **Kode Utama**

Berikut adalah preview kode utama yang digunakan untuk analisis (disederhanakan untuk keperluan preview saja). Kode lengkap tersedia dalam file proyek ini:

```python
import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import os
from datetime import datetime

# Unduh dataset dari Kaggle
path = kagglehub.dataset_download("nypd/vehicle-collisions")
df = pd.read_csv(os.path.join(path, "database.csv"), parse_dates=["DATE"])

# Preprocessing
df.dropna(subset=["DATE"], inplace=True)  # Menghapus NaT di DATE
df["BOROUGH"].fillna("UNKNOWN", inplace=True)
df["LATITUDE"].fillna(df["LATITUDE"].mean(), inplace=True)
# ... (lanjutkan untuk kolom lainnya)

# Hapus duplikat
df.drop_duplicates(inplace=True)

# Normalisasi dan Standarisasi
scaler = MinMaxScaler()
df["Total Victims"] = df["PERSONS INJURED"] + df["PERSONS KILLED"]
df["Total Victims Normalized"] = scaler.fit_transform(df[["Total Victims"]])

# Deteksi Outlier
z_score_scaler = StandardScaler()
z_scores = np.abs(z_score_scaler.fit_transform(df[["Total Victims"]]))
df_no_outliers = df[z_scores <= 3].copy()

# Visualisasi
plt.figure()
sns.histplot(df["Total Victims"])
plt.savefig("total_victims_distribution.png")

# Dashboard
fig = px.pie(df, names="BOROUGH", title="Distribusi Kecelakaan per Borough")
pio.write_html(fig, "borough_distribution.html")

# Simpan dataset
df_no_outliers.to_csv("vehicle_collisions_processed.csv", index=False)
```

**Catatan**: Kode di atas adalah contoh singkat. Kode lengkap mencakup lebih banyak langkah, seperti deteksi *outlier* pada `LATITUDE` dan `LONGITUDE`, pembuatan dashboard temporal, dan analisis korelasi.

---

## **Metodologi**

### **Tahapan KDD**
1. **Seleksi Data**: Dataset "NYPD Vehicle Collisions" dipilih dari Kaggle.
2. **Preprocessing**:
   - **Penanganan *Missing Values***: Imputasi dengan rata-rata (numerik) dan "UNKNOWN"/"NONE" (kategorikal).
   - **Penghapusan Duplikat**: Menghapus 63.644 duplikat (13,32% dari data awal).
   - **Normalisasi dan Standarisasi**: Menggunakan Min-Max Scaling dan Z-score pada `Total Victims`.
   - **Pengecekan Konsistensi**: Menyeragamkan teks menjadi huruf kapital dan menghapus spasi berlebih.
   - **Deteksi *Outlier***: Menghapus 5.827 *outlier* menggunakan Z-score (*threshold* 3).
3. **Transformasi**: Mengelompokkan data berdasarkan waktu, wilayah, dan kategori.
4. **Data Mining**: Analisis distribusi, tren, dan korelasi.
5. **Evaluasi**: Menyusun *insight* bisnis berdasarkan hasil analisis.

### **Alat dan Teknik**
- **Bahasa Pemrograman**: Python.
- **Pustaka**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly.
- **Metode Analisis**: Statistik deskriptif, Z-score, korelasi Pearson.

---

## **Hasil Analisis**

### **Ringkasan Data**
- **Baris Awal**: 477.732.
- **Baris Setelah Preprocessing**: 408.261.
- **Missing Values Ditangani**: 4.254.154 nilai.
- **Duplikat Dihapus**: 63.644 baris.
- **Outlier Dihapus**: 5.827 baris.

### **Temuan Utama**
1. **Distribusi Wilayah**:
   - Brooklyn (23,35%) dan Manhattan (18,88%) merupakan wilayah dengan kecelakaan tertinggi.
2. **Jenis Kendaraan**:
   - Kendaraan penumpang (60,36%) dan SUV (22,13%) paling sering terlibat.
3. **Faktor Penyebab**:
   - "Driver Inattention/Distraction" (14,87%) dan "Unspecified" (48,49%) adalah faktor utama.
4. **Tren Temporal**:
   - Puncak kecelakaan terjadi pada jam 14:00-17:00 dan hari Jumat.
5. **Distribusi Korban**:
   - Pengemudi (54.722 korban) paling banyak terdampak, diikuti pejalan kaki (20.575 korban).
6. **Lokasi**:
   - Kecelakaan terkonsentrasi di pusat kota (Manhattan, Brooklyn, Queens).

### **Visualisasi**
- **Statis**: Histogram, *pie chart*, *bar plot*, *scatter plot*, dan *heatmap* disimpan sebagai file PNG.
- **Interaktif**: Tiga dashboard HTML mencakup analisis multidimensi, temporal, dan korban.

---

## **Insight Bisnis untuk Pemerintah NYC**

1. **Fokus pada Faktor Penyebab**: Tingkatkan kampanye kesadaran tentang "Driver Inattention/Distraction" dan perbaiki pelaporan faktor penyebab.
2. **Waktu Rawan**: Tingkatkan patroli pada jam 14:00-17:00 dan hari Jumat.
3. **Lokasi Rawan**: Evaluasi infrastruktur di Brooklyn dan Manhattan, tambah lampu lalu lintas atau kamera pengawas.
4. **Jenis Kendaraan**: Regulasikan pengemudi kendaraan penumpang dan SUV dengan pelatihan keselamatan.
5. **Perlindungan Korban**: Tingkatkan fasilitas untuk pejalan kaki dan pengendara sepeda di wilayah padat.

---

## **Lisensi**

Proyek ini bersifat open-source dan dapat digunakan untuk tujuan pendidikan. Dataset asli berasal dari Kaggle dan patuh pada ketentuan lisensi Kaggle. Visualisasi dan kode kustom di bawah lisensi MIT.

---

## **Catatan Tambahan**

- Pastikan koneksi internet aktif saat mengunduh dataset dari Kaggle.
- Jika ada error terkait `kaleido`, instal paket tersebut untuk mendukung ekspor PNG.
- Laporan lengkap tersedia di file `reports.pdf` untuk referensi lebih lanjut.
---
