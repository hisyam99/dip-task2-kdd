import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import sys
from datetime import datetime

sns.set(style="whitegrid")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"vehicle_collision_analysis_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"analysis_report_{timestamp}.txt")

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

f = open(output_file, 'w')
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, f)

def save_plot(fig, filename):
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches='tight')
    print(f"Plot disimpan sebagai: {filepath}")

def save_dashboard(fig, filename_base):
    html_path = os.path.join(output_dir, f"{filename_base}_{timestamp}.html")
    png_path = os.path.join(output_dir, f"{filename_base}_{timestamp}.png")
    fig.write_html(html_path)
    print(f"Dashboard disimpan sebagai HTML: {html_path}")
    try:
        pio.write_image(fig, png_path, format='png', width=1200, height=800)
        print(f"Dashboard disimpan sebagai PNG: {png_path}")
    except ValueError as e:
        print(f"Gagal menyimpan dashboard sebagai PNG: {str(e)}")
        print("Silakan instal kaleido dengan 'pip install -U kaleido' untuk mendukung ekspor PNG.")

print("Mengunduh dataset dari Kaggle...")
path = kagglehub.dataset_download("nypd/vehicle-collisions")
print(f"Path ke file dataset: {path}")

csv_file = None
for file in os.listdir(path):
    if file.endswith('.csv'):
        csv_file = os.path.join(path, file)
        break

if csv_file is None:
    raise FileNotFoundError("File CSV tidak ditemukan di direktori yang diunduh!")
print(f"File CSV yang digunakan: {csv_file}")

print("\nMemuat dataset...")
df = pd.read_csv(csv_file)
print("Mengonversi DATE ke format datetime saat memuat...")
df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y', errors='coerce')
print(f"Tipe data DATE setelah konversi awal: {df['DATE'].dtype}")
print("Contoh 5 baris pertama DATE setelah konversi awal:")
print(df['DATE'].head().to_string())
print(f"Jumlah nilai NaT di DATE: {df['DATE'].isna().sum()}")
print("Dataset awal:")
df.info(verbose=True, memory_usage=True, show_counts=True)
print("\nKolom yang tersedia:")
print(df.columns.tolist())
print("\nStatistik deskriptif awal:")
print(df.describe(include='all').to_string())
print("\n")

df_original = df.copy()

print("Menangani missing values...")
missing_before = df.isnull().sum()
print("Jumlah missing values sebelum preprocessing:")
print(missing_before.to_string())
print("\nPersentase missing values sebelum preprocessing:")
print((missing_before / len(df) * 100).to_string())

df.loc[:, 'BOROUGH'] = df['BOROUGH'].fillna('Unknown')
df.loc[:, 'ZIP CODE'] = df['ZIP CODE'].fillna(df['ZIP CODE'].mode()[0])
df.loc[:, 'ON STREET NAME'] = df['ON STREET NAME'].fillna('Unknown')
df.loc[:, 'CROSS STREET NAME'] = df['CROSS STREET NAME'].fillna('Unknown')
df.loc[:, 'LATITUDE'] = df['LATITUDE'].fillna(df['LATITUDE'].mean())
df.loc[:, 'LONGITUDE'] = df['LONGITUDE'].fillna(df['LONGITUDE'].mean())
df.loc[:, 'LOCATION'] = df['LOCATION'].fillna(df['LATITUDE'].astype(str) + ', ' + df['LONGITUDE'].astype(str))
df.loc[:, 'OFF STREET NAME'] = df['OFF STREET NAME'].fillna('Unknown')
df.loc[:, 'PERSONS INJURED'] = df['PERSONS INJURED'].fillna(0)
df.loc[:, 'PERSONS KILLED'] = df['PERSONS KILLED'].fillna(0)
df.loc[:, 'PEDESTRIANS INJURED'] = df['PEDESTRIANS INJURED'].fillna(0)
df.loc[:, 'PEDESTRIANS KILLED'] = df['PEDESTRIANS KILLED'].fillna(0)
df.loc[:, 'CYCLISTS INJURED'] = df['CYCLISTS INJURED'].fillna(0)
df.loc[:, 'CYCLISTS KILLED'] = df['CYCLISTS KILLED'].fillna(0)
df.loc[:, 'MOTORISTS INJURED'] = df['MOTORISTS INJURED'].fillna(0)
df.loc[:, 'MOTORISTS KILLED'] = df['MOTORISTS KILLED'].fillna(0)
df.loc[:, 'VEHICLE 1 TYPE'] = df['VEHICLE 1 TYPE'].fillna('Unknown')
df.loc[:, 'VEHICLE 2 TYPE'] = df['VEHICLE 2 TYPE'].fillna('None')
df.loc[:, 'VEHICLE 3 TYPE'] = df['VEHICLE 3 TYPE'].fillna('None')
df.loc[:, 'VEHICLE 4 TYPE'] = df['VEHICLE 4 TYPE'].fillna('None')
df.loc[:, 'VEHICLE 5 TYPE'] = df['VEHICLE 5 TYPE'].fillna('None')
df.loc[:, 'VEHICLE 1 FACTOR'] = df['VEHICLE 1 FACTOR'].fillna('Unspecified')
df.loc[:, 'VEHICLE 2 FACTOR'] = df['VEHICLE 2 FACTOR'].fillna('Unspecified')
df.loc[:, 'VEHICLE 3 FACTOR'] = df['VEHICLE 3 FACTOR'].fillna('Unspecified')
df.loc[:, 'VEHICLE 4 FACTOR'] = df['VEHICLE 4 FACTOR'].fillna('Unspecified')
df.loc[:, 'VEHICLE 5 FACTOR'] = df['VEHICLE 5 FACTOR'].fillna('Unspecified')

missing_after = df.isnull().sum()
print("\nJumlah missing values setelah imputasi:")
print(missing_after.to_string())
print("\nPersentase missing values setelah imputasi:")
print((missing_after / len(df) * 100).to_string())

print("\nMenghapus duplikat...")
duplicates_before = df.duplicated(subset=['DATE', 'TIME', 'LATITUDE', 'LONGITUDE']).sum()
print(f"Jumlah duplikat sebelum: {duplicates_before}")
print(f"Persentase duplikat sebelum: {duplicates_before / len(df) * 100:.2f}%")
df = df.drop_duplicates(subset=['DATE', 'TIME', 'LATITUDE', 'LONGITUDE'], keep='first').copy()
duplicates_after = df.duplicated(subset=['DATE', 'TIME', 'LATITUDE', 'LONGITUDE']).sum()
print(f"Jumlah duplikat setelah: {duplicates_after}")
print(f"Persentase duplikat setelah: {duplicates_after / len(df) * 100:.2f}%")

print("\nMelakukan normalisasi dan standarisasi...")
df.loc[:, 'Total Victims'] = df['PERSONS INJURED'] + df['PERSONS KILLED']
min_max_scaler = MinMaxScaler()
df.loc[:, 'Total Victims MinMax'] = min_max_scaler.fit_transform(df[['Total Victims']])
z_score_scaler = StandardScaler()
df.loc[:, 'Total Victims ZScore'] = z_score_scaler.fit_transform(df[['Total Victims']])
print("Statistik Total Victims (Original):")
print(df['Total Victims'].describe().to_string())
print("\nStatistik Total Victims MinMax:")
print(df['Total Victims MinMax'].describe().to_string())
print("\nStatistik Total Victims ZScore:")
print(df['Total Victims ZScore'].describe().to_string())

print("\nMemeriksa dan menyeragamkan konsistensi teks...")
df.loc[:, 'BOROUGH'] = df['BOROUGH'].str.upper()
df.loc[:, 'ON STREET NAME'] = df['ON STREET NAME'].str.upper()
df.loc[:, 'CROSS STREET NAME'] = df['CROSS STREET NAME'].str.upper()
df.loc[:, 'OFF STREET NAME'] = df['OFF STREET NAME'].str.upper()
df.loc[:, 'VEHICLE 1 TYPE'] = df['VEHICLE 1 TYPE'].str.upper()
df.loc[:, 'VEHICLE 2 TYPE'] = df['VEHICLE 2 TYPE'].str.upper()
df.loc[:, 'VEHICLE 3 TYPE'] = df['VEHICLE 3 TYPE'].str.upper()
df.loc[:, 'VEHICLE 4 TYPE'] = df['VEHICLE 4 TYPE'].str.upper()
df.loc[:, 'VEHICLE 5 TYPE'] = df['VEHICLE 5 TYPE'].str.upper()
df.loc[:, 'VEHICLE 1 FACTOR'] = df['VEHICLE 1 FACTOR'].str.upper()
df.loc[:, 'VEHICLE 2 FACTOR'] = df['VEHICLE 2 FACTOR'].str.upper()
df.loc[:, 'VEHICLE 3 FACTOR'] = df['VEHICLE 3 FACTOR'].str.upper()
df.loc[:, 'VEHICLE 4 FACTOR'] = df['VEHICLE 4 FACTOR'].str.upper()
df.loc[:, 'VEHICLE 5 FACTOR'] = df['VEHICLE 5 FACTOR'].str.upper()
print("Contoh 5 baris pertama setelah penyeragaman teks (kolom teks saja):")
print(df[['BOROUGH', 'ON STREET NAME', 'VEHICLE 1 TYPE', 'VEHICLE 1 FACTOR']].head().to_string())

print("\nMendeteksi outlier menggunakan Z-score...")
z_score_scaler = StandardScaler()
z_scores = np.abs(z_score_scaler.fit_transform(df[['Total Victims']]))
threshold = 3
outliers = df[z_scores > threshold]
df_no_outliers = df[z_scores <= threshold].copy()
print(f"Jumlah outlier yang terdeteksi (Z-score > {threshold}): {len(outliers)}")
print(f"Persentase outlier: {len(outliers) / len(df) * 100:.2f}%")
print("Statistik Total Victims setelah menghapus outlier:")
print(df_no_outliers['Total Victims'].describe().to_string())

print("\nMendeteksi outlier pada LATITUDE dan LONGITUDE menggunakan Z-score...")
lat_z_scores = np.abs(z_score_scaler.fit_transform(df_no_outliers[['LATITUDE']]))
long_z_scores = np.abs(z_score_scaler.fit_transform(df_no_outliers[['LONGITUDE']]))
lat_outliers = df_no_outliers[lat_z_scores > 3]
long_outliers = df_no_outliers[long_z_scores > 3]
print(f"Jumlah outlier LATITUDE (Z-score > 3): {len(lat_outliers)}")
print(f"Jumlah outlier LONGITUDE (Z-score > 3): {len(long_outliers)}")
df_no_outliers = df_no_outliers[(lat_z_scores <= 3) & (long_z_scores <= 3)].copy()
print(f"Baris setelah menghapus outlier LATITUDE dan LONGITUDE: {len(df_no_outliers)}")

print("\nVerifikasi tipe data DATE sebelum time series:")
print(f"Tipe data DATE: {df_no_outliers['DATE'].dtype}")
print("Contoh 5 baris pertama DATE:")
print(df_no_outliers['DATE'].head().to_string())
print(f"Jumlah nilai NaT di DATE: {df_no_outliers['DATE'].isna().sum()}")

print("\nMembuat visualisasi dan analisis teks...")

print("\nAnalisis Missing Values:")
print("Missing values sebelum imputasi (top 5):")
print(missing_before.nlargest(5).to_string())
print("Missing values setelah imputasi (top 5):")
print(missing_after.nlargest(5).to_string())
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.barplot(x=missing_before.index, y=missing_before.values, color='red')
plt.title('Missing Values Sebelum Imputasi')
plt.xticks(rotation=90)
plt.subplot(1, 2, 2)
sns.barplot(x=missing_after.index, y=missing_after.values, color='green')
plt.title('Missing Values Sesudah Imputasi')
plt.xticks(rotation=90)
plt.tight_layout()
save_plot(plt, "missing_values_comparison.png")
plt.show()
plt.close()

print("\nAnalisis Distribusi Total Victims:")
print("Statistik sebelum outlier:")
print((df_original['PERSONS INJURED'] + df_original['PERSONS KILLED']).describe().to_string())
print("Statistik setelah outlier:")
print(df_no_outliers['Total Victims'].describe().to_string())
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df_original['PERSONS INJURED'] + df_original['PERSONS KILLED'], bins=30, kde=True, color='orange')
plt.title('Distribusi Total Victims Sebelum')
plt.subplot(1, 2, 2)
sns.histplot(df_no_outliers['Total Victims'], bins=30, kde=True, color='blue')
plt.title('Distribusi Total Victims Sesudah (No Outliers)')
plt.tight_layout()
save_plot(plt, "total_victims_distribution.png")
plt.show()
plt.close()

print("\nDistribusi Kecelakaan per Borough:")
borough_counts = df_no_outliers['BOROUGH'].value_counts()
print(borough_counts.to_string())
print(f"Persentase per Borough:\n{(borough_counts / len(df_no_outliers) * 100).to_string()}")
plt.figure(figsize=(8, 8))
plt.pie(borough_counts, labels=borough_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribusi Kecelakaan per Borough')
save_plot(plt, "borough_distribution.png")
plt.show()
plt.close()

print("\nTop 10 Jenis Kendaraan dalam Kecelakaan:")
vehicle_counts = df_no_outliers['VEHICLE 1 TYPE'].value_counts().head(10)
print(vehicle_counts.to_string())
print(f"Persentase Top 10 Jenis Kendaraan:\n{(vehicle_counts / len(df_no_outliers) * 100).to_string()}")
plt.figure(figsize=(12, 6))
sns.barplot(x=vehicle_counts.index, y=vehicle_counts.values, palette='viridis')
plt.title('Top 10 Jenis Kendaraan dalam Kecelakaan')
plt.xticks(rotation=45)
save_plot(plt, "top_10_vehicle_types.png")
plt.show()
plt.close()

print("\nTop 10 Faktor Penyebab Kecelakaan:")
factor_counts = df_no_outliers['VEHICLE 1 FACTOR'].value_counts().head(10)
print(factor_counts.to_string())
print(f"Persentase Top 10 Faktor Penyebab:\n{(factor_counts / len(df_no_outliers) * 100).to_string()}")
plt.figure(figsize=(12, 6))
sns.barplot(x=factor_counts.index, y=factor_counts.values, palette='magma')
plt.title('Top 10 Faktor Penyebab Kecelakaan')
plt.xticks(rotation=45)
save_plot(plt, "top_10_vehicle_factors.png")
plt.show()
plt.close()

print("\nAnalisis Lokasi Kecelakaan:")
print("Statistik Latitude:")
print(df_no_outliers['LATITUDE'].describe().to_string())
print("Statistik Longitude:")
print(df_no_outliers['LONGITUDE'].describe().to_string())
plt.figure(figsize=(10, 8))
sns.scatterplot(x='LONGITUDE', y='LATITUDE', hue='Total Victims', size='Total Victims',
                data=df_no_outliers, palette='coolwarm', alpha=0.5)
plt.title('Lokasi Kecelakaan Berdasarkan Latitude dan Longitude (2016-2017)')
plt.xlim(-74.25, -73.7)
plt.ylim(40.5, 41.0)
save_plot(plt, "accident_locations.png")
plt.show()
plt.close()

print("\nKorelasi Antar Variabel Numerik:")
numeric_cols = ['LATITUDE', 'LONGITUDE', 'PERSONS INJURED', 'PERSONS KILLED', 'PEDESTRIANS INJURED',
                'PEDESTRIANS KILLED', 'CYCLISTS INJURED', 'CYCLISTS KILLED', 'MOTORISTS INJURED', 'MOTORISTS KILLED']
corr = df_no_outliers[numeric_cols].corr()
print(corr.to_string())
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Heatmap Korelasi Antar Variabel Numerik')
save_plot(plt, "correlation_heatmap.png")
plt.show()
plt.close()

print("\nMembuat time series plot...")
df_no_outliers.loc[:, 'MONTH'] = df_no_outliers['DATE'].dt.to_period('M')
monthly_counts = df_no_outliers['MONTH'].value_counts().sort_index()
print("\nJumlah Kecelakaan per Bulan:")
print(monthly_counts.to_string())
plt.figure(figsize=(12, 6))
monthly_counts.plot(kind='line', marker='o', color='purple')
plt.title('Jumlah Kecelakaan per Bulan')
plt.xticks(rotation=45)
save_plot(plt, "monthly_accident_trend.png")
plt.show()
plt.close()

print("\nMembuat dashboard interaktif multidimensi...")
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Distribusi Kecelakaan per Borough', 'Top 10 Jenis Kendaraan', 'Top 10 Faktor Penyebab', 'Tren Kecelakaan per Bulan'),
    specs=[ [{'type': 'pie'}, {'type': 'bar'}], [{'type': 'bar'}, {'type': 'scatter'}] ]
)
fig.add_trace(go.Pie(labels=borough_counts.index, values=borough_counts.values, name='Borough', textinfo='percent+label', hole=0.3), row=1, col=1)
fig.add_trace(go.Bar(x=vehicle_counts.index, y=vehicle_counts.values, name='Jenis Kendaraan', marker_color='royalblue'), row=1, col=2)
fig.add_trace(go.Bar(x=factor_counts.index, y=factor_counts.values, name='Faktor Penyebab', marker_color='firebrick'), row=2, col=1)
fig.add_trace(go.Scatter(x=monthly_counts.index.astype(str), y=monthly_counts.values, mode='lines+markers', name='Jumlah Kecelakaan', line=dict(color='purple', width=2)), row=2, col=2)

df_no_outliers['HOUR'] = pd.to_datetime(df_no_outliers['TIME'], errors='coerce').dt.hour
df_no_outliers['DAY_OF_WEEK'] = df_no_outliers['DATE'].dt.day_name()
df_no_outliers['MONTH_NAME'] = df_no_outliers['DATE'].dt.month_name()
df_no_outliers['YEAR'] = df_no_outliers['DATE'].dt.year

fig.update_layout(height=800, width=1200, title_text="Dashboard Multidimensi Analisis Kecelakaan NYC", showlegend=True,
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig.update_xaxes(tickangle=45, row=1, col=2)
fig.update_xaxes(tickangle=45, row=2, col=1)
fig.update_xaxes(title_text="Bulan", row=2, col=2)
fig.update_yaxes(title_text="Jumlah Kecelakaan", row=1, col=2)
fig.update_yaxes(title_text="Jumlah Kecelakaan", row=2, col=1)
fig.update_yaxes(title_text="Jumlah Kecelakaan", row=2, col=2)
save_dashboard(fig, "multidimensional_dashboard")
fig.show()

print("\nMembuat dashboard analisis temporal...")
fig_time = make_subplots(rows=1, cols=3, subplot_titles=('Kecelakaan per Jam', 'Kecelakaan per Hari', 'Kecelakaan per Bulan'),
                         specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]])

hourly_counts = df_no_outliers['HOUR'].value_counts().sort_index()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_counts = df_no_outliers['DAY_OF_WEEK'].value_counts().reindex(day_order)
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
monthly_name_counts = df_no_outliers['MONTH_NAME'].value_counts().reindex(month_order)

print("\nKecelakaan per Jam:")
print(hourly_counts.to_string())
print("\nKecelakaan per Hari:")
print(daily_counts.to_string())
print("\nKecelakaan per Bulan:")
print(monthly_name_counts.to_string())

fig_time.add_trace(go.Bar(x=hourly_counts.index, y=hourly_counts.values, name='Per Jam', marker_color='darkcyan'), row=1, col=1)
fig_time.add_trace(go.Bar(x=daily_counts.index, y=daily_counts.values, name='Per Hari', marker_color='darkviolet'), row=1, col=2)
fig_time.add_trace(go.Bar(x=monthly_name_counts.index, y=monthly_name_counts.values, name='Per Bulan', marker_color='darkslateblue'), row=1, col=3)

fig_time.update_layout(height=500, width=1400, title_text="Dashboard Analisis Temporal Kecelakaan NYC", showlegend=True)
fig_time.update_xaxes(title_text="Jam", row=1, col=1)
fig_time.update_xaxes(title_text="Hari", tickangle=45, row=1, col=2)
fig_time.update_xaxes(title_text="Bulan", tickangle=45, row=1, col=3)
fig_time.update_yaxes(title_text="Jumlah Kecelakaan", row=1, col=1)
fig_time.update_yaxes(title_text="Jumlah Kecelakaan", row=1, col=2)
fig_time.update_yaxes(title_text="Jumlah Kecelakaan", row=1, col=3)
save_dashboard(fig_time, "temporal_dashboard")
fig_time.show()

# Dashboard Korban (Diperbarui)
print("\nMembuat dashboard tambahan...")
fig_victims = make_subplots(rows=1, cols=2, subplot_titles=('Korban Berdasarkan Jenis', 'Perbandingan Cedera vs Kematian'),
                            specs=[[{'type': 'pie'}, {'type': 'bar'}]])

victim_types = {
    'Pejalan Kaki': df_no_outliers['PEDESTRIANS INJURED'].sum() + df_no_outliers['PEDESTRIANS KILLED'].sum(),
    'Pengendara Sepeda': df_no_outliers['CYCLISTS INJURED'].sum() + df_no_outliers['CYCLISTS KILLED'].sum(),
    'Pengemudi': df_no_outliers['MOTORISTS INJURED'].sum() + df_no_outliers['MOTORISTS KILLED'].sum()
}
injury_death = {
    'Pejalan Kaki Cedera': df_no_outliers['PEDESTRIANS INJURED'].sum(),
    'Pejalan Kaki Meninggal': df_no_outliers['PEDESTRIANS KILLED'].sum(),
    'Pengendara Sepeda Cedera': df_no_outliers['CYCLISTS INJURED'].sum(),
    'Pengendara Sepeda Meninggal': df_no_outliers['CYCLISTS KILLED'].sum(),
    'Pengemudi Cedera': df_no_outliers['MOTORISTS INJURED'].sum(),
    'Pengemudi Meninggal': df_no_outliers['MOTORISTS KILLED'].sum()
}

print("\nKorban Berdasarkan Jenis:")
for k, v in victim_types.items():
    print(f"{k}: {v}")
print("\nPerbandingan Cedera vs Kematian:")
for k, v in injury_death.items():
    print(f"{k}: {v}")

fig_victims.add_trace(go.Pie(labels=list(victim_types.keys()), values=list(victim_types.values()), name='Jenis Korban', textinfo='percent+label', marker_colors=['gold', 'mediumturquoise', 'darkorange']), row=1, col=1)
fig_victims.add_trace(go.Bar(x=list(injury_death.keys()), y=list(injury_death.values()), name='Cedera vs Meninggal', marker_color=['lightskyblue', 'darkblue', 'lightgreen', 'darkgreen', 'lightsalmon', 'darkred']), row=1, col=2)

fig_victims.update_layout(height=500, width=1200, title_text="Dashboard Analisis Korban Kecelakaan NYC", showlegend=True)
fig_victims.update_xaxes(title_text="Kategori", tickangle=45, row=1, col=2)
fig_victims.update_yaxes(title_text="Jumlah Korban", row=1, col=2)
save_dashboard(fig_victims, "victims_dashboard")
fig_victims.show()

print("\nMenyimpan dataset yang telah diproses...")
processed_file = os.path.join(output_dir, f"vehicle_collisions_processed_{timestamp}.csv")
df_no_outliers.to_csv(processed_file, index=False)
print(f"Dataset disimpan sebagai: {processed_file}")

print("\nRingkasan preprocessing:")
print(f"Baris awal: {len(df_original)}")
print(f"Baris setelah preprocessing: {len(df_no_outliers)}")
print(f"Missing values dihapus/ditangani: {missing_before.sum() - missing_after.sum()}")
print(f"Duplikat dihapus: {duplicates_before - duplicates_after}")
print(f"Outlier dihapus: {len(df) - len(df_no_outliers)}")

sys.stdout = original_stdout
f.close()
