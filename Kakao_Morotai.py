import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as ticker  # Import ticker untuk FixedLocator

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    csv_path = 'D:/Anggi/data/data_kakao.csv'
    data = pd.read_csv(csv_path)
    return data

# Fungsi-fungsi analisis
def analyze_yearly_production(df):
    yearly_production = df.groupby('tahun')['produksi_pertahun'].agg(['mean', 'sum']).round(2)
    return yearly_production

def analyze_top_regions(df):
    top_regions = df.groupby('wilayah')['produksi_pertahun'].agg(['mean', 'sum']).round(2).sort_values('sum', ascending=False)
    return top_regions

def analyze_rain_production(df):
    rain_production = df.groupby('curah_hujan')['produksi_pertahun'].agg(['mean', 'count']).round(2)
    return rain_production

def analyze_market_demand(df):
    market_demand = df.groupby('permintaan_pasar').agg({
        'produksi_pertahun': 'mean',
        'harga': 'mean',
        'wilayah': 'count'
    }).round(2)
    return market_demand

def analyze_price_per_region(df):
    price_analysis = df.groupby('wilayah').agg({
        'harga': ['mean', 'min', 'max'],
        'permintaan_pasar': lambda x: x.value_counts().index[0]
    }).round(2)
    return price_analysis

def analyze_correlation(df):
    correlation = df[['produksi_pertahun', 'harga', 'luas_lahan_hektar', 'tingkat_konsumsi_perkapita_perkg']].corr()
    return correlation

def analyze_potential_regions(df):
    potential_regions = df.groupby('wilayah').agg({
        'produksi_pertahun': 'mean',
        'permintaan_pasar': lambda x: x.value_counts().index[0],
        'tingkat_konsumsi_perkapita_perkg': 'mean',
        'harga': 'mean'
    }).round(2)
    potential_regions['skor_potensi'] = (
        (potential_regions['produksi_pertahun'] / potential_regions['produksi_pertahun'].max()) * 0.3 +
        (potential_regions['tingkat_konsumsi_perkapita_perkg'] / potential_regions['tingkat_konsumsi_perkapita_perkg'].max()) * 0.3 +
        (potential_regions['harga'] / potential_regions['harga'].max()) * 0.4
    ).round(2)
    return potential_regions

def generate_recommendations(df):
    """
    Menghasilkan rekomendasi implementasi strategi berdasarkan analisis wilayah dan risiko.
    """
    # Contoh sederhana: rekomendasi berdasarkan skor potensi
    potential_regions = analyze_potential_regions(df)
    recommendations = potential_regions.sort_values('skor_potensi', ascending=False)
    return recommendations

def main():
    # Memuat data
    data = load_data()

    # Mengubah data kategorikal menjadi numerik untuk analisis korelasi
    label_encoder = LabelEncoder()
    data['curah_hujan_encoded'] = label_encoder.fit_transform(data['curah_hujan'])
    data['permintaan_pasar_encoded'] = label_encoder.fit_transform(data['permintaan_pasar'])

    # Tab untuk menampilkan analisis
    tabs = st.tabs([
        "Analisis Data Produksi dan Permintaan Kakao", 
        "Analisis Data Kakao", 
        "Analisis Risiko dan Rekomendasi Implementasi", 
        "Analisis Peluang Pasar Kakao di Pulau Morotai", 
        "Kesimpulan Utama"
    ])

    # Tab 1: Analisis Data Produksi dan Permintaan Kakao
    with tabs[0]:
        st.header("Analisis Data Produksi dan Permintaan Kakao")
        
        # Trend Produksi per Tahun
        st.subheader("Trend Produksi per Tahun")
        yearly_production = analyze_yearly_production(data)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=data, x='tahun', y='produksi_pertahun', hue='wilayah', marker='o', ax=ax)
        ax.set_title("Trend Produksi Kakao per Tahun")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Produksi (kg)")
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)  # Menutup plot untuk menghindari kebocoran memori
        st.markdown("""
        **Kesimpulan:**
        - Grafik menunjukkan fluktuasi produksi kakao dari tahun ke tahun. Puncak produksi terjadi pada tahun tertentu (misalnya, 2022), sementara produksi terendah terjadi pada tahun lainnya (misalnya, 2023).
        - Produksi rata-rata per tahun bervariasi, dengan tahun tertentu menunjukkan produksi rata-rata yang lebih tinggi dibandingkan tahun lainnya.
        """)
        
        # Wilayah dengan Produksi Tertinggi
        st.subheader("Wilayah dengan Produksi Tertinggi")
        top_regions = analyze_top_regions(data).head()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_regions.index, y=top_regions['sum'], hue=top_regions.index, palette='viridis', legend=False, ax=ax)
        ax.set_title("Produksi Kakao per Wilayah")
        ax.set_xlabel("Wilayah")
        ax.set_ylabel("Total Produksi (kg)")
        ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(top_regions.index))))  # Atur posisi tick
        ax.set_xticklabels(top_regions.index, rotation=45)  # Atur label tick dengan rotasi
        st.pyplot(fig)
        plt.close(fig)  # Menutup plot untuk menghindari kebocoran memori
        st.markdown("""
        **Kesimpulan:**
        - Grafik batang menunjukkan wilayah dengan produksi tertinggi. Misalnya, Halmahera Utara memiliki produksi tertinggi, diikuti oleh wilayah lainnya.
        - Wilayah dengan produksi tertinggi memiliki potensi besar untuk pengembangan lebih lanjut.
        """)
        
        # Pengaruh Curah Hujan terhadap Produksi
        st.subheader("Pengaruh Curah Hujan terhadap Produksi")
        rain_production = analyze_rain_production(data)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=rain_production.index, y=rain_production['mean'], hue=rain_production.index, palette='coolwarm', legend=False, ax=ax)
        ax.set_title("Pengaruh Curah Hujan terhadap Produksi Kakao")
        ax.set_xlabel("Curah Hujan")
        ax.set_ylabel("Rata-Rata Produksi (kg)")
        st.pyplot(fig)
        plt.close(fig)  # Menutup plot untuk menghindari kebocoran memori
        st.markdown("""
        **Kesimpulan:**
        - Grafik menunjukkan bahwa curah hujan sedang memberikan dampak positif pada produksi kakao, dengan rata-rata produksi tertinggi.
        - Curah hujan rendah dan tinggi menghasilkan rata-rata produksi yang lebih rendah.
        """)
        
        # Analisis Permintaan Pasar
        st.subheader("Analisis Permintaan Pasar")
        market_demand = analyze_market_demand(data)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=market_demand.index, y=market_demand['produksi_pertahun'], hue=market_demand.index, palette='cool', legend=False, ax=ax)
        ax.set_title("Distribusi Permintaan Pasar")
        ax.set_xlabel("Kategori Permintaan")
        ax.set_ylabel("Rata-Rata Produksi (kg)")
        ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(market_demand.index))))  # Atur posisi tick
        ax.set_xticklabels(market_demand.index, rotation=45)  # Atur label tick dengan rotasi
        st.pyplot(fig)
        plt.close(fig)  # Menutup plot untuk menghindari kebocoran memori
        st.markdown("""
        **Kesimpulan:**
        - Grafik menunjukkan bahwa permintaan pasar tidak selalu selaras dengan produksi kakao. Permintaan pasar rendah memiliki produksi rata-rata lebih tinggi dibanding permintaan pasar tinggi.
        - Harga rata-rata cenderung menurun seiring meningkatnya permintaan pasar.
        """)
        
        # Harga per Wilayah
        st.subheader("Harga per Wilayah")
        price_analysis = analyze_price_per_region(data)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=price_analysis.index, y=price_analysis['harga']['mean'], hue=price_analysis.index, palette='magma', legend=False, ax=ax)
        ax.set_title("Harga Rata-Rata Kakao per Wilayah")
        ax.set_xlabel("Wilayah")
        ax.set_ylabel("Harga (Rp/kg)")
        ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(price_analysis.index))))  # Atur posisi tick
        ax.set_xticklabels(price_analysis.index, rotation=45)  # Atur label tick dengan rotasi
        st.pyplot(fig)
        plt.close(fig)  # Menutup plot untuk menghindari kebocoran memori
        st.markdown("""
        **Kesimpulan:**
        - Grafik menunjukkan harga rata-rata kakao per wilayah. Misalnya, Halmahera Timur memiliki harga rata-rata tertinggi, sementara Halmahera Utara memiliki harga rata-rata terendah.
        - Harga rata-rata mungkin mencerminkan volume produksi yang lebih tinggi di wilayah tertentu.
        """)
        
        # Analisis Korelasi
        st.subheader("Analisis Korelasi")
        correlation = analyze_correlation(data)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Korelasi antara Produksi, Curah Hujan, Harga, dan Permintaan Pasar")
        st.pyplot(fig)
        plt.close(fig)  # Menutup plot untuk menghindari kebocoran memori
        st.markdown("""
        **Kesimpulan:**
        - Heatmap korelasi menunjukkan hubungan antara variabel produksi, harga, luas lahan, dan permintaan pasar.
        - Hubungan antara produksi per tahun dengan harga menunjukkan korelasi yang lemah, mengindikasikan bahwa peningkatan produksi tidak secara langsung memengaruhi harga.
        """)
        
        # Wilayah Paling Potensial
        st.subheader("Wilayah Paling Potensial")
        potential_regions = analyze_potential_regions(data).sort_values('skor_potensi', ascending=False).head()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=potential_regions.index, y=potential_regions['skor_potensi'], hue=potential_regions.index, palette='plasma', legend=False, ax=ax)
        ax.set_title("Wilayah dengan Potensi Produksi Tertinggi")
        ax.set_xlabel("Wilayah")
        ax.set_ylabel("Skor Potensi")
        ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(potential_regions.index))))  # Atur posisi tick
        ax.set_xticklabels(potential_regions.index, rotation=45)  # Atur label tick dengan rotasi
        st.pyplot(fig)
        plt.close(fig)  # Menutup plot untuk menghindari kebocoran memori
        st.markdown("""
        **Kesimpulan:**
        - Grafik batang menunjukkan wilayah dengan skor potensi tertinggi. Misalnya, Halmahera Tengah dan Halmahera Utara dinilai paling potensial untuk pengembangan kakao.
        - Wilayah ini memiliki rata-rata produksi tinggi, konsumsi per kapita stabil, dan harga rata-rata yang kompetitif.
        """)

    # Tab 2: Analisis Data Kakao
    with tabs[1]:
        st.header("Analisis Data Kakao")
        
        # Analisis Seasonality
        st.subheader("Analisis Seasonality (Pola Produksi Berdasarkan Curah Hujan)")
        seasonality_data = data.groupby(['tahun', 'curah_hujan'])['produksi_pertahun'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=seasonality_data, x='tahun', y='produksi_pertahun', hue='curah_hujan', marker='o', ax=ax)
        ax.set_title("Analisis Seasonality")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Produksi (kg)")
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)  # Menutup plot untuk menghindari kebocoran memori
        st.markdown("""
        **Kesimpulan:**
        - Grafik garis menunjukkan pola produksi berdasarkan curah hujan. Produksi tertinggi terjadi pada kondisi curah hujan sedang, sementara produksi terendah terjadi pada curah hujan tinggi.
        - Harga rata-rata tertinggi tercatat pada curah hujan sedang, sedangkan harga terendah terjadi pada curah hujan tinggi.
        """)
        
        # Proyeksi Permintaan dan Produksi
        st.subheader("Proyeksi Permintaan dan Produksi")
        yearly_production = analyze_yearly_production(data)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=yearly_production.reset_index(), x='tahun', y='sum', marker='o', ax=ax)
        ax.set_title("Proyeksi Permintaan dan Produksi")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Total Produksi (kg)")
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)  # Menutup plot untuk menghindari kebocoran memori
        st.markdown("""
        **Kesimpulan:**
        - Grafik garis menunjukkan proyeksi produksi kakao. Produksi diproyeksikan menurun hingga tahun tertentu, menunjukkan perlunya upaya untuk meningkatkan produktivitas.
        """)
        
        # Analisis Kompetisi (Market Share)
        st.subheader("Analisis Kompetisi (Market Share)")
        market_share = data.groupby('wilayah')['produksi_pertahun'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=market_share['wilayah'], y=market_share['produksi_pertahun'], hue=market_share['wilayah'], palette='viridis', legend=False, ax=ax)
        ax.set_title("Analisis Kompetisi (Market Share)")
        ax.set_xlabel("Wilayah")
        ax.set_ylabel("Total Produksi (kg)")
        ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(market_share['wilayah']))))  # Atur posisi tick
        ax.set_xticklabels(market_share['wilayah'], rotation=45)  # Atur label tick dengan rotasi
        st.pyplot(fig)
        plt.close(fig)  # Menutup plot untuk menghindari kebocoran memori
        st.markdown("""
        **Kesimpulan:**
        - Grafik batang menunjukkan pangsa pasar per wilayah. Misalnya, Halmahera Utara memiliki pangsa pasar terbesar, diikuti oleh wilayah lainnya.
        - Wilayah dengan pangsa pasar besar memiliki kontribusi signifikan terhadap total produksi.
        """)
        
        # Analisis Faktor Harga
        st.subheader("Analisis Faktor Harga")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=data, x='produksi_pertahun', y='harga', hue='wilayah', ax=ax)
        ax.set_title("Analisis Faktor Harga")
        ax.set_xlabel("Produksi (kg)")
        ax.set_ylabel("Harga (Rp/kg)")
        st.pyplot(fig)
        plt.close(fig)  # Menutup plot untuk menghindari kebocoran memori
        st.markdown("""
        **Kesimpulan:**
        - Scatter plot menunjukkan hubungan antara produksi dan harga. Korelasi antara produksi per tahun dan harga menunjukkan pengaruh yang sangat lemah.
        - Faktor lain seperti luas lahan dan tingkat kesuburan tanah juga tidak menunjukkan hubungan signifikan dengan harga.
        """)

    # Tab 3: Analisis Risiko dan Rekomendasi Implementasi
    with tabs[2]:
        st.header("Analisis Risiko dan Rekomendasi Implementasi")
        
        # Analisis Skor Wilayah
        st.subheader("Analisis Skor Wilayah")
        potential_regions = analyze_potential_regions(data).sort_values('skor_potensi', ascending=False).head()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=potential_regions.index, y=potential_regions['skor_potensi'], hue=potential_regions.index, palette='plasma', legend=False, ax=ax)
        ax.set_title("Analisis Skor Wilayah")
        ax.set_xlabel("Wilayah")
        ax.set_ylabel("Skor Potensi")
        ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(potential_regions.index))))  # Atur posisi tick
        ax.set_xticklabels(potential_regions.index, rotation=45)  # Atur label tick dengan rotasi
        st.pyplot(fig)
        plt.close(fig)  # Menutup plot untuk menghindari kebocoran memori
        st.markdown("""
        **Kesimpulan:**
        - Grafik batang menunjukkan skor potensi per wilayah. Wilayah dengan skor tinggi memiliki potensi besar untuk pengembangan kakao.
        - Wilayah dengan skor menengah memiliki potensi untuk ditingkatkan dengan intervensi yang tepat.
        """)
        
        # Analisis Risiko Produksi
        st.subheader("Analisis Risiko Produksi")
        risk_data = data.groupby('wilayah')['produksi_pertahun'].std().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=risk_data['wilayah'], y=risk_data['produksi_pertahun'], hue=risk_data['wilayah'], palette='coolwarm', legend=False, ax=ax)
        ax.set_title("Analisis Risiko Produksi")
        ax.set_xlabel("Wilayah")
        ax.set_ylabel("Standar Deviasi Produksi (kg)")
        ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(risk_data['wilayah']))))  # Atur posisi tick
        ax.set_xticklabels(risk_data['wilayah'], rotation=45)  # Atur label tick dengan rotasi
        st.pyplot(fig)
        plt.close(fig)  # Menutup plot untuk menghindari kebocoran memori
        st.markdown("""
        **Kesimpulan:**
        - Grafik batang menunjukkan standar deviasi produksi per wilayah. Wilayah dengan standar deviasi tinggi menunjukkan risiko produksi yang lebih besar.
        - Wilayah ini mungkin memerlukan strategi mitigasi risiko untuk meningkatkan stabilitas produksi.
        """)
        
        # Rekomendasi Implementasi
        st.subheader("Rekomendasi Implementasi")
        recommendations = generate_recommendations(data)
        st.write(recommendations)
        st.markdown("""
        **Rekomendasi Implementasi:**
        - **Ekspansi agresif, fokus peningkatan kapasitas** untuk wilayah unggulan.
        - **Pengembangan bertahap, fokus efisiensi** untuk wilayah potensial.
        - **Evaluasi ulang strategi, fokus perbaikan fundamental** untuk wilayah berkembang.
        """)

    # Tab 4: Analisis Peluang Pasar Kakao di Pulau Morotai
    with tabs[3]:
        st.header("Analisis Peluang Pasar Kakao di Pulau Morotai")
        
        # Contoh analisis peluang pasar
        st.subheader("Peluang Pasar Berdasarkan Permintaan")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=data, x='permintaan_pasar', y='produksi_pertahun', hue='permintaan_pasar', palette='cool', legend=False, ax=ax)
        ax.set_title("Peluang Pasar Berdasarkan Permintaan")
        ax.set_xlabel("Kategori Permintaan")
        ax.set_ylabel("Produksi (kg)")
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown("""
        **Kesimpulan:**
        - Peluang pasar terbesar ada di wilayah dengan permintaan tinggi.
        - Wilayah dengan permintaan sedang memiliki potensi untuk ditingkatkan melalui strategi pemasaran.
        """)

    # Tab 5: Kesimpulan Utama
    with tabs[4]:
        st.header("Kesimpulan Utama")
        
        # Contoh kesimpulan utama
        st.markdown("""
        **Kesimpulan Utama:**
        - Produksi kakao di Pulau Morotai memiliki potensi besar untuk dikembangkan.
        - Peningkatan produksi dan pemasaran dapat meningkatkan profitabilitas.
        - Manajemen risiko dan diversifikasi produk diperlukan untuk mengurangi fluktuasi harga.
        """)

# Panggil fungsi main() untuk menjalankan dashboard
if __name__ == "__main__":
    main()