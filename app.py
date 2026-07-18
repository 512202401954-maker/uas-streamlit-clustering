import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="Analisis Clustering Cacat Produk",
    page_icon="📊",
    layout="wide"
)

# ==========================
# MEMBACA DATASET
# ==========================
df = pd.read_excel("defects_data.xlsx")

# ==========================
# SIDEBAR
# ==========================
menu = st.sidebar.selectbox(
    "📋 Pilih Menu",
    [
        "🏠 Home",
        "📂 Dataset",
        "📊 EDA",
        "🎯 K-Means Clustering",
        "🌳 Hierarchical Clustering",
        "💡 Interpretasi",
        "📈 Insight Bisnis"
    ]
)

# ==========================
# HOME
# ==========================
if menu == "🏠 Home":

    st.title("📊 Analisis Clustering Cacat Produk Industri Manufaktur")

    st.write("""
Selamat datang di aplikasi analisis clustering cacat produk industri manufaktur.

Aplikasi ini menggunakan metode K-Means Clustering dan Hierarchical Clustering
untuk mengelompokkan data cacat produk berdasarkan karakteristiknya.

Aplikasi ini dibuat sebagai tugas UAS Project Kecerdasan Buatan.
""")

    st.markdown("---")

    st.markdown("""
### 👨‍🎓 Informasi Pengembang

- **Nama** : Cipta Bagus Rahmawan
- **NIM** : E1202401954
- **Program Studi** : Teknik Industri
- **Universitas** : Universitas Dian Nuswantoro
- **Mata Kuliah** : Project Kecerdasan Buatan
- **Semester** : 4
""")

st.markdown("---")

st.caption("© 2025/2026 Cipta Bagus Rahmawan | Project Kecerdasan Buatan | Universitas Dian Nuswatoro")

# ==========================
# DATASET
# ==========================
elif menu == "📂 Dataset":

    st.title("📂 Dataset")

    st.subheader("Informasi Dataset")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Jumlah Baris", df.shape[0])

    with col2:
        st.metric("Jumlah Kolom", df.shape[1])

    st.markdown("---")

    st.subheader("5 Data Pertama")
    st.dataframe(df.head())

    st.markdown("---")

    st.subheader("Informasi Kolom")
    info_df = pd.DataFrame({
        "Nama Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str)
    })

    st.dataframe(info_df)

    st.markdown("---")

    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe(include="all"))

# ==========================
# MENU LAIN
# ==========================
elif menu == "📊 EDA":

    import matplotlib.pyplot as plt

    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("Distribusi Tingkat Severity")

    severity_count = df["severity"].value_counts()

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(severity_count.index, severity_count.values)
    ax.set_xlabel("Severity")
    ax.set_ylabel("Jumlah")
    ax.set_title("Distribusi Severity")

    st.pyplot(fig)


    st.markdown("---")

    st.subheader("Distribusi Jenis Cacat")

    defect_count = df["defect_type"].value_counts()

    fig2, ax2 = plt.subplots(figsize=(7,4))
    ax2.bar(defect_count.index, defect_count.values)
    ax2.set_xlabel("Jenis Cacat")
    ax2.set_ylabel("Jumlah")
    ax2.set_title("Distribusi Defect Type")

    st.pyplot(fig2)


    st.markdown("---")

    st.subheader("Ringkasan Data")

    st.dataframe(df.describe(include="all"))

elif menu == "🎯 K-Means Clustering":

    st.title("🎯 K-Means Clustering")

    # Mapping severity
    severity_mapping = {
        "Minor": 1,
        "Moderate": 2,
        "Critical": 3
    }

    df["severity_score"] = df["severity"].map(severity_mapping)

    df["repair_cost"] = pd.to_numeric(df["repair_cost"], errors="coerce")

    df = df.dropna(subset=["repair_cost"])

    X = df[["repair_cost", "severity_score"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    df["cluster_kmeans"] = kmeans.fit_predict(X_scaled)

    st.subheader("Jumlah Data Tiap Cluster")

    st.dataframe(
        df["cluster_kmeans"].value_counts().reset_index().rename(
            columns={
                "index": "Cluster",
                "cluster_kmeans": "Jumlah Data"
            }
        )
    )

    st.subheader("Visualisasi Cluster")

    fig, ax = plt.subplots(figsize=(9,6))

    sns.scatterplot(
        data=df,
        x="repair_cost",
        y="severity_score",
        hue="cluster_kmeans",
        palette="Set1",
        s=70,
        ax=ax
    )

    ax.set_xlabel("Repair Cost")
    ax.set_ylabel("Severity Score")

    st.pyplot(fig)

    st.subheader("Profil Cluster")

    profil = df.groupby("cluster_kmeans")[["repair_cost","severity_score"]].mean()

    st.dataframe(profil)

elif menu == "🌳 Hierarchical Clustering":

    from sklearn.cluster import AgglomerativeClustering

    st.title("🌳 Hierarchical Clustering")

    severity_mapping = {
        "Minor": 1,
        "Moderate": 2,
        "Critical": 3
    }

    df["severity_score"] = df["severity"].map(severity_mapping)

    df["repair_cost"] = pd.to_numeric(df["repair_cost"], errors="coerce")

    data = df.dropna(subset=["repair_cost"])

    X = data[["repair_cost", "severity_score"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    hc = AgglomerativeClustering(
        n_clusters=3
    )

    data["cluster_hc"] = hc.fit_predict(X_scaled)

    st.subheader("Jumlah Data Tiap Cluster")

    st.dataframe(
        data["cluster_hc"]
        .value_counts()
        .sort_index()
        .reset_index(name="Jumlah Data")
        .rename(columns={"index":"Cluster"})
    )

    st.subheader("Visualisasi Hierarchical Clustering")

    fig, ax = plt.subplots(figsize=(9,6))

    sns.scatterplot(
        data=data,
        x="repair_cost",
        y="severity_score",
        hue="cluster_hc",
        palette="Set2",
        s=70,
        ax=ax
    )

    ax.set_xlabel("Repair Cost")
    ax.set_ylabel("Severity Score")

    st.pyplot(fig)

    st.subheader("Profil Cluster")

    profil = data.groupby("cluster_hc")[["repair_cost","severity_score"]].mean()

    st.dataframe(profil)

elif menu == "💡 Interpretasi":

    st.title("💡 Interpretasi Hasil Clustering")

    st.markdown("""
### Cluster 0
- Memiliki biaya perbaikan relatif tinggi.
- Tingkat keparahan cacat cenderung sedang hingga tinggi.
- Memerlukan perhatian lebih dalam proses quality control.

### Cluster 1
- Memiliki biaya perbaikan sedang.
- Didominasi cacat dengan tingkat keparahan tinggi.
- Perlu dilakukan evaluasi proses produksi pada tahap inspeksi.

### Cluster 2
- Memiliki biaya perbaikan paling rendah.
- Mayoritas merupakan cacat ringan.
- Dapat dijadikan acuan sebagai kondisi produksi yang baik.
""")

elif menu == "📈 Insight Bisnis":

    st.title("📈 Insight Bisnis")

    st.markdown("""
### Rekomendasi

1. Prioritaskan penanganan cluster dengan biaya perbaikan tertinggi.

2. Tingkatkan proses inspeksi pada tahap produksi yang sering menghasilkan cacat kritis.

3. Gunakan hasil clustering sebagai dasar penjadwalan preventive maintenance.

4. Evaluasi metode inspeksi yang memiliki tingkat cacat tinggi.

5. Monitoring biaya perbaikan secara berkala untuk menekan kerugian produksi.
""")