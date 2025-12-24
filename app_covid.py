import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

# ======================
# KONFIGURASI HALAMAN
# ======================
st.set_page_config(
    page_title="Dashboard COVID-19",
    layout="wide"
)

st.title("📊 Dashboard Analisis & Prediksi COVID-19")
st.markdown(
    "**Indikator:** Kasus COVID-19 harian per 1 juta penduduk  \n"
    "**Sumber Data:** Our World in Data (OWID)"
)

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv(
        "daily-new-confirmed-covid-19-cases-per-million-people.csv"
    )
    df["Day"] = pd.to_datetime(df["Day"])
    return df

df = load_data()


# ======================
# SIDEBAR
# ======================
st.sidebar.header("⚙️ Pengaturan")

negara = st.sidebar.selectbox(
    "Pilih Negara",
    sorted(df["Entity"].unique())
)

periode_prediksi = st.sidebar.slider(
    "Periode Prediksi (hari)",
    min_value=7,
    max_value=60,
    value=30
)

# ======================
# FILTER DATA
# ======================
df_negara = df[df["Entity"] == negara].copy()

kolom_kasus = df.columns[-1]
df_negara = df_negara[["Day", kolom_kasus]].dropna()

# ======================
# METRIK UTAMA
# ======================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🌍 Negara", negara)

with col2:
    st.metric("📅 Jumlah Hari Data", len(df_negara))

with col3:
    st.metric(
        "📌 Nilai Terakhir",
        f"{df_negara[kolom_kasus].iloc[-1]:.2f}"
    )

# ======================
# GRAFIK TREN
# ======================
st.subheader("📈 Tren Kasus COVID-19")

fig = px.line(
    df_negara,
    x="Day",
    y=kolom_kasus,
    labels={
        "Day": "Tanggal",
        kolom_kasus: "Kasus per 1 Juta Penduduk"
    },
    title=f"Kasus COVID-19 Harian per 1 Juta Penduduk di {negara}"
)

st.plotly_chart(fig, use_container_width=True)

# ======================
# ANALISIS PREDIKSI (ARIMA)
# ======================
st.subheader("🔮 Prediksi Kasus COVID-19")

try:
    y = df_negara.set_index("Day")[kolom_kasus]

    # Model ARIMA
    model = ARIMA(y, order=(1, 1, 1))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=periode_prediksi)

    df_forecast = pd.DataFrame({
        "Tanggal": forecast.index,
        "Prediksi Kasus per 1 Juta": forecast.values
    })

    fig2 = px.line(
        df_forecast,
        x="Tanggal",
        y="Prediksi Kasus per 1 Juta",
        title=f"Prediksi {periode_prediksi} Hari ke Depan di {negara}"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.success("✅ Prediksi berhasil dibuat")

except Exception as e:
    st.error("❌ Data tidak cukup untuk dilakukan prediksi")
    st.text(e)

# ======================
# DATA MENTAH
# ======================
with st.expander("📄 Lihat Data Mentah"):
    st.dataframe(df_negara.tail(100))
