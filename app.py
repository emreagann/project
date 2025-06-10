import streamlit as st
import pandas as pd

# Dilsel Değerler ve Katsayılar
linguistic_vars = {
    "VB": [0.2, 0.2, 0.1, 0.65, 0.8, 0.85, 0.45, 0.8, 0.7],
    "B": [0.35, 0.35, 0.1, 0.5, 0.75, 0.8, 0.5, 0.75, 0.65],
    "MB": [0.5, 0.3, 0.5, 0.5, 0.35, 0.45, 0.45, 0.3, 0.6],
    "M": [0.4, 0.45, 0.5, 0.4, 0.45, 0.5, 0.35, 0.4, 0.45],
    "MG": [0.6, 0.45, 0.5, 0.2, 0.15, 0.25, 0.1, 0.25, 0.15],
    "G": [0.7, 0.75, 0.8, 0.15, 0.2, 0.25, 0.1, 0.15, 0.2],
    "VG": [0.95, 0.9, 0.95, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05],
}

# Skor fonksiyonu
def score_function(values):
    alpha_alpha, alpha_beta, alpha_gamma, beta_alpha, beta_beta, beta_gamma, gamma_alpha, gamma_beta, gamma_gamma = values
    score = (1 / 12) * (
        8 * alpha_alpha + 2 * alpha_beta + alpha_gamma - 
        beta_alpha - 2 * beta_beta - beta_gamma - 
        gamma_alpha - 2 * gamma_beta - gamma_gamma
    )
    return score

# Sayısal değerlere dönüştürme
def get_valid_numeric_values(value):
    numeric_values = linguistic_vars.get(value)
    if numeric_values is not None:
        return numeric_values
    return [0] * 9  # Geçersiz bir değer geldiğinde sıfırlarla doldurulacak

# Uygulama
st.title('Dilsel Değer Dönüşümü ve Skor Hesaplama')

uploaded_file = st.file_uploader("Excel Dosyasını Yükle", type="xlsx")

if uploaded_file is not None:
    # Excel dosyasını oku
    df = pd.read_excel(uploaded_file, sheet_name='Alternatives')
    alternatives_values = df.iloc[:, 2:]

    # Her bir dilsel değeri sayısal değerlere dönüştür
    valid_numeric_values_df = alternatives_values.applymap(lambda x: get_valid_numeric_values(x))

    # Skor hesaplama işlemi
    scores_df = valid_numeric_values_df.applymap(score_function)

    # Decision Makers sayısına göre ortalama hesaplama
    dm_count = len(df)  # Karar verici sayısını alıyoruz
    scores_df['Average'] = scores_df.mean(axis=1)

    # Skorları görüntüle
    st.write("Hesaplanan Skorlar (Ortalama alınmış)")
    st.dataframe(scores_df)
