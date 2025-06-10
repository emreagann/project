import streamlit as st
import pandas as pd
import numpy as np

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

# MABAC: Normalize and compute BAA (Border Approximation Area)
def normalize_data(df, criteria_type):
    if criteria_type == 'benefit':
        return (df - df.min()) / (df.max() - df.min())
    elif criteria_type == 'cost':
        return (df.max() - df) / (df.max() - df.min())
    return df

def calculate_BAA(normalized_df):
    # Convert DataFrame to numeric if it's not
    return normalized_df.apply(pd.to_numeric, errors='coerce').prod(axis=0)**(1/len(normalized_df))

def calculate_distances(normalized_df, BAA):
    # Ensure the input is a numeric array or DataFrame
    normalized_df = normalized_df.apply(pd.to_numeric, errors='coerce')
    return np.linalg.norm(normalized_df - BAA, axis=1)

# Uygulama
st.title('Dilsel Değer Dönüşümü ve Skor Hesaplama')

uploaded_file = st.file_uploader("Excel Dosyasını Yükle", type="xlsx")

if uploaded_file is not None:
    # Excel dosyasını oku
    df = pd.read_excel(uploaded_file, sheet_name='Alternatives')
    weights_df = pd.read_excel(uploaded_file, sheet_name='Weights')  # Weights sayfasını oku
    alternatives_values = df.iloc[:, 2:]  # Dilsel değerler

    # Weights sayfasından kriter türlerini al
    criteria_types = weights_df['Type'].tolist()  # 'Type' sütununda kriter türlerini al

    # Her bir dilsel değeri sayısal değerlere dönüştür
    valid_numeric_values_df = alternatives_values.applymap(lambda x: get_valid_numeric_values(x))

    # Normalize et ve MABAC için BAA hesapla
    normalized_df = pd.DataFrame()

    # Kriter türlerine göre her bir kriteri normalize et
    for i, criteria_type in enumerate(criteria_types):
        # İlgili kriter türünü al ve normalize et
        normalized_df.iloc[:, i] = normalize_data(valid_numeric_values_df.iloc[:, i], criteria_type)

    # BAA hesapla
    BAA = calculate_BAA(normalized_df)

    # Mesafeleri hesapla
    distances = calculate_distances(normalized_df, BAA)

    # Sonuçları dataframe'e ekle
    df['MABAC Score'] = distances

    # Alternatifleri MABAC skoruna göre sırala
    df['Rank'] = df['MABAC Score'].rank(ascending=False)  # Skorları büyükten küçüğe sırala

    # Skorları ve sıralamayı göster
    st.write("MABAC Skorları ve Sıralama")
    st.dataframe(df[['MABAC Score', 'Rank']])
