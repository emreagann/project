import streamlit as st
import pandas as pd
import numpy as np

# Linguistic variables and their corresponding T2NN values
linguistic_vars = {
    "VB": [0.2, 0.2, 0.1, 0.65, 0.8, 0.85, 0.45, 0.8, 0.7],
    "B": [0.35, 0.35, 0.1, 0.5, 0.75, 0.8, 0.5, 0.75, 0.65],
    "MB": [0.5, 0.3, 0.5, 0.5, 0.35, 0.45, 0.45, 0.3, 0.6],
    "M": [0.4, 0.45, 0.5, 0.4, 0.45, 0.5, 0.35, 0.4, 0.45],
    "MG": [0.6, 0.45, 0.5, 0.2, 0.15, 0.25, 0.1, 0.25, 0.15],
    "G": [0.7, 0.75, 0.8, 0.15, 0.2, 0.25, 0.1, 0.15, 0.2],
    "VG": [0.95, 0.9, 0.95, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05],
}

def score_function(values):
    alpha_alpha, alpha_beta, alpha_gamma, beta_alpha, beta_beta, beta_gamma, gamma_alpha, gamma_beta, gamma_gamma = values
    score = (1 / 12) * (
        8 * alpha_alpha + 2 * alpha_beta + alpha_gamma -
        beta_alpha - 2 * beta_beta - beta_gamma -
        gamma_alpha - 2 * gamma_beta - gamma_gamma
    )
    return score

def get_valid_numeric_values(value):
    numeric_values = linguistic_vars.get(str(value).strip())
    if numeric_values is not None:
        return score_function(numeric_values)
    return 0

def normalize_data(df, criteria_type):
    if criteria_type == 'benefit':
        return (df - df.min()) / (df.max() - df.min())
    elif criteria_type == 'cost':
        return (df.max() - df) / (df.max() - df.min())
    return df

def apply_weights(normalized_df, weights):
    if len(weights) != normalized_df.shape[1]:
        st.error("Ağırlıklar ve kriter sayısı uyuşmuyor!")
        return normalized_df
    return normalized_df * weights

def calculate_BAA(weighted_df):
    BAA = weighted_df.prod(axis=0) ** (1 / len(weighted_df))
    return BAA

def calculate_distances(weighted_df, BAA):
    diff = weighted_df.subtract(BAA, axis=1)
    squared_sum = (diff ** 2).sum(axis=1)
    distances = np.sqrt(squared_sum)
    return distances

st.title('T2NN MABAC Hesaplama Aracı')

uploaded_file = st.file_uploader("Excel dosyası yükleyin", type="xlsx")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, sheet_name='Alternatives')
    weights_df = pd.read_excel(uploaded_file, sheet_name='Weights')

    # İlk iki sütun A1 adı ve kriter adları olsun (örnek: A1, C1, C2...)
    alternatives = df.iloc[:, :2]
    criteria_values = df.iloc[:, 2:]  # DM1-DM4 verileri

    # Dilsel ifadeleri sayısala çevirip ortalamasını al
    num_criteria = criteria_values.shape[1] // 4
    decision_matrix = pd.DataFrame(index=alternatives.index)

    for i in range(num_criteria):
        group = criteria_values.iloc[:, i*4:(i+1)*4]
        numeric_scores = group.applymap(get_valid_numeric_values)
        decision_matrix[f'C{i+1}'] = numeric_scores.mean(axis=1)

    criteria_types = weights_df['Type'].tolist()
    weights = weights_df['Weight'].tolist()

    # Normalizasyon
    normalized_df = pd.DataFrame(index=decision_matrix.index)
    for i, col in enumerate(decision_matrix.columns):
        normalized_df[col] = normalize_data(decision_matrix[col], criteria_types[i])

    # Ağırlıklı normalizasyon
    weighted_df = apply_weights(normalized_df, weights)

    # MABAC hesaplamaları
    BAA = calculate_BAA(weighted_df)
    distances = calculate_distances(weighted_df, BAA)

    # Sonuçları göster
    alternatives['MABAC Score'] = distances
    alternatives['Rank'] = alternatives['MABAC Score'].rank(ascending=False)

    st.write("Karar Matrisi (Ortalamalar):")
    st.dataframe(decision_matrix)

    st.write("MABAC Skorları ve Sıralama:")
    st.dataframe(alternatives[['MABAC Score', 'Rank']])
