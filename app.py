import streamlit as st
import pandas as pd
import numpy as np

# Güncellenmiş linguistic_vars (Tablo A.2 - alternatif değerlendirme)
linguistic_vars = {
    "VB": [0.20, 0.20, 0.10, 0.65, 0.80, 0.85, 0.45, 0.80, 0.70],
    "B":  [0.35, 0.35, 0.10, 0.50, 0.75, 0.80, 0.50, 0.75, 0.65],
    "MB": [0.50, 0.30, 0.50, 0.50, 0.35, 0.45, 0.45, 0.30, 0.60],
    "M":  [0.40, 0.45, 0.50, 0.40, 0.45, 0.50, 0.35, 0.40, 0.45],
    "MG": [0.60, 0.45, 0.50, 0.20, 0.15, 0.25, 0.10, 0.25, 0.15],
    "G":  [0.70, 0.75, 0.80, 0.15, 0.20, 0.25, 0.10, 0.15, 0.20],
    "VG": [0.95, 0.90, 0.95, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05],
}

def score_function(values):
    a1, a2, a3, b1, b2, b3, g1, g2, g3 = values
    return (1 / 12) * (8 + (a1 + 2 * a2 + a3) - (b1 + 2 * b2 + b3) - (g1 + 2 * g2 + g3))


def get_valid_numeric_values(value):
    value = str(value).strip()
    return score_function(linguistic_vars[value]) if value in linguistic_vars else 0

def normalize_data(series, criteria_type):
    if series.max() == series.min():
        return 0
    if criteria_type.lower() == 'benefit':
        return (series - series.min()) / (series.max() - series.min())
    elif criteria_type.lower() == 'cost':
        return (series.max() - series) / (series.max() - series.min())
    return series

def apply_weights(normalized_df, weights):
    return normalized_df.multiply(weights, axis=1)

def calculate_BAA(weighted_df):
    return weighted_df.prod(axis=0) ** (1 / len(weighted_df))

def calculate_difference_matrix(weighted_df, BAA):
    return weighted_df - BAA

def calculate_scores(diff_df):
    return diff_df.sum(axis=1)

st.title("T2NN + MABAC Karar Destek Sistemi")

uploaded_file = st.file_uploader("Excel dosyasını yükleyin", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="Alternatives")
    weights_df = pd.read_excel(uploaded_file, sheet_name="Weights")

    n_criteria = len(weights_df)
    n_dms = 4
    n_alternatives = len(df) // n_dms
    criteria_names = weights_df["Criteria No"].tolist()

    final_matrix = pd.DataFrame(columns=criteria_names)
    alternatives = []

    for i in range(n_alternatives):
        group = df.iloc[i * n_dms:(i + 1) * n_dms][criteria_names]
        scored = group.applymap(get_valid_numeric_values)
        avg_scores = scored.mean(axis=0)
        final_matrix.loc[i] = avg_scores
        alt_name = df.iloc[i * n_dms, 0]
        alternatives.append(alt_name.strip() if isinstance(alt_name, str) else f"A{i+1}")

    final_matrix.index = alternatives

    # Normalizasyon
    normalized_df = pd.DataFrame(index=final_matrix.index)
    for i, col in enumerate(final_matrix.columns):
        normalized_df[col] = normalize_data(final_matrix[col], weights_df.iloc[i]["Type"])

    # Ağırlıklandırma
    weights = weights_df["Weight"].values
    weighted_df = apply_weights(normalized_df, weights)

    # MABAC hesaplama
    BAA = calculate_BAA(weighted_df)
    difference_df = calculate_difference_matrix(weighted_df, BAA)
    scores = calculate_scores(difference_df)

    result_df = pd.DataFrame({
        "Alternative": final_matrix.index,
        "MABAC Score": scores,
        "Rank": scores.rank(ascending=False).astype(int)
    }).sort_values(by="Rank")

    # Gösterimler
    st.subheader("1. Karar Matrisi (T2NN Ortalama Skorlar):")
    st.dataframe(final_matrix)

    st.subheader("2. Normalize Matris:")
    st.dataframe(normalized_df)

    st.subheader("3. Ağırlıklı Normalize Matris:")
    st.dataframe(weighted_df)

    st.subheader("4. Border Approximation Area (BAA):")
    st.write(BAA)

    st.subheader("5. Mesafe Matrisi (V - BAA):")
    st.dataframe(difference_df)

    st.subheader("6. MABAC Skorları ve Sıralama:")
    st.dataframe(result_df)

    st.success("Hesaplamalar tamamlandı. Tüm matrisler gösterildi.")
