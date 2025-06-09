import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dilsel terimlerin T2NN karşılıkları (alternatifler)
linguistic_to_t2nn_alternatives = {
    "VB": ((0.20, 0.20, 0.10), (0.65, 0.80, 0.85), (0.45, 0.80, 0.70)),
    "B": ((0.35, 0.35, 0.10), (0.50, 0.75, 0.80), (0.50, 0.75, 0.65)),
    "MB": ((0.50, 0.30, 0.50), (0.50, 0.35, 0.45), (0.45, 0.30, 0.60)),
    "M": ((0.40, 0.45, 0.50), (0.40, 0.45, 0.50), (0.35, 0.40, 0.45)),
    "MG": ((0.60, 0.45, 0.50), (0.20, 0.15, 0.25), (0.10, 0.25, 0.15)),
    "G": ((0.70, 0.75, 0.80), (0.15, 0.20, 0.25), (0.10, 0.15, 0.20)),
    "VG": ((0.95, 0.90, 0.95), (0.10, 0.10, 0.05), (0.05, 0.05, 0.05))
}

# Dilsel terimlerin T2NN karşılıkları (ağırlıklar)
linguistic_to_t2nn_weights = {
    "L": ((0.20, 0.30, 0.20), (0.60, 0.70, 0.80), (0.45, 0.75, 0.75)),
    "ML": ((0.40, 0.30, 0.25), (0.45, 0.55, 0.40), (0.45, 0.60, 0.55)),
    "M": ((0.50, 0.55, 0.55), (0.40, 0.45, 0.55), (0.35, 0.40, 0.35)),
    "H": ((0.80, 0.75, 0.70), (0.20, 0.15, 0.30), (0.15, 0.10, 0.20)),
    "VH": ((0.90, 0.85, 0.95), (0.10, 0.15, 0.10), (0.05, 0.05, 0.10))
}

# Skor fonksiyonu
def t2nn_score(t2nn):
    T, I, F = t2nn
    return (1 / 12) * (
        8 + T[0] + 2 * T[1] + T[2]
        - I[0] - 2 * I[1] - I[2]
        - F[0] - 2 * F[1] - F[2]
    )

# Normalize et
def normalize(values, type_):
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return [1.0] * len(values)
    if type_ == "benefit":
        return [(v - min_v) / (max_v - min_v) for v in values]
    else:
        return [(max_v - v) / (max_v - min_v) for v in values]

# Ağırlıklı matris
def weighted_matrix(norm_matrix, weights):
    return np.array(norm_matrix) * np.array(weights)

def border_area_calc(V):
    return np.prod(V, axis=0) ** (1 / V.shape[0])

def distance_matrix_calc(V, B):
    return V - B

def final_scores(D):
    return np.sum(D, axis=1)

st.title("T2NN MABAC Yöntemi Uygulaması")
uploaded_file = st.file_uploader("Excel dosyasını yükleyin", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    data_df = pd.read_excel(xls, sheet_name=0, header=None)
    weight_df = pd.read_excel(xls, sheet_name=1, header=None)
    criteria = [c for c in data_df.columns if isinstance(c, str) and c.startswith("C")]
    alternatives = data_df.iloc[:, 0].dropna().unique().tolist()

    # Kullanıcıdan kriter türü seçimi
    st.subheader("Kriter Türlerini Seçin (Benefit veya Cost)")
    criteria_types = {}
    for crit in criteria:
        criteria_types[crit] = st.selectbox(f"{crit} türü:", ["benefit", "cost"], key=f"type_{crit}")

    # Alternatifler için T2NN skoru hesapla (4 karar vericinin ortalaması)
    alt_scores = []
    for alt in alternatives:
        alt_rows = data_df[data_df[0] == alt].iloc[:, 2:]
        alt_scores.append([
            np.mean([
                t2nn_score(linguistic_to_t2nn_alternatives[val])
                for val in alt_rows[c] if val in linguistic_to_t2nn_alternatives
            ]) for c in alt_rows.columns
        ])

    # Kriter ağırlıkları için T2NN skoru (4 uzman ortalaması)
    weight_matrix = weight_df.iloc[:, 1:]
    weight_scores = [
        np.mean([
            t2nn_score(linguistic_to_t2nn_weights[val])
            for val in weight_matrix[c] if val in linguistic_to_t2nn_weights
        ]) for c in weight_matrix.columns
    ]

    # Normalize et
    norm_matrix = []
    for j, crit in enumerate(criteria):
        col = [row[j] for row in alt_scores]
        norm_matrix.append(normalize(col, criteria_types[crit]))
    norm_matrix = np.array(norm_matrix).T

    # Ağırlıklı
    V = weighted_matrix(norm_matrix, weight_scores)
    B = border_area_calc(V)
    D = distance_matrix_calc(V, B)
    scores = final_scores(D)

    result_df = pd.DataFrame({"Alternatif": alternatives, "Skor": scores})
    result_df = result_df.sort_values(by="Skor", ascending=False)

    st.subheader("Sonuç: Alternatif Sıralaması")
    st.dataframe(result_df)

    st.subheader("Skor Dağılımı")
    fig, ax = plt.subplots()
    ax.bar(result_df['Alternatif'].astype(str), result_df['Skor'])
    st.pyplot(fig)
