import streamlit as st
import pandas as pd
import numpy as np

# Linguistic T2NN definitions
linguistic_to_t2nn_alternatives = {
    "VB": ((0.20, 0.20, 0.10), (0.65, 0.80, 0.85), (0.45, 0.80, 0.70)),
    "B":  ((0.35, 0.35, 0.10), (0.50, 0.75, 0.80), (0.50, 0.75, 0.65)),
    "MB": ((0.50, 0.30, 0.50), (0.50, 0.35, 0.45), (0.45, 0.30, 0.60)),
    "M":  ((0.40, 0.45, 0.50), (0.40, 0.45, 0.50), (0.35, 0.40, 0.45)),
    "MG": ((0.60, 0.45, 0.50), (0.20, 0.15, 0.25), (0.10, 0.25, 0.15)),
    "G":  ((0.70, 0.75, 0.80), (0.15, 0.20, 0.25), (0.10, 0.15, 0.20)),
    "VG": ((0.95, 0.90, 0.95), (0.10, 0.10, 0.05), (0.05, 0.05, 0.05))
}

linguistic_to_t2nn_weights = {
    "L": ((0.20, 0.30, 0.20), (0.60, 0.70, 0.80), (0.45, 0.75, 0.75)),
    "ML": ((0.40, 0.30, 0.25), (0.45, 0.55, 0.40), (0.45, 0.60, 0.55)),
    "M": ((0.50, 0.55, 0.55), (0.40, 0.45, 0.55), (0.35, 0.40, 0.35)),
    "H": ((0.80, 0.75, 0.70), (0.20, 0.15, 0.30), (0.15, 0.10, 0.20)),
    "VH": ((0.90, 0.85, 0.95), (0.10, 0.15, 0.10), (0.05, 0.05, 0.10))
}

# Score function from Definition 4
def t2nn_score(T, I, F):
    return (1/12) * (
        8 +
        (T[0] + 2*T[1] + T[2]) -
        (I[0] + 2*I[1] + I[2]) -
        (F[0] + 2*F[1] + F[2])
    )

# Normalize scores
def normalize(scores, typ):
    scores = np.array(scores)
    if typ == 'benefit':
        return (scores - scores.min()) / (scores.max() - scores.min())
    else:  # cost
        return (scores.max() - scores) / (scores.max() - scores.min())

# MABAC steps
def weighted_matrix(norm_matrix, weights):
    return norm_matrix * weights

def border_area_calc(V):
    return np.prod(V, axis=0) ** (1 / V.shape[0])

def distance_matrix_calc(V, B):
    return V - B

def final_scores(D):
    return np.sum(D, axis=1)

# Streamlit App
st.title("MABAC Yöntemi – T2NN ile")

uploaded_file = st.file_uploader("Excel dosyasını yükleyin", type=["xlsx"])
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    data_df = pd.read_excel(xls, sheet_name=0, header=None)
    weight_df = pd.read_excel(xls, sheet_name=1, header=None)

    # Alternatif isimleri
    alternatives = data_df.iloc[:, 0].dropna().unique().tolist()

    # Kriterler
    criteria = data_df.iloc[1, 2:].tolist()

    # Kullanıcıdan kriter türlerini al
    st.subheader("Kriter türlerini belirtin (Benefit/Cost)")
    criterion_types = {}
    for c in criteria:
        criterion_types[c] = st.selectbox(f"{c} türü:", ["benefit", "cost"], key=c)

    # Ortalama skor matrisini oluştur (alternatif başına)
    score_matrix = []
    for alt in alternatives:
        alt_rows = data_df[data_df.iloc[:, 0] == alt]
        rows = data_df.loc[alt_rows.index[0]: alt_rows.index[0]+3, 2:]
        scores = []
        for col in rows.columns:
            terms = rows[col].tolist()
            t2nn_values = [linguistic_to_t2nn_alternatives[str(term)] for term in terms]
            avg_T = np.mean([x[0] for x in t2nn_values], axis=0)
            avg_I = np.mean([x[1] for x in t2nn_values], axis=0)
            avg_F = np.mean([x[2] for x in t2nn_values], axis=0)
            score = t2nn_score(avg_T, avg_I, avg_F)
            scores.append(score)
        score_matrix.append(scores)

    score_matrix = np.array(score_matrix)

    # Normalize et
    norm_matrix = []
    for i, c in enumerate(criteria):
        col = score_matrix[:, i]
        norm = normalize(col, criterion_types[c])
        norm_matrix.append(norm)
    norm_matrix = np.array(norm_matrix).T  # m x n

    # Ağırlıklar
    weight_scores = []
    for i, c in enumerate(criteria):
        dm_weights = weight_df.iloc[:, i+1].tolist()
        t2nns = [linguistic_to_t2nn_weights.get(str(w).strip(), ((0,0,0),(0,0,0),(0,0,0))) for w in dm_weights]
        avg_T = np.mean([x[0] for x in t2nns], axis=0)
        avg_I = np.mean([x[1] for x in t2nns], axis=0)
        avg_F = np.mean([x[2] for x in t2nns], axis=0)
        score = t2nn_score(avg_T, avg_I, avg_F)
        weight_scores.append(score)

    # MABAC işlemleri
    V = weighted_matrix(norm_matrix, weight_scores)
    B = border_area_calc(V)
    D = distance_matrix_calc(V, B)
    scores = final_scores(D)

    # Gösterim
    result_df = pd.DataFrame({"Alternatif": alternatives, "Skor": scores})
    result_df = result_df.sort_values(by="Skor", ascending=False)

    st.subheader("Sonuç: Alternatif Sıralaması")
    st.dataframe(result_df)
