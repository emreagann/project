import streamlit as st
import pandas as pd
import numpy as np



def t2nn_score(T, I, F):
    return (1/12) * ((8 + (T[0] + 2*T[1] + T[2]) - (I[0] + 2*I[1] + I[2]) - (F[0] + 2*F[1] + F[2])))

def normalize(scores, typ):
    scores = np.array(scores)
    if scores.max() == scores.min():
        return np.ones_like(scores)
    if typ == "benefit":
        return (scores - scores.min()) / (scores.max() - scores.min())
    else:
        return (scores.max() - scores) / (scores.max() - scores.min())

def weighted_matrix(norm_matrix, weights):
    return norm_matrix * weights

def border_area_calc(V):
    if V.shape[0] == 0:
        raise ValueError("Alternatif sayısı sıfır.")
    return np.prod(V, axis=0) ** (1 / V.shape[0])

def distance_matrix_calc(V, B):
    return V - B

def final_scores(D):
    return np.sum(D, axis=1)

def load_linguistic_t2nn_dict(df, label_col=0, start_col=1):
    t2nn_dict = {}
    for i in range(len(df)):
        label = str(df.iloc[i, label_col]).strip().upper()
        try:
            T = tuple(map(float, df.iloc[i, start_col:start_col+3]))
            I = tuple(map(float, df.iloc[i, start_col+3:start_col+6]))
            F = tuple(map(float, df.iloc[i, start_col+6:start_col+9]))
            t2nn_dict[label] = (T, I, F)
        except:
            continue
    return t2nn_dict

# ---------------------------
# Streamlit Başlangıç
# ---------------------------

st.title("MABAC Yöntemi – Type-2 Neutrosophic Sayılarla")

uploaded_file = st.file_uploader("Excel dosyasını yükleyin", type=["xlsx"])
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)

    data_df = pd.read_excel(xls, sheet_name="Alternatives", header=None)
    weight_df = pd.read_excel(xls, sheet_name="Weights", header=None)

    # --- Linguistic Tabloları Yükle ---
    alt_ling_table = data_df.iloc[-7:, :]  # Alternatives sayfasının en altındaki T2NN tablosu
    linguistic_to_t2nn_alternatives = load_linguistic_t2nn_dict(alt_ling_table, label_col=0, start_col=1)

    weight_ling_table = weight_df.iloc[2:7, :]  # Weights sayfasında VH–L tablosu
    linguistic_to_t2nn_weights = load_linguistic_t2nn_dict(weight_ling_table, label_col=2, start_col=3)

    # --- Kriter Türlerini Al ---
    criteria_types_raw = weight_df.iloc[10:28, :2]  # C1–C18 + tür
    criterion_types = {
        str(row[0]).strip(): str(row[1]).strip().lower()
        for _, row in criteria_types_raw.iterrows()
    }

    # --- Alternatifleri ve indekslerini al ---
    criteria = [f"C{i+1}" for i in range(18)]
    alternatives = []
    alt_indices = []
    for i, val in enumerate(data_df.iloc[:, 0]):
        if isinstance(val, str) and val.strip().startswith("A"):
            name = val.strip(":")
            if name.lower() != "alternatives":
                alternatives.append(name)
                alt_indices.append(i)

    # --- Skor Matrisi Hesapla ---
    score_matrix = []
    for idx in alt_indices:
        if idx + 3 >= len(data_df):
            continue
        rows = data_df.iloc[idx:idx+4, 2:]
        scores = []
        for col in rows.columns:
            terms = rows[col].tolist()
            t2nn_values = [
                linguistic_to_t2nn_alternatives.get(str(term).strip().upper())
                for term in terms
                if pd.notna(term) and str(term).strip().upper() in linguistic_to_t2nn_alternatives
            ]
            if len(t2nn_values) > 0:
                avg_T = np.mean([x[0] for x in t2nn_values], axis=0)
                avg_I = np.mean([x[1] for x in t2nn_values], axis=0)
                avg_F = np.mean([x[2] for x in t2nn_values], axis=0)
                score = t2nn_score(avg_T, avg_I, avg_F)
            else:
                score = 0
            scores.append(score)
        score_matrix.append(scores)

    score_matrix = np.array(score_matrix)

    # --- Normalize Et ---
    norm_matrix = []
    for i, c in enumerate(criteria):
        col = score_matrix[:, i]
        typ = criterion_types.get(c, "benefit")  # default: benefit
        norm = normalize(col, typ)
        norm_matrix.append(norm)
    norm_matrix = np.array(norm_matrix).T

    # --- Ağırlık Hesapla ---
    weight_scores = []
    for i, c in enumerate(criteria):
        weights = weight_df.iloc[0:4, i+2].tolist()  # DM1–DM4 satırları, 2. sütundan başla
        t2nns = [
            linguistic_to_t2nn_weights.get(str(w).strip().upper(), ((0,0,0), (0,0,0), (0,0,0)))
            for w in weights
        ]
        avg_T = np.mean([x[0] for x in t2nns], axis=0)
        avg_I = np.mean([x[1] for x in t2nns], axis=0)
        avg_F = np.mean([x[2] for x in t2nns], axis=0)
        score = t2nn_score(avg_T, avg_I, avg_F)
        weight_scores.append(score)

    # --- MABAC ---
    V = weighted_matrix(norm_matrix, weight_scores)
    B = border_area_calc(V)
    D = distance_matrix_calc(V, B)
    scores = final_scores(D)
    theta_scores = scores / np.sum(scores)

    # --- Sonuç Göster ---
    result_df = pd.DataFrame({
        "Alternatif": alternatives,
        "Skor": scores.round(4),
        "Normalize Skor": theta_scores.round(4)
    }).sort_values(by="Skor", ascending=False)

    st.subheader("Sonuç: Alternatif Sıralaması")
    st.dataframe(result_df.reset_index(drop=True))
