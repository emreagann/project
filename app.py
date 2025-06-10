import streamlit as st
import pandas as pd
import numpy as np

# Skor fonksiyonu
def t2nn_score(T, I, F):
    return (1/12) * ((8 + (T[0] + 2*T[1] + T[2]) - (I[0] + 2*I[1] + I[2]) - (F[0] + 2*F[1] + F[2])))

# Normalize fonksiyonu
def normalize(scores, typ):
    scores = np.array(scores)
    if scores.max() == scores.min():
        return np.ones_like(scores)
    if typ == "benefit":
        return (scores - scores.min()) / (scores.max() - scores.min())
    else:
        return (scores.max() - scores) / (scores.max() - scores.min())

# MABAC fonksiyonları
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

# Başlangıç
st.title("MABAC Yöntemi – T2NN ile Otomatik Excel Okuma")

uploaded_file = st.file_uploader("Excel dosyasını yükleyin", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)

    data_df = pd.read_excel(xls, sheet_name="Alternatives", header=None)
    weight_df = pd.read_excel(xls, sheet_name="Weights", header=None)
    types_df = pd.read_excel(xls, sheet_name="Types")
    ling_df = pd.read_excel(xls, sheet_name="Linguistic")

    # 1. Alternatifler
    criteria = [f"C{i+1}" for i in range(18)]
    alternatives = []
    alt_indices = []
    for i, val in enumerate(data_df.iloc[:, 0]):
        if isinstance(val, str) and val.strip().startswith("A"):
            name = val.strip(":")
            if name.lower() != "alternatives":
                alternatives.append(name)
                alt_indices.append(i)

    # 2. Kriter türleri
    criterion_types = {
        row["Criterion"]: row["Type"].strip().lower()
        for _, row in types_df.iterrows()
    }

    # 3. Linguistic sözlüklerini kur
    alt_dict = {}
    weight_dict = {}

    for _, row in ling_df.iterrows():
        name = row["Label"].strip()
        T = (row["T1"], row["T2"], row["T3"])
        I = (row["I1"], row["I2"], row["I3"])
        F = (row["F1"], row["F2"], row["F3"])
        if row["Use"] == "alternative":
            alt_dict[name] = (T, I, F)
        elif row["Use"] == "weight":
            weight_dict[name] = (T, I, F)

    # 4. Alternatif skor matrisini oluştur
    score_matrix = []
    for idx in alt_indices:
        if idx + 3 >= len(data_df):  # 4 karar verici gerekiyor
            st.warning(f"{alternatives[alt_indices.index(idx)]} eksik satır, atlandı.")
            continue
        rows = data_df.iloc[idx:idx+4, 2:]
        scores = []
        for col in rows.columns:
            terms = rows[col].tolist()
            t2nn_values = [
                alt_dict.get(str(term).strip())
                for term in terms
                if pd.notna(term) and alt_dict.get(str(term).strip()) is not None
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

    # 5. Normalize et
    norm_matrix = []
    for i, c in enumerate(criteria):
        col = score_matrix[:, i]
        norm = normalize(col, criterion_types[c])
        norm_matrix.append(norm)
    norm_matrix = np.array(norm_matrix).T

    # 6. Ağırlıkları hesapla
    weight_scores = []
    for i, c in enumerate(criteria):
        weights = weight_df.iloc[:, i+1].tolist()
        t2nns = [weight_dict.get(str(w).strip(), ((0,0,0), (0,0,0), (0,0,0))) for w in weights]
        avg_T = np.mean([x[0] for x in t2nns], axis=0)
        avg_I = np.mean([x[1] for x in t2nns], axis=0)
        avg_F = np.mean([x[2] for x in t2nns], axis=0)
        score = t2nn_score(avg_T, avg_I, avg_F)
        weight_scores.append(score)

    # 7. MABAC
    V = weighted_matrix(norm_matrix, weight_scores)
    B = border_area_calc(V)
    D = distance_matrix_calc(V, B)
    scores = final_scores(D)
    theta = scores / np.sum(scores)

    # 8. Sonuç
    result_df = pd.DataFrame({
        "Alternatif": alternatives,
        "MABAC Skor": scores.round(4),
        "Normalize Skor": theta.round(4)
    }).sort_values(by="MABAC Skor", ascending=False)

    st.subheader("Sonuçlar – Alternatif Sıralaması")
    st.dataframe(result_df.reset_index(drop=True))
