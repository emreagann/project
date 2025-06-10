import streamlit as st
import pandas as pd
import numpy as np

# --------------------- Fonksiyonlar ---------------------

# T2NN skor formülü
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

# MABAC işlemleri
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

# Excel'den linguistic ifadeleri okuyan fonksiyon
def get_linguistic_t2nn_dict(df):
    result = {}
    for i in range(len(df)):
        label = str(df.iloc[i, 0]).strip()
        try:
            T = tuple(map(float, df.iloc[i, 1:4]))
            I = tuple(map(float, df.iloc[i, 4:7]))
            F = tuple(map(float, df.iloc[i, 7:10]))
            result[label] = (T, I, F)
        except:
            continue
    return result

# --------------------- Arayüz ---------------------

st.title("MABAC Yöntemi – Type-2 Neutrosophic Sayılarla")

uploaded_file = st.file_uploader("Excel dosyasını yükleyin", type=["xlsx"])
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)

    # Sayfaları oku
    data_df = pd.read_excel(xls, sheet_name="Alternatives", header=None)
    weight_df = pd.read_excel(xls, sheet_name="Criteria Weights", header=None)

    # Linguistic T2NN tablolarını oku
    alt_ling_df = pd.read_excel(xls, sheet_name="Alternatives", skiprows=30)
    weight_ling_df = pd.read_excel(xls, sheet_name="Criteria Weights", skiprows=6, nrows=5)

    linguistic_to_t2nn_alternatives = get_linguistic_t2nn_dict(alt_ling_df)
    linguistic_to_t2nn_weights = get_linguistic_t2nn_dict(weight_ling_df)

    # Kriter adları
    criteria = [f"C{i+1}" for i in range(18)]

    # Alternatifler ve indeksler
    alternatives = []
    alt_indices = []
    for i, val in enumerate(data_df.iloc[:, 0]):
        if isinstance(val, str) and val.strip().startswith("A") and val.lower() != "alternatives":
            alternatives.append(val.strip(":"))
            alt_indices.append(i)

    # Kriter türlerini Excel'den al
    criterion_type_row = weight_df[weight_df.iloc[:, 0] == "Type"].iloc[0]
    criterion_types = {}
    for i, c in enumerate(criteria):
        typ = criterion_type_row[i + 1]  # İlk sütun başlık
        criterion_types[c] = str(typ).strip().lower()

    # T2NN skor matrisini oluştur
   score_matrix = []
for idx in alt_indices:
    # Son satırdan sonra 3 satır daha var mı kontrol et
    if idx + 3 >= len(data_df):
        st.warning(f"{alternatives[alt_indices.index(idx)]} için yeterli karar verici satırı yok. Atlandı.")
        continue

    rows = data_df.iloc[idx:idx+4, 2:]  # 4 karar verici verisi alınıyor
    scores = []

    for col in rows.columns:
        terms = rows[col].tolist()
        t2nn_values = [
            linguistic_to_t2nn_alternatives.get(str(term).strip())
            for term in terms
            if pd.notna(term) and linguistic_to_t2nn_alternatives.get(str(term).strip()) is not None
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

    # Normalize et
    norm_matrix = []
    for i, c in enumerate(criteria):
        col = score_matrix[:, i]
        norm = normalize(col, criterion_types[c])
        norm_matrix.append(norm)
    norm_matrix = np.array(norm_matrix).T

    # Ağırlık skorları
    weight_scores = []
    for i, c in enumerate(criteria):
        weights = weight_df.iloc[0:4, i+1].tolist()
        t2nns = [
            linguistic_to_t2nn_weights.get(str(w).strip(), ((0,0,0), (0,0,0), (0,0,0)))
            for w in weights
        ]
        avg_T = np.mean([x[0] for x in t2nns], axis=0)
        avg_I = np.mean([x[1] for x in t2nns], axis=0)
        avg_F = np.mean([x[2] for x in t2nns], axis=0)
        score = t2nn_score(avg_T, avg_I, avg_F)
        weight_scores.append(score)

    # MABAC
    V = weighted_matrix(norm_matrix, weight_scores)
    B = border_area_calc(V)
    D = distance_matrix_calc(V, B)
    scores = final_scores(D)
    theta_scores = scores / np.sum(scores)

    # Sonuç
    result_df = pd.DataFrame({
        "Alternatif": alternatives,
        "Skor": scores.round(4),
        "Normalize Skor": theta_scores.round(4)
    }).sort_values(by="Skor", ascending=False)

    st.subheader("Sonuç: Alternatif Sıralaması")
    st.dataframe(result_df.reset_index(drop=True))
