import streamlit as st
import pandas as pd

# --- T2NN sözlükleri ---
alternative_linguistic_vars = {
    "VB": [0.20, 0.20, 0.10, 0.65, 0.80, 0.85, 0.45, 0.80, 0.70],
    "B":  [0.35, 0.35, 0.10, 0.50, 0.75, 0.80, 0.50, 0.75, 0.65],
    "MB": [0.50, 0.30, 0.50, 0.50, 0.35, 0.45, 0.45, 0.30, 0.60],
    "M":  [0.40, 0.45, 0.50, 0.40, 0.45, 0.50, 0.35, 0.40, 0.45],
    "MG": [0.60, 0.45, 0.50, 0.20, 0.15, 0.25, 0.10, 0.25, 0.15],
    "G":  [0.70, 0.75, 0.80, 0.15, 0.20, 0.25, 0.10, 0.15, 0.20],
    "VG": [0.95, 0.90, 0.95, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05],
}

weight_linguistic_vars = {
    "L":  [(0.20, 0.30, 0.20), (0.60, 0.70, 0.80), (0.45, 0.75, 0.75)],
    "ML": [(0.40, 0.30, 0.25), (0.45, 0.55, 0.40), (0.45, 0.60, 0.55)],
    "M":  [(0.50, 0.55, 0.55), (0.40, 0.45, 0.55), (0.35, 0.40, 0.35)],
    "H":  [(0.80, 0.75, 0.70), (0.20, 0.15, 0.30), (0.15, 0.10, 0.20)],
    "VH": [(0.90, 0.85, 0.95), (0.10, 0.15, 0.10), (0.05, 0.05, 0.10)],
}

# --- Yardımcı Fonksiyonlar ---

def score_function(values):
    a1, a2, a3, b1, b2, b3, g1, g2, g3 = values
    return (1 / 12) * (8 + (a1 + 2 * a2 + a3) - (b1 + 2 * b2 + b3) - (g1 + 2 * g2 + g3))

def score_from_merged_t2nn(t2nn):
    (a1, a2, a3), (b1, b2, b3), (g1, g2, g3) = t2nn
    return (1 / 12) * (8 + (a1 + 2*a2 + a3) - (b1 + 2*b2 + b3) - (g1 + 2*g2 + g3))

def get_t2nn_from_linguistic(value):
    if value in alternative_linguistic_vars:
        nums = alternative_linguistic_vars[value]
        return tuple(tuple(nums[i:i+3]) for i in range(0, 9, 3))
    return ((0,0,0), (0,0,0), (0,0,0))

def get_weight_t2nn_from_linguistic(value):
    return weight_linguistic_vars.get(value.strip(), ((0,0,0), (0,0,0), (0,0,0)))

def merge_t2nn_vectors(t2nn_list):
    n = len(t2nn_list)
    merged = []
    for i in range(3):  # T, I, F
        avg = tuple(sum(vec[i][j] for vec in t2nn_list) / n for j in range(3))
        merged.append(avg)
    return tuple(merged)

def t2nn_addition(a, b):
    return tuple(
        tuple(a[i][j] + b[i][j] for j in range(3)) for i in range(3)
    )

def combine_weights_t2nns(weight_list):
    combined = weight_list[0]
    for w in weight_list[1:]:
        combined = t2nn_addition(combined, w)
    return tuple(tuple(x / 4 for x in comp) for comp in combined)

# --- Streamlit Arayüzü ---

st.title("T2NN MABAC Alternatif ve Ağırlık Skorlama")

uploaded_file = st.file_uploader("Excel dosyanızı yükleyin (.xlsx)", type=["xlsx"])

if uploaded_file:
    # Dosyayı oku
    xls = pd.ExcelFile(uploaded_file)

    # Alternatives: MultiIndex index + MultiIndex columns
   # Alternatives sayfasını oku (header'ları al ama index'leme yapma)
    raw_df = pd.read_excel(xls, "Alternatives", header=[0, 1])

# İlk iki sütunu ("Alternative", "DM") alıp eksik yerleri yukarıdan doldur
    raw_df[['Alternative', 'DM']] = raw_df.iloc[:, :2].fillna(method='ffill')

# Veriyi ayır: Alternatif + DM kolonlarını index yap, geri kalanlar skorlar
    data = raw_df.drop(columns=['Alternative', 'DM'])
    data.index = pd.MultiIndex.from_frame(raw_df[['Alternative', 'DM']])
    data.columns.names = ['Criteria', 'DM']

    # Sonuç olarak bu bizim asıl alternatif matrisimiz olacak
    alt_df = data

    
    # Weights: Kriter isimleri satır, DM'ler sütun
    wt_df = pd.read_excel(xls, "Weights", index_col=0)

    st.subheader("Yüklenen Alternatif Verileri")
    st.dataframe(alt_df.reset_index())  # index'i göstererek daha okunabilir hale getiriyoruz

    st.subheader("Yüklenen Ağırlık Verileri")
    st.dataframe(wt_df)

    # Alternatif skor matrisini hesapla
    alternatives = alt_df.index.get_level_values(0).unique()
    criteria = alt_df.columns.get_level_values(0).unique()
    decision_makers = alt_df.columns.get_level_values(1).unique()

    alt_scores = pd.DataFrame(index=alternatives, columns=criteria)

    for alt in alternatives:
        for crit in criteria:
            t2nns = []
            for dm in decision_makers:
                val = alt_df.loc[(alt, dm), (crit, dm)]
                t2nns.append(get_t2nn_from_linguistic(val))
            merged = merge_t2nn_vectors(t2nns)
            score = score_from_merged_t2nn(merged)
            alt_scores.loc[alt, crit] = round(score, 4)

    st.subheader("Ortalama Karar Matrisi (Skorlar)")
    st.dataframe(alt_scores)

    # Ağırlık skorlarını hesapla
    weight_scores = pd.Series(index=wt_df.index, dtype=float)
    for crit in wt_df.index:
        weight_list = [get_weight_t2nn_from_linguistic(wt_df.loc[crit, dm]) for dm in wt_df.columns]
        combined = combine_weights_t2nns(weight_list)
        weight_scores[crit] = round(score_from_merged_t2nn(combined), 4)

    st.subheader("Kriter Ağırlıkları (Skorlar)")
    st.dataframe(weight_scores)
