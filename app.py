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
    if pd.isna(value):
        return ((0, 0, 0), (0, 0, 0), (0, 0, 0))
    return tuple(tuple(alternative_linguistic_vars.get(value.strip(), [0]*9)[i:i+3]) for i in range(0, 9, 3))

def get_weight_t2nn_from_linguistic(value):
    if pd.isna(value):
        return ((0, 0, 0), (0, 0, 0), (0, 0, 0))
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
    return tuple(tuple(x / len(weight_list) for x in comp) for comp in combined)

def zero_out_I_and_F(t2nn):
    T, _, _ = t2nn
    zero = (0.0, 0.0, 0.0)
    return (T, zero, zero)
def combine_multiple_decision_makers(alt_df, decision_makers, criteria, alternatives):
    combined_results = {}

    # Her alternatif ve kriter için karar vericilerin birleşimini hesapla
    for alt in alternatives:
        for crit in criteria:
            t2nns = []
            
            # Tüm karar vericiler için T2NN'leri al
            for dm in decision_makers:
                try:
                    val = alt_df.loc[(alt, dm), (crit, dm)]
                except KeyError:
                    val = None
                t2nns.append(get_t2nn_from_linguistic(val))
            
            # Karar vericiler arasındaki birleşimi yap
            merged_t2nn = merge_t2nn_vectors(t2nns)  # Bu fonksiyon ⊕ işlemini yapar

            # Skoru hesapla
            score = score_from_merged_t2nn(merged_t2nn)
            combined_results[(alt, crit)] = round(score, 4)
    
    return combined_results

# --- Streamlit Arayüzü ---
st.title("T2NN MABAC Alternatif ve Ağırlık Skorlama")

uploaded_file = st.file_uploader("Excel dosyanızı yükleyin (.xlsx)", type=["xlsx"])

if uploaded_file:
    # Dosyayı oku
    xls = pd.ExcelFile(uploaded_file)

    # Alternatives sayfasını oku
    raw_df = pd.read_excel(xls, "Alternatives", header=[0, 1])
    raw_df[['Alternative', 'DM']] = raw_df.iloc[:, :2].fillna(method='ffill')
    data = raw_df.drop(columns=['Alternative', 'DM'])
    data.index = pd.MultiIndex.from_frame(raw_df[['Alternative', 'DM']])
    data.columns.names = ['Criteria', 'DM']
    alt_df = data

    # Weights sayfasını oku
    wt_df = pd.read_excel(xls, "Weights", index_col=0)
    wt_df = wt_df[wt_df.index.notna()]  # NaN index varsa çıkar

    # --- GÖRSELLER ---
    st.subheader("Yüklenen Alternatif Verileri")
    st.dataframe(alt_df.reset_index())

    st.subheader("Yüklenen Ağırlık Verileri")
    st.dataframe(wt_df)

    # --- Alternatif skorlarını hesapla ---
    alternatives = alt_df.index.get_level_values(0).unique()
    criteria = alt_df.columns.get_level_values(0).unique()
    decision_makers = alt_df.columns.get_level_values(1).unique()

    # Alternatif skorlarını hesapla
    alt_scores = pd.DataFrame(index=alternatives, columns=criteria)

    for alt in alternatives:
        for crit in criteria:
            t2nns = []
            
            # Her DM için T2NN değerlerini alıyoruz
            for dm in decision_makers:
                try:
                    val = alt_df.loc[(alt, dm), (crit, dm)]
                except KeyError:
                    val = None
                t2nns.append(get_t2nn_from_linguistic(val))
            
            # T2NN'leri birleştir
            merged_t2nn = merge_t2nn_vectors(t2nns)

            # Skoru hesapla
            score = score_from_merged_t2nn(merged_t2nn)
            alt_scores.loc[alt, crit] = round(score, 4)

    st.subheader("Ortalama Karar Matrisi (Skorlar)")
    st.dataframe(alt_scores)

    # --- Ağırlık skorlarını hesapla ---
    weight_scores = pd.Series(index=wt_df.index, dtype=float)

    # Genişletilmiş tablo: T, I, F + skor + normalize
    detailed_weights = pd.DataFrame(columns=[
        'T1', 'T2', 'T3', 'I1', 'I2', 'I3', 'F1', 'F2', 'F3', 'Score', 'Normalized'
    ], index=wt_df.index)

    for crit in wt_df.index:
        weight_list = [get_weight_t2nn_from_linguistic(wt_df.loc[crit, dm]) for dm in wt_df.columns]

        # T2NN'leri topla ve ortala
        combined = combine_weights_t2nns(weight_list)

        # I ve F'yi sıfırla
        adjusted = zero_out_I_and_F(combined)

        # Skoru hesapla
        score = score_from_merged_t2nn(adjusted)

        # Detaylı tabloya ekle
        (t1, t2, t3), (i1, i2, i3), (f1, f2, f3) = combined
        detailed_weights.loc[crit] = [t1, t2, t3, i1, i2, i3, f1, f2, f3, score, 0]

    # Normalize et
    total_score = detailed_weights['Score'].sum()
    detailed_weights['Normalized'] = detailed_weights['Score'] / total_score

    # Görselleştir
    st.subheader("Birleştirilmiş T2NN Ağırlıklar + Skor + Normalize")
    st.dataframe(detailed_weights.style.format(precision=4))

    # İsteğe bağlı: sadece skorları gösteren sade tablo
    st.subheader("Kriter Ağırlıkları (Skorlar)")
    st.dataframe(detailed_weights[['Score']].style.format(precision=4))
