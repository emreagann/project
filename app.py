import streamlit as st
import pandas as pd
import numpy as np

# --- T2NN sınıfı ve skor fonksiyonu ---
class T2NN:
    def __init__(self, T, I, F):
        self.T = T  # (T1, T2, T3)
        self.I = I  # (I1, I2, I3)
        self.F = F  # (F1, F2, F3)

    def score(self):
        T = sum([self.T[0], 2*self.T[1], self.T[2]])
        I = sum([self.I[0], 2*self.I[1], self.I[2]])
        F = sum([self.F[0], 2*self.F[1], self.F[2]])
        return (8 + T - I - F) / 12

# --- Yardımcı fonksiyonlar ---
def convert_range_to_t2n(a, b):
    m = (a + b) / 2
    T = (a/10, m/10, b/10)
    I = (0.0125, 0.0125, 0.0125)
    F = (1 - b/10, 1 - m/10, 1 - a/10)
    return T2NN(T, I, F)

def normalize_minmax(values, benefit=True):
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return [0 for _ in values]
    if benefit:
        return [(v - min_v)/(max_v - min_v) for v in values]
    else:
        return [(max_v - v)/(max_v - min_v) for v in values]

# --- Streamlit arayüzü ---
st.title("Type-2 Neutrosophic MABAC Uygulaması")

uploaded_file = st.file_uploader("Excel dosyasını yükleyin", type=[".xlsx"])

if uploaded_file is None:
    st.warning("Lütfen karar matrisi içeren Excel dosyasını yükleyin.")
    st.stop()

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    decision_matrix = pd.read_excel(xls, sheet_name="Alternatives", header=1, index_col=0)
    weights = pd.read_excel(xls, sheet_name="Criteria Weights")
    sub_criteria = pd.read_excel(xls, sheet_name="Sub-Criteria")

    # Sütun isimlerini normalize et
    sub_criteria.columns = sub_criteria.columns.str.strip().str.lower()
    weights.columns = weights.columns.str.strip().str.lower()

    criteria = decision_matrix.columns.tolist()
    alternatives = decision_matrix.index.tolist()

    # Kriter türleri
    types = dict(zip(sub_criteria['criteria no'], sub_criteria['sub-criteria attributes']))
    evals = dict(zip(sub_criteria['criteria no'], sub_criteria['evaluation perspective']))
    weights_dict = dict(zip(weights['criteria no'], weights['weight']))

    # --- Normalize matris oluştur ---
    norm_scores = {crit: [] for crit in criteria}

    for crit in criteria:
        is_quant = evals[crit] == "quantitative"
        is_benefit = types[crit].lower() == "benefit"
        col_scores = []
        for alt in alternatives:
            val = decision_matrix.loc[alt, crit]
            if is_quant:
                if isinstance(val, str) and '-' in val:
                    val = str(val).replace('–', '-')
                    a, b = map(float, val.split('-'))
                else:
                    a = b = float(str(val).replace('–', '-'))
                t2nn = convert_range_to_t2n(a, b)
                score = t2nn.score()
            else:
                score = float(str(val).replace('–', '-'))
            col_scores.append(score)
        norm_scores[crit] = normalize_minmax(col_scores, benefit=is_benefit)

    norm_df = pd.DataFrame(norm_scores, columns=criteria, index=alternatives)
    st.subheader("Normalize Edilmiş Karar Matrisi")
    st.dataframe(norm_df.style.format("{:.4f}"))

    # --- Weighted normalize matris ---
    weighted_df = norm_df.copy()
    for crit in criteria:
        weighted_df[crit] = weighted_df[crit] * weights_dict[crit]

    st.subheader("Weighted Normalize Matris (V)")
    st.dataframe(weighted_df.style.format("{:.4f}"))

    # --- Border (B) ---
    B = weighted_df.apply(lambda col: col.prod()**(1/len(col)), axis=0)

    # --- Mesafe Matrisi Q = V - B ---
    Q = weighted_df - B
    st.subheader("MABAC Mesafe Matrisi (Q = V - B)")
    st.dataframe(Q.style.format("{:.4f}"))

    # --- MABAC Skorları ---
    scores = Q.sum(axis=1)
    results = pd.DataFrame({"Scores": scores, "Ranking": scores.rank(ascending=False).astype(int)}, index=alternatives)
    st.subheader("MABAC Results and Ranking")
    st.dataframe(results.sort_values("Ranking"))
