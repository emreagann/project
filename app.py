import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from t2 import T2NeutrosophicNumber

st.title("MABAC for Ship Fuel Selection using T2 Neutrosophic Numbers")

def convert_range_to_t2n(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if '-' in value or '–' in value:
        value = value.replace('–', '-')
        parts = value.split('-')
        if len(parts) == 2:
            try:
                low = float(parts[0].replace(',', '.'))
                high = float(parts[1].replace(',', '.'))
                if low == high:
                    return low
                mid = (low + high) / 2
                indet = (high - low) / 2
                truth = (low, mid, high)
                indeterminacy = (indet, indet, indet)
                falsity = tuple(1 - t for t in truth)
                return T2NeutrosophicNumber(truth, indeterminacy, falsity)
            except:
                return None
    try:
        return float(value.replace(',', '.'))
    except:
        return None

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df_alt = pd.read_excel(uploaded_file, sheet_name="Alternatives", index_col=0, skiprows=1)
        df_info = pd.read_excel(uploaded_file, sheet_name="Criteria Weights")
    except:
        st.error("Dosya okunamadı veya beklenen sayfalar bulunamadı.")
        st.stop()

    criteria = df_alt.index.tolist()
    alternatives = df_alt.columns.tolist()
    data_raw = df_alt.values.T  

    X = np.array([[convert_range_to_t2n(cell) for cell in row] for row in data_raw], dtype=object)
    score_matrix = np.array([
        [
            np.mean(cell.truth) if isinstance(cell, T2NeutrosophicNumber)
            else float(cell) if isinstance(cell, (int, float))
            else 0.0
            for cell in row
        ]
        for row in X
    ])

    df_info.columns = df_info.columns.str.strip().str.lower()
    if "weight" not in df_info.columns or "type" not in df_info.columns or "criteria no" not in df_info.columns:
        st.error("Criteria Weights sayfasında gerekli sütunlar eksik (criteria no, type, weight).")
        st.stop()

    try:
        weights = []
        types = []
        for crit in criteria:
            df_info["criteria no"] = df_info["criteria no"].astype(str).str.strip().str.upper()
            crit_info = df_info[df_info["criteria no"] == crit.strip().upper()]
            if crit_info.empty:
                st.error(f"{crit} için ağırlık/tip bilgisi bulunamadı.")
                st.stop()
            weights.append(float(str(crit_info.iloc[0]["weight"]).replace(',', '.')))
            types.append(crit_info.iloc[0]["type"].strip().lower())
    except Exception as e:
        st.error(f"Ağırlık ve tür bilgileri alınırken hata: {e}")
        st.stop()

    X_norm = np.zeros_like(score_matrix)
    for j, ctype in enumerate(types):
        col = score_matrix[:, j]
        min_val, max_val = np.min(col), np.max(col)
        if max_val == min_val:
            X_norm[:, j] = 0
        elif ctype == "benefit":
            X_norm[:, j] = (col - min_val) / (max_val - min_val)
        else:
            X_norm[:, j] = (max_val - col) / (max_val - min_val)

    V_numeric = X_norm * np.array(weights)

    
    G_vector = np.prod(V_numeric, axis=0) ** (1 / V_numeric.shape[0])


    Distance_matrix = V_numeric - G_vector
    Total_scores = Distance_matrix.sum(axis=1)


    df_original = pd.DataFrame(score_matrix, index=alternatives, columns=criteria)
    df_norm = pd.DataFrame(X_norm, index=alternatives, columns=criteria)
    df_weighted = pd.DataFrame(V_numeric, index=alternatives, columns=criteria)
    df_border = pd.DataFrame(G_vector.reshape(1, -1), columns=criteria)
    df_distance = pd.DataFrame(Distance_matrix, index=alternatives, columns=criteria)
    df_scores = pd.DataFrame({"TOTAL SCORE": Total_scores}, index=alternatives).sort_values(by="TOTAL SCORE", ascending=False)


    st.subheader("Original Decision Matrix (Performance Values)")
    st.dataframe(df_original.style.format("{:.3f}"))

    st.subheader("Normalized Matrix")
    st.dataframe(df_norm.style.format("{:.3f}"))

    st.subheader("Weighted Normalized Matrix (V)")
    st.dataframe(df_weighted.style.format("{:.3f}"))

    st.subheader("Border Approximation Area (G)")
    st.dataframe(df_border.style.format("{:.3f}"))

    st.subheader("Distance Matrix (V - G)")
    st.dataframe(df_distance.style.format("{:.3f}"))

    st.subheader("MABAC Total Scores")
    st.dataframe(df_scores.style.format("{:.4f}"))

    # Grafik
    fig, ax = plt.subplots()
    ax.bar(df_scores.index, df_scores["TOTAL SCORE"], color="steelblue")
    ax.set_ylabel("Score")
    ax.set_title("Alternative Comparison")
    st.pyplot(fig)

    best = df_scores.iloc[0]
    st.success(f"Best Alternative: **{best.name}** with Score: **{best['TOTAL SCORE']:.4f}**")

