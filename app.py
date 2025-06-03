import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from t2 import T2NeutrosophicNumber, classic_to_t2n

st.title("T2 Neutrosophic MABAC for Ship Fuel Selection")

def convert_range_to_mean(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str) and ('-' in value or '–' in value):
        value = value.replace('–', '-')
        parts = value.split('-')
        try:
            nums = [float(p.replace(',', '.')) for p in parts]
            return sum(nums) / len(nums)
        except:
            return np.nan
    else:
        try:
            return float(str(value).replace(',', '.').replace(' ', ''))
        except:
            return np.nan

if uploaded_file := st.file_uploader("Upload your Excel file", type=["xlsx"]):

    df_raw = pd.read_excel(uploaded_file, sheet_name=0, header=1)
    try:
        df_info = pd.read_excel(uploaded_file, sheet_name=1)
    except:
        st.stop()

    criteria = df_raw.iloc[:, 0].tolist()
    alternatives = df_raw.columns[1:].tolist()

    types = []
    st.subheader("Select Criterion Type (Benefit or Cost)")
    for crit in criteria:
        t = st.selectbox(f"{crit}:", ["benefit", "cost"], key=crit)
        types.append(t)

    try:
        weights = [float(str(w).replace(',', '.')) for w in df_info["Weight"]]
    except:
        st.stop()

    # --- Verileri işle ---
    data_raw = df_raw.iloc[:, 1:].values
    X = np.array([[convert_range_to_mean(cell) for cell in row] for row in data_raw], dtype=float)

    st.subheader("Raw Data (converted ranges to means)")
    st.dataframe(pd.DataFrame(X, index=criteria, columns=alternatives))

    X_norm = np.zeros_like(X)
    for j in range(len(criteria)):
        col = X[j, :]
        min_val = np.nanmin(col)
        max_val = np.nanmax(col)
        if max_val == min_val:
            X_norm[j, :] = 0
        else:
            if types[j] == "benefit":
                X_norm[j, :] = (col - min_val) / (max_val - min_val)
            else:
                X_norm[j, :] = (max_val - col) / (max_val - min_val)
    X_norm = np.nan_to_num(X_norm)

    X_t2n = [[classic_to_t2n(X_norm[j, i]) for j in range(len(criteria))] for i in range(len(alternatives))]
    weights_t2n = [classic_to_t2n(w, indeterminacy=0.1) for w in weights]

    V = []
    for i in range(len(alternatives)):
        row = []
        for j in range(len(criteria)):
            row.append(X_t2n[i][j] * weights_t2n[j])
        V.append(row)

    g = []
    for j in range(len(criteria)):
        sum_t = np.mean([V[i][j].truth for i in range(len(alternatives))])
        sum_i = np.mean([V[i][j].indeterminacy for i in range(len(alternatives))])
        sum_f = np.mean([V[i][j].falsity for i in range(len(alternatives))])
        g.append(T2NeutrosophicNumber(sum_t, sum_i, sum_f))

    def t2n_sub(a, b):
        return T2NeutrosophicNumber(
            max(0, a.truth - b.truth),
            max(0, a.indeterminacy - b.indeterminacy),
            max(0, a.falsity - b.falsity)
        )

    Q = []
    for i in range(len(alternatives)):
        row = []
        for j in range(len(criteria)):
            row.append(t2n_sub(V[i][j], g[j]))
        Q.append(row)

    scores = []
    for i in range(len(alternatives)):
        total = T2NeutrosophicNumber(0, 0, 0)
        for j in range(len(criteria)):
            total = total + Q[i][j]
        scores.append(total.score())

    df_results = pd.DataFrame({
        "Alternative": alternatives,
        "Score": scores
    }).sort_values("Score", ascending=False)

    st.subheader("MABAC T2 Neutrosophic Results")
    st.dataframe(df_results)

    fig, ax = plt.subplots()
    ax.bar(df_results["Alternative"], df_results["Score"], color="green")
    ax.set_ylabel("Score")
    ax.set_title("Alternatives Comparison")
    st.pyplot(fig)
