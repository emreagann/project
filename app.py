import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from t2 import T2NeutrosophicNumber, t2nn_score, normalize_t2nn

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
                mid = (low + high) / 2
                indet = (high - low) / 2
                truth = (low, mid, high)
                indeterminacy = (indet, indet, indet)
                falsity = tuple(1 - t for t in truth)
                return T2NeutrosophicNumber(truth, indeterminacy, falsity)
            except:
                return None
    else:
        try:
            val = float(value.replace(',', '.'))
            truth = (val, val, val)
            indeterminacy = (0.0, 0.0, 0.0)
            falsity = (1 - val, 1 - val, 1 - val)
            return T2NeutrosophicNumber(truth, indeterminacy, falsity)
        except:
            return None


uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df_alt = pd.read_excel(uploaded_file, sheet_name="Alternatives", index_col=0, skiprows=1)
        df_weights = pd.read_excel(uploaded_file, sheet_name="Criteria Weights")
        df_sub = pd.read_excel(uploaded_file, sheet_name="Sub-Criteria")
    except:
        st.stop()

    criteria = df_alt.index.tolist()
    alternatives = df_alt.columns.tolist()
    data_raw = df_alt.values.T

    X = np.array([[convert_range_to_t2n(cell) for cell in row] for row in data_raw], dtype=object)

    df_weights.columns = df_weights.columns.str.strip().str.lower()
    df_sub.columns = df_sub.columns.str.strip().str.lower()

    required_weight_cols = {"criteria no", "weight"}
    required_sub_cols = {"criteria no", "sub-criteria attributes"}
    if not required_weight_cols.issubset(df_weights.columns) or not required_sub_cols.issubset(df_sub.columns):
        st.error("Criteria Weights sheet must include: 'criteria no', 'weight'; Sub-Criteria sheet must include: 'criteria no', 'sub-criteria attributes'")
        st.stop()

    df_weights["criteria no"] = df_weights["criteria no"].astype(str).str.strip().str.upper()
    df_sub["criteria no"] = df_sub["criteria no"].astype(str).str.strip().str.upper()

    weights = []
    types = []

    for crit in criteria:
        crit_code = crit.strip().upper()
        weight_row = df_weights[df_weights["criteria no"] == crit_code]
        sub_row = df_sub[df_sub["criteria no"] == crit_code]

        if weight_row.empty or sub_row.empty:
            st.error(f"{crit} için ağırlık veya perspective bilgisi eksik.")
            st.stop()

        weights.append(float(str(weight_row.iloc[0]["weight"]).replace(',', '.')))
        attribute = sub_row.iloc[0]["sub-criteria attributes"].strip().lower()
        if attribute not in {"benefit", "cost"}:
            st.error(f"{crit} için geçersiz sub-criteria attribute: {attribute}")
            st.stop()
        types.append(attribute)

    X_norm_obj = np.empty_like(X, dtype=object)
    for j in range(len(criteria)):
        col = [x[j] for x in X]
        col_valid = [v for v in col if isinstance(v, T2NeutrosophicNumber)]

        if not col_valid:
            st.error(f"{criteria[j]} sütununda geçerli T2NN değeri yok. Verileri kontrol et.")
            st.stop()

        min_val = T2NeutrosophicNumber(
            truth=tuple(min(v.truth[i] for v in col_valid) for i in range(3)),
            indeterminacy=tuple(min(v.indeterminacy[i] for v in col_valid) for i in range(3)),
            falsity=tuple(min(v.falsity[i] for v in col_valid) for i in range(3)),
        )
        max_val = T2NeutrosophicNumber(
            truth=tuple(max(v.truth[i] for v in col_valid) for i in range(3)),
            indeterminacy=tuple(max(v.indeterminacy[i] for v in col_valid) for i in range(3)),
            falsity=tuple(max(v.falsity[i] for v in col_valid) for i in range(3)),
        )
        for i in range(len(alternatives)):
            X_norm_obj[i, j] = normalize_t2nn(X[i, j], min_val, max_val, types[j])

    X_norm = np.array([[t2nn_score(cell) for cell in row] for row in X_norm_obj])

    V_numeric = X_norm * np.array(weights)
    G_vector = np.prod(V_numeric, axis=0) ** (1 / V_numeric.shape[0])
    Distance_matrix = V_numeric - G_vector
    Total_scores = Distance_matrix.sum(axis=1)

    df_original = pd.DataFrame([[t2nn_score(cell) for cell in row] for row in X], index=alternatives, columns=criteria)
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

    fig, ax = plt.subplots()
    ax.bar(df_scores.index, df_scores["TOTAL SCORE"], color="steelblue")
    ax.set_ylabel("Score")
    ax.set_title("Alternative Comparison")
    st.pyplot(fig)

    best = df_scores.iloc[0]
    st.success(f"Best Alternative: **{best.name}** with Score: **{best['TOTAL SCORE']:.4f}**")
