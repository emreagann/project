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
    df_raw = pd.read_excel(uploaded_file, sheet_name=0, header=1)
    first_column_values = df_raw.iloc[:, 0].astype(str).str.upper()

    if all(val.startswith("A") for val in first_column_values[:3]):
        df_raw = df_raw.set_index(df_raw.columns[0])
        df_raw = df_raw.transpose()
        df_raw.insert(0, "Criteria", df_raw.index)
        df_raw.reset_index(drop=True, inplace=True)

    try:
        df_info = pd.read_excel(uploaded_file, sheet_name=1)
        if any(col.lower().startswith(("truth", "t", "indeterminacy", "i", "falsity", "f")) for col in df_raw.columns):
            st.stop()

    except:
        st.error("Info sheet not found.")
        st.stop()

    criteria = df_raw.iloc[:, 0].tolist()
    alternatives = df_raw.columns[1:].tolist()
    df_info.columns = df_info.columns.str.strip().str.lower()

    types = df_info["type"].tolist()

    weights_col = "weight" if "weight" in df_info.columns else "weights"
    weights = [float(str(w).replace(',', '.')) for w in df_info[weights_col]]

    data_raw = df_raw.iloc[:, 1:].values
    X = np.array([[convert_range_to_t2n(cell) for cell in row] for row in data_raw], dtype=object).T

else:
    st.subheader("Manual Data Entry")

    alt_count = st.number_input("Number of Alternatives", min_value=1, step=1)
    crit_count = st.number_input("Number of Criteria", min_value=1, step=1)

    alternatives = []
    criteria = []
    with st.form("data_entry_form"):
        for i in range(int(alt_count)):
            alt = st.text_input(f"Alternative {i+1}", key=f"alt_{i}")
            alternatives.append(alt if alt else f"Alt{i+1}")

        for j in range(int(crit_count)):
            crit = st.text_input(f"Criterion {j+1}", key=f"crit_{j}")
            criteria.append(crit if crit else f"Crit{j+1}")

        types = [st.selectbox(f"{c} type", ["benefit", "cost"], key=c) for c in criteria]
        weights = [st.number_input(f"Weight for {c}", min_value=0.0, max_value=1.0, step=0.01) for c in criteria]

        data_matrix = []
        for j in range(int(crit_count)):
            row = []
            for i in range(int(alt_count)):
                val = st.text_input(f"Value for {alternatives[i]} - {criteria[j]}", key=f"val_{j}_{i}")
                row.append(val)
            data_matrix.append(row)

        submitted = st.form_submit_button("Calculate")
        if not submitted:
            st.stop()

    X = np.array([[convert_range_to_t2n(cell) for cell in row] for row in data_matrix], dtype=object).T

score_matrix = np.array([
    [
        np.mean(cell.truth) if isinstance(cell, T2NeutrosophicNumber)
        else float(cell) if isinstance(cell, (int, float)) 
        else 0.0
        for cell in row
    ]
    for row in X
])


X_norm = np.array(score_matrix, copy=True)

X_norm = np.zeros_like(score_matrix)
for j, ctype in enumerate(types):
    col = score_matrix[:, j]
    min_val, max_val = np.min(col), np.max(col)
    
    if max_val == min_val:
        X_norm[:, j] = 0 
    elif ctype.strip().lower() == "benefit":
        X_norm[:, j] = (col - min_val) / (max_val - min_val)
    else: 
        X_norm[:, j] = (max_val - col) / (max_val - min_val)


V_numeric = np.zeros_like(X_norm)
for i in range(len(alternatives)):
    for j in range(len(criteria)):
        V_numeric[i, j] = X_norm[i, j] * weights[j]

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
st.dataframe(df_original.style.format("{:.9f}"))

st.subheader("Normalized Matrix")
st.dataframe(df_norm.style.format("{:.4f}"))

st.subheader("Weighted Normalized Matrix (V)")
st.dataframe(df_weighted.style.format("{:.4f}"))

st.subheader("Border Approximation Area (G)")
st.dataframe(df_border.style.format("{:.6f}"))

st.subheader("Distance Matrix (V - G)")
st.dataframe(df_distance.style.format("{:.6f}"))

st.subheader("MABAC Total Scores")
st.dataframe(df_scores.style.format("{:.6f}"))

fig, ax = plt.subplots()
ax.bar(df_scores.index, df_scores["TOTAL SCORE"], color="steelblue")
ax.set_ylabel("Score")
ax.set_title("Alternative Comparison")
st.pyplot(fig)

best = df_scores.iloc[0]
st.success(f"Best Alternative: **{best.name}** with Score: **{best['TOTAL SCORE']:.4f}**") 
