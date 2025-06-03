import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from t2 import T2NeutrosophicNumber, classic_to_t2n

st.title("T2 Neutrosophic MABAC for Decision Making")
st.subheader("Select a method to enter data")
input_method = st.radio("How to enter data?", ["Excel Upload", "Manual Entry"])

proceed = False
criteria, alternatives, weights, types, X = [], [], [], [], None

def detect_structure(df):
    first_col = df.iloc[:, 0]
    other_cols = df.iloc[:, 1:]
    
    if all(isinstance(x, str) for x in first_col) and all(other_cols.applymap(lambda x: isinstance(x, (int, float))).all()):
        return 'alternatives_in_rows'
    
    first_row = df.iloc[0, :]
    other_rows = df.iloc[1:, :]
    
    if all(isinstance(x, str) for x in first_row) and all(other_rows.applymap(lambda x: isinstance(x, (int, float))).all().all()):
        return 'alternatives_in_columns'
    
    return 'unknown'
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

if input_method == "Excel Upload":
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file:
     df_raw = pd.read_excel(uploaded_file, sheet_name=0, header=None)
     df_info = pd.read_excel(uploaded_file, sheet_name=1)

     structure = detect_structure(df_raw)
     st.write("Detected structure:", structure)

     if structure == 'alternatives_in_rows':
        criteria = df_raw.columns[1:].tolist()
        alternatives = df_raw.iloc[:, 0].tolist()
        data_raw = df_raw.iloc[:, 1:].values

     elif structure == 'alternatives_in_columns':
        alternatives = df_raw.columns[1:].tolist()
        criteria = df_raw.iloc[:, 0].tolist()
        data_raw = df_raw.iloc[:, 1:].values
        data_raw = data_raw.T

     if "Type" in df_info.columns:
        types = [str(t).strip().lower() for t in df_info["Type"].dropna()]

     weight_columns = [col for col in df_info.columns if str(col).strip().lower().startswith("c")]
     weight_row = df_info.iloc[0]
     weights = []
     for col in weight_columns:
        val = weight_row[col]
        try:
            weights.append(float(str(val).replace(",", ".")))
        except:
            weights.append(0.0)

     X = np.array([[convert_range_to_mean(cell) for cell in row] for row in data_raw], dtype=float)  # Burada artık data_raw var
     proceed = True

elif input_method == "Manual Entry":
    num_criteria = st.number_input("Number of criteria", min_value=1, step=1, format="%d")
    num_alternatives = st.number_input("Number of alternatives", min_value=1, step=1, format="%d")

    alternatives = []
    st.subheader("Enter Alternative Names")
    for j in range(num_alternatives):
        alt = st.text_input(f"Alternative {j+1} Name", key=f"alt_{j}")
        alternatives.append(alt)

    criteria, types, weights = [], [], []
    st.subheader("Enter Criteria")
    for i in range(num_criteria):
        crit = st.text_input(f"Criterion {i+1} Name", key=f"crit_{i}")
        criteria.append(crit)
        c_type = st.selectbox(f"{crit} type", ["benefit", "cost"], key=f"type_{i}")
        types.append(c_type)
        weight = st.number_input(f"{crit} weight", min_value=0.0, max_value=1.0, step=0.001, format="%.3f", key=f"weight_{i}")
        weights.append(weight)

    st.subheader("Enter Criterion Performance Values")
    X = np.zeros((num_criteria, num_alternatives))
    for i in range(num_criteria):
        for j in range(num_alternatives):
            val_str = st.text_input(f"Value of {criteria[i]} for {alternatives[j]}", key=f"val_{i}_{j}")
            X[i, j] = convert_range_to_mean(val_str)

    proceed = True

if proceed:
    st.subheader("Decision Matrix (Performance Values)")
    st.dataframe(pd.DataFrame(X, index=criteria, columns=alternatives), width=400, height=250)

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
    print(f"len(X_t2n) = {len(X_t2n)}")
    for idx, row in enumerate(X_t2n):
     print(f"len(X_t2n[{idx}]) = {len(row)}")
    print(f"len(weights_t2n) = {len(weights_t2n)}")
    print(f"len(alternatives) = {len(alternatives)}, len(criteria) = {len(criteria)}")
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
    })

    st.subheader("MABAC T2 Neutrosophic Results")
    st.dataframe(df_results)

    fig, ax = plt.subplots()
    ax.bar(df_results["Alternative"], df_results["Score"], color="green")
    ax.set_ylabel("Score")
    ax.set_title("Alternatives Comparison")
    st.pyplot(fig)

    st.subheader("Optional: View Intermediate Matrices")

    if st.checkbox("Show V Matrix"):
        df_v = pd.DataFrame([[str(cell) for cell in row] for row in V], index=alternatives, columns=criteria)
        st.dataframe(df_v)

    if st.checkbox("Show G Vector"):
        df_g = pd.DataFrame([str(gj) for gj in g], index=criteria, columns=["G Value"])
        st.dataframe(df_g)

    if st.checkbox("Show Q Matrix"):
        df_q = pd.DataFrame([[str(cell) for cell in row] for row in Q], index=alternatives, columns=criteria)
        st.dataframe(df_q)
