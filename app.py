import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from t2 import T2NeutrosophicNumber, classic_to_t2n
import re

st.title("T2 Neutrosophic MABAC for Ship Fuel Selection")
def detect_excel_references(df):
    def is_excel_ref(s):
        if not isinstance(s, str):
            return False
        return bool(re.match(r"^[A-Z]+[0-9]+$", s.strip().upper()))
    
    first_row = df.iloc[0, 1:]
    first_col = df.iloc[1:, 0]
    
    first_row_refs = all(is_excel_ref(x) for x in first_row)
    first_col_refs = all(is_excel_ref(x) for x in first_col)
    
    return first_row_refs, first_col_refs
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

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file, sheet_name=0, header=None)
    
    first_row_refs, first_col_refs = detect_excel_references(df_raw)
    
    if first_row_refs:
        header_row = 1
    else:
        header_row = 0
    
    df_raw = pd.read_excel(uploaded_file, sheet_name=0, header=header_row)
    
    try:
        df_info = pd.read_excel(uploaded_file, sheet_name=1)
    except:
        st.error("Not founded.")
        st.stop()
    
    criteria = df_raw.iloc[:, 0].tolist()
    alternatives = df_raw.columns[1:].tolist()

    df_info.columns = df_info.columns.str.strip().str.lower()

    types = df_info["type"].tolist()

    if "weight" in df_info.columns:
        weights_col = "weight"
    elif "weights" in df_info.columns:
        weights_col = "weights"

    weights = [float(str(w).replace(',', '.')) for w in df_info[weights_col]]


    data_raw = df_raw.iloc[:, 1:].values
    X = np.array([[convert_range_to_mean(cell) for cell in row] for row in data_raw.T], dtype=float)

else:
    st.subheader("Manual data entry")

    alt_count = st.number_input("Number of Alternatives", min_value=1, step=1)
    crit_count = st.number_input("Number of Criteria", min_value=1, step=1)

    alternatives = []
    criteria = []

    with st.form("data_entry_form"):
        st.write("Enter Alternatives")
        for i in range(int(alt_count)):
            alt = st.text_input(f"Alternative {i+1} name", key=f"alt_{i}")
            alternatives.append(alt if alt else f"Alt{i+1}")

        st.write("Enter Criteria")
        for j in range(int(crit_count)):
            crit = st.text_input(f"Criterion {j+1} name", key=f"crit_{j}")
            criteria.append(crit if crit else f"Crit{j+1}")

        st.write("Select Criteria Type (Benefit or Cost)")
        types = []
        for crit in criteria:
            t = st.selectbox(f"{crit}:", ["benefit", "cost"], key=crit)
            types.append(t)

        st.write("Enter Criteria Weights")
        weights = []
        for crit in criteria:
           w = st.number_input(f"Weight for {crit}", min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
           weights.append(w)

        st.write("Enter Performance Values (Decision Matrix)")
        data_matrix = []
        for j in range(int(crit_count)):
            row = []
            for i in range(int(alt_count)):
                val = st.text_input(f"Value for Alternative '{alternatives[i]}' - Criterion '{criteria[j]}'", key=f"val_{j}_{i}")
                row.append(val)
            data_matrix.append(row)

        submitted = st.form_submit_button("Process")
        if not submitted:
            st.stop()

    X = np.array([[convert_range_to_mean(cell) for cell in row] for row in data_matrix], dtype=float)

X_norm = np.zeros_like(X)
for j in range(len(criteria)):
    col = X[:, j] 
    min_val = np.nanmin(col)
    max_val = np.nanmax(col)
    if max_val == min_val:
        X_norm[:, j] = 0
    else:
        if types[j] == "benefit":
            X_norm[:, j] = (col - min_val) / (max_val - min_val)
        else:
            X_norm[:, j] = (max_val - col) / (max_val - min_val)

X_norm = np.nan_to_num(X_norm)

X_t2n = [[classic_to_t2n(X_norm[i][j]) for j in range(len(criteria))] for i in range(len(alternatives))]
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

Q = []
for i in range(len(alternatives)):
    row = []
    for j in range(len(criteria)):
        row.append(V[i][j] - g[j])
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



st.subheader("Original Decision Matrix (Performance Values)")
df_original = pd.DataFrame(X, index=alternatives, columns=criteria)
st.dataframe(df_original.style.format("{:.4f}"))


st.subheader("MABAC T2 Neutrosophic Final Results")
st.dataframe(df_results)

fig, ax = plt.subplots()
ax.bar(df_results["Alternative"], df_results["Score"], color="green")
ax.set_ylabel("Score")
ax.set_title("Alternatives Comparison")
st.pyplot(fig)
st.subheader("Border Approximation G (Average Neutrosophic Value)")
df_g = pd.DataFrame(
    {criteria[j]: [f"T: {round(g[j].truth, 3)} | I: {round(g[j].indeterminacy, 3)} | F: {round(g[j].falsity, 3)}"] for j in range(len(criteria))}
).T
df_g.columns = ["G (T2N Avg)"]
st.dataframe(df_g)

st.subheader("Difference Matrix Q = V - G")
df_q = pd.DataFrame(
    [[str(cell) for cell in row] for row in Q],
    columns=criteria,
    index=alternatives
)
st.dataframe(df_q)
st.write(f"Alternatives detected: {alternatives}")
