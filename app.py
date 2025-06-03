import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from t2 import T2NeutrosophicNumber, classic_to_t2n

st.title("T2 Neutrosophic MABAC for Ship Fuel Selection")

input_mode = st.radio("Choose Input Mode", ["Upload Excel", "Manual Input"])

def convert_range_to_mean(value):
    if pd.isna(value): return np.nan
    if isinstance(value, str) and ('-' in value or '–' in value):
        value = value.replace('–', '-')
        parts = value.split('-')
        try:
            nums = [float(p.replace(',', '.')) for p in parts]
            return sum(nums) / len(nums)
        except: return np.nan
    try:
        return float(str(value).replace(',', '.').replace(' ', ''))
    except: return np.nan

if input_mode == "Upload Excel":
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file:
        df_raw = pd.read_excel(uploaded_file, sheet_name=0, header=1)
        try:
            df_info = pd.read_excel(uploaded_file, sheet_name=1)
        except:
            st.error("Second sheet with weights not found!")
            st.stop()

        criteria = df_raw.iloc[:, 0].tolist()
        alternatives = df_raw.columns[1:].tolist()
        types = [st.selectbox(f"{crit}:", ["benefit", "cost"], key=crit) for crit in criteria]

        try:
            weights = [float(str(w).replace(',', '.')) for w in df_info["Weight"]]
        except:
            st.error("Could not parse weights from Excel.")
            st.stop()

        data_raw = df_raw.iloc[:, 1:].values
        X = np.array([[convert_range_to_mean(cell) for cell in row] for row in data_raw], dtype=float)

else:
    st.subheader("Manual Input")
    num_criteria = st.number_input("Number of Criteria", min_value=1, value=3)
    num_alternatives = st.number_input("Number of Alternatives", min_value=1, value=3)

    criteria = [st.text_input(f"Criterion {i+1} Name", value=f"C{i+1}") for i in range(num_criteria)]
    alternatives = [st.text_input(f"Alternative {i+1} Name", value=f"A{i+1}") for i in range(num_alternatives)]
    types = [st.selectbox(f"{criteria[i]} Type:", ["benefit", "cost"], key=f"type_{i}") for i in range(num_criteria)]

    weights = []
    st.subheader("Enter Criteria Weights")
    for crit in criteria:
        weights.append(st.number_input(f"Weight for {crit}", min_value=0.0, format="%.4f"))

    st.subheader("Enter Decision Matrix")
    X = np.zeros((num_criteria, num_alternatives))
    for i in range(num_criteria):
        for j in range(num_alternatives):
            X[i, j] = st.number_input(f"Value for {criteria[i]}, {alternatives[j]}", key=f"dm_{i}_{j}")

st.subheader("Decision Matrix")
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
        max(0, a.falsity - b.falsity))

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

df_results = pd.DataFrame({"Alternative": alternatives, "Score": scores}).sort_values("Score", ascending=False)


st.subheader("MABAC T2 Neutrosophic Results")
st.dataframe(df_results)

fig, ax = plt.subplots()
ax.bar(df_results["Alternative"], df_results["Score"], color="green")
ax.set_ylabel("Score")
ax.set_title("Alternatives Comparison")
st.pyplot(fig)

if st.checkbox("Show Normalized Matrix (X_norm)"):
    st.dataframe(pd.DataFrame(X_norm, index=criteria, columns=alternatives))

if st.checkbox("Show Weighted Matrix (V)"):
    st.write([[str(val) for val in row] for row in V])

if st.checkbox("Show Border Area (G)"):
    st.write([str(val) for val in g])

if st.checkbox("Show Q Matrix"):
    st.write([[str(val) for val in row] for row in Q])

if st.checkbox("Compare with Article Results"):
    expected = st.text_area("Enter expected scores (comma-separated)")
    try:
        expected_vals = [float(x.strip()) for x in expected.split(",")]
        if len(expected_vals) == len(scores):
            diffs = [abs(scores[i] - expected_vals[i]) for i in range(len(scores))]
            st.write("Differences:", diffs)
        else:
            st.error("Number of expected scores does not match number of alternatives")
    except:
        st.error("Invalid input for expected scores")
