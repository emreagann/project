import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
st.title("MABAC for Ship Fuel Selection")

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
    df_raw = pd.read_excel(uploaded_file, sheet_name=0, header=1)
    first_column_values = df_raw.iloc[:, 0].astype(str).str.upper()

    if all(val.startswith("A") for val in first_column_values[:3]):
        df_raw = df_raw.set_index(df_raw.columns[0])
        df_raw = df_raw.transpose()
        df_raw.insert(0, "Criteria", df_raw.index)
        df_raw.reset_index(drop=True, inplace=True)
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
    X = np.array([[convert_range_to_mean(cell) for cell in row] for row in data_raw], dtype=float).T

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

for j, (crit, ctype) in enumerate(zip(criteria, types)):
    col = X[:, j]
    min_val = np.nanmin(col)
    max_val = np.nanmax(col)

    if max_val == min_val:
        X_norm[:, j] = 0.0
    else:
        ctype_clean = ctype.strip().lower()
        if ctype_clean == "benefit":
            X_norm[:, j] = (col - min_val) / (max_val - min_val)
        elif ctype_clean == "cost":
            X_norm[:, j] = (max_val - col) / (max_val - min_val)

X_norm = np.nan_to_num(X_norm)

V_numeric = np.zeros_like(X_norm)
for i in range(len(alternatives)):
    for j in range(len(criteria)):
        V_numeric[i, j] = X_norm[i, j] * weights[j]

df_weighted = pd.DataFrame(V_numeric, index=alternatives, columns=criteria)

G_vector = np.prod(V_numeric + 1e-10, axis=0) ** (1 / V_numeric.shape[0])
df_border = pd.DataFrame(G_vector.reshape(1, -1), columns=criteria)

Distance_matrix = V_numeric - G_vector
Total_scores = Distance_matrix.sum(axis=1)
df_distance = pd.DataFrame(Distance_matrix, index=alternatives, columns=criteria)
df_scores = pd.DataFrame({"TOTAL SCORE": Total_scores}, index=alternatives).sort_values(by="TOTAL SCORE", ascending=False)







st.subheader("Original Decision Matrix (Performance Values)")
df_original = pd.DataFrame(X, index=alternatives, columns=criteria)
st.dataframe(df_original.style.format("{:.4f}"))

st.subheader("Normalized Matrix (Benefit/Cost Adjusted)")
df_norm = pd.DataFrame(X_norm, index=alternatives, columns=criteria)
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
ax.bar(df_scores.index, df_scores["TOTAL SCORE"], color="blue")
ax.set_ylabel("Score")
ax.set_title("Alternatives Comparison")
st.pyplot(fig)
best_alt = df_scores.sort_values(by="TOTAL SCORE", ascending=False).iloc[0]
st.success(f"Best Alternative: **{best_alt.name}** with Score: **{best_alt['TOTAL SCORE']:.4f}**")



