import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="T2NN-MABAC Decision Support", layout="wide")
st.title("T2NN-MABAC Decision Support System")

# Step 1: Mode selection
mode = st.radio("Select Input Mode", ["Manual Input", "Upload Excel File"])

# Step 2: Editable Linguistic Scale
st.sidebar.header("Linguistic Scale Mapping")
def default_linguistic_mapping():
    return {
        "VL": 0.2,
        "L": 0.4,
        "M": 0.6,
        "G": 0.8,
        "VG": 1.0
    }

linguistic_mapping = {}
st.sidebar.write("Define the mapping from linguistic terms to scores:")
defs = default_linguistic_mapping()
for label, default in defs.items():
    linguistic_mapping[label] = st.sidebar.number_input(f"{label}", min_value=0.0, max_value=1.0, value=default, step=0.1)

# Step 3: Input handling
data = None
if mode == "Upload Excel File":
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file, header=0)
        data = df
        st.write("### Uploaded Data")
        st.dataframe(df)
else:
    st.write("### Manual Input")
    n_alternatives = st.number_input("Number of Alternatives", min_value=1, value=3)
    n_criteria = st.number_input("Number of Criteria", min_value=1, value=3)
    n_dms = 4

    manual_data = []
    for a in range(n_alternatives):
        alt_row = []
        for c in range(n_criteria):
            dm_values = []
            for d in range(n_dms):
                dm_value = st.selectbox(
                    f"Alt {a+1}, Crit {c+1}, DM {d+1}", list(linguistic_mapping.keys()),
                    key=f"A{a}_C{c}_DM{d}"
                )
                dm_values.append(linguistic_mapping[dm_value])
            avg_score = np.mean(dm_values)
            alt_row.append(avg_score)
        manual_data.append(alt_row)
    data = pd.DataFrame(manual_data, columns=[f"C{i+1}" for i in range(n_criteria)])
    st.write("### Averaged T2NN Score Matrix")
    st.dataframe(data)

# Proceed only if data is available
if data is not None:
    st.subheader("Step 4: MABAC Processing")

    # User input for criteria types and weights
    st.write("#### Define Criteria Types and Weights")
    col1, col2 = st.columns(2)
    with col1:
        criterion_types = [
            st.selectbox(f"Criterion {col}", ["Benefit", "Cost"], key=f"ctype_{col}")
            for col in data.columns
        ]
    with col2:
        criterion_weights = [
            st.number_input(f"Weight for {col}", min_value=0.0, max_value=1.0, value=1.0/len(data.columns), key=f"w_{col}")
            for col in data.columns
        ]

    # Normalize matrix
    norm_data = data.copy()
    for i, col in enumerate(data.columns):
        if criterion_types[i] == "Benefit":
            norm_data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        else:  # Cost
            norm_data[col] = (data[col].max() - data[col]) / (data[col].max() - data[col].min())

    st.write("#### Normalized Matrix")
    st.dataframe(norm_data)

    # Weighted normalized matrix (G matrix)
    weighted_matrix = norm_data.copy()
    for i, col in enumerate(data.columns):
        weighted_matrix[col] = norm_data[col] * criterion_weights[i]

    st.write("#### Weighted Normalized Matrix (G)")
    st.dataframe(weighted_matrix)

    # Border Approximation Area (BAA)
    baa = weighted_matrix.mean(axis=0)
    st.write("#### Border Approximation Area (BAA)")
    st.dataframe(pd.DataFrame([baa], index=["BAA"]))

    # Distance Matrix
    distance_matrix = weighted_matrix - baa
    st.write("#### Distance Matrix")
    st.dataframe(distance_matrix)

    # Final MABAC Scores
    mabac_scores = distance_matrix.sum(axis=1)
    results = pd.DataFrame({"Alternative": [f"A{i+1}" for i in range(len(mabac_scores))], "MABAC Score": mabac_scores})
    results = results.sort_values("MABAC Score", ascending=False)

    st.write("### Final Ranking")
    st.dataframe(results.reset_index(drop=True))
