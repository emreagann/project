import streamlit as st
import pandas as pd
import numpy as np

class T2NN:
    def __init__(self, T, I, F):
        self.T = T
        self.I = I
        self.F = F

    def score(self):
        t_score = (self.T[0] + 2 * self.T[1] + self.T[2])
        i_score = (self.I[0] + 2 * self.I[1] + self.I[2])
        f_score = (self.F[0] + 2 * self.F[1] + self.F[2])
        return (1 / 12) * (8 + t_score - i_score - f_score)

def convert_range_to_t2n(a, b):
    if a == b:
        T = (a / 10, a / 10, a / 10)
        I = (0.0125, 0.0125, 0.0125)
        F = (1 - a / 10, 1 - a / 10, 1 - a / 10)
    else:
        m = (a + b) / 2
        T = (a / 10, m / 10, b / 10)
        I = (0.0125, 0.0125, 0.0125)
        F = (1 - b / 10, 1 - m / 10, 1 - a / 10)
    return T2NN(T, I, F)


def normalize_minmax(values, benefit=True):
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return [0 for _ in values]
    if benefit:
        return [(v - min_v)/(max_v - min_v) for v in values]
    else:
        return [(max_v - v)/(max_v - min_v) for v in values]

st.title("Type-2 Neutrosophic MABAC Application")
input_mode = st.radio("Select input method:", ["Upload from Excel", "Manual Entry"])

if input_mode == "Upload from Excel":
    uploaded_file = st.file_uploader("Upload an Excel file", type=[".xlsx"])
    if uploaded_file is None:
        st.warning("Please upload an Excel file that includes the decision matrix.")
        st.stop()

    xls = pd.ExcelFile(uploaded_file)
    decision_matrix = pd.read_excel(xls, sheet_name="Alternatives", header=1, index_col=0)
    weights_df = pd.read_excel(xls, sheet_name="Criteria Weights")
    sub_criteria = pd.read_excel(xls, sheet_name="Sub-Criteria")

    sub_criteria.columns = sub_criteria.columns.str.strip().str.lower()
    weights_df.columns = weights_df.columns.str.strip().str.lower()

    criteria = decision_matrix.columns.tolist()
    alternatives = decision_matrix.index.tolist()
    types = dict(zip(sub_criteria['criteria no'], sub_criteria['sub-criteria attributes']))
    evals = dict(zip(sub_criteria['criteria no'], sub_criteria['evaluation perspective']))
    weights_dict = dict(zip(weights_df['criteria no'], weights_df['weight']))

elif input_mode == "Manual Entry":
    num_criteria = st.number_input("Number of criteria", min_value=1,)
    num_alternatives = st.number_input("Number of alternatives", min_value=1,)

    with st.expander("Criteria Names and Weights"):
        criteria = [f"C{i+1}" for i in range(num_criteria)]
        weights_dict = {c: st.number_input(f"Weight for {c}", min_value=0.0, max_value=1.0, step=0.001, format="%.3f") for c in criteria}

    with st.expander("Criteria Types and Evaluation Perspective"):
        types = {c: st.selectbox(f"{c} type", ["benefit", "cost"], key=f"type_{c}") for c in criteria}
        evals = {c: st.selectbox(f"{c} evaluation", ["quantitative", "qualitative"], key=f"eval_{c}") for c in criteria}

    st.subheader("Decision Matrix")
    matrix_data = pd.DataFrame(
        [[0.0 for _ in range(num_criteria)] for _ in range(num_alternatives)],
        columns=criteria,
        index=[f"A{i+1}" for i in range(num_alternatives)]
    )
    decision_matrix = st.data_editor(matrix_data, num_rows="dynamic", key="manual_input_matrix")
    alternatives = decision_matrix.index.tolist()

norm_scores = {crit: [] for crit in criteria}

for crit in criteria:
    is_quant = evals[crit] == "quantitative"
    is_benefit = types[crit].lower() == "benefit"
    col_scores = []
    for alt in alternatives:
        val = decision_matrix.loc[alt, crit]
        try:
            val = str(val).replace('–', '-').strip()
            if is_quant:
                if '-' in val:
                    parts = val.split('-')
                    if len(parts) == 2 and all(p.strip() != '' for p in parts):
                        a, b = map(float, parts)
                    else:
                        raise ValueError(f"Invalid interval format: '{val}'")
                else:
                    a = b = float(val)
                t2nn = convert_range_to_t2n(a, b)
                score = t2nn.score()
            else:
                score = float(val)
        except Exception as e:
            score = 0 
        col_scores.append(score)
    norm_scores[crit] = normalize_minmax(col_scores, benefit=is_benefit)


norm_df = pd.DataFrame(norm_scores, columns=criteria, index=alternatives)
st.subheader("Normalized Decision Matrix")
st.dataframe(norm_df.style.format("{:.4f}"))

weighted_df = norm_df.copy()
for crit in criteria:
    weighted_df[crit] = weighted_df[crit] * weights_dict[crit]

st.subheader("Weighted Normalized Matrix (V)")
st.dataframe(weighted_df.style.format("{:.4f}"))

B = weighted_df.apply(lambda col: col.prod()**(1/len(col)), axis=0)
st.subheader("Border Approximation Area (B)")
st.dataframe(B.to_frame().style.format("{:.4f}"))
Q = weighted_df - B
st.subheader("MABAC Distance Matrix (Q = V - B)")
st.dataframe(Q.style.format("{:.4f}"))

scores = Q.sum(axis=1)
results = pd.DataFrame({"Scores": scores, "Ranking": scores.rank(ascending=False).astype(int)}, index=alternatives)
st.subheader("MABAC Results and Ranking")
st.dataframe(results.sort_values("Ranking"))
