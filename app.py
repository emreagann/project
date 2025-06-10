import streamlit as st
import pandas as pd
import numpy as np

# Linguistic vars for alternatives (Tablo A.2)
linguistic_vars = {
    "VB": [0.20, 0.20, 0.10, 0.65, 0.80, 0.85, 0.45, 0.80, 0.70],
    "B":  [0.35, 0.35, 0.10, 0.50, 0.75, 0.80, 0.50, 0.75, 0.65],
    "MB": [0.50, 0.30, 0.50, 0.50, 0.35, 0.45, 0.45, 0.30, 0.60],
    "M":  [0.40, 0.45, 0.50, 0.40, 0.45, 0.50, 0.35, 0.40, 0.45],
    "MG": [0.60, 0.45, 0.50, 0.20, 0.15, 0.25, 0.10, 0.25, 0.15],
    "G":  [0.70, 0.75, 0.80, 0.15, 0.20, 0.25, 0.10, 0.15, 0.20],
    "VG": [0.95, 0.90, 0.95, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05],
}

# Linguistic vars for criteria weights (Tablo A.1)
criteria_linguistic_weights = {
    "L":  [(0.20, 0.30, 0.20), (0.60, 0.70, 0.80), (0.45, 0.75, 0.75)],
    "ML": [(0.40, 0.30, 0.25), (0.45, 0.55, 0.40), (0.45, 0.60, 0.55)],
    "M":  [(0.50, 0.55, 0.55), (0.40, 0.45, 0.55), (0.35, 0.40, 0.35)],
    "H":  [(0.80, 0.75, 0.70), (0.20, 0.15, 0.30), (0.15, 0.10, 0.20)],
    "VH": [(0.90, 0.85, 0.95), (0.10, 0.15, 0.10), (0.05, 0.05, 0.10)]
}

def t2nn_add(t1, t2):
    result = []
    for i in range(3):  # T, I, F blocks
        a1, a2, a3 = t1[i]
        b1, b2, b3 = t2[i]
        result.append((
            a1 + b1 - a1 * b1,
            a2 + b2 - a2 * b2,
            a3 + b3 - a3 * b3
        ))
    return tuple(result)

def score_function(values):
    a1, a2, a3, b1, b2, b3, g1, g2, g3 = values
    return (1 / 12) * (8 + (a1 + 2 * a2 + a3) - (b1 + 2 * b2 + b3) - (g1 + 2 * g2 + g3))

def score_t2nn_weight(t2nn):
    (a1, a2, a3), (b1, b2, b3), (c1, c2, c3) = t2nn
    return (1 / 12) * (8 + (a1 + 2 * a2 + a3) - (b1 + 2 * b2 + b3) - (c1 + 2 * c2 + c3))

def get_valid_numeric_values(value):
    value = str(value).strip()
    return score_function(linguistic_vars[value]) if value in linguistic_vars else 0

def normalize_data(series, criteria_type):
    if series.max() == series.min():
        return 0
    if criteria_type.lower() == 'benefit':
        return (series - series.min()) / (series.max() - series.min())
    elif criteria_type.lower() == 'cost':
        return (series.max() - series) / (series.max() - series.min())
    return series

def apply_weights(normalized_df, weights):
    return normalized_df.multiply(weights, axis=1)

def calculate_BAA(weighted_df):
    return weighted_df.prod(axis=0) ** (1 / len(weighted_df))

def calculate_difference_matrix(weighted_df, BAA):
    return weighted_df - BAA

def calculate_scores(diff_df):
    return diff_df.sum(axis=1)

st.title("T2NN MABAC CALCULATION")

input_method = st.radio("Select input method:", ("Upload Excel File", "Manual Entry"))

if input_method == "Upload Excel File":
    uploaded_file = st.file_uploader("Upload your excel file", type="xlsx")
    if uploaded_file:
        df = pd.read_excel(uploaded_file, sheet_name="Alternatives")
        weights_df_raw = pd.read_excel(uploaded_file, sheet_name="Weights")

        # Detect number of decision makers
        if "Alternative" in df.columns:
            df.rename(columns={"Alternative": "Alternatives"}, inplace=True)

       criteria_names = weights_df_raw.columns[1:].tolist()

        # Process criteria weights: average all DMs' T2NNs
        merged_weights = {}
        for crit in criteria_names:
            t2nn_sum = None
            for i in range(1, 5):  # DM1–DM4
                term = weights_df_raw.loc[weights_df_raw['Criteria No'] == crit, f'DM{i}'].values[0]
                if isinstance(term, str) and term.strip() in criteria_linguistic_weights:
                    t2nn = criteria_linguistic_weights[term.strip()]
                    if t2nn_sum is None:
                        t2nn_sum = t2nn
                    else:
                        t2nn_sum = t2nn_add(t2nn_sum, t2nn)
            if t2nn_sum:
                merged_weights[crit] = score_t2nn_weight(t2nn_sum)
            else:
                merged_weights[crit] = 0

        weights_df = pd.DataFrame({
            "Criteria No": criteria_names,
            "Weight": [merged_weights[c] for c in criteria_names],
            "Type": weights_df_raw["Type"].tolist()
        })

# Manuel giriş kısmı henüz güncellenmedi — istenirse entegre edilebilir.
