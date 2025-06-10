import streamlit as st
import pandas as pd
import numpy as np

import streamlit as st
import pandas as pd
import numpy as np

# Linguistic variables for Alternatives (from Table A.2)
alternative_linguistic_vars = {
    "VB": [0.20, 0.20, 0.10, 0.65, 0.80, 0.85, 0.45, 0.80, 0.70],
    "B":  [0.35, 0.35, 0.10, 0.50, 0.75, 0.80, 0.50, 0.75, 0.65],
    "MB": [0.50, 0.30, 0.50, 0.50, 0.35, 0.45, 0.45, 0.30, 0.60],
    "M":  [0.40, 0.45, 0.50, 0.40, 0.45, 0.50, 0.35, 0.40, 0.45],
    "MG": [0.60, 0.45, 0.50, 0.20, 0.15, 0.25, 0.10, 0.25, 0.15],
    "G":  [0.70, 0.75, 0.80, 0.15, 0.20, 0.25, 0.10, 0.15, 0.20],
    "VG": [0.95, 0.90, 0.95, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05],
}

# Linguistic variables for Criteria Weights (from Table A.1)
criteria_linguistic_weights = {
    "VL": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0, 0.6, 0.7, 0.8],
    "L":  [0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.5, 0.6, 0.7],
    "ML": [0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.4, 0.5, 0.6],
    "M":  [0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.3, 0.4, 0.5],
    "MH": [0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.2, 0.3, 0.4],
    "H":  [0.5, 0.6, 0.7, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3],
    "VH": [0.6, 0.7, 0.8, 0.2, 0.3, 0.4, 0.0, 0.1, 0.2],
}


def score_function(values):
    a1, a2, a3, b1, b2, b3, g1, g2, g3 = values
    return (1 / 12) * (8 + (a1 + 2 * a2 + a3) - (b1 + 2 * b2 + b3) - (g1 + 2 * g2 + g3))

def t2nn_addition(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def t2nn_average(vectors):
    n = len(vectors)
    summed = vectors[0]
    for vec in vectors[1:]:
        summed = t2nn_addition(summed, vec)
    return [x / n for x in summed]

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

        if "Alternative" in df.columns:
            df.rename(columns={"Alternative": "Alternatives"}, inplace=True)

        criteria_names = weights_df_raw.columns[1:].tolist()

        # Process criteria weights: average all DMs' T2NNs
        merged_weights = {}
        for crit in criteria_names:
            t2nn_list = []
            for i in range(weights_df_raw.shape[0]):
                val = weights_df_raw.loc[i, crit]
                if isinstance(val, str) and val.strip() in linguistic_vars:
                    t2nn_list.append(linguistic_vars[val.strip()])
            if t2nn_list:
                avg = t2nn_average(t2nn_list)
                merged_weights[crit] = avg

        # Compute score weights
        weight_scores = {crit: score_function(vec) for crit, vec in merged_weights.items()}
        weights = pd.Series(weight_scores)
        weights /= weights.sum()

        weights_df = pd.DataFrame({
            "Criteria No": list(weight_scores.keys()),
            "Weight": list(weights.values),
            "Type": ["benefit"] * len(weight_scores)  # or custom if available
        })

        # Decision matrix
        alt_counts = df['Alternatives'].value_counts()
        n_dms = alt_counts.iloc[0]
        n_alternatives = len(alt_counts)
        criteria_names = weights_df["Criteria No"].tolist()

        final_matrix = pd.DataFrame(columns=criteria_names)

        for alt in df['Alternatives'].unique():
            group = df[df['Alternatives'] == alt][criteria_names]
            scored = group.applymap(lambda x: score_function(linguistic_vars.get(str(x).strip(), [0]*9)))
            avg_scores = scored.mean(axis=0)
            final_matrix.loc[alt] = avg_scores

        normalized_df = pd.DataFrame(index=final_matrix.index)
        for i, col in enumerate(final_matrix.columns):
            normalized_df[col] = normalize_data(final_matrix[col], weights_df.iloc[i]["Type"])

        weighted_df = apply_weights(normalized_df, weights_df["Weight"].values)
        BAA = calculate_BAA(weighted_df)
        difference_df = calculate_difference_matrix(weighted_df, BAA)
        scores = calculate_scores(difference_df)

        result_df = pd.DataFrame({
            "Alternative": final_matrix.index,
            "MABAC Score": scores,
            "Rank": scores.rank(ascending=False).astype(int)
        }).sort_values(by="Rank")

        st.subheader("Decision Matrix (T2NN Score Averages)")
        st.dataframe(final_matrix)

        st.subheader("Normalized Matrix")
        st.dataframe(normalized_df)

        st.subheader("Weighted Normalized Matrix")
        st.dataframe(weighted_df)

        st.subheader("Border Approximation Area")
        st.write(BAA)

        st.subheader("Distance Matrix")
        st.dataframe(difference_df)

        st.subheader("MABAC SCORE AND RANKING")
        st.dataframe(result_df)

        st.success("Calculation complete. All matrices displayed.")
