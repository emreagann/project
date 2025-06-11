import streamlit as st
import pandas as pd

# --- T2NN sözlükleri --- (Tablo A.1 ve A.2'den alındı)
alternative_linguistic_vars = {
    "VB": [0.20, 0.20, 0.10, 0.65, 0.80, 0.85, 0.45, 0.80, 0.70],
    "B":  [0.35, 0.35, 0.10, 0.50, 0.75, 0.80, 0.50, 0.75, 0.65],
    "MB": [0.50, 0.30, 0.50, 0.50, 0.35, 0.45, 0.45, 0.30, 0.60],
    "M":  [0.40, 0.45, 0.50, 0.40, 0.45, 0.50, 0.35, 0.40, 0.45],
    "MG": [0.60, 0.45, 0.50, 0.20, 0.15, 0.25, 0.10, 0.25, 0.15],
    "G":  [0.70, 0.75, 0.80, 0.15, 0.20, 0.25, 0.10, 0.15, 0.20],
    "VG": [0.95, 0.90, 0.95, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05],
}

weight_linguistic_vars = {
    "L":  [(0.20, 0.30, 0.20), (0.60, 0.70, 0.80), (0.45, 0.75, 0.75)],
    "ML": [(0.40, 0.30, 0.25), (0.45, 0.55, 0.40), (0.45, 0.60, 0.55)],
    "M":  [(0.50, 0.55, 0.55), (0.40, 0.45, 0.55), (0.35, 0.40, 0.35)],
    "H":  [(0.80, 0.75, 0.70), (0.20, 0.15, 0.30), (0.15, 0.10, 0.20)],
    "VH": [(0.90, 0.85, 0.95), (0.10, 0.15, 0.10), (0.05, 0.05, 0.10)],
}

# --- Yardımcı Fonksiyonlar ---
def get_t2nn_from_linguistic(value, is_weight=False):
    """Convert linguistic value to T2NN vector."""
    if pd.isna(value):
        return ((0, 0, 0), (0, 0, 0), (0, 0, 0))
    if is_weight:
        return tuple(tuple(weight_linguistic_vars.get(value.strip(), [0]*9)[i:i+3]) for i in range(0, 9, 3))
    else:
        return tuple(tuple(alternative_linguistic_vars.get(value.strip(), [0]*9)[i:i+3]) for i in range(0, 9, 3))

def merge_t2nn_vectors(t2nn_list):
    """Combine multiple T2NN vectors."""
    n = len(t2nn_list)
    merged = []
    for i in range(3):  # T, I, F
        avg = tuple(sum(vec[i][j] for vec in t2nn_list) / n for j in range(3))
        merged.append(avg)
    return tuple(merged)

def score_from_merged_t2nn(t2nn, is_benefit=True):
    """Calculate score from merged T2NN vector."""
    (a1, a2, a3), (b1, b2, b3), (g1, g2, g3) = t2nn
    score = (1 / 12) * (8 + (a1 + 2*a2 + a3) - (b1 + 2*b2 + b3) - (g1 + 2*g2 + g3))
    return score if is_benefit else -score

def min_max_normalization(df, criteria, criteria_types):
    """Apply min-max normalization to criteria columns."""
    normalized_df = df.copy()
    
    for crit in criteria:
        if criteria_types[crit] == "Benefit":
            col_max = df[crit].max()
            col_min = df[crit].min()
            normalized_df[crit] = (df[crit] - col_min) / (col_max - col_min)
        
        elif criteria_types[crit] == "Cost":
            col_max = df[crit].max()
            col_min = df[crit].min()
            normalized_df[crit] = (col_max - df[crit]) / (col_max - col_min)
    
    return normalized_df

def weighted_normalization(normalized_df, weights, criteria):
    """Apply weighted normalization to the decision matrix."""
    weighted_df = normalized_df.copy()
    for crit in criteria:
        weighted_df[crit] = normalized_df[crit] * weights[crit]  # Apply weight
    return weighted_df

def calculate_baa(weighted_df):
    """Calculate the Border Approximation Area (BAA)."""
    baa = []
    for idx, row in weighted_df.iterrows():
        baa_value = 1
        for crit in row:
            baa_value *= crit ** (1 / len(weighted_df.columns))  # Multiply each normalized value
        baa.append(baa_value)
    return baa

def calculate_distance_matrix(baa_values, weighted_df):
    """Calculate the distance matrix between alternatives."""
    distance_matrix = []
    for idx, baa_val in enumerate(baa_values):
        distances = []
        for row in weighted_df.iterrows():
            distance = abs(baa_val - row[1].sum())  # Ideal alternative distance
            distances.append(distance)
        distance_matrix.append(distances)
    return distance_matrix

def calculate_mabac_score(distance_matrix, baa_values):
    """Calculate the MABAC score for each alternative."""
    mabac_scores = []
    for i in range(len(distance_matrix)):
        score = 0
        for j in range(len(distance_matrix[i])):
            score += distance_matrix[i][j] * baa_values[j]  # Multiply distance by BAA
        mabac_scores.append(score)
    return mabac_scores

# --- Streamlit Arayüzü ---
st.title("T2NN MABAC Alternatif ve Ağırlık Skorlama")

input_type = st.radio("Select Input Type", ("Excel", "Manual"))

if input_type == "Excel":
    uploaded_file = st.file_uploader("Upload your Excel file (.xlsx)", type=["xlsx"])

    if uploaded_file:
        xls = pd.ExcelFile(uploaded_file)

        raw_df = pd.read_excel(xls, "Alternatives", header=[0, 1])
        raw_df[['Alternative', 'DM']] = raw_df.iloc[:, :2].fillna(method='ffill')
        data = raw_df.drop(columns=['Alternative', 'DM'])
        data.index = pd.MultiIndex.from_frame(raw_df[['Alternative', 'DM']])
        data.columns.names = ['Criteria', 'DM']
        alt_df = data

        wt_df = pd.read_excel(xls, "Weights", index_col=0)
        wt_df = wt_df[wt_df.index.notna()]

        st.subheader("Loaded Alternatives Data")
        st.dataframe(alt_df.reset_index())

        st.subheader("Loaded Weights Data")
        st.dataframe(wt_df)

        alternatives = alt_df.index.get_level_values(0).unique()
        criteria = alt_df.columns.get_level_values(0).unique()
        decision_makers = alt_df.columns.get_level_values(1).unique()

        # Kriterler için sadece benefit/cost seçimi yapılır
        criteria_types = {}
        for crit in criteria:
            criteria_types[crit] = st.radio(f"Select if {crit} is Benefit or Cost", ("Benefit", "Cost"))

        # Normalizasyonu uygula
        normalized_df = min_max_normalization(alt_df, criteria, criteria_types)

        # Ağırlıklı normalizasyonu uygula
        weighted_df = weighted_normalization(normalized_df, wt_df, criteria)

        # BAA hesaplamasını yap
        baa_values = calculate_baa(weighted_df)

        # Mesafe matrisini hesapla
        distance_matrix = calculate_distance_matrix(baa_values, weighted_df)

        # MABAC skoru hesapla
        mabac_scores = calculate_mabac_score(distance_matrix, baa_values)

        # Sonuçları göster
        alt_scores = pd.DataFrame(mabac_scores, index=alternatives, columns=["MABAC Score"])
        st.subheader("MABAC Scores for Alternatives")
        st.dataframe(alt_scores)

elif input_type == "Manual":
    st.subheader("Manual Entry Section for Alternatives and Weights")

    alternatives = st.text_input("Enter Alternatives (comma separated)").split(",")
    criteria = st.text_input("Enter Criteria (comma separated)").split(",")
    decision_makers = st.text_input("Enter Decision Makers (comma separated)").split(",")

    # Kriterler için sadece benefit/cost seçimi yapılır
    criteria_types = {}
    for crit in criteria:
        criteria_types[crit] = st.radio(f"Select if {crit} is Benefit or Cost", ("Benefit", "Cost"))

    # Alternatif ve kriterler için manuel değer girişi
    manual_data = {}
    for alt in alternatives:
        for crit in criteria:
            for dm in decision_makers:
                manual_data[(alt.strip(), crit.strip(), dm.strip())] = st.text_input(f"Enter linguistic value for {alt.strip()} under {crit.strip()} by {dm.strip()}", "")

    manual_df = pd.DataFrame(manual_data)
    manual_df.index = pd.MultiIndex.from_tuples(manual_data.keys(), names=["Alternative", "Criteria", "DM"])

    # Normalizasyon, ağırlıklı normalizasyon ve skor hesaplamaları
    normalized_df = min_max_normalization(manual_df, criteria, criteria_types)
    weighted_df = weighted_normalization(normalized_df, manual_df, criteria)
    baa_values = calculate_baa(weighted_df)
    distance_matrix = calculate_distance_matrix(baa_values, weighted_df)
    mabac_scores = calculate_mabac_score(distance_matrix, baa_values)

    # Sonuçları göster
    alt_scores = pd.DataFrame(mabac_scores, index=alternatives, columns=["MABAC Score"])
    st.subheader("MABAC Scores for Alternatives")
    st.dataframe(alt_scores)
