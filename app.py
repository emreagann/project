import streamlit as st
import pandas as pd

# --- T2NN sözlükleri ---
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
def score_function(values, is_benefit=True):
    a1, a2, a3, b1, b2, b3, g1, g2, g3 = values
    score = (1 / 12) * (8 + (a1 + 2 * a2 + a3) - (b1 + 2 * b2 + b3) - (g1 + 2 * g2 + g3))
    return score if is_benefit else -score

def score_from_merged_t2nn(t2nn, is_benefit=True):
    (a1, a2, a3), (b1, b2, b3), (g1, g2, g3) = t2nn
    score = (1 / 12) * (8 + (a1 + 2*a2 + a3) - (b1 + 2*b2 + b3) - (g1 + 2*g2 + g3))
    return score if is_benefit else -score

def get_t2nn_from_linguistic(value):
    if pd.isna(value):
        return ((0, 0, 0), (0, 0, 0), (0, 0, 0))
    return tuple(tuple(alternative_linguistic_vars.get(value.strip(), [0]*9)[i:i+3]) for i in range(0, 9, 3))

def get_weight_t2nn_from_linguistic(value):
    if pd.isna(value):
        return ((0, 0, 0), (0, 0, 0), (0, 0, 0))
    return weight_linguistic_vars.get(value.strip(), ((0,0,0), (0,0,0), (0,0,0)))

def merge_t2nn_vectors(t2nn_list):
    n = len(t2nn_list)
    merged = []
    for i in range(3):  # T, I, F
        avg = tuple(sum(vec[i][j] for vec in t2nn_list) / n for j in range(3))
        merged.append(avg)
    return tuple(merged)

def t2nn_addition(a, b):
    return tuple(
        tuple(a[i][j] + b[i][j] for j in range(3)) for i in range(3)
    )

def combine_weights_t2nns(weight_list):
    combined = weight_list[0]
    for w in weight_list[1:]:
        combined = t2nn_addition(combined, w)
    return tuple(tuple(x / len(weight_list) for x in comp) for comp in combined)

def zero_out_I_and_F(t2nn):
    T, _, _ = t2nn
    zero = (0.0, 0.0, 0.0)
    return (T, zero, zero)

def combine_multiple_decision_makers(alt_df, decision_makers, criteria, alternatives, is_benefit=True):
    combined_results = {}

    for alt in alternatives:
        for crit in criteria:
            t2nns = []
            for dm in decision_makers:
                try:
                    val = alt_df.loc[(alt, dm), (crit, dm)]
                except KeyError:
                    val = None
                t2nns.append(get_t2nn_from_linguistic(val))
            
            merged_t2nn = merge_t2nn_vectors(t2nns)
            score = score_from_merged_t2nn(merged_t2nn, is_benefit)
            combined_results[(alt, crit)] = round(score, 4)
    
    return combined_results

def combine_weights(alt_df, decision_makers, criteria):
    combined_weights = {}

    for crit in criteria:
        weight_list = []
        
        for dm in decision_makers:
            try:
                val = alt_df.loc[(crit, dm)]
            except KeyError:
                val = None
            weight_list.append(get_weight_t2nn_from_linguistic(val))
        
        merged_weight = combine_weights_t2nns(weight_list)
        score = score_from_merged_t2nn(merged_weight)
        combined_weights[crit] = round(score, 4)
    
    return combined_weights

# --- Streamlit Arayüzü ---
st.title("T2NN MABAC Alternatif ve Ağırlık Skorlama")

# Kullanıcıya seçim yaptırmak için seçenek ekleyelim: Excel veya manuel giriş
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

        # Kriterler için benefit/cost seçimi
        criteria_types = {}
        for crit in criteria:
            criteria_types[crit] = st.radio(f"Select if {crit} is Benefit or Cost", ("Benefit", "Cost"))

        # Decision Type
        decision_type = st.radio("Select Decision Type", ("Benefit", "Cost"))
        is_benefit = decision_type == "Benefit"

        # Karar vericilerin birleşik sonuçlarını hesapla
        combined_results = combine_multiple_decision_makers(alt_df, decision_makers, criteria, alternatives, is_benefit)

        # Sonuçları görüntüle
        alt_scores = pd.DataFrame.from_dict(combined_results, orient='index', columns=["Score"])

        st.subheader("Decision Matrix (Combined Scores)")
        st.dataframe(alt_scores)

        # Ağırlıkların birleşik skorlarını hesapla
        combined_weights = combine_weights(wt_df, decision_makers, criteria)

        # Ağırlık skorlarını görüntüle
        weight_scores = pd.DataFrame.from_dict(combined_weights, orient='index', columns=["Score"])

        st.subheader("Combined Weights (Scores)")
        st.dataframe(weight_scores)

elif input_type == "Manual":
    st.subheader("Manual Entry Section for Alternatives and Weights")

    # Kullanıcıdan manuel veri alalım
    alternatives = st.text_input("Enter Alternatives (comma separated)").split(",")
    criteria = st.text_input("Enter Criteria (comma separated)").split(",")
    decision_makers = st.text_input("Enter Decision Makers (comma separated)").split(",")

    # Kriterler için benefit/cost seçimi
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

    combined_results = combine_multiple_decision_makers(manual_df, decision_makers, criteria, alternatives, is_benefit)
    alt_scores = pd.DataFrame.from_dict(combined_results, orient='index', columns=["Score"])

    st.subheader("Decision Matrix (Combined Scores)")
    st.dataframe(alt_scores)

    combined_weights = combine_weights(manual_df, decision_makers, criteria)
    weight_scores = pd.DataFrame.from_dict(combined_weights, orient='index', columns=["Score"])

    st.subheader("Combined Weights (Scores)")
    st.dataframe(weight_scores)
