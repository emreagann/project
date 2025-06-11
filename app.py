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
        # Kriterin türüne göre normalizasyon yapıyoruz
        if criteria_types[crit] == "Benefit":
            # NaN kontrolü yapalım
            col_max = df[crit].max(skipna=True)  # NaN değerleri atla
            col_min = df[crit].min(skipna=True)  # NaN değerleri atla
            normalized_df[crit] = (df[crit] - col_min) / (col_max - col_min) if col_max != col_min else 0
        
        elif criteria_types[crit] == "Cost":
            col_max = df[crit].max(skipna=True)  # NaN değerleri atla
            col_min = df[crit].min(skipna=True)  # NaN değerleri atla
            normalized_df[crit] = (col_max - df[crit]) / (col_max - col_min) if col_max != col_min else 0
    
    return normalized_df

def combine_multiple_decision_makers(alt_df, decision_makers, criteria, alternatives, criteria_types):
    combined_results = {}

    for alt in alternatives:
        for crit in criteria:
            t2nns = []
            is_benefit = criteria_types[crit] == "Benefit"  # Benefit veya Cost
            for dm in decision_makers:
                try:
                    # Her karar vericinin dilsel değeri T2NN'ye dönüştürülür
                    val = alt_df.loc[(alt, dm), (crit, dm)]
                except KeyError:
                    val = None
                t2nns.append(get_t2nn_from_linguistic(val))
            
            # Karar vericilerin T2NN vektörleri birleştirilir
            merged_t2nn = merge_t2nn_vectors(t2nns)
            score = score_from_merged_t2nn(merged_t2nn, is_benefit)
            combined_results[(alt, crit)] = round(score, 4)
    
    return combined_results

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
