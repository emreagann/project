import pandas as pd
import numpy as np
import streamlit as st

alternative_linguistic_vars = {
    "VB": {"alpha1": 0.20, "alpha2": 0.20, "alpha3": 0.10, "beta1": 0.65, "beta2": 0.80, "beta3": 0.85, "gamma1": 0.45, "gamma2": 0.80, "gamma3": 0.70},
    "B": {"alpha1": 0.35, "alpha2": 0.35, "alpha3": 0.10, "beta1": 0.50, "beta2": 0.75, "beta3": 0.80, "gamma1": 0.50, "gamma2": 0.75, "gamma3": 0.65},
    "MB": {"alpha1": 0.50, "alpha2": 0.30, "alpha3": 0.50, "beta1": 0.50, "beta2": 0.35, "beta3": 0.45, "gamma1": 0.45, "gamma2": 0.30, "gamma3": 0.60},
    "M": {"alpha1": 0.40, "alpha2": 0.45, "alpha3": 0.50, "beta1": 0.40, "beta2": 0.45, "beta3": 0.50, "gamma1": 0.35, "gamma2": 0.40, "gamma3": 0.45},
    "MG": {"alpha1": 0.60, "alpha2": 0.45, "alpha3": 0.50, "beta1": 0.20, "beta2": 0.15, "beta3": 0.25, "gamma1": 0.10, "gamma2": 0.25, "gamma3": 0.15},
    "G": {"alpha1": 0.70, "alpha2": 0.75, "alpha3": 0.80, "beta1": 0.15, "beta2": 0.20, "beta3": 0.25, "gamma1": 0.10, "gamma2": 0.15, "gamma3": 0.20},
    "VG": {"alpha1": 0.95, "alpha2": 0.90, "alpha3": 0.95, "beta1": 0.10, "beta2": 0.10, "beta3": 0.05, "gamma1": 0.05, "gamma2": 0.05, "gamma3": 0.05}
}

weight_linguistic_vars = {
    "L": {
        "alpha": [0.20, 0.30, 0.20],
        "beta":  [0.60, 0.70, 0.80],
        "gamma": [0.45, 0.75, 0.75]
    },
    "ML": {
        "alpha": [0.40, 0.30, 0.25],
        "beta":  [0.45, 0.55, 0.40],
        "gamma": [0.45, 0.60, 0.55]
    },
    "M": {
        "alpha": [0.50, 0.55, 0.55],
        "beta":  [0.40, 0.45, 0.55],
        "gamma": [0.35, 0.40, 0.35]
    },
    "H": {
        "alpha": [0.80, 0.75, 0.70],
        "beta":  [0.20, 0.15, 0.30],
        "gamma": [0.15, 0.10, 0.20]
    },
    "VH": {
        "alpha": [0.90, 0.85, 0.95],
        "beta":  [0.10, 0.15, 0.10],
        "gamma": [0.05, 0.05, 0.10]
    }
}

def linguistic_to_numeric(value, linguistic_vars):
    if isinstance(value, str):
        value = value.strip()
    vars = linguistic_vars.get(value, None)
    if vars:
        return [vars[f'alpha{i}'] for i in range(1, 4)], [vars[f'beta{i}'] for i in range(1, 4)], [vars[f'gamma{i}'] for i in range(1, 4)]
    else:
        return [0, 0, 0], [0, 0, 0], [0, 0, 0]
def generate_tif_table(alternatives_df, criteria_names, num_dms):
    columns = pd.MultiIndex.from_product([
        criteria_names,
        ['T', 'I', 'F'],
        ['α', 'β', 'γ']
    ], names=['Criteria', 'Type', 'Index'])
    
    tif_df = pd.DataFrame(index=range(num_dms), columns=columns)
    
    for dm in range(1, num_dms + 1):
        for crit in criteria_names:
            for alt_idx, alt_name in enumerate(alternatives_df['Alternatives']):
                value = alternatives_df.loc[alt_idx, f"{crit}_DM{dm}"]
                alpha, beta, gamma = linguistic_to_numeric(value, alternative_linguistic_vars)
                
                tif_df.loc[dm-1, (crit, 'T', 'α')] = alpha[0]
                tif_df.loc[dm-1, (crit, 'T', 'β')] = alpha[1]
                tif_df.loc[dm-1, (crit, 'T', 'γ')] = alpha[2]
                
                tif_df.loc[dm-1, (crit, 'I', 'α')] = beta[0]
                tif_df.loc[dm-1, (crit, 'I', 'β')] = beta[1]
                tif_df.loc[dm-1, (crit, 'I', 'γ')] = beta[2]
                
                tif_df.loc[dm-1, (crit, 'F', 'α')] = gamma[0]
                tif_df.loc[dm-1, (crit, 'F', 'β')] = gamma[1]
                tif_df.loc[dm-1, (crit, 'F', 'γ')] = gamma[2]
    
    tif_df.index = [f'DM{i+1}' for i in range(num_dms)]
    
    return tif_df
def weight_linguistic_to_numeric(value, weight_linguistic_vars):
    if isinstance(value, str):
        value = value.strip()
    vars = weight_linguistic_vars.get(value, None)
    if vars:
        alpha = vars['alpha']
        beta = vars['beta']
        gamma = vars['gamma']
        return alpha, beta, gamma
    else:
        return [0, 0, 0], [0, 0, 0], [0, 0, 0]
def combine_weight_values(weight_values, weight_linguistic_vars):
    alphas, betas, gammas = [], [], []
    for val in weight_values:
        alpha, beta, gamma = weight_linguistic_to_numeric(val, weight_linguistic_vars)
        alphas.append(alpha)
        betas.append(beta)
        gammas.append(gamma)
    
    combined_alpha = alphas[0]
    for i in range(1, len(alphas)):
        combined_alpha = [
            combined_alpha[j] + alphas[i][j] - (np.prod([a[j] for a in alphas])) for j in range(3)
        ]

    combined_beta = betas[0]
    for i in range(1, len(betas)):
        combined_beta = [
            combined_beta[j] * betas[i][j] for j in range(3)
        ]
        
    combined_gamma = gammas[0]
    for i in range(1, len(gammas)):
        combined_gamma = [
            combined_gamma[j] * gammas[i][j] for j in range(3)
        ]
    
    return combined_alpha, combined_beta, combined_gamma

def get_combined_weights_df(weight_df, weight_linguistic_vars):
    criteria = [col.strip() for col in weight_df.columns if col not in ['Decision Makers'] and not col.startswith('Unnamed')]
    combined_weights = {}
    
    for crit in criteria:
        values = weight_df[crit].tolist()
        combined_alpha, combined_beta, combined_gamma = combine_weight_values(values, weight_linguistic_vars)
        combined_weights[crit] = {
            'alpha': combined_alpha,
            'beta': combined_beta,
            'gamma': combined_gamma
        }
    return combined_weights

def combine_alternativevalues(row, num_criteria):
    combined_values = {}
    for i in range(1, num_criteria + 1):
        for j in range(1, 5):
            value_c1 = row[f'C{i}_DM{j}']  
            value_c2 = row[f'C{i+1}_DM{j}']  
            
            alpha_c1, beta_c1, gamma_c1 = linguistic_to_numeric(value_c1, alternative_linguistic_vars)
            alpha_c2, beta_c2, gamma_c2 = linguistic_to_numeric(value_c2, alternative_linguistic_vars)
            
            combined_values[f'alpha1_combined_DM{i}_DM{j}'] = alpha_c1[0] + alpha_c2[0] - (alpha_c1[0] * alpha_c2[0])
            combined_values[f'alpha2_combined_DM{i}_DM{j}'] = alpha_c1[1] + alpha_c2[1] - (alpha_c1[1] * alpha_c2[1])
            combined_values[f'alpha3_combined_DM{i}_DM{j}'] = alpha_c1[2] + alpha_c2[2] - (alpha_c1[2] * alpha_c2[2])
            
            combined_values[f'beta1_combined_DM{i}_DM{j}'] = beta_c1[0] * beta_c2[0]
            combined_values[f'beta2_combined_DM{i}_DM{j}'] = beta_c1[1] * beta_c2[1]
            combined_values[f'beta3_combined_DM{i}_DM{j}'] = beta_c1[2] * beta_c2[2]
            
            combined_values[f'gamma1_combined_DM{i}_DM{j}'] = gamma_c1[0] * gamma_c2[0]
            combined_values[f'gamma2_combined_DM{i}_DM{j}'] = gamma_c1[1] * gamma_c2[1]
            combined_values[f'gamma3_combined_DM{i}_DM{j}'] = gamma_c1[2] * gamma_c2[2]
    
    return combined_values


def calculate_score(values, num_criteria):
    alpha_sum = 0
    beta_sum = 0
    gamma_sum = 0
    
    for i in range(1, num_criteria + 1):
        alpha_sum += values[f'alpha1_combined_DM{i}'] + 2 * values[f'alpha2_combined_DM{i}'] + values[f'alpha3_combined_DM{i}']
        beta_sum += values[f'beta1_combined_DM{i}'] + 2 * values[f'beta2_combined_DM{i}'] + values[f'beta3_combined_DM{i}']
        gamma_sum += values[f'gamma1_combined_DM{i}'] + 2 * values[f'gamma2_combined_DM{i}'] + values[f'gamma3_combined_DM{i}']
    
    score = (1 / 12) * (8 + alpha_sum - beta_sum - gamma_sum)
    return score



def calculate_alternative_scores(alternatives_df):
    scores = []
    for index, row in alternatives_df.iterrows():
        alternative = row['Alternatives']
        combined_values = combine_alternativevalues(row)
        score = calculate_score(combined_values)
        scores.append({'Alternative': alternative, 'Score': score})
    return pd.DataFrame(scores)

def transform_alternatives_df(df, num_criteria):
    df = df[['Alternatives', 'Decision Makers'] + [f'C{i}' for i in range(1, num_criteria + 1)]]
    df['Alternatives'] = df['Alternatives'].fillna(method='ffill')
    df['DM'] = df['Decision Makers'].str.extract(r'DM\s*:?\s*(\d+)').astype(int)
    result = {}
    for alt in df['Alternatives'].unique():
        alt_df = df[df['Alternatives'] == alt]
        row = {'Alternatives': alt}
        for _, r in alt_df.iterrows():
            for i in range(1, num_criteria + 1):
                row[f'C{i}_DM{r.DM}'] = r[f'C{i}']
        result[alt] = row
    return pd.DataFrame(result.values())


def combine_weight_values(weight_values, weight_linguistic_vars):
    alphas, betas, gammas = [], [], []
    for val in weight_values:
        alpha, beta, gamma = weight_linguistic_to_numeric(val, weight_linguistic_vars)
        alphas.append(alpha)
        betas.append(beta)
        gammas.append(gamma)
    combined_alpha = alphas[0]
    combined_beta = betas[0]
    combined_gamma = gammas[0]
    for i in range(1, len(alphas)):
        combined_alpha = [
            combined_alpha[j] + alphas[i][j] - (combined_alpha[j] * alphas[i][j]) for j in range(3)
        ]
        combined_beta = [
            combined_beta[j] * betas[i][j] for j in range(3)
        ]
        combined_gamma = [
            combined_gamma[j] * gammas[i][j] for j in range(3)
        ]
    return combined_alpha, combined_beta, combined_gamma

def get_combined_weights_df(weight_df, weight_linguistic_vars):
    criteria = [col.strip() for col in weight_df.columns if col not in ['Decision Makers'] and not col.startswith('Unnamed')]
    combined_weights = {}
    for crit in criteria:
        values = weight_df[crit].tolist()
        combined_alpha, combined_beta, combined_gamma = combine_weight_values(values, weight_linguistic_vars)
        combined_weights[crit] = {
            'alpha': combined_alpha,
            'beta': combined_beta,
            'gamma': combined_gamma
        }
    return combined_weights

def normalize_decision_matrix(data, criteria_types):
    normalized_matrix = np.zeros_like(data, dtype=float)
    for j, crit in enumerate(criteria_types):
        if criteria_types[crit] == 'Benefit':
            normalized_matrix[:, j] = (data[:, j] - data[:, j].min()) / (data[:, j].max() - data[:, j].min())
        else: 
            normalized_matrix[:, j] = (data[:, j].max() - data[:, j]) / (data[:, j].max() - data[:, j].min())
    return normalized_matrix


def weighted_decision_matrix(normalized_matrix, combined_weights, criteria):
    weighted_matrix = np.zeros_like(normalized_matrix, dtype=float)
    for i, alt in enumerate(normalized_matrix):
        for j, crit in enumerate(criteria):
            weight = np.mean(combined_weights[crit]['alpha']) + np.mean(combined_weights[crit]['beta']) + np.mean(combined_weights[crit]['gamma'])
            weighted_matrix[i, j] = normalized_matrix[i, j] * weight
    return weighted_matrix


def border_approximation_area(weighted_matrix):
    m = len(weighted_matrix) 
    baa = []
    
    for j in range(weighted_matrix.shape[1]): 
        product = 1
        for i in range(m): 
            product *= weighted_matrix[i, j]
        baa.append(product ** (1/m))  
    
    return baa


def distance_matrix(weighted_matrix, baa):
    dist_matrix = np.zeros_like(weighted_matrix, dtype=float)
    for i in range(weighted_matrix.shape[0]): 
        dist_matrix[i, :] = np.abs(weighted_matrix[i, :] - baa)
    return dist_matrix


def mabac_score(distance_matrix):
    scores = distance_matrix.sum(axis=1)
    return scores


def mabac(alternatives_df, combined_weights, criteria_types, num_criteria):
    alternatives = alternatives_df['Alternatives'].tolist()
    criteria = [f'C{i}' for i in range(1, num_criteria + 1)]
    m = len(alternatives)
    n = len(criteria)

    st.write("### T-I-F (Truth-Indeterminacy-Falsity) Values")
    tif_df = generate_tif_table(alternatives_df, criteria, 4)  
    st.dataframe(tif_df.style.format("{:.2f}"))

    data = []
    for idx, row in alternatives_df.iterrows():
        row_data = []
        for crit in criteria:
            vals = [row.get(f'{crit}_DM{dm}', None) for dm in range(1, 5)]
            numeric_vals = [linguistic_to_numeric(v, alternative_linguistic_vars)[0][0] for v in vals if v is not None]
            row_data.append(np.mean(numeric_vals))
        data.append(row_data)
    data = np.array(data)

    st.write("### Step 1: Normalize the Decision Matrix")
    normalized_matrix = normalize_decision_matrix(data, criteria_types)
    normalized_df = pd.DataFrame(normalized_matrix, index=alternatives, columns=criteria)
    st.dataframe(normalized_df)
    
    st.write("### Step 2: Weighted Decision Matrix")
    weighted_matrix = weighted_decision_matrix(normalized_matrix, combined_weights, criteria)
    weighted_df = pd.DataFrame(weighted_matrix, index=alternatives, columns=criteria)
    st.dataframe(weighted_df)
    
    st.write("### Step 3: Border Approximation Area (BAA)")
    baa = border_approximation_area(weighted_matrix)
    baa_df = pd.DataFrame([baa], columns=criteria, index=['BAA'])
    st.dataframe(baa_df)
    
    st.write("### Step 4: Distance Matrix")
    dist_matrix = distance_matrix(weighted_matrix, baa)
    dist_df = pd.DataFrame(dist_matrix, index=alternatives, columns=criteria)
    st.dataframe(dist_df)
    
    st.write("### Step 5: MABAC Scores")
    scores = mabac_score(dist_matrix)
    result_df = pd.DataFrame({'Alternative': alternatives, 'Score': scores})
    result_df = result_df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    result_df['Rank'] = result_df.index + 1
    st.dataframe(result_df)

    return result_df

st.title("T2NN Mabac Calculation System")

input_method = st.radio("Select a Enterence:", ["Upload your Excel File", "Manual Enterence"])
def get_criteria_from_excel(df):
     criteria_columns = [col for col in df.columns if col.startswith('C')]
     return criteria_columns
if input_method == "Upload your Excel File":
    uploaded_file = st.file_uploader("Upload your Excel File", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name=None)
        alternatives_df = df['Alternatives']
        criteria_columns = get_criteria_from_excel(alternatives_df)
        num_criteria = len(criteria_columns)
        alternatives_df = transform_alternatives_df(alternatives_df, num_criteria)
        alternatives = alternatives_df['Alternatives'].tolist()

        if 'Weights' in df:
            weights_df = df['Weights']
            criteria = [col.strip() for col in weights_df.columns if col not in ['Decision Makers'] and not col.startswith('Unnamed')]
            criteria_types = {}
            for idx, crit in enumerate(criteria):
                crit = crit.strip()
                unique_key = f"type_{crit}_{idx}_{hash(crit)}"
                criteria_types[crit] = st.selectbox(
                    f"{crit} type",
                    options=["Benefit", "Cost"],
                    key=unique_key
                )
            combined_weights = get_combined_weights_df(weights_df, weight_linguistic_vars)
        elif 'Criteria Weights' in df:
            weights_df = df['Criteria Weights']
            criteria = [col.strip() for col in weights_df.columns if col not in ['Decision Makers'] and not col.startswith('Unnamed')]
            criteria_types = {}
            for idx, crit in enumerate(criteria):
                crit = crit.strip()
                unique_key = f"type_{crit}_{idx}_{hash(crit)}"
                criteria_types[crit] = st.selectbox(
                    f"{crit} criteria type",
                    options=["Benefit", "Cost"],
                    key=unique_key
                )
            combined_weights = get_combined_weights_df(weights_df, weight_linguistic_vars)
        
        st.write("Results")
        results = mabac(alternatives_df, combined_weights, criteria_types, num_criteria)

else:
    st.subheader("Manual Data Enterence")
    
    num_alternatives = st.number_input("Alternative Number:", min_value=2, value=3, key="num_alternatives")
    
    num_criteria = st.number_input("Criteria Number:", min_value=2, value=2, key="num_criteria")
    
    num_dms = st.number_input("Decision Maker Number:", min_value=1, value=4, key="num_dms")
    
    st.subheader("Alternative Names")
    alternative_names = []
    for i in range(num_alternatives):
        name = st.text_input(f"Alternative {i+1} Name:", key=f"alt_name_{i}")
        alternative_names.append(name)
    

    st.subheader("Criteria Names")
    criteria_names = []
    criteria_types = {}
   

    for i in range(num_criteria):
        name = st.text_input(f"Criteria {i+1} Name:", key=f"crit_name_{i}")
        criteria_names.append(name)
        criteria_types[name] = st.selectbox(
            f"Criteria {i+1} Type:",
            options=["Benefit", "Cost"],
            key=f"crit_type_{i}"
        )
    
    st.subheader("Alternative Evaulation")
    alternatives_data = []
    
    for alt_idx, alt_name in enumerate(alternative_names):
        st.write(f"### {alt_name} Evaulation")
        alt_data = {"Alternatives": alt_name}
        
        for crit_idx, crit_name in enumerate(criteria_names):
            for dm in range(1, num_dms + 1):
                value = st.selectbox(
                    f"{crit_name} - DM{dm} Evaulation:",
                    options=list(alternative_linguistic_vars.keys()),
                    key=f"eval_alt_{alt_idx}_crit_{crit_idx}_dm_{dm}"
                )
                alt_data[f"{crit_name}_DM{dm}"] = value
        
        alternatives_data.append(alt_data)
    
    st.subheader("Criteria Weights")
    weights_data = []
    
    for dm in range(1, num_dms + 1):
        st.write(f"### DM{dm} Weight Evaulations")
        dm_data = {"Decision Makers": f"DM{dm}"}
        
        for crit_idx, crit_name in enumerate(criteria_names):
            weight = st.selectbox(
                f"{crit_name} Weight:",
                options=list(weight_linguistic_vars.keys()),
                key=f"weight_dm_{dm}_crit_{crit_idx}"
            )
            dm_data[crit_name] = weight
        
        weights_data.append(dm_data)
    
    if st.button("Calculate", key="calculate_button"):
        alternatives_df = pd.DataFrame(alternatives_data)
        weights_df = pd.DataFrame(weights_data)
        
        combined_weights = get_combined_weights_df(weights_df, weight_linguistic_vars)
        
        st.write("### Alternatives Score")
        results = mabac(alternatives_df, combined_weights, criteria_types, num_criteria)
