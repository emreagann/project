import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# T2NN skor hesaplama fonksiyonu (Definition 4)
def t2nn_score(t2nn):
    T, I, F = t2nn
    return (1/12) * (
        8 + (T[0] + 2*T[1] + T[2]) - (I[0] + 2*I[1] + I[2]) - (F[0] + 2*F[1] + F[2])
    )

# T2NN dönüşüm fonksiyonu
def convert_to_t2nn(value, term_dict):
    return term_dict.get(value, ((0, 0, 0), (0, 0, 0), (0, 0, 0)))

# Normalizasyon fonksiyonu
def normalize_values(values, value_type='benefit'):
    min_val, max_val = min(values), max(values)
    if min_val == max_val:
        return [1 for _ in values]
    if value_type == 'benefit':
        return [(v - min_val) / (max_val - min_val) for v in values]
    elif value_type == 'cost':
        return [(max_val - v) / (max_val - min_val) for v in values]
    return values

# Ağırlıklı normalize karar matrisi
def weighted_normalized_matrix(normalized_matrix, weights):
    df = pd.DataFrame(normalized_matrix)
    weighted = df * weights
    return weighted.to_numpy()

# Border area hesaplama
def calculate_border_area(weighted_matrix):
    return np.prod(weighted_matrix, axis=0) ** (1 / weighted_matrix.shape[0])

# Mesafe matrisi
def calculate_distance_matrix(weighted_matrix, border_area):
    return weighted_matrix - border_area

# Final skorlar
def final_scores(distance_matrix):
    return np.sum(distance_matrix, axis=1)

# Streamlit arayüzü
st.title("MABAC Yöntemi Uygulaması")

# T2NN dilsel terimler
linguistic_to_t2nn_criteria = {
    "Low": ((0.20, 0.30, 0.20), (0.60, 0.70, 0.80), (0.45, 0.75, 0.75)),
    "MediumLow": ((0.40, 0.30, 0.25), (0.45, 0.55, 0.40), (0.45, 0.60, 0.55)),
    "Medium": ((0.50, 0.55, 0.55), (0.40, 0.45, 0.55), (0.35, 0.40, 0.35)),
    "High": ((0.80, 0.75, 0.70), (0.20, 0.15, 0.30), (0.15, 0.10, 0.20)),
    "VeryHigh": ((0.90, 0.85, 0.95), (0.10, 0.15, 0.10), (0.05, 0.05, 0.10))
}

linguistic_to_t2nn_alternatives = {
    "VeryBad": ((0.20, 0.20, 0.10), (0.65, 0.80, 0.85), (0.45, 0.80, 0.70)),
    "Bad": ((0.35, 0.35, 0.10), (0.50, 0.75, 0.80), (0.50, 0.75, 0.65)),
    "MediumBad": ((0.50, 0.30, 0.50), (0.50, 0.35, 0.45), (0.45, 0.30, 0.60)),
    "Medium": ((0.40, 0.45, 0.50), (0.40, 0.45, 0.50), (0.35, 0.40, 0.45)),
    "MediumGood": ((0.60, 0.45, 0.50), (0.20, 0.15, 0.25), (0.10, 0.25, 0.15)),
    "Good": ((0.70, 0.75, 0.80), (0.15, 0.20, 0.25), (0.10, 0.15, 0.20)),
    "VeryGood": ((0.95, 0.90, 0.95), (0.10, 0.10, 0.05), (0.05, 0.05, 0.05))
}

uploaded_file = st.file_uploader("Veri dosyasını yükleyin (Excel veya CSV)", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.write("Excel Dosyasından Yüklenen Veri Tablosu:")
    st.dataframe(data)

    alternatives = data.iloc[:, 0].tolist()
    criteria = data.columns[1:].tolist()

    # Otomatik kriter türleri belirleniyor (sabit: benefit/cost bilgisi excelde)
    cost_criteria = ["C4", "C5", "C8", "C9"]
    benefit_criteria = [c for c in criteria if c not in cost_criteria]

    st.subheader("Kriter Ağırlıklarını Girin")
    criteria_weights = {}
    for criterion in criteria:
        weight = st.selectbox(f"{criterion} için ağırlık girin", list(linguistic_to_t2nn_criteria.keys()))
        criteria_weights[criterion] = convert_to_t2nn(weight, linguistic_to_t2nn_criteria)

    transformed_values = {}
    for criterion in criteria:
        linguistic_values = data[criterion].tolist()
        transformed_values[criterion] = [convert_to_t2nn(value, linguistic_to_t2nn_alternatives) for value in linguistic_values]

    normalized_matrix = {}
    for criterion, values in transformed_values.items():
        kind = 'cost' if criterion in cost_criteria else 'benefit'
        normalized_matrix[criterion] = normalize_values([val[0][0] for val in values], value_type=kind)

    weight_vector = [t2nn_score(w) for w in criteria_weights.values()]

    weighted_matrix = weighted_normalized_matrix(normalized_matrix, weight_vector)
    border_area = calculate_border_area(weighted_matrix)
    distance_matrix = calculate_distance_matrix(weighted_matrix, border_area)
    scores = final_scores(distance_matrix)

    st.write("Alternatiflerin Nihai Sıralamaları:")
    sorted_scores = pd.DataFrame({"Alternatif": alternatives, "Skor": scores})
    sorted_scores = sorted_scores.sort_values(by="Skor", ascending=False)
    st.dataframe(sorted_scores)

    st.subheader("Alternatiflerin Skor Dağılımı")
    fig, ax = plt.subplots()
    ax.bar(sorted_scores['Alternatif'], sorted_scores['Skor'])
    st.pyplot(fig)

else:
    st.write("Excel dosyasını yüklemediniz. Lütfen verileri manuel olarak girin.")
