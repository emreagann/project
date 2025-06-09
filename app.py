import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# T2NN dönüşüm fonksiyonu
def convert_to_t2nn(value, term_dict):
    return term_dict.get(value, ((0, 0, 0), (0, 0, 0), (0, 0, 0)))

# Normalizasyon fonksiyonu
def normalize_values(values, value_type='benefit'):
    min_val, max_val = min(values), max(values)

    # Eğer min_val ile max_val aynıysa, tüm değerleri 1 olarak kabul et
    if min_val == max_val:
        return [1 for v in values]

    if value_type == 'benefit':
        return [(v - min_val) / (max_val - min_val) for v in values]
    elif value_type == 'cost':
        return [(max_val - v) / (max_val - min_val) for v in values]
    return values

# Ağırlıklı normalizasyon karar matrisi
def weighted_normalized_matrix(normalized_matrix, weights):
    weighted_matrix = np.array([list(normalized_matrix.values())]).T * np.array(weights)
    return weighted_matrix

# Border area hesaplama fonksiyonu
def calculate_border_area(weighted_matrix):
    return np.prod(weighted_matrix, axis=0) ** (1 / weighted_matrix.shape[0])

# Mesafe matrisi hesaplama
def calculate_distance_matrix(weighted_matrix, border_area):
    return weighted_matrix - border_area

# Sonuçları hesaplama
def final_scores(distance_matrix):
    return np.sum(distance_matrix, axis=1)

# Streamlit arayüzü
st.title("MABAC Yöntemi Uygulaması")

# T2NN dilsel terimlerini içeren sözlük
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

# Kullanıcıdan Excel dosyasını yükleme ya da manuel giriş yapma
uploaded_file = st.file_uploader("Veri dosyasını yükleyin (Excel veya CSV)", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    st.write("Excel Dosyasından Yüklenen Veri Tablosu:")
    st.dataframe(data)

    # Alternatifler ve kriterler verilerini işleme
    alternatives = data.iloc[:, 0]  # İlk sütun alternatifler
    criteria = data.columns[1:]     # Kriterler

    # Kriter türleri (cost veya benefit) verisi alalım
    criteria_types = st.multiselect("Kriter türlerini seçin (Benefit veya Cost)", criteria)
    
    # Kullanıcıdan kriter ağırlıklarını manuel girme
    st.subheader("Kriter Ağırlıklarını Girin")
    criteria_weights = {}
    for criterion in criteria:
        weight = st.selectbox(f"{criterion} için ağırlık girin", ["Low", "MediumLow", "Medium", "High", "VeryHigh"])
        criteria_weights[criterion] = convert_to_t2nn(weight, linguistic_to_t2nn_criteria)

    # T2NN dönüşüm ve normalizasyon işlemi
    transformed_values = {}
    for criterion in criteria:
        linguistic_values = data[criterion].tolist()
        transformed_values[criterion] = [convert_to_t2nn(value, linguistic_to_t2nn_criteria) for value in linguistic_values]

    normalized_matrix = {}
    for criterion, values in transformed_values.items():
        if criterion in criteria_types:
            normalized_matrix[criterion] = normalize_values([value[0][0] for value in values], value_type='benefit')  # Benefit türü
        else:
            normalized_matrix[criterion] = normalize_values([value[0][0] for value in values], value_type='cost')  # Cost türü
    
    # MABAC işlemleri
    weighted_matrix = weighted_normalized_matrix(normalized_matrix, list(criteria_weights.values()))
    border_area = calculate_border_area(weighted_matrix)
    distance_matrix = calculate_distance_matrix(weighted_matrix, border_area)
    scores = final_scores(distance_matrix)

    # Sonuçları görselleştirme
    st.write("Alternatiflerin Nihai Sıralamaları:")
    sorted_scores = pd.DataFrame({"Alternatif": alternatives, "Skor": scores})
    sorted_scores = sorted_scores.sort_values(by="Skor", ascending=False)
    st.dataframe(sorted_scores)

    # Grafikle gösterim
    st.subheader("Alternatiflerin Skor Dağılımı")
    fig, ax = plt.subplots()
    ax.bar(sorted_scores['Alternatif'], sorted_scores['Skor'])
    st.pyplot(fig)

else:
    st.write("Excel dosyasını yüklemediniz. Lütfen verileri manuel olarak girin.")

    # Manuel veri girişi
    num_alternatives = st.number_input("Alternatif sayısını girin", min_value=1, value=3)
    num_criteria = st.number_input("Kriter sayısını girin", min_value=1, value=3)

    alternatives = []
    for i in range(num_alternatives):
        alt = st.text_input(f"Alternatif {i+1} ismini girin", f"A{i+1}")
        alternatives.append(alt)

    criteria = []
    for i in range(num_criteria):
        crit = st.text_input(f"Kriter {i+1} ismini girin", f"C{i+1}")
        criteria.append(crit)

    # Alternatifler ve kriterler için dilsel terimler girişi
    decision_matrix = {}

    for i in range(num_criteria):
        values = []
        st.write(f"{criteria[i]} için alternatiflerin değerlerini girin (örn: 'Low', 'Medium', 'High')")
        for j in range(num_alternatives):
            value = st.selectbox(f"Alternatif {alternatives[j]} için {criteria[i]} değeri", ["VeryBad", "Bad", "MediumBad", "Medium", "MediumGood", "Good", "VeryGood"])
            values.append(value)
        decision_matrix[criteria[i]] = values

    # Kriter ağırlıkları girişi
    weights = []
    st.write("Kriter ağırlıklarını girin (örn: 'Low', 'Medium', 'High')")
    for i in range(num_criteria):
        weight = st.selectbox(f"{criteria[i]} için ağırlık", ["Low", "MediumLow", "Medium", "High", "VeryHigh"])
        weights.append(convert_to_t2nn(weight, linguistic_to_t2nn_criteria))

    # T2NN dönüşümü ve normalizasyon işlemi
    transformed_values = {}
    for criterion in criteria:
        transformed_values[criterion] = [convert_to_t2nn(value, linguistic_to_t2nn_alternatives) for value in decision_matrix[criterion]]

    normalized_matrix = {}
    for criterion, values in transformed_values.items():
        normalized_matrix[criterion] = normalize_values([value[0][0] for value in values], value_type='benefit')

    # MABAC işlemleri
    weighted_matrix = weighted_normalized_matrix(normalized_matrix, [weight[0][0] for weight in weights])
    border_area = calculate_border_area(weighted_matrix)
    distance_matrix = calculate_distance_matrix(weighted_matrix, border_area)
    scores = final_scores(distance_matrix)

    # Sonuçları görselleştirme
    st.write("Alternatiflerin Nihai Sıralamaları:")
    sorted_scores = pd.DataFrame({"Alternatif": alternatives, "Skor": scores})
    sorted_scores = sorted_scores.sort_values(by="Skor", ascending=False)
    st.dataframe(sorted_scores)

    # Grafikle gösterim
    st.subheader("Alternatiflerin Skor Dağılımı")
    fig, ax = plt.subplots()
    ax.bar(sorted_scores['Alternatif'], sorted_scores['Skor'])
    st.pyplot(fig)
