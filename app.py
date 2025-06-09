import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# T2NN dönüşüm fonksiyonu
def convert_to_t2nn(value_range):
    a, b = value_range
    m = (a + b) / 2
    T = (a / 10, m / 10, b / 10)
    I = (0.0125, 0.0125, 0.0125)
    F = (1 - b / 10, 1 - m / 10, 1 - a / 10)
    return T, I, F

# Normalizasyon fonksiyonu
def normalize_values(values, value_type='benefit'):
    min_val, max_val = min(values), max(values)
    if value_type == 'benefit':
        return [(v - min_val) / (max_val - min_val) for v in values]
    elif value_type == 'cost':
        return [(max_val - v) / (max_val - min_val) for v in values]
    return values

# Ağırlıklı normalizasyon karar matrisi
def weighted_normalized_matrix(normalized_matrix, weights):
    return np.array([list(normalized_matrix.values())]).T * np.array(weights)

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

# Veri dosyasını yükleme
uploaded_file = st.file_uploader("Veri dosyasını yükleyin (Excel veya CSV)", type=["csv", "xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    st.write("Veri Tablosu:")
    st.dataframe(data)

    # Alternatifler ve kriterler verilerini işleme
    alternatives = data.iloc[:, 0]  # İlk sütun alternatifler
    criteria = data.columns[1:]     # Kriterler

    # Kriter türleri (cost veya benefit) verisi alalım
    criteria_types = st.multiselect("Kriter türlerini seçin (Benefit veya Cost)", criteria)
    
    # Kullanıcıdan kriter ağırlıklarını alma
    weights = st.slider("Kriter ağırlıklarını girin (toplam 1 olmalı)", 0.0, 1.0, (0.2, 0.2, 0.2, 0.2), step=0.05)

    # T2NN dönüşüm ve normalizasyon işlemi
    transformed_values = {}
    for criterion in criteria:
        criterion_values = data[criterion].tolist()
        transformed_values[criterion] = convert_to_t2nn(criterion_values)
    
    normalized_matrix = {}
    for criterion, values in transformed_values.items():
        if criterion in criteria_types:
            normalized_matrix[criterion] = normalize_values(values, value_type='benefit')
        else:
            normalized_matrix[criterion] = normalize_values(values, value_type='cost')
    
    # MABAC işlemleri
    weighted_matrix = weighted_normalized_matrix(normalized_matrix, weights)
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
