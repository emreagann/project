import streamlit as st
import pandas as pd

# Skor fonksiyonunu tanımlayalım
def calculate_score(linguistic_scores):
    # linguistic_scores: [αα, αβ, αY, βα, ββ, βY, γα, γβ, γγ]
    alpha_alpha, alpha_beta, alpha_gamma, beta_alpha, beta_beta, beta_gamma, gamma_alpha, gamma_beta, gamma_gamma = linguistic_scores
    return (1/12) * (8 + (alpha_alpha + 2*alpha_beta + alpha_gamma) - (beta_alpha + 2*beta_beta + beta_gamma) - (gamma_alpha + 2*gamma_beta + gamma_gamma))

# Normalizasyon işlemi
def normalize_matrix(matrix, is_benefit=True):
    if is_benefit:
        # Fayda normalizasyonu (max-min normalizasyonu)
        return matrix.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    else:
        # Maliyet normalizasyonu (max-min normalizasyonu)
        return matrix.apply(lambda x: (x.max() - x) / (x.max() - x.min()), axis=0)

# Streamlit arayüzü
st.title("MABAC Karar Matrisi Hesaplama")

uploaded_file = st.file_uploader("Excel dosyasını yükleyin", type=["xlsx"])

if uploaded_file is not None:
    # Excel dosyasını oku
    df = pd.read_excel(uploaded_file, sheet_name="Alternatives")
    weights_df = pd.read_excel(uploaded_file, sheet_name="Weights")

    # Alternatifler ve Ağırlıklar sayfalarındaki sütun adlarını kontrol et
    st.write("Alternatives Sayfasındaki Sütunlar:", df.columns)
    st.write("Weights Sayfasındaki Sütunlar:", weights_df.columns)

    # Alternatifler sayfasındaki linguistik değerleri alalım
    linguistic_values_alternatives = {}
    for index, row in df.iterrows():
        linguistic_values_alternatives[row['Alternatives']] = row[1:].values.tolist()  # Alternatifin linguistik değerlerini al

    # Weights sayfasındaki linguistik değerleri alalım
    linguistic_values_weights = {}
    for index, row in weights_df.iterrows():
        linguistic_values_weights[row[weights_df.columns[0]]] = row[1:].values.tolist()  # Ağırlıkların linguistik değerlerini al (ilk sütun adı)
    
    # Linguistik terimleri sayılara dönüştürme fonksiyonu
    def convert_linguistic_to_score(value, linguistic_values):
        return linguistic_values.get(value, [0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Alternatifler için skorlama işlemi (T2NN hesaplama)
    score_matrix = []
    for row in df.itertuples():
        alternative_scores = []
        for i, decision_maker in enumerate(row[1:], 1):
            linguistic_score = convert_linguistic_to_score(decision_maker, linguistic_values_alternatives)
            score = calculate_score(linguistic_score)  # Skoru hesapla
            alternative_scores.append(score)
        score_matrix.append(alternative_scores)

    # Alternatiflerin T2NN sonucu
    score_df = pd.DataFrame(score_matrix, columns=[f"DM{i}" for i in range(1, 5)], index=df['Alternatives'])
    st.write("Alternatifler için T2NN Skor Matrisi:", score_df)

    # Normalizasyon (Fayda veya Maliyet)
    is_benefit = st.radio("Normalizasyon Türünü Seçin:", ("Fayda", "Maliyet"))
    is_benefit = is_benefit == "Fayda"
    
    # Alternatifler için normalizasyon
    normalized_score_df = normalize_matrix(score_df, is_benefit)
    st.write("Normalizasyon Sonrası Alternatifler Karar Matrisi:", normalized_score_df)

    # Weights için skorlama işlemi (T2NN hesaplama)
    weight_matrix = []
    for row in weights_df.itertuples():
        weight_scores = []
        for i, decision_maker in enumerate(row[1:], 1):
            linguistic_score = convert_linguistic_to_score(decision_maker, linguistic_values_weights)
            score = calculate_score(linguistic_score)  # Skoru hesapla
            weight_scores.append(score)
        weight_matrix.append(weight_scores)

    # Ağırlıkların T2NN sonucu
    weight_df = pd.DataFrame(weight_matrix, columns=[f"DM{i}" for i in range(1, 5)], index=weights_df[weights_df.columns[0]])
    st.write("Ağırlıklar için T2NN Skor Matrisi:", weight_df)

    # Weights için normalizasyon
    normalized_weight_df = normalize_matrix(weight_df, is_benefit)
    st.write("Normalizasyon Sonrası Ağırlıklar Karar Matrisi:", normalized_weight_df)

    # Ağırlıklı karar matrisini hesaplama
    weighted_decision_matrix = normalized_score_df.multiply(normalized_weight_df, axis=0)
    st.write("Ağırlıklı Karar Matrisi:", weighted_decision_matrix)
    
    # Ortalamayı alarak T2NN karar matrisi oluşturma
    avg_scores = weighted_decision_matrix.mean(axis=1)
    st.write("T2NN Karar Matrisi (Ortalamalar):", avg_scores)
