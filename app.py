import streamlit as st
import pandas as pd
import numpy as np

# Skor fonksiyonunu tanımlayalım
def calculate_score(linguistic_scores):
    # linguistic_scores: [αα, αβ, αY, βα, ββ, βY, γα, γβ, γγ]
    # Bu değerler eksikse varsayılan olarak 0 kullanacağız
    try:
        alpha_alpha, alpha_beta, alpha_gamma, beta_alpha, beta_beta, beta_gamma, gamma_alpha, gamma_beta, gamma_gamma = linguistic_scores
    except ValueError:
        st.error("Linguistic scores veri formatında bir hata oluştu!")
        return np.nan

    return (1/12) * (8 + (alpha_alpha + 2*alpha_beta + alpha_gamma) - (beta_alpha + 2*beta_beta + beta_gamma) - (gamma_alpha + 2*gamma_beta + gamma_gamma))

# Normalizasyon işlemi (Fayda / Maliyet)
def normalize_matrix(matrix, is_benefit=True):
    if is_benefit:
        # Fayda normalizasyonu (max-min normalizasyonu)
        return matrix.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    else:
        # Maliyet normalizasyonu (max-min normalizasyonu)
        return matrix.apply(lambda x: (x.max() - x) / (x.max() - x.min()), axis=0)

# Sınır yaklaşım alanı mesafesi (MABAC skorlarını oluşturma)
def border_approximation_area_distance(matrix):
    # Mesafe hesaplama
    return matrix.apply(lambda x: np.sum(np.abs(x - matrix.mean(axis=1))), axis=1)

# Streamlit arayüzü
st.title("MABAC Karar Matrisi Hesaplama")

uploaded_file = st.file_uploader("Excel dosyasını yükleyin", type=["xlsx"])

if uploaded_file is not None:
    # Excel dosyasını oku
    df = pd.read_excel(uploaded_file, sheet_name="Alternatives")
    weights_df = pd.read_excel(uploaded_file, sheet_name="Weights")

    # Alternatifler sayfasındaki linguistik değerleri alalım
    linguistic_values_alternatives = {
        'Very Bad (VB)': [0.2, 0.2, 0.1, 0.65, 0.8, 0.85, 0.45, 0.8, 0.7],
        'Bad (B)': [0.35, 0.35, 0.35, 0.75, 0.8, 0.9, 0.5, 0.75, 0.65],
        'Medium Bad (MB)': [0.4, 0.45, 0.4, 0.75, 0.85, 0.95, 0.6, 0.8, 0.7],
        'Medium (M)': [0.4, 0.45, 0.5, 0.8, 0.85, 0.9, 0.6, 0.8, 0.75],
        'Medium Good (MG)': [0.6, 0.55, 0.5, 0.85, 0.9, 0.95, 0.75, 0.85, 0.8],
        'Good (G)': [0.7, 0.7, 0.75, 0.9, 0.95, 1.0, 0.85, 0.9, 0.85],
        'Very Good (VG)': [0.95, 0.95, 0.9, 1.0, 1.0, 1.0, 0.95, 0.95, 0.9]
    }

    linguistic_values_weights = {
        'Low (L)': [0.2, 0.3, 0.2, 0.6, 0.75, 0.85, 0.45, 0.8, 0.7],
        'Medium Low (ML)': [0.4, 0.3, 0.25, 0.45, 0.55, 0.6, 0.5, 0.65, 0.55],
        'Medium (M)': [0.5, 0.55, 0.55, 0.4, 0.45, 0.5, 0.6, 0.75, 0.7],
        'High (H)': [0.8, 0.75, 0.7, 0.2, 0.25, 0.3, 0.85, 0.9, 0.8],
        'Very High (VH)': [0.9, 0.85, 0.95, 0.15, 0.1, 0.1, 0.95, 0.9, 0.85]
    }

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
    
    # Sınır yaklaşım alanı mesafesi hesaplama (MABAC puanı)
    distance_matrix = border_approximation_area_distance(weighted_decision_matrix)
    st.write("Sınır Yaklaşım Alanı Mesafesi:", distance_matrix)
