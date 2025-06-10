import streamlit as st
import pandas as pd

# Skor fonksiyonunu güncelleyelim
def calculate_score(linguistic_scores):
    alpha_alpha, alpha_beta, alpha_gamma, beta_alpha, beta_beta, beta_gamma, gamma_alpha, gamma_beta, gamma_gamma = linguistic_scores
    return (1/12) * (8 + (alpha_alpha + 2*alpha_beta + alpha_gamma) - (beta_alpha + 2*beta_beta + beta_gamma) - (gamma_alpha + 2*gamma_beta + gamma_gamma))

# Streamlit arayüzü
st.title("MABAC Karar Matrisi Hesaplama")

uploaded_file = st.file_uploader("Excel dosyasını yükleyin", type=["xlsx"])

if uploaded_file is not None:
    # Excel dosyasını oku
    df = pd.read_excel(uploaded_file, sheet_name="Alternatives")
    weights_df = pd.read_excel(uploaded_file, sheet_name="Weights")
    
    # Linguistik değerler (Alternatives sayfasından)
    linguistic_values = {
        'Very Bad (VB)': [0.2, 0.2, 0.1, 0.65, 0.8, 0.85, 0.45, 0.8, 0.7],
        'Bad (B)': [0.35, 0.35, 0.35, 0.75, 0.8, 0.9, 0.5, 0.75, 0.65],
        'Medium Bad (MB)': [0.4, 0.45, 0.4, 0.75, 0.85, 0.95, 0.6, 0.8, 0.7],
        'Medium (M)': [0.4, 0.45, 0.5, 0.8, 0.85, 0.9, 0.6, 0.8, 0.75],
        'Medium Good (MG)': [0.6, 0.55, 0.5, 0.85, 0.9, 0.95, 0.75, 0.85, 0.8],
        'Good (G)': [0.7, 0.7, 0.75, 0.9, 0.95, 1.0, 0.85, 0.9, 0.85],
        'Very Good (VG)': [0.95, 0.95, 0.9, 1.0, 1.0, 1.0, 0.95, 0.95, 0.9]
    }

    # Linguistik terimleri sayılara dönüştürme fonksiyonu
    def convert_linguistic_to_score(value, linguistic_values):
        return linguistic_values.get(value, [0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Alternatifler ve karar vericiler için skorlama işlemi
    score_matrix = []
    for row in df.itertuples():
        alternative_scores = []
        for i, decision_maker in enumerate(row[1:], 1):
            linguistic_score = convert_linguistic_to_score(decision_maker, linguistic_values)
            score = calculate_score(linguistic_score)  # Skoru hesapla
            alternative_scores.append(score)
        score_matrix.append(alternative_scores)

    # Skor matrisini dataframe olarak göster
    score_df = pd.DataFrame(score_matrix, columns=[f"DM{i}" for i in range(1, 5)], index=df['Alternatives'])
    st.write("Karar Vericilerin Skor Matrisi:", score_df)

    # Ağırlıkları işleme (linguistikse)
    weight_scores = []
    for col in weights_df.columns:
        weight_values = weights_df[col].apply(lambda x: convert_linguistic_to_score(x, linguistic_values))
        weight_scores.append([calculate_score(weight) for weight in weight_values])

    # Ağırlıklı karar matrisini oluşturma
    weight_df = pd.DataFrame(weight_scores, columns=weights_df.columns, index=weights_df.index)
    st.write("Ağırlıklı Karar Matrisi:", weight_df)

    # Ortalamayı alarak T2NN karar matrisi oluşturma
    avg_scores = score_df.mean(axis=1)
    st.write("T2NN Karar Matrisi:", avg_scores)
