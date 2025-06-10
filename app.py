import streamlit as st
import pandas as pd

# Skor fonksiyonunu tanımlayalım
def calculate_score(alpha, beta, gamma):
    return (1/12) * (8 + (alpha + 2*beta + gamma) - (beta + 2*gamma + gamma))

# Streamlit arayüzü
st.title("MABAC Karar Matrisi Hesaplama")

uploaded_file = st.file_uploader("Excel dosyasını yükleyin", type=["xlsx"])

if uploaded_file is not None:
    # Excel dosyasını oku
    df = pd.read_excel(uploaded_file, sheet_name="Alternatives")
    weights_df = pd.read_excel(uploaded_file, sheet_name="Weights")
    linguistic_df = pd.read_excel(uploaded_file, sheet_name="Linguistic Variables")

    # Linguistik değişkenleri dinamik olarak alalım
    linguistic_values = {}
    for index, row in linguistic_df.iterrows():
        linguistic_values[row['Linguistic Variables']] = row[['(αα,αβ,αγ)', '(βα,ββ,βγ)', '(γα,γβ,γγ)']].values.tolist()

    # Linguistik terimleri sayılara dönüştürme fonksiyonu
    def convert_linguistic_to_score(value, linguistic_values):
        return linguistic_values.get(value, [0, 0, 0])

    # Alternatifler ve karar vericiler için skorlama işlemi
    score_matrix = []
    for row in df.itertuples():
        alternative_scores = []
        for i, decision_maker in enumerate(row[1:], 1):
            linguistic_score = convert_linguistic_to_score(decision_maker, linguistic_values)
            score = calculate_score(*linguistic_score)  # Skoru hesapla
            alternative_scores.append(score)
        score_matrix.append(alternative_scores)

    # Skor matrisini dataframe olarak göster
    score_df = pd.DataFrame(score_matrix, columns=[f"DM{i}" for i in range(1, 5)], index=df['Alternatives'])
    st.write("Karar Vericilerin Skor Matrisi:", score_df)

    # Ağırlıkları işleme (linguistikse)
    weight_scores = []
    for col in weights_df.columns:
        weight_values = weights_df[col].apply(lambda x: convert_linguistic_to_score(x, linguistic_values))
        weight_scores.append([calculate_score(*weight) for weight in weight_values])

    # Ağırlıklı karar matrisini oluşturma
    weight_df = pd.DataFrame(weight_scores, columns=weights_df.columns, index=weights_df.index)
    st.write("Ağırlıklı Karar Matrisi:", weight_df)

    # Ortalamayı alarak T2NN karar matrisi oluşturma
    avg_scores = score_df.mean(axis=1)
    st.write("T2NN Karar Matrisi:", avg_scores)
