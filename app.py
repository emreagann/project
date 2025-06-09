import streamlit as st
import pandas as pd
import numpy as np
from t2 import T2NeutrosophicNumber

# T2NN skor hesaplama fonksiyonu
def t2nn_score(t2nn):
    T = np.mean(t2nn.truth)
    I = np.mean(t2nn.indeterminacy)
    F = np.mean(t2nn.falsity)
    return (T + (1 - I) + (1 - F)) / 3

# Normalize fonksiyonu
def norm(x, min_val, max_val, is_benefit=True):
    if is_benefit:
        return (x - min_val) / (max_val - min_val) if max_val != min_val else 0
    else:
        return (max_val - x) / (max_val - min_val) if max_val != min_val else 0

# Aralıkları T2NN'e çevirme fonksiyonu
def convert_range_to_t2nn(value):
    if isinstance(value, str) and ('-' in value or '–' in value):
        value = value.replace('–', '-')
        try:
            a_str, b_str = value.split('-')
            a = float(a_str.replace(',', '.'))
            b = float(b_str.replace(',', '.'))
            m = (a + b) / 2

            a /= 10
            m /= 10
            b /= 10

            T = (a, m, b)
            I = (0.0125, 0.0125, 0.0125)
            F = (1 - b, 1 - m, 1 - a)

            return T2NeutrosophicNumber(T, I, F)
        except:
            return None
    else:
        try:
            x = float(str(value).replace(',', '.'))
            T = (x/10, x/10, x/10)
            I = (0.0125, 0.0125, 0.0125)
            F = (1 - x/10, 1 - x/10, 1 - x/10)
            return T2NeutrosophicNumber(T, I, F)
        except:
            return None

# Excel'den verileri yükle
def load_excel(file):
    xls = pd.ExcelFile(file)
    df_alternatives = pd.read_excel(xls, sheet_name="Alternatives")
    df_weights = pd.read_excel(xls, sheet_name="Criteria Weights")
    df_sub = pd.read_excel(xls, sheet_name="Sub-Criteria")

    criteria = df_weights["criteria"].tolist()
    weights = df_weights["weight"].tolist()
    attributes = df_sub["sub-criteria attributes"].tolist()
    perspectives = df_sub["evaluation perspective"].tolist()

    data_raw = df_alternatives[criteria].values.tolist()
    alternatives = df_alternatives["alternative"].tolist()

    # Karar matrisi oluştur
    X = np.empty((len(alternatives), len(criteria)), dtype=object)

    for j, crit in enumerate(criteria):
        for i in range(len(alternatives)):
            val = data_raw[i][j]
            if perspectives[j] == "quantitative":
                X[i, j] = convert_range_to_t2nn(val)
            else:
                try:
                    X[i, j] = float(str(val).replace(',', '.'))
                except:
                    X[i, j] = np.nan

    # Normalizasyon
    X_norm = np.empty_like(X, dtype=object)

    for j in range(len(criteria)):
        ctype = attributes[j]
        if perspectives[j] == "quantitative":
            col = [X[i, j] for i in range(len(alternatives))]
            min_val = T2NeutrosophicNumber(
                tuple(min(v.truth[i] for v in col) for i in range(3)),
                tuple(max(v.indeterminacy[i] for v in col) for i in range(3)),  # max I
                tuple(max(v.falsity[i] for v in col) for i in range(3))         # max F
            )
            max_val = T2NeutrosophicNumber(
                tuple(max(v.truth[i] for v in col) for i in range(3)),
                tuple(min(v.indeterminacy[i] for v in col) for i in range(3)),  # min I
                tuple(min(v.falsity[i] for v in col) for i in range(3))         # min F
            )
            for i in range(len(alternatives)):
                value = X[i, j]
                t = [norm(value.truth[i], min_val.truth[i], max_val.truth[i], ctype == "benefit") for i in range(3)]
                i_ = [norm(value.indeterminacy[i], min_val.indeterminacy[i], max_val.indeterminacy[i], ctype == "cost") for i in range(3)]
                f = [norm(value.falsity[i], min_val.falsity[i], max_val.falsity[i], ctype == "cost") for i in range(3)]
                X_norm[i, j] = T2NeutrosophicNumber(t, i_, f)
        else:
            col = [X[i, j] for i in range(len(alternatives))]
            min_val = min(col)
            max_val = max(col)
            for i in range(len(alternatives)):
                value = X[i, j]
                norm_val = norm(value, min_val, max_val, ctype == "benefit")
                X_norm[i, j] = norm_val

    return alternatives, criteria, weights, X_norm
