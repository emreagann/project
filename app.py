import streamlit as st
import pandas as pd
import numpy as np

# Linguistic variables for alternative evaluations (Table A.2)
alternative_linguistic_vars = {
    "VB": [0.20, 0.20, 0.10, 0.65, 0.80, 0.85, 0.45, 0.80, 0.70],
    "B":  [0.35, 0.35, 0.10, 0.50, 0.75, 0.80, 0.50, 0.75, 0.65],
    "MB": [0.50, 0.30, 0.50, 0.50, 0.35, 0.45, 0.45, 0.30, 0.60],
    "M":  [0.40, 0.45, 0.50, 0.40, 0.45, 0.50, 0.35, 0.40, 0.45],
    "MG": [0.60, 0.45, 0.50, 0.20, 0.15, 0.25, 0.10, 0.25, 0.15],
    "G":  [0.70, 0.75, 0.80, 0.15, 0.20, 0.25, 0.10, 0.15, 0.20],
    "VG": [0.95, 0.90, 0.95, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05],
}

# Corrected linguistic variables for criteria weights (Table A.1)
weight_linguistic_vars = {
    "VL": [(0.0, 0.0, 0.1), (0.95, 1.0, 1.0), (0.95, 1.0, 1.0)],
    "L":  [(0.1, 0.2, 0.3), (0.7, 0.8, 0.9), (0.7, 0.8, 0.9)],
    "ML": [(0.2, 0.3, 0.4), (0.6, 0.7, 0.8), (0.6, 0.7, 0.8)],
    "M":  [(0.3, 0.4, 0.5), (0.5, 0.6, 0.7), (0.5, 0.6, 0.7)],
    "MH": [(0.4, 0.5, 0.6), (0.4, 0.5, 0.6), (0.4, 0.5, 0.6)],
    "H":  [(0.5, 0.6, 0.7), (0.3, 0.4, 0.5), (0.3, 0.4, 0.5)],
    "VH": [(0.6, 0.7, 0.8), (0.2, 0.3, 0.4), (0.2, 0.3, 0.4)],
    "VVH": [(0.7, 0.8, 0.9), (0.1, 0.2, 0.3), (0.1, 0.2, 0.3)]
}

def score_function(values):
    a1, a2, a3, b1, b2, b3, g1, g2, g3 = values
    return (1 / 12) * (8 + (a1 + 2 * a2 + a3) - (b1 + 2 * b2 + b3) - (g1 + 2 * g2 + g3))

def score_from_merged_t2nn(t2nn):
    (a1, a2, a3), (b1, b2, b3), (g1, g2, g3) = t2nn
    return (1 / 12) * (8 + (a1 + 2*a2 + a3) - (b1 + 2*b2 + b3) - (g1 + 2*g2 + g3))

def merge_t2nn_vectors(t2nn_list):
    n = len(t2nn_list)
    merged = []
    for i in range(3):  # T, I, F
        avg = tuple(sum(vec[i][j] for vec in t2nn_list) / n for j in range(3))
        merged.append(avg)
    return tuple(merged)

def get_valid_numeric_values(value):
    value = str(value).strip()
    return score_function(alternative_linguistic_vars[value]) if value in alternative_linguistic_vars else 0

def normalize_data(series, criteria_type):
    if series.max() == series.min():
        return 0
    if criteria_type.lower() == 'benefit':
        return (series - series.min()) / (series.max() - series.min())
    elif criteria_type.lower() == 'cost':
        return (series.max() - series) / (series.max() - series.min())
    return series

def apply_weights(normalized_df, weights):
    return normalized_df.multiply(weights, axis=1)

def calculate_BAA(weighted_df):
    return weighted_df.prod(axis=0) ** (1 / len(weighted_df))

def calculate_difference_matrix(weighted_df, BAA):
    return weighted_df - BAA

def calculate_scores(diff_df):
    return diff_df.sum(axis=1)

# The rest of the application logic should be updated to use these mappings for both alternatives and weight scoring.
