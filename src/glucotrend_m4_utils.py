"""Shared GlucoBench feature preparation for Module 4 analyses."""

import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATASET = os.path.join(BASE_DIR, "dataset", "GlucoBench_benchmark_dataset_RAW.csv")
M4_OUTPUT = os.path.join(BASE_DIR, "outputs", "GlucoTrend_Module_4_Outputs")
FEATURES = ["glucose", "heart_rate", "stress_level", "hbA1c", "insulin_sensitivity"]


def load_scaled_features(sample_size=None, random_state=42):
    if not os.path.exists(RAW_DATASET):
        raise FileNotFoundError(f"GlucoBench dataset not found: {RAW_DATASET}")
    data = pd.read_csv(RAW_DATASET)
    columns = [column for column in FEATURES if column in data.columns]
    if len(columns) < 2:
        raise ValueError("At least two Module 4 numeric features are required.")
    frame = data[columns].replace([np.inf, -np.inf], np.nan)
    frame = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(frame), columns=columns)
    source_rows = data
    if sample_size and len(frame) > sample_size:
        selected = frame.sample(sample_size, random_state=random_state)
        source_rows = data.loc[selected.index].reset_index(drop=True)
        frame = selected.reset_index(drop=True)
    scaled = StandardScaler().fit_transform(frame)
    os.makedirs(M4_OUTPUT, exist_ok=True)
    return source_rows, frame, scaled
