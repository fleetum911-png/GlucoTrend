# --------------------------------------------
# GlucoTrend Dataset Preprocessing
# --------------------------------------------

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------------------------
# CHOOSE YOUR TARGET FRAMING HERE
# One of: "glucose_lead_1" (regression), "target_binary",
#         "target_ada3", "target_5class", "target_trend"
# --------------------------------------------
TARGET_COLUMN = "target_ada3"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(BASE_DIR, "dataset", "target_engineered_M2.csv")
output_file = os.path.join(BASE_DIR, "dataset", "final_preprocess_M2.csv")

if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"{input_file} not found. Run target_variable_engineering_M2.py first."
    )

df = pd.read_csv(input_file)

# Create a copy so original dataset remains unchanged
processed_df = df.copy()

print("Original Dataset Shape:", processed_df.shape)

# --------------------------------------------
# Drop Fully-Empty Columns and Non-Feature Identifiers
# --------------------------------------------
processed_df.drop(columns=["notes"], errors="ignore", inplace=True)

ALL_TARGET_CANDIDATES = [
    "glucose_lead_1", "glucose_lead_3", "glucose_lead_6",
    "target_binary", "target_ada3", "target_5class", "target_trend"
]
OTHER_TARGETS = [c for c in ALL_TARGET_CANDIDATES if c != TARGET_COLUMN]

# --------------------------------------------
# Remove Duplicate Records
# --------------------------------------------
processed_df.drop_duplicates(inplace=True)

# --------------------------------------------
# Handle Missing Values
# --------------------------------------------
numeric_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    processed_df[col] = processed_df[col].fillna(processed_df[col].median())

categorical_cols = processed_df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])

# --------------------------------------------
# Clean Text Data
# --------------------------------------------
for col in categorical_cols:
    processed_df[col] = processed_df[col].str.strip()
    processed_df[col] = processed_df[col].str.lower()

# --------------------------------------------
# Label Encoding (categorical predictors only - not user_id/timestamp)
# --------------------------------------------
encoder = LabelEncoder()
skip_encoding = {"user_id", "timestamp"}

for col in categorical_cols:
    if col in skip_encoding:
        continue
    processed_df[col] = encoder.fit_transform(processed_df[col])

# --------------------------------------------
# Feature Scaling (predictors only - target excluded so it stays interpretable)
# --------------------------------------------
scale_cols = [c for c in numeric_cols if c not in ALL_TARGET_CANDIDATES]

scaler = StandardScaler()
processed_df[scale_cols] = scaler.fit_transform(processed_df[scale_cols])

# --------------------------------------------
# Drop the target framings you are NOT using this run, and drop
# identifier columns not suited as model features
# --------------------------------------------
processed_df.drop(columns=OTHER_TARGETS, errors="ignore", inplace=True)
processed_df.drop(columns=["user_id", "timestamp"], errors="ignore", inplace=True)

# --------------------------------------------
# Save Preprocessed Dataset
# --------------------------------------------
processed_df.to_csv(output_file, index=False)

print("\nPreprocessing Completed Successfully!")
print("Target column used:", TARGET_COLUMN)
print("Original Dataset Shape :", df.shape)
print("Processed Dataset Shape:", processed_df.shape)
print("Saved File :", output_file)
