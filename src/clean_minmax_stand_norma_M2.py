# ---------------------------------------------------------
# Numeric column pre-process Techniques
# Mean, Median, mode
# Feature Scaling, Standardization, and Normalization
# Save all results in ONE CSV file (clean_minmax_stand_norma_M2.csv)
# ------------------------------------------------------------
# check the scikit-learn library is installed or not
# if not install with the command "pip install scikit-learn"

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Load Dataset (target-engineered, NOT raw)
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "dataset", "target_engineered_M2.csv")

df = pd.read_csv(file_path)

# "notes" is 100% empty (mean of an all-NaN column is still NaN, so it
# can never be mean-imputed) - drop it before anything else.
df = df.drop(columns=["notes"], errors="ignore")

print("Original Dataset")
print("------------------------")
print(df.head())

print("Dataset Shape:", df.shape)
print("\nData Types:")
print("------------------------")
print(df.dtypes)

print("\nMissing Values:")
print("------------------------")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nDuplicate Records:", df.duplicated().sum())

# ---------------------------------------------------
# Step 2: Remove Duplicate Records
# ---------------------------------------------------
df = df.drop_duplicates()

# ---------------------------------------------------
# Step 3: Handle Missing Values
# ---------------------------------------------------
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

for column in numerical_columns:
    df[column] = df[column].fillna(df[column].mean())

categorical_columns = df.select_dtypes(include=['object']).columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# ---------------------------------------------------
# Step 4: Remove Leading and Trailing Spaces
# ---------------------------------------------------
for column in categorical_columns:
    df[column] = df[column].str.strip()

# ------------------------------------------------------------
# Select Numeric Columns
# NOTE: this scales every numeric column, INCLUDING the engineered
# regression targets (glucose_lead_1/3/6). That is fine for
# inspecting distributions here, but when you build the actual
# model, scale X and y separately (fit the scaler on train only,
# and usually leave the target unscaled for interpretability).
# ------------------------------------------------------------
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

print("\nNumeric Columns:")
print(list(numeric_columns))

# ------------------------------------------------------------
# Standardization (Z-score): Mean = 0, Standard Deviation = 1
# ------------------------------------------------------------
standard_scaler = StandardScaler()
standardized = standard_scaler.fit_transform(df[numeric_columns])

for i, col in enumerate(numeric_columns):
    df[col + "_Standardized"] = standardized[:, i]

# ------------------------------------------------------------
# Feature Scaling (Min-Max Scaling): Values between 0 and 1
# ------------------------------------------------------------
minmax_scaler = MinMaxScaler()
scaled = minmax_scaler.fit_transform(df[numeric_columns])

for i, col in enumerate(numeric_columns):
    df[col + "_Scaled"] = scaled[:, i]

# ------------------------------------------------------------
# Normalization (L2 Normalization): Each row becomes a unit vector
# ------------------------------------------------------------
normalizer = Normalizer(norm='l2')
normalized = normalizer.fit_transform(df[numeric_columns])

for i, col in enumerate(numeric_columns):
    df[col + "_Normalized"] = normalized[:, i]

# ------------------------------------------------------------
# Display Results after pre-processing (Verify Dataset)
# ------------------------------------------------------------
print("\nDisplay Results after Preprocessed Dataset")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nMissing Values After Preprocessing")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\nDuplicate Records After Preprocessing")
print(df.duplicated().sum())

# ---------------------------------------------------
# Step 8: Save Preprocessed Dataset
# ---------------------------------------------------
output_file = os.path.join(BASE_DIR, "dataset", "clean_minmax_stand_norma_M2.csv")
df.to_csv(output_file, index=False)

print("\nPreprocessed dataset saved successfully:", output_file)

# to display histogram of preprocessed data
pf = pd.read_csv(output_file)
pf[list(numeric_columns)].hist(figsize=(14, 12), bins=10, edgecolor='black')
plt.suptitle("Histogram of Preprocessed GlucoTrend Dataset")
plt.tight_layout()
plt.show()
