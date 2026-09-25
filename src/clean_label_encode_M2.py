import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# ==========================================================
# Load GlucoTrend Dataset (target-engineered, NOT raw)
# Original file is NOT modified
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, "dataset", "target_engineered_M2.csv"))

# Create a copy for processing
data = df.copy()

# Identifier / target columns are never fed into a feature encoder
NON_FEATURE_COLS = [
    "user_id", "timestamp", "notes",
    "target_binary", "target_ada3", "target_5class", "target_trend"
]

# ==========================================================
# 1. Remove Leading and Trailing Spaces
# ==========================================================
for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].str.strip()

# ==========================================================
# 2. Identify Missing Values
# ==========================================================
print("Missing Values Before Cleaning:")
print(data.isnull().sum()[data.isnull().sum() > 0])

# ==========================================================
# 3. Remove Duplicate Records
# ==========================================================
before = data.shape[0]
data = data.drop_duplicates()
after = data.shape[0]
print("\nDuplicate Records Removed:", before - after)

# ==========================================================
# Separate Numerical and Categorical FEATURE Columns
# ==========================================================
num_cols = [c for c in data.select_dtypes(include=np.number).columns if c not in NON_FEATURE_COLS]
cat_cols = [c for c in data.select_dtypes(exclude=np.number).columns if c not in NON_FEATURE_COLS]

print("\nNumerical feature columns:", num_cols)
print("Categorical feature columns:", cat_cols)

# ==========================================================
# 4. Fill Missing Numerical Values with Mean
# ==========================================================
if len(num_cols) > 0:
    num_imputer = SimpleImputer(strategy="mean")
    data[num_cols] = num_imputer.fit_transform(data[num_cols])

# ==========================================================
# 5. Fill Missing Categorical Values with Mode
# ==========================================================
if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

# ==========================================================
# 6. Label Encoding
# ==========================================================
label_encoders = {}

for col in cat_cols:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
    label_encoders[col] = encoder

# ==========================================================
# 7. Check Missing Values After Cleaning
# ==========================================================
print("\nMissing Values After Cleaning:")
print(data.isnull().sum()[data.isnull().sum() > 0])

# ==========================================================
# Save Result
# ==========================================================
output_file = os.path.join(BASE_DIR, "dataset", "clean_label_encode_M2.csv")
data.to_csv(output_file, index=False)

print("\n======================================")
print("Original dataset is NOT modified.")
print("Label Encoding completed successfully.")
print("Output file:", output_file)
print("======================================")
