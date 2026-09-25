import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# ==========================================================
# Load GlucoTrend Dataset (target-engineered, NOT raw)
# Original file is NOT modified
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, "dataset", "target_engineered_M2.csv"))

data = df.copy()

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
# 2. Remove Duplicate Records
# ==========================================================
before = data.shape[0]
data = data.drop_duplicates()
print("Duplicate Records Removed:", before - data.shape[0])

# ==========================================================
# Separate Numerical and Categorical FEATURE Columns
# ==========================================================
num_cols = [c for c in data.select_dtypes(include=np.number).columns if c not in NON_FEATURE_COLS]
cat_cols = [c for c in data.select_dtypes(exclude=np.number).columns if c not in NON_FEATURE_COLS]

# exercise_intensity has a natural order: none < low < moderate < high
# This is the one column where ORDINAL encoding is more appropriate
# than label/one-hot - the category order carries real meaning.
ordered_categories = {
    "exercise_intensity": ["none", "low", "moderate", "high"],
}

print("\nCategorical feature columns to ordinal encode:", cat_cols)

# ==========================================================
# 3. Fill Missing Values
# ==========================================================
if len(num_cols) > 0:
    data[num_cols] = SimpleImputer(strategy="mean").fit_transform(data[num_cols])

if len(cat_cols) > 0:
    data[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(data[cat_cols])

# ==========================================================
# 4. Ordinal Encoding
# ==========================================================
categories_list = []
for col in cat_cols:
    if col in ordered_categories:
        categories_list.append(ordered_categories[col])
    else:
        # No natural order known -> encoder infers alphabetical order
        categories_list.append(sorted(data[col].unique().tolist()))

if len(cat_cols) > 0:
    encoder = OrdinalEncoder(categories=categories_list, handle_unknown="use_encoded_value", unknown_value=-1)
    data[cat_cols] = encoder.fit_transform(data[cat_cols])

# ==========================================================
# 5. Check Missing Values After Cleaning
# ==========================================================
print("\nMissing Values After Cleaning:")
print(data.isnull().sum()[data.isnull().sum() > 0])

# ==========================================================
# Save Result
# ==========================================================
output_file = os.path.join(BASE_DIR, "dataset", "clean_ordinal_encode_M2.csv")
data.to_csv(output_file, index=False)

print("\n======================================")
print("Original dataset is NOT modified.")
print("Ordinal Encoding completed successfully.")
print("Output file:", output_file)
print("======================================")
