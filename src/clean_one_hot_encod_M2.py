import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
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

print("\nCategorical feature columns to one-hot encode:", cat_cols)

# ==========================================================
# 3. Fill Missing Values
# ==========================================================
if len(num_cols) > 0:
    data[num_cols] = SimpleImputer(strategy="mean").fit_transform(data[num_cols])

if len(cat_cols) > 0:
    data[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(data[cat_cols])

# ==========================================================
# 4. One-Hot Encoding
# ==========================================================
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

if len(cat_cols) > 0:
    encoded_array = encoder.fit_transform(data[cat_cols])
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(cat_cols),
        index=data.index
    )
else:
    encoded_df = pd.DataFrame(index=data.index)

# ==========================================================
# 5. Merge Back With Remaining Columns (kept but not re-encoded)
# ==========================================================
keep_cols = [c for c in data.columns if c not in cat_cols]
final_output = pd.concat([data[keep_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

print("\nShape before one-hot encoding:", data.shape)
print("Shape after one-hot encoding:", final_output.shape)

# ==========================================================
# Save Result
# ==========================================================
output_file = os.path.join(BASE_DIR, "dataset", "clean_one_hot_encoding_M2.csv")
final_output.to_csv(output_file, index=False)

print("\n======================================")
print("Original dataset is NOT modified.")
print("One-Hot Encoding completed successfully.")
print("Output file:", output_file)
print("======================================")
