import os
import pandas as pd
import numpy as np

# ==========================================================
# Load GlucoTrend Dataset (target-engineered, NOT raw)
# Original dataset is NOT modified
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
# 2. Identify Missing Values
# ==========================================================
print("Missing Values Before Cleaning:")
print(data.isnull().sum()[data.isnull().sum() > 0])

# ==========================================================
# 3. Remove Duplicate Records
# ==========================================================
duplicate_count = data.duplicated().sum()
data = data.drop_duplicates()
print("\nDuplicate Records Removed:", duplicate_count)

# ==========================================================
# 4. Identify Numerical and Categorical FEATURE Columns
# ==========================================================
num_cols = [c for c in data.select_dtypes(include=np.number).columns if c not in NON_FEATURE_COLS]
cat_cols = [c for c in data.select_dtypes(exclude=np.number).columns if c not in NON_FEATURE_COLS]

print("\nNumerical Columns:", num_cols)
print("Categorical Columns:", cat_cols)

# ==========================================================
# 5. Fill Missing Numerical Values with Mean
# ==========================================================
for col in num_cols:
    data[col] = data[col].fillna(data[col].mean())

# ==========================================================
# 6. Fill Missing Categorical Values with Mode
# ==========================================================
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# ==========================================================
# 7. Select Target Column For Encoding
# ==========================================================
# Target-mean encoding needs a NUMERIC target. Of the 5 engineered
# targets, "target_binary" is the closest analogue to placement's
# numeric "PlacementStatus" column, so it is mapped to 0/1 here.
# Swap TARGET_FOR_ENCODING to "glucose" if you'd rather encode
# categories by their average glucose level instead.
data["target_binary_numeric"] = data["target_binary"].map({"In_Range": 0, "Out_of_Range": 1})
TARGET_FOR_ENCODING = "target_binary_numeric"

# ==========================================================
# 8. Apply Target Encoding Using Pandas
# ==========================================================
target_encoded_df = pd.DataFrame()

for col in cat_cols:
    mean_encoding = data.groupby(col)[TARGET_FOR_ENCODING].mean()
    target_encoded_df["Target_" + col] = data[col].map(mean_encoding)

# ==========================================================
# 9. Merge Numerical Columns and Target Encoded Columns
# ==========================================================
final_output = pd.concat(
    [
        data[num_cols + NON_FEATURE_COLS].reset_index(drop=True),
        target_encoded_df.reset_index(drop=True)
    ],
    axis=1
)

# ==========================================================
# 10. Check Missing Values After Processing
# ==========================================================
print("\nMissing Values After Cleaning:")
print(final_output.isnull().sum()[final_output.isnull().sum() > 0])

# ==========================================================
# 11. Save Final Result
# ==========================================================
output_file = os.path.join(BASE_DIR, "dataset", "clean_target_encode_M2.csv")
final_output.to_csv(output_file, index=False)

print("\n======================================")
print("Target Encoding Completed Successfully")
print("Original dataset is NOT modified")
print("Output file:", output_file)
print("======================================")
