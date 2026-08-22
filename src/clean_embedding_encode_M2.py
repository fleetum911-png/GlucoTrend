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
# 7. Pandas-Based Embedding Encoding
# ==========================================================
embedding_output = pd.DataFrame()
embedding_size = 3  # Number of embedding dimensions

for col in cat_cols:
    categories = data[col].unique()

    embedding_matrix = {}
    for index, category in enumerate(categories):
        vector = np.zeros(embedding_size)
        vector[index % embedding_size] = 1
        embedding_matrix[category] = vector

    embeddings = data[col].map(embedding_matrix)

    embedding_df = pd.DataFrame(
        embeddings.tolist(),
        columns=[f"Embedding_{col}_1", f"Embedding_{col}_2", f"Embedding_{col}_3"]
    )

    embedding_output = pd.concat([embedding_output, embedding_df], axis=1)

# ==========================================================
# 8. Merge Numerical Columns + Embeddings + Targets
# ==========================================================
final_output = pd.concat(
    [
        data[num_cols + NON_FEATURE_COLS].reset_index(drop=True),
        embedding_output.reset_index(drop=True)
    ],
    axis=1
)

# ==========================================================
# 9. Check Missing Values After Processing
# ==========================================================
print("\nMissing Values After Cleaning:")
print(final_output.isnull().sum()[final_output.isnull().sum() > 0])

# ==========================================================
# 10. Save Result
# ==========================================================
output_file = os.path.join(BASE_DIR, "dataset", "clean_embedded_encode_M2.csv")
final_output.to_csv(output_file, index=False)

print("\n======================================")
print("Embedding Encoding Completed")
print("Original dataset is NOT modified")
print("Output File:", output_file)
print("======================================")
