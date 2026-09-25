import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, "dataset", "GlucoBench_benchmark_dataset_RAW.csv"))

# 2. Retrieve data in different ways

# View the first 5 rows
print("--- First 5 Rows ---")
print(df.head())

# Print specific columns
print("----print 6 columns----")
subset = df.iloc[:, 0:6]
print(subset)

# 2. Identify missing values per column
missing_counts = df.isnull().sum()
print("-----Missing Values Per Column:----------")
print(missing_counts)
print("-" * 40)

# 3. Detect duplicate rows
# Keeps the first occurrence and marks subsequent duplicates as True
duplicate_rows = df[df.duplicated()]
print(f"Total duplicate rows detected: {len(duplicate_rows)}")
print(duplicate_rows)
print("-" * 40)

# 3b. Basic shape / dtype / patient overview (GlucoTrend-specific)
print("Dataset Shape:", df.shape)
print("\nUnique Patients (user_id):", df["user_id"].nunique())
print("\nRows per patient:")
print(df["user_id"].value_counts())
print("-" * 40)
print("\nGlucose Summary Statistics:")
print(df["glucose"].describe())

# 4. Produce a missingness heatmap
plt.figure(figsize=(10, 6))
# cbar=False hides the color bar; yticklabels=False hides row numbers for cleaner visuals;
# cmap="viridis" Uses a purple-to-yellow color scale for high contrast.
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("Missing Values Heatmap - GlucoTrend")
plt.tight_layout()
plt.show()
