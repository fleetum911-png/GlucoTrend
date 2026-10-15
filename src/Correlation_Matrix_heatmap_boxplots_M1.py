import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the output directory exists
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(BASE_DIR, "outputs", "Boxplots_correlation")
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------------
# 1. Load GlucoBench Dataset
# -------------------------------------------------------------
df = pd.read_csv(os.path.join(BASE_DIR, "dataset", "GlucoBench_benchmark_dataset_RAW.csv"))

# "notes" is 100% empty in this dataset - drop before any numeric analysis
df = df.drop(columns=["notes"], errors="ignore")

print("Dataset Loaded Successfully. Shape:", df.shape)

# -------------------------------------------------------------
# 2. Compute Correlation Matrix & Generate Heatmap
# -------------------------------------------------------------
# Select only numerical columns for correlation
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numerical_cols].corr()

# Print matrix to terminal
print("\n--- Correlation Matrix ---")
print(corr_matrix)

# Create Heatmap
plt.figure(figsize=(14, 12))
ax = plt.gca()
heatmap = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(corr_matrix.index)), corr_matrix.index, fontsize=7)
for row in range(len(corr_matrix.index)):
    for column in range(len(corr_matrix.columns)):
        ax.text(column, row, f"{corr_matrix.iloc[row, column]:.2f}", ha="center", va="center", fontsize=6)
plt.title("Correlation Heatmap of Numerical Features - GlucoTrend", fontsize=14, fontweight="bold")
plt.xticks(rotation=90, fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()

# Export Heatmap
heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()
print(f"Exported heatmap to: {heatmap_path}")

# -------------------------------------------------------------
# 3. Produce Boxplots: Numerical Features vs target_ada3
# -------------------------------------------------------------
# NOTE: target_ada3 only exists AFTER running target_variable_engineering_M2.py
# If it isn't available yet, fall back to a raw clinical cut of glucose itself.
target_col = "target_ada3"

if target_col not in df.columns:
    print(f"\n'{target_col}' not found in raw data (expected - it is engineered).")
    print("Run target_variable_engineering_M2.py first for target-vs-feature boxplots.")
    print("Falling back to plotting features against a quick glucose-band cut.")
    df[target_col] = pd.cut(
        df["glucose"], bins=[0, 70, 180, 1000],
        labels=["Hypoglycemia", "Target Range", "Hyperglycemia"]
    )

feature_cols = [c for c in numerical_cols if c != "glucose"]

for col in feature_cols[:12]:  # keep the run fast; expand as needed
    fig, ax = plt.subplots(figsize=(6, 5))
    df.boxplot(column=col, by=target_col, ax=ax, grid=False)
    plt.title(f"{col} vs {target_col}", fontsize=12, fontweight="bold")
    fig.suptitle("")
    plt.xlabel(target_col)
    plt.ylabel(col)
    plt.tight_layout()

    boxplot_filename = f"boxplot_{col}_vs_{target_col}.png"
    boxplot_path = os.path.join(output_dir, boxplot_filename)
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    print(f"Exported boxplot to: {boxplot_path}")

print("\nAll EDA tasks completed successfully!")
