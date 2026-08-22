import os
import pandas as pd

# ============================================================
# GLUCOTREND DATASET
# HANDLING MISSING VALUES USING PANDAS ONLY
#
# Techniques:
# 1. Deletion
# 2. Mean Imputation
# 3. Median Imputation
# 4. Forward-Fill (time-series appropriate for CGM lag features)
# 5. Missing Indicator Features
#
# Input is target_engineered_M2.csv (built by
# target_variable_engineering_M2.py), NOT the raw file, so the
# engineered target columns are carried through the pipeline.
# Original files are NOT modified. All results are stored in
# ONE CSV file.
# ============================================================

# ------------------------------------------------------------
# 1. READ DATASET
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(BASE_DIR, "dataset", "target_engineered_M2.csv")
output_file = os.path.join(BASE_DIR, "dataset", "clean_missing_imputer_M2.csv")

if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"{input_file} not found. Run target_variable_engineering_M2.py first."
    )

df = pd.read_csv(input_file, parse_dates=["timestamp"])

print("=" * 70)
print("GLUCOTREND DATASET - BEFORE CLEANING")
print("=" * 70)
print(df.head())
print("\nDataset Shape:", df.shape)

# ------------------------------------------------------------
# 2. CHECK MISSING VALUES
# ------------------------------------------------------------
print("\nMissing Values in Dataset:")
print(df.isnull().sum()[df.isnull().sum() > 0])

df_original = df.copy()

# ============================================================
# STEP 0: DROP FULLY-EMPTY COLUMNS
# ============================================================
# "notes" is 100% null in this dataset - nothing to impute from it.
fully_empty_cols = [c for c in df_original.columns if df_original[c].isnull().all()]
print("\nFully-empty columns dropped:", fully_empty_cols)
df_base = df_original.drop(columns=fully_empty_cols)

# ============================================================
# METHOD 1: DELETION
# ============================================================
df_deletion = df_base.dropna().copy()

print("\n" + "=" * 70)
print("1. DELETION")
print("=" * 70)
print("Original rows:", len(df_base))
print("Rows after deletion:", len(df_deletion))
print("Rows deleted:", len(df_base) - len(df_deletion))

deletion_indicator = df_base.isnull().any(axis=1).astype(int)

# ============================================================
# METHOD 2: MEAN IMPUTATION (glucose_lag_*/ glucose_lead_* etc.)
# ============================================================
df_mean = df_base.copy()
numeric_columns = df_mean.select_dtypes(include="number").columns.tolist()

for column in numeric_columns:
    if df_mean[column].isnull().any():
        mean_value = df_mean[column].mean()
        df_mean[column] = df_mean[column].fillna(mean_value)
        print("Mean used for", column, "=", round(mean_value, 3))

# ============================================================
# METHOD 3: MEDIAN IMPUTATION
# ============================================================
df_median = df_base.copy()

for column in numeric_columns:
    if df_median[column].isnull().any():
        median_value = df_median[column].median()
        df_median[column] = df_median[column].fillna(median_value)
        print("Median used for", column, "=", median_value)

# ============================================================
# METHOD 4: FORWARD-FILL (time-series appropriate)
# ============================================================
# glucose_lag_1/3/6 and glucose_lead_1/3/6 are only missing at the
# very start/end of EACH patient's series (no earlier/later reading
# exists yet). A per-patient forward/backward fill respects the
# time-series structure far better than a global mean for this data.
df_ffill = df_base.sort_values(["user_id", "timestamp"]).copy()
lag_lead_cols = [c for c in df_ffill.columns if c.startswith("glucose_lag_") or c.startswith("glucose_lead_")]
df_ffill[lag_lead_cols] = df_ffill.groupby("user_id")[lag_lead_cols].transform(
    lambda s: s.ffill().bfill()
)

# ============================================================
# METHOD 5: MISSING INDICATOR FEATURES
# ============================================================
df_final = df_ffill.copy()
for column in df_base.columns:
    if df_base[column].isnull().any():
        df_final[f"{column}_was_missing"] = df_original[column].isnull().astype(int)

df_final["any_missing_flag"] = deletion_indicator.values

# ============================================================
# CHECK MISSING VALUES AFTER CLEANING
# ============================================================
print("\nMissing Values After Cleaning:")
print(df_final.isnull().sum()[df_final.isnull().sum() > 0])

# ============================================================
# SAVE RESULT
# ============================================================
df_final.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("MISSING VALUE HANDLING COMPLETED SUCCESSFULLY")
print("=" * 70)
print("Output file:", output_file)
