"""
GlucoTrend - main.py

Entry point that runs the full src/ pipeline in order, mirroring
placement_prediction's project layout. Run this once from the project
root to regenerate every dataset/ and outputs/ artifact from scratch:

    python main.py

Each stage is a standalone script under src/ (same convention as
placement_prediction) - you can also run any of them individually.
"""

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

PIPELINE = [
    "Dataset_Load_identify_missing_values_M1.py",
    "Correlation_Matrix_heatmap_boxplots_M1.py",
    "target_variable_engineering_M2.py",   # builds all 5 candidate targets
    "clean_missing_imputer_M2.py",
    "clean_label_encode_M2.py",
    "clean_one_hot_encod_M2.py",
    "clean_ordinal_encod_M2.py",
    "clean_target_encode_M2.py",
    "clean_embedding_encode_M2.py",
    "clean_minmax_stand_norma_M2.py",
    "final_preprocess_M2.py",              # set TARGET_COLUMN inside this file first
]


def run_stage(script_name):
    script_path = os.path.join(SRC_DIR, script_name)
    print("\n" + "=" * 70)
    print(f"RUNNING: {script_name}")
    print("=" * 70)
    result = subprocess.run([sys.executable, script_path], cwd=SRC_DIR)
    if result.returncode != 0:
        print(f"\nStage failed: {script_name} (exit code {result.returncode})")
        sys.exit(result.returncode)


if __name__ == "__main__":
    print("GlucoTrend pipeline starting...")
    for stage in PIPELINE:
        run_stage(stage)
    print("\nAll stages completed. See dataset/ and outputs/ for results.")
    print("To launch the web app instead, run: python app.py")
