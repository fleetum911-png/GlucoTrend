# ================================================================
#                          Module_3
#              Decision_Tree_Classifier_M3.py
# ================================================================
# DECISION TREE CLASSIFIER
# GLUCOSE-RANGE CLASSIFICATION USING GLUCOTREND DATASET
# ================================================================
#
# IMPORTANT:
# The target-engineered dataset is NEVER modified.
# All preprocessing is performed on copies / inside a pipeline.
#
# LEAKAGE NOTE (GlucoTrend-specific - placement_prediction did not
# need this because PlacementStatus was an independent raw column):
# target_binary / target_ada3 / target_5class / target_trend are all
# DERIVED directly from "glucose" (and target_trend from
# "glucose_roc_per_min"). Those source columns - plus the OTHER
# unused target candidates - are dropped from the feature set below
# so the model has to learn from real predictors (insulin, carbs,
# activity, history) instead of trivially re-deriving the label.
#
# OUTPUTS:
#      1. Accuracy
#      2. Precision
#      3. Recall
#      4. F1 Score
#      5. Confusion Matrix
#      6. Performance Graph
#      7. Actual vs Predicted Chart
#      8. Class Distribution Chart
#      9. Feature Importance Chart
#     10. Decision Tree Visualization
#     11. Classification Report
#     12. Test Predictions
#     13. Trained Model
#     14. Decision Tree Parameters
#
# Output folders are created ONLY if they do not already exist.
# ================================================================

# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ================================================================
# 2. DATASET PATH
# ================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "target_engineered_M2.csv")

# ================================================================
# 3. OUTPUT MAIN FOLDER
# ================================================================
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "Decision_Tree_Classifier_M3_Outputs")

# ================================================================
# 4. OUTPUT SUBFOLDERS
# ================================================================
METRICS_FOLDER = os.path.join(OUTPUT_FOLDER, "metrics")
PREDICTIONS_FOLDER = os.path.join(OUTPUT_FOLDER, "predictions")
CONFUSION_FOLDER = os.path.join(OUTPUT_FOLDER, "confusion_matrix")
CHARTS_FOLDER = os.path.join(OUTPUT_FOLDER, "charts")
TREE_FOLDER = os.path.join(OUTPUT_FOLDER, "decision_tree")
FEATURE_FOLDER = os.path.join(OUTPUT_FOLDER, "feature_importance")
MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, "model")

# ================================================================
# 5. CREATE FOLDERS ONLY IF THEY DO NOT EXIST
# ================================================================
folders = [
    OUTPUT_FOLDER, METRICS_FOLDER, PREDICTIONS_FOLDER,
    CONFUSION_FOLDER, CHARTS_FOLDER, TREE_FOLDER, FEATURE_FOLDER, MODEL_FOLDER
]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("Created folder:", folder)
    else:
        print("Folder already exists:", folder)

# ================================================================
# 6. PROGRAM HEADER
# ================================================================
print("\n")
print("=" * 80)
print("              DECISION TREE - GLUCOSE RANGE CLASSIFICATION")
print("=" * 80)

# ================================================================
# 7. CHECK DATASET
# ================================================================
if not os.path.exists(DATASET_PATH):
    print("\nERROR: Dataset not found.")
    print("\nRun target_variable_engineering_M2.py first. Expected path:")
    print(DATASET_PATH)
    raise SystemExit

print("\nDataset found successfully.")

# ================================================================
# 8. LOAD DATASET
# ================================================================
df = pd.read_csv(DATASET_PATH)
print("\nDataset loaded successfully.")
print("Rows     :", df.shape[0])
print("Columns :", df.shape[1])

# ================================================================
# 9. CREATE A COPY
# ================================================================
data = df.copy()

# ================================================================
# 10. DISPLAY DATASET INFORMATION
# ================================================================
print("\nDataset columns:")
print(list(data.columns))
print("\nFirst 5 records:")
print(data.head())

# ================================================================
# 11. SELECT TARGET COLUMN
# ================================================================
# Choose ONE of: "target_binary", "target_ada3", "target_5class", "target_trend"
# target_binary is used by default - best-supported (no empty classes).
TARGET_COLUMN = "target_binary"

if TARGET_COLUMN not in data.columns:
    print("\nERROR: Target column could not be detected.")
    print("\nAvailable columns:")
    for column in data.columns:
        print(column)
    print("\nSet TARGET_COLUMN manually in the program.")
    raise SystemExit

target_column = TARGET_COLUMN
print("\nTarget column:", target_column)

# ================================================================
# 12. REMOVE MISSING TARGET VALUES
# ================================================================
data_model = data.dropna(subset=[target_column]).copy()
print("\nRecords used for modeling:", len(data_model))

# ================================================================
# 13. DROP LEAKAGE COLUMNS (see note at top of file)
# ================================================================
ALL_TARGET_CANDIDATES = ["target_binary", "target_ada3", "target_5class", "target_trend"]
OTHER_TARGETS = [c for c in ALL_TARGET_CANDIDATES if c != target_column]

LEAKAGE_SOURCE_COLUMNS = {
    "target_binary": ["glucose"],
    "target_ada3": ["glucose"],
    "target_5class": ["glucose"],
    "target_trend": ["glucose_roc_per_min"],
}
leakage_cols = LEAKAGE_SOURCE_COLUMNS.get(target_column, [])

IDENTIFIER_COLUMNS = ["user_id", "timestamp", "notes"]
REGRESSION_TARGETS = ["glucose_lead_1", "glucose_lead_3", "glucose_lead_6"]

drop_cols = OTHER_TARGETS + leakage_cols + IDENTIFIER_COLUMNS + REGRESSION_TARGETS
print("\nDropped as leakage / identifiers / other targets:", drop_cols)

# ================================================================
# 14. SEPARATE FEATURES AND TARGET
# ================================================================
X = data_model.drop(columns=[target_column] + drop_cols, errors="ignore").copy()
y = data_model[target_column].copy()

# ================================================================
# 15. TARGET DISTRIBUTION
# ================================================================
print("\nTarget class distribution:")
print(y.value_counts())

# ================================================================
# 16. REMOVE COMPLETELY EMPTY FEATURES
# ================================================================
empty_columns = X.columns[X.isnull().all()].tolist()
if len(empty_columns) > 0:
    print("\nCompletely empty columns:", empty_columns)
    X = X.drop(columns=empty_columns)

# ================================================================
# 17. IDENTIFY NUMERIC FEATURES
# ================================================================
numeric_features = X.select_dtypes(include=np.number).columns.tolist()

# ================================================================
# 18. IDENTIFY CATEGORICAL FEATURES
# ================================================================
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("\nNumeric features:")
print(numeric_features)
print("\nCategorical features:")
print(categorical_features)

# ================================================================
# 19. NUMERIC PREPROCESSING
# ================================================================
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

# ================================================================
# 20. CATEGORICAL PREPROCESSING
# ================================================================
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# ================================================================
# 21. COLUMN TRANSFORMER
# ================================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

# ================================================================
# 22. TRAIN-TEST SPLIT
# ================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))

# ================================================================
# 23. CREATE DECISION TREE
# ================================================================
decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# ================================================================
# 24. CREATE MODEL PIPELINE
# ================================================================
model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", decision_tree)])

# ================================================================
# 25. TRAIN DECISION TREE
# ================================================================
print("\nTraining Decision Tree...")
model.fit(X_train, y_train)
print("Decision Tree training completed.")

# ================================================================
# 26. PREDICTION
# ================================================================
print("\nGenerating predictions...")
y_pred = model.predict(X_test)
print("Prediction completed.")

# ================================================================
# 27-30. METRICS
# ================================================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

# ================================================================
# 31. DISPLAY PERFORMANCE
# ================================================================
print("\n")
print("=" * 80)
print("                 DECISION TREE PERFORMANCE")
print("=" * 80)
print(f"\nAccuracy   : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")
print("\nPerformance Percentage:")
print(f"Accuracy   : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall     : {recall * 100:.2f}%")
print(f"F1 Score     : {f1 * 100:.2f}%")

# ================================================================
# 32. SAVE METRICS
# ================================================================
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [accuracy, precision, recall, f1],
    "Percentage": [accuracy * 100, precision * 100, recall * 100, f1 * 100]
})
metrics_df.to_csv(os.path.join(METRICS_FOLDER, "decision_tree_metrics.csv"), index=False)

# ================================================================
# 33. CLASSIFICATION REPORT
# ================================================================
classification_report_result = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
classification_report_df = pd.DataFrame(classification_report_result).transpose()
classification_report_df.to_csv(os.path.join(METRICS_FOLDER, "classification_report.csv"))

# ================================================================
# 34. CONFUSION MATRIX
# ================================================================
cm = confusion_matrix(y_test, y_pred)
tree_classifier = model.named_steps["classifier"]
class_labels = tree_classifier.classes_

print("\n")
print("=" * 80)
print("                   CONFUSION MATRIX")
print("=" * 80)
print(cm)

# ================================================================
# 35. SAVE CONFUSION MATRIX CSV
# ================================================================
cm_df = pd.DataFrame(
    cm,
    index=["Actual_" + str(label) for label in class_labels],
    columns=["Predicted_" + str(label) for label in class_labels]
)
cm_df.to_csv(os.path.join(CONFUSION_FOLDER, "confusion_matrix.csv"))

# ================================================================
# 36. CONFUSION MATRIX GRAPH
# ================================================================
plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks(range(len(class_labels)), class_labels, rotation=45)
plt.yticks(range(len(class_labels)), class_labels)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(CONFUSION_FOLDER, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()

# ================================================================
# 37. PERFORMANCE GRAPH
# ================================================================
metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
metric_values = [accuracy * 100, precision * 100, recall * 100, f1 * 100]

plt.figure(figsize=(10, 6))
bars = plt.bar(metric_names, metric_values)
plt.title("Decision Tree Performance")
plt.xlabel("Evaluation Metric")
plt.ylabel("Score (%)")
plt.ylim(0, 100)
for bar, value in zip(bars, metric_values):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.2f}%", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_FOLDER, "performance_graph.png"), dpi=300, bbox_inches="tight")
plt.close()

# ================================================================
# 38. ACTUAL VS PREDICTED CHART
# ================================================================
actual_counts = y_test.value_counts()
predicted_counts = pd.Series(y_pred).value_counts()
comparison_df = pd.DataFrame({"Actual": actual_counts, "Predicted": predicted_counts}).fillna(0)
comparison_df = comparison_df.reindex(class_labels)

plt.figure(figsize=(9, 6))
x = np.arange(len(class_labels))
width = 0.35
plt.bar(x - width / 2, comparison_df["Actual"], width, label="Actual")
plt.bar(x + width / 2, comparison_df["Predicted"], width, label="Predicted")
plt.xlabel("Glucose Range Class")
plt.ylabel("Number of Readings")
plt.title("Actual vs Predicted Glucose Range")
plt.xticks(x, class_labels)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_FOLDER, "actual_vs_predicted.png"), dpi=300, bbox_inches="tight")
plt.close()

# ================================================================
# 39. CLASS DISTRIBUTION CHART
# ================================================================
class_counts = y.value_counts()
plt.figure(figsize=(8, 6))
plt.bar(class_counts.index.astype(str), class_counts.values)
plt.xlabel("Glucose Range Class")
plt.ylabel("Number of Readings")
plt.title("Glucose Range Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_FOLDER, "class_distribution.png"), dpi=300, bbox_inches="tight")
plt.close()

# ================================================================
# 40. FEATURE NAMES AFTER ENCODING
# ================================================================
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# ================================================================
# 41-42. FEATURE IMPORTANCE
# ================================================================
feature_importances = tree_classifier.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": feature_names, "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)
feature_importance_df.to_csv(os.path.join(FEATURE_FOLDER, "feature_importance.csv"), index=False)

# ================================================================
# 43. FEATURE IMPORTANCE GRAPH
# ================================================================
top_features = feature_importance_df.head(15).sort_values(by="Importance")
plt.figure(figsize=(10, 7))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Decision Tree Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(FEATURE_FOLDER, "feature_importance.png"), dpi=300, bbox_inches="tight")
plt.close()

# ================================================================
# 44-45. DECISION TREE VISUALIZATION
# ================================================================
print("\n")
print("=" * 80)
print("                DECISION TREE VISUALIZATION")
print("=" * 80)
print("\nRendering Decision Tree...")

plt.figure(figsize=(30, 18))
plot_tree(
    tree_classifier,
    feature_names=feature_names,
    class_names=[str(label) for label in class_labels],
    filled=True,
    rounded=True,
    proportion=False,
    precision=2,
    fontsize=8
)
plt.title("Decision Tree - Glucose Range Classification", fontsize=20)
plt.tight_layout()
tree_image_path = os.path.join(TREE_FOLDER, "decision_tree_diagram.png")
plt.savefig(tree_image_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nDecision Tree diagram saved at:")
print(tree_image_path)

# ================================================================
# 47. SAVE TEST PREDICTIONS
# ================================================================
test_predictions = X_test.copy()
test_predictions["Actual"] = y_test.values
test_predictions["Predicted"] = y_pred
test_predictions.to_csv(os.path.join(PREDICTIONS_FOLDER, "test_predictions.csv"), index=False)

# ================================================================
# 48. SAVE TRAINED MODEL
# ================================================================
model_path = os.path.join(MODEL_FOLDER, "decision_tree_model.pkl")
with open(model_path, "wb") as file:
    pickle.dump(model, file)

# ================================================================
# 49. SAVE DECISION TREE PARAMETERS
# ================================================================
parameters_df = pd.DataFrame({
    "Parameter": ["Algorithm", "Target", "Criterion", "Maximum Depth",
                  "Minimum Samples Split", "Minimum Samples Leaf", "Random State"],
    "Value": ["Decision Tree Classifier", target_column, decision_tree.criterion,
              decision_tree.max_depth, decision_tree.min_samples_split,
              decision_tree.min_samples_leaf, decision_tree.random_state]
})
parameters_df.to_csv(os.path.join(METRICS_FOLDER, "decision_tree_parameters.csv"), index=False)

# ================================================================
# 50. FINAL SUMMARY
# ================================================================
print("\n")
print("=" * 80)
print("         DECISION TREE COMPLETED SUCCESSFULLY")
print("=" * 80)
print("\nOriginal target-engineered dataset was NOT modified.")
print("\nDataset used:")
print(DATASET_PATH)
print("\nAll outputs stored in:")
print(OUTPUT_FOLDER)
print("\n")
print("FINAL PERFORMANCE")
print("-" * 50)
print(f"Accuracy   : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall     : {recall * 100:.2f}%")
print(f"F1 Score   : {f1 * 100:.2f}%")
print("\n")
print("=" * 80)
print("                       PROGRAM FINISHED")
print("=" * 80)
