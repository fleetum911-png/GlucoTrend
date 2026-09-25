# ================================================================
#                          Module_3
#                AdaBoost_Classifier_M3.py
# ================================================================
# ADABOOST CLASSIFIER
# GLUCOSE-RANGE CLASSIFICATION USING GLUCOTREND DATASET
# ================================================================
#
# IMPORTANT: The target-engineered dataset is NEVER modified.
# All preprocessing is performed on copies / inside a pipeline.
#
# LEAKAGE NOTE: see Decision_Tree_Classifier_M3.py - "glucose" (and
# "glucose_roc_per_min" for target_trend) are dropped from features
# because the engineered targets are derived directly from them.
#
# OUTPUTS:
#   1. Accuracy            6. Performance Graph        11. Classification Report
#   2. Precision           7. Actual vs Predicted       12. Test Predictions
#   3. Recall              8. Class Distribution        13. Trained Model
#   4. F1 Score            9. Feature Importance Chart  14. AdaBoost Parameters
#   5. Confusion Matrix    10. AdaBoost Base Tree Visualization
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
from sklearn.ensemble import AdaBoostClassifier
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
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "AdaBoost_Classifier_M3_Outputs")

# ================================================================
# 4. OUTPUT SUBFOLDERS
# ================================================================
METRICS_FOLDER = os.path.join(OUTPUT_FOLDER, "metrics")
PREDICTIONS_FOLDER = os.path.join(OUTPUT_FOLDER, "predictions")
CONFUSION_FOLDER = os.path.join(OUTPUT_FOLDER, "confusion_matrix")
CHARTS_FOLDER = os.path.join(OUTPUT_FOLDER, "charts")
TREE_FOLDER = os.path.join(OUTPUT_FOLDER, "adaboost_base_tree")
FEATURE_FOLDER = os.path.join(OUTPUT_FOLDER, "feature_importance")
MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, "model")

# ================================================================
# 5. CREATE OUTPUT FOLDERS
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
print("              ADABOOST - GLUCOSE RANGE CLASSIFICATION")
print("=" * 80)

# ================================================================
# 7. CHECK DATASET
# ================================================================
if not os.path.exists(DATASET_PATH):
    print("\nERROR: Dataset not found.")
    print("Run target_variable_engineering_M2.py first. Expected path:")
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

data = df.copy()

# ================================================================
# 9. SELECT TARGET COLUMN
# ================================================================
TARGET_COLUMN = "target_binary"

if TARGET_COLUMN not in data.columns:
    print("\nERROR: Target column could not be detected.")
    raise SystemExit

target_column = TARGET_COLUMN
print("\nTarget column:", target_column)

# ================================================================
# 10. REMOVE MISSING TARGET VALUES
# ================================================================
data_model = data.dropna(subset=[target_column]).copy()
print("\nRecords used for modeling:", len(data_model))

# ================================================================
# 11. DROP LEAKAGE COLUMNS
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
# 12. SEPARATE FEATURES AND TARGET
# ================================================================
X = data_model.drop(columns=[target_column] + drop_cols, errors="ignore").copy()
y = data_model[target_column].copy()

print("\nTarget class distribution:")
print(y.value_counts())

# ================================================================
# 13. REMOVE COMPLETELY EMPTY FEATURES
# ================================================================
empty_columns = X.columns[X.isnull().all()].tolist()
if len(empty_columns) > 0:
    print("\nCompletely empty columns:", empty_columns)
    X = X.drop(columns=empty_columns)

# ================================================================
# 14. IDENTIFY NUMERIC / CATEGORICAL FEATURES
# ================================================================
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("\nNumeric features:")
print(numeric_features)
print("\nCategorical features:")
print(categorical_features)

# ================================================================
# 15-16. PREPROCESSING PIPELINES
# ================================================================
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# ================================================================
# 17. COLUMN TRANSFORMER
# ================================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

# ================================================================
# 18. TRAIN-TEST SPLIT
# ================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# ================================================================
# 19. CREATE BASE (WEAK) DECISION TREE
# ================================================================
# AdaBoost uses weak decision trees as base estimators.
base_tree = DecisionTreeClassifier(
    max_depth=2,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# ================================================================
# 20. CREATE ADABOOST MODEL
# ================================================================
# For current scikit-learn versions, estimator= is used.
try:
    adaboost = AdaBoostClassifier(
        estimator=base_tree,
        n_estimators=100,
        learning_rate=0.8,
        random_state=42
    )
except TypeError:
    # Compatibility with older scikit-learn versions
    adaboost = AdaBoostClassifier(
        base_estimator=base_tree,
        n_estimators=100,
        learning_rate=0.8,
        random_state=42
    )

# ================================================================
# 21. CREATE PIPELINE
# ================================================================
model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", adaboost)])

# ================================================================
# 22. TRAIN ADABOOST
# ================================================================
print("\nTraining AdaBoost...")
model.fit(X_train, y_train)
print("AdaBoost training completed.")

# ================================================================
# 23. PREDICTION
# ================================================================
print("\nGenerating predictions...")
y_pred = model.predict(X_test)
print("Prediction completed.")

# ================================================================
# 24-27. METRICS
# ================================================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n" + "=" * 80)
print("                 ADABOOST PERFORMANCE")
print("=" * 80)
print(f"\nAccuracy   : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall     : {recall * 100:.2f}%")
print(f"F1 Score   : {f1 * 100:.2f}%")

# ================================================================
# 28. SAVE METRICS
# ================================================================
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [accuracy, precision, recall, f1],
    "Percentage": [accuracy * 100, precision * 100, recall * 100, f1 * 100]
})
metrics_df.to_csv(os.path.join(METRICS_FOLDER, "adaboost_metrics.csv"), index=False)

# ================================================================
# 29. CLASSIFICATION REPORT
# ================================================================
classification_report_result = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
pd.DataFrame(classification_report_result).transpose().to_csv(
    os.path.join(METRICS_FOLDER, "classification_report.csv")
)

# ================================================================
# 30. CONFUSION MATRIX
# ================================================================
cm = confusion_matrix(y_test, y_pred)
adaboost_classifier = model.named_steps["classifier"]
class_labels = adaboost_classifier.classes_

print("\n" + "=" * 80)
print("                   CONFUSION MATRIX")
print("=" * 80)
print(cm)

cm_df = pd.DataFrame(
    cm,
    index=["Actual_" + str(label) for label in class_labels],
    columns=["Predicted_" + str(label) for label in class_labels]
)
cm_df.to_csv(os.path.join(CONFUSION_FOLDER, "confusion_matrix.csv"))

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("AdaBoost - Confusion Matrix")
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
# 31. PERFORMANCE GRAPH
# ================================================================
metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
metric_values = [accuracy * 100, precision * 100, recall * 100, f1 * 100]
plt.figure(figsize=(10, 6))
bars = plt.bar(metric_names, metric_values)
plt.title("AdaBoost Performance")
plt.xlabel("Evaluation Metric")
plt.ylabel("Score (%)")
plt.ylim(0, 100)
for bar, value in zip(bars, metric_values):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.2f}%", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_FOLDER, "performance_graph.png"), dpi=300, bbox_inches="tight")
plt.close()

# ================================================================
# 32. ACTUAL VS PREDICTED CHART
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
plt.title("Actual vs Predicted Glucose Range - AdaBoost")
plt.xticks(x, class_labels)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_FOLDER, "actual_vs_predicted.png"), dpi=300, bbox_inches="tight")
plt.close()

# ================================================================
# 33. CLASS DISTRIBUTION CHART
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
# 34. FEATURE NAMES AFTER ENCODING
# ================================================================
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# ================================================================
# 35-36. FEATURE IMPORTANCE
# ================================================================
feature_importances = adaboost_classifier.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": feature_names, "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)
feature_importance_df.to_csv(os.path.join(FEATURE_FOLDER, "feature_importance.csv"), index=False)

top_features = feature_importance_df.head(15).sort_values(by="Importance")
plt.figure(figsize=(10, 7))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 AdaBoost Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(FEATURE_FOLDER, "feature_importance.png"), dpi=300, bbox_inches="tight")
plt.close()

# ================================================================
# 37. ADABOOST BASE TREE VISUALIZATION
# ================================================================
# AdaBoost contains many weak decision trees - visualize the first one.
print("\n" + "=" * 80)
print("                ADABOOST BASE TREE VISUALIZATION")
print("=" * 80)
print("\nRendering first AdaBoost base decision tree...")

plt.figure(figsize=(20, 12))
plot_tree(
    adaboost_classifier.estimators_[0],
    feature_names=feature_names,
    class_names=[str(label) for label in class_labels],
    filled=True,
    rounded=True,
    proportion=False,
    precision=2,
    fontsize=9
)
plt.title("AdaBoost - First Base Decision Tree", fontsize=18)
plt.tight_layout()
tree_image_path = os.path.join(TREE_FOLDER, "adaboost_base_tree.png")
plt.savefig(tree_image_path, dpi=300, bbox_inches="tight")
plt.close()
print("\nAdaBoost base tree saved at:", tree_image_path)

# ================================================================
# 38. SAVE TEST PREDICTIONS
# ================================================================
test_predictions = X_test.copy()
test_predictions["Actual"] = y_test.values
test_predictions["Predicted"] = y_pred
test_predictions.to_csv(os.path.join(PREDICTIONS_FOLDER, "test_predictions.csv"), index=False)

# ================================================================
# 39. SAVE TRAINED MODEL
# ================================================================
with open(os.path.join(MODEL_FOLDER, "adaboost_model.pkl"), "wb") as file:
    pickle.dump(model, file)

# ================================================================
# 40. SAVE ADABOOST PARAMETERS
# ================================================================
parameters_df = pd.DataFrame({
    "Parameter": ["Algorithm", "Target", "Base Estimator", "N Estimators",
                  "Learning Rate", "Random State"],
    "Value": ["AdaBoost Classifier", target_column, "Decision Tree (max_depth=2)",
              adaboost_classifier.n_estimators, adaboost_classifier.learning_rate,
              adaboost_classifier.random_state]
})
parameters_df.to_csv(os.path.join(METRICS_FOLDER, "adaboost_parameters.csv"), index=False)

# ================================================================
# 41. FINAL SUMMARY
# ================================================================
print("\n" + "=" * 80)
print("         ADABOOST COMPLETED SUCCESSFULLY")
print("=" * 80)
print("\nAll outputs stored in:", OUTPUT_FOLDER)
print("\nFINAL PERFORMANCE")
print("-" * 50)
print(f"Accuracy   : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall     : {recall * 100:.2f}%")
print(f"F1 Score   : {f1 * 100:.2f}%")
print("\n" + "=" * 80)
print("                       PROGRAM FINISHED")
print("=" * 80)
