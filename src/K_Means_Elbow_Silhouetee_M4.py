# ============================================================
#                          Module_4
#              K_Means_Elbow_Silhouetee_M4.py
# ============================================================
# GLUCOTREND - PREPROCESSED DATASET
# K-MEANS AND K-MEANS++ CLUSTERING
#
# Features (glucose "behavior profile" clustering, GlucoTrend's
# analogue of placement_prediction's CGPA/Backlogs/Internships/
# Aptitude academic-profile features):
#    glucose
#    heart_rate
#    stress_level
#    hbA1c
#
# Methods:
#    1. K-Means
#    2. K-Means++
#    3. Elbow Method
#    4. Silhouette Score
#    5. PCA visualization
#
# IMPORTANT:
# - Uses the PREPROCESSED dataset (final_preprocess_M2.csv)
# - Original dataset is NOT modified
# - Outputs are stored in separate folders
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    accuracy_score
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. INPUT FILE
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "dataset", "final_preprocess_M2.csv")

# ============================================================
# 2. OUTPUT FOLDER
# ============================================================
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "K_Means_K++Means_Elbow_Silhoute_M4_Outputs")

# ============================================================
# 3. CREATE SEPARATE OUTPUT FOLDERS
# ============================================================
KMEANS_FOLDER = os.path.join(OUTPUT_FOLDER, "KMeans")
KMEANS_PP_FOLDER = os.path.join(OUTPUT_FOLDER, "KMeansPlusPlus")
ELBOW_FOLDER = os.path.join(OUTPUT_FOLDER, "Elbow")
SILHOUETTE_FOLDER = os.path.join(OUTPUT_FOLDER, "Silhouette")
ACCURACY_FOLDER = os.path.join(OUTPUT_FOLDER, "Accuracy")

for folder in [OUTPUT_FOLDER, KMEANS_FOLDER, KMEANS_PP_FOLDER, ELBOW_FOLDER, SILHOUETTE_FOLDER, ACCURACY_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ============================================================
# 4. LOAD PREPROCESSED DATASET
# ============================================================
print("=" * 75)
print("K-MEANS AND K-MEANS++")
print("GLUCOTREND - PREPROCESSED DATASET")
print("=" * 75)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        "\nPreprocessed dataset was not found.\n\nRun final_preprocess_M2.py first. Expected path:\n"
        + INPUT_FILE
    )

data = pd.read_csv(INPUT_FILE)
print("\nPreprocessed dataset loaded successfully.")
print("Rows    :", data.shape[0])
print("Columns :", data.shape[1])

# ============================================================
# 5. DISPLAY COLUMNS
# ============================================================
print("\nAvailable columns:")
print("-" * 75)
for i, column in enumerate(data.columns, 1):
    print(i, ".", column)

# ============================================================
# 6. COLUMN-NAME NORMALIZATION
# ============================================================


def normalize_column_name(column):
    return str(column).strip().lower().replace("_", "").replace(" ", "").replace("-", "")


normalized_columns = {normalize_column_name(column): column for column in data.columns}

# ============================================================
# 7. REQUIRED FEATURES
# ============================================================
required_features = ["glucose", "heart_rate", "stress_level", "hbA1c"]

# ============================================================
# 8. FIND FEATURES
# ============================================================
feature_columns = []
for feature in required_features:
    normalized_feature = normalize_column_name(feature)
    if normalized_feature in normalized_columns:
        feature_columns.append(normalized_columns[normalized_feature])
    else:
        raise ValueError(
            f"\nRequired column '{feature}' was not found.\n\nAvailable columns:\n"
            + "\n".join(data.columns.astype(str))
        )

# ============================================================
# 9. DISPLAY SELECTED FEATURES
# ============================================================
print("\n" + "=" * 75)
print("FEATURES USED FOR CLUSTERING")
print("=" * 75)
for feature in feature_columns:
    print("-", feature)

# ============================================================
# 10. SELECT FOUR FEATURES
# ============================================================
X = data[feature_columns].copy()

# ============================================================
# 11. CONVERT TO NUMERIC
# ============================================================
for column in feature_columns:
    X[column] = pd.to_numeric(X[column], errors="coerce")

# ============================================================
# 12. HANDLE INFINITE VALUES
# ============================================================
X = X.replace([np.inf, -np.inf], np.nan)

# ============================================================
# 13. HANDLE MISSING VALUES
# ============================================================
print("\nMissing values:")
print(X.isnull().sum())

for column in feature_columns:
    X[column] = X[column].fillna(X[column].median())

print("\nMissing values after preprocessing:")
print(X.isnull().sum())

# ============================================================
# 14. STANDARDISATION
# ============================================================
# Even though this is a preprocessed dataset, K-Means is
# distance-based. Therefore, the four clustering variables
# are standardised before clustering.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nFeature standardisation completed.")

# ============================================================
# 15. K RANGE
# ============================================================
K_VALUES = range(2, 11)

# ============================================================
# 16. K-MEANS - ELBOW AND SILHOUETTE
# ============================================================
print("\n" + "=" * 75)
print("K-MEANS ANALYSIS")
print("=" * 75)

kmeans_inertia = []
kmeans_silhouette = []
for k in K_VALUES:
    model = KMeans(n_clusters=k, init="random", n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)
    kmeans_inertia.append(inertia)
    kmeans_silhouette.append(silhouette)
    print(f"K = {k:2d} | Inertia = {inertia:.4f} | Silhouette = {silhouette:.4f}")

# ============================================================
# 17. K-MEANS++ - ELBOW AND SILHOUETTE
# ============================================================
print("\n" + "=" * 75)
print("K-MEANS++ ANALYSIS")
print("=" * 75)

kmeans_pp_inertia = []
kmeans_pp_silhouette = []
for k in K_VALUES:
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)
    kmeans_pp_inertia.append(inertia)
    kmeans_pp_silhouette.append(silhouette)
    print(f"K = {k:2d} | Inertia = {inertia:.4f} | Silhouette = {silhouette:.4f}")

# ============================================================
# 18. SELECT BEST K
# ============================================================
# Maximum Silhouette Score is used to select K automatically.
# The Elbow graph should also be inspected.
best_k_kmeans = list(K_VALUES)[np.argmax(kmeans_silhouette)]
best_k_kmeans_pp = list(K_VALUES)[np.argmax(kmeans_pp_silhouette)]

print("\n" + "=" * 75)
print("SELECTED NUMBER OF CLUSTERS")
print("=" * 75)
print("K-Means best K   :", best_k_kmeans)
print("K-Means++ best K :", best_k_kmeans_pp)

# ============================================================
# 19-22. ELBOW / SILHOUETTE GRAPHS
# ============================================================


def line_chart(x_vals, y_vals, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(list(x_vals), y_vals, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(list(x_vals))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


line_chart(K_VALUES, kmeans_inertia, "Number of Clusters (K)", "Inertia",
           "Elbow Method - K-Means", os.path.join(ELBOW_FOLDER, "elbow_kmeans.png"))
line_chart(K_VALUES, kmeans_pp_inertia, "Number of Clusters (K)", "Inertia",
           "Elbow Method - K-Means++", os.path.join(ELBOW_FOLDER, "elbow_kmeans_plus_plus.png"))
line_chart(K_VALUES, kmeans_silhouette, "Number of Clusters (K)", "Silhouette Score",
           "Silhouette Score - K-Means", os.path.join(SILHOUETTE_FOLDER, "silhouette_kmeans.png"))
line_chart(K_VALUES, kmeans_pp_silhouette, "Number of Clusters (K)", "Silhouette Score",
           "Silhouette Score - K-Means++", os.path.join(SILHOUETTE_FOLDER, "silhouette_kmeans_plus_plus.png"))

# ============================================================
# 23. FINAL K-MEANS
# ============================================================
kmeans_model = KMeans(n_clusters=best_k_kmeans, init="random", n_init=10, random_state=42)
kmeans_labels = kmeans_model.fit_predict(X_scaled)

# ============================================================
# 24. FINAL K-MEANS++
# ============================================================
kmeans_pp_model = KMeans(n_clusters=best_k_kmeans_pp, init="k-means++", n_init=10, random_state=42)
kmeans_pp_labels = kmeans_pp_model.fit_predict(X_scaled)

# ============================================================
# 25-26. SAVE CLUSTERED DATA
# ============================================================
kmeans_result = X.copy()
kmeans_result["KMeans_Cluster"] = kmeans_labels + 1
kmeans_result.to_csv(os.path.join(KMEANS_FOLDER, "kmeans_clustered_data.csv"), index=False)

kmeans_pp_result = X.copy()
kmeans_pp_result["KMeansPlusPlus_Cluster"] = kmeans_pp_labels + 1
kmeans_pp_result.to_csv(os.path.join(KMEANS_PP_FOLDER, "kmeans_plus_plus_clustered_data.csv"), index=False)

# ============================================================
# 27-28. FINAL METRICS
# ============================================================
km_silhouette = silhouette_score(X_scaled, kmeans_labels)
km_calinski = calinski_harabasz_score(X_scaled, kmeans_labels)
km_davies = davies_bouldin_score(X_scaled, kmeans_labels)

pp_silhouette = silhouette_score(X_scaled, kmeans_pp_labels)
pp_calinski = calinski_harabasz_score(X_scaled, kmeans_pp_labels)
pp_davies = davies_bouldin_score(X_scaled, kmeans_pp_labels)

# ============================================================
# 29-30. SAVE METRICS
# ============================================================
pd.DataFrame({
    "Method": ["K-Means"], "Number_of_Clusters": [best_k_kmeans],
    "Inertia": [kmeans_model.inertia_], "Silhouette_Score": [km_silhouette],
    "Calinski_Harabasz_Score": [km_calinski], "Davies_Bouldin_Score": [km_davies]
}).to_csv(os.path.join(KMEANS_FOLDER, "kmeans_metrics.csv"), index=False)

pd.DataFrame({
    "Method": ["K-Means++"], "Number_of_Clusters": [best_k_kmeans_pp],
    "Inertia": [kmeans_pp_model.inertia_], "Silhouette_Score": [pp_silhouette],
    "Calinski_Harabasz_Score": [pp_calinski], "Davies_Bouldin_Score": [pp_davies]
}).to_csv(os.path.join(KMEANS_PP_FOLDER, "kmeans_plus_plus_metrics.csv"), index=False)

# ============================================================
# 31-32. 3-D CLUSTERING GRAPHS
# ============================================================


def plot_3d(x_data, labels, title, save_path):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        x_data[feature_columns[0]], x_data[feature_columns[1]], x_data[feature_columns[2]],
        c=labels, s=20, alpha=0.7
    )
    ax.set_xlabel(feature_columns[0])
    ax.set_ylabel(feature_columns[1])
    ax.set_zlabel(feature_columns[2])
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# Four features cannot be directly plotted in a normal 3-D graph.
# The first three (glucose, heart_rate, stress_level) are shown here.
# All FOUR features are used for actual clustering.
plot_3d(X, kmeans_labels, "K-Means Clustering", os.path.join(KMEANS_FOLDER, "kmeans_3D_clustering.png"))
plot_3d(X, kmeans_pp_labels, "K-Means++ Clustering", os.path.join(KMEANS_PP_FOLDER, "kmeans_plus_plus_3D_clustering.png"))

# ============================================================
# 33. PCA 2-D VISUALISATION
# ============================================================
# PCA is ONLY used here to visualize the four-dimensional
# clustering results in two dimensions. PCA is NOT used to
# perform the clustering.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


def pca_scatter(labels, title, save_path):
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=20, alpha=0.7)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(title)
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


pca_scatter(kmeans_labels, "K-Means Clusters - PCA Visualization", os.path.join(KMEANS_FOLDER, "kmeans_PCA_clustering.png"))
pca_scatter(kmeans_pp_labels, "K-Means++ Clusters - PCA Visualization",
            os.path.join(KMEANS_PP_FOLDER, "kmeans_plus_plus_PCA_clustering.png"))

# ============================================================
# 36. CLUSTER COUNTS
# ============================================================
print("\n" + "=" * 75)
print("K-MEANS CLUSTER COUNTS")
print("=" * 75)
print(kmeans_result["KMeans_Cluster"].value_counts().sort_index())

print("\n" + "=" * 75)
print("K-MEANS++ CLUSTER COUNTS")
print("=" * 75)
print(kmeans_pp_result["KMeansPlusPlus_Cluster"].value_counts().sort_index())

# ============================================================
# 37. FIND GLUCOSE-RANGE TARGET (GlucoTrend's analogue of
# placement's Placement/Placed/Status/Target lookup)
# ============================================================
target_candidates = ["target_ada3", "target_binary", "target_5class", "target_trend"]
target_column = None
for candidate in target_candidates:
    normalized_candidate = normalize_column_name(candidate)
    if normalized_candidate in normalized_columns:
        target_column = normalized_columns[normalized_candidate]
        break

# ============================================================
# 38. CLUSTER MATCHING ACCURACY
# ============================================================


def cluster_matching_accuracy(true_values, cluster_values):
    true_values = pd.Series(true_values).reset_index(drop=True)
    cluster_values = pd.Series(cluster_values).reset_index(drop=True)
    predicted_values = np.empty(len(cluster_values), dtype=object)
    for cluster in np.unique(cluster_values):
        indexes = np.where(cluster_values == cluster)[0]
        cluster_targets = true_values.iloc[indexes]
        majority = cluster_targets.mode()
        if len(majority) > 0:
            predicted_values[indexes] = majority.iloc[0]
    # Compare as strings - true_values may already be label-encoded (int)
    # by final_preprocess_M2.py, while predicted_values is built as a
    # generic object array; casting both avoids a dtype mismatch in
    # accuracy_score's target-type inference.
    return accuracy_score(true_values.astype(str), pd.Series(predicted_values).astype(str))


# ============================================================
# 39. CALCULATE ACCURACY IF TARGET EXISTS
# ============================================================
if target_column is not None:
    print("\n" + "=" * 75)
    print("GLUCOSE-RANGE CLUSTER MATCHING ACCURACY")
    print("=" * 75)

    target = data[target_column].copy()
    valid = target.notna()
    target_valid = target[valid].reset_index(drop=True)
    km_labels_valid = pd.Series(kmeans_labels, index=data.index)[valid].reset_index(drop=True)
    pp_labels_valid = pd.Series(kmeans_pp_labels, index=data.index)[valid].reset_index(drop=True)

    km_accuracy = cluster_matching_accuracy(target_valid, km_labels_valid)
    pp_accuracy = cluster_matching_accuracy(target_valid, pp_labels_valid)

    print("Target column:", target_column)
    print(f"K-Means   : {km_accuracy * 100:.2f}%")
    print(f"K-Means++ : {pp_accuracy * 100:.2f}%")

    pd.DataFrame({
        "Method": ["K-Means", "K-Means++"],
        "Cluster_Matching_Accuracy": [km_accuracy, pp_accuracy],
        "Accuracy_Percentage": [km_accuracy * 100, pp_accuracy * 100]
    }).to_csv(os.path.join(ACCURACY_FOLDER, "glucose_range_cluster_matching_accuracy.csv"), index=False)
else:
    print("\nNo target_ada3/target_binary/target_5class/target_trend column was found.")
    print("Accuracy is not calculated because K-Means is an unsupervised algorithm.")

# ============================================================
# 40. FINAL RESULTS
# ============================================================
print("\n" + "=" * 75)
print("FINAL CLUSTERING RESULTS")
print("=" * 75)
print("\nK-MEANS")
print("-" * 40)
print("Best K:", best_k_kmeans)
print("Inertia:", round(kmeans_model.inertia_, 4))
print("Silhouette:", round(km_silhouette, 4))
print("Calinski-Harabasz:", round(km_calinski, 4))
print("Davies-Bouldin:", round(km_davies, 4))

print("\nK-MEANS++")
print("-" * 40)
print("Best K:", best_k_kmeans_pp)
print("Inertia:", round(kmeans_pp_model.inertia_, 4))
print("Silhouette:", round(pp_silhouette, 4))
print("Calinski-Harabasz:", round(pp_calinski, 4))
print("Davies-Bouldin:", round(pp_davies, 4))

# ============================================================
# 41. OUTPUT LOCATION
# ============================================================
print("\n" + "=" * 75)
print("OUTPUTS SAVED")
print("=" * 75)
print("\nMain folder:", OUTPUT_FOLDER)
print("K-Means:", KMEANS_FOLDER)
print("K-Means++:", KMEANS_PP_FOLDER)
print("Elbow:", ELBOW_FOLDER)
print("Silhouette:", SILHOUETTE_FOLDER)
print("Accuracy:", ACCURACY_FOLDER)
print("\nOriginal/preprocessed input dataset was NOT modified.")
print("\nPROGRAM COMPLETED SUCCESSFULLY.")
