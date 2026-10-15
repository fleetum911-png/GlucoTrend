"""Agglomerative hierarchical clustering on a GlucoBench sample."""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans
from sklearn.metrics import silhouette_score
from glucotrend_m4_utils import M4_OUTPUT, load_scaled_features

_, features, scaled = load_scaled_features(sample_size=2500)
model = AgglomerativeClustering(n_clusters=4, linkage="ward")
labels = model.fit_predict(scaled)
result = features.reset_index(drop=True).copy()
result["cluster"] = labels
result.to_csv(os.path.join(M4_OUTPUT, "hierarchical_clusters.csv"), index=False)
score = silhouette_score(scaled, labels)
pd.DataFrame([{"sample_records": len(result), "clusters": int(pd.Series(labels).nunique()), "silhouette": score}]).to_csv(
    os.path.join(M4_OUTPUT, "hierarchical_metrics.csv"), index=False)
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(result.glucose, result.heart_rate, c=labels, cmap="viridis", s=12, alpha=.65)
ax.set(title="Agglomerative clusters (sampled GlucoBench readings)", xlabel="Glucose (mg/dL)", ylabel="Heart rate (bpm)")
fig.colorbar(scatter, ax=ax, label="Cluster")
fig.tight_layout(); fig.savefig(os.path.join(M4_OUTPUT, "hierarchical_clusters.png"), dpi=180); plt.close(fig)
print(f"Hierarchical clustering complete: silhouette {score:.3f}; results: {M4_OUTPUT}")

# Divisive hierarchical approximation: recursively split clusters with bisecting k-means.
divisive = BisectingKMeans(n_clusters=4, random_state=42, n_init=5).fit_predict(scaled)
divisive_result = features.reset_index(drop=True).copy()
divisive_result["cluster"] = divisive
divisive_result.to_csv(os.path.join(M4_OUTPUT, "divisive_clusters.csv"), index=False)
divisive_score = silhouette_score(scaled, divisive)
pd.DataFrame([{"sample_records": len(divisive_result), "clusters": int(pd.Series(divisive).nunique()),
               "silhouette": divisive_score}]).to_csv(os.path.join(M4_OUTPUT, "divisive_metrics.csv"), index=False)
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(divisive_result.glucose, divisive_result.heart_rate, c=divisive, cmap="viridis", s=12, alpha=.65)
ax.set(title="Divisive clusters (sampled GlucoBench readings)", xlabel="Glucose (mg/dL)", ylabel="Heart rate (bpm)")
fig.colorbar(scatter, ax=ax, label="Cluster")
fig.tight_layout(); fig.savefig(os.path.join(M4_OUTPUT, "divisive_clusters.png"), dpi=180); plt.close(fig)
print(f"Divisive clustering complete: silhouette {divisive_score:.3f}; results: {M4_OUTPUT}")
