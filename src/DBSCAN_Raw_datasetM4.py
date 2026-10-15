"""Cluster GlucoBench readings with DBSCAN after numeric scaling."""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from glucotrend_m4_utils import M4_OUTPUT, load_scaled_features

_, features, scaled = load_scaled_features(sample_size=2500)
labels = DBSCAN(eps=0.9, min_samples=12, n_jobs=-1).fit_predict(scaled)
result = features.reset_index(drop=True).copy()
result["cluster"] = labels
result.to_csv(os.path.join(M4_OUTPUT, "dbscan_clusters.csv"), index=False)
cluster_ids = set(labels) - {-1}
silhouette = silhouette_score(scaled, labels, sample_size=min(1000, len(labels)), random_state=42) if len(cluster_ids) > 1 else float("nan")
pd.DataFrame([{"sample_records": len(result), "clusters": len(cluster_ids),
               "noise_records": int((labels == -1).sum()), "silhouette": silhouette}]).to_csv(
    os.path.join(M4_OUTPUT, "dbscan_metrics.csv"), index=False)
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(result.glucose, result.heart_rate, c=labels, cmap="tab10", s=10, alpha=.65)
ax.set(title="DBSCAN clusters (sampled GlucoBench readings)", xlabel="Glucose (mg/dL)", ylabel="Heart rate (bpm)")
fig.colorbar(scatter, ax=ax, label="Cluster (-1 = noise)")
fig.tight_layout(); fig.savefig(os.path.join(M4_OUTPUT, "dbscan_clusters.png"), dpi=180); plt.close(fig)
print(f"DBSCAN complete: {len(cluster_ids)} clusters, {int((labels == -1).sum())} noise readings; results: {M4_OUTPUT}")
