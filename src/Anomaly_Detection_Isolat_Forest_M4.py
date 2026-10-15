"""Detect unusual GlucoBench readings with Isolation Forest."""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from glucotrend_m4_utils import M4_OUTPUT, load_scaled_features

_, features, scaled = load_scaled_features()
model = IsolationForest(contamination=0.03, random_state=42, n_jobs=-1)
labels = model.fit_predict(scaled)
scores = model.decision_function(scaled)
result = features.copy()
result["anomaly"] = labels == -1
result["anomaly_score"] = scores
result.to_csv(os.path.join(M4_OUTPUT, "isolation_forest_results.csv"), index=False)
summary = pd.DataFrame([{"records": len(result), "anomalies": int((labels == -1).sum()),
                         "anomaly_rate_pct": round(float((labels == -1).mean() * 100), 2)}])
summary.to_csv(os.path.join(M4_OUTPUT, "isolation_forest_summary.csv"), index=False)
fig, ax = plt.subplots(figsize=(7, 4))
result["anomaly"].value_counts().reindex([False, True], fill_value=0).plot.bar(ax=ax, color=["#0f9488", "#dc4b4b"])
ax.set(title="GlucoBench anomaly detection", xlabel="Anomaly", ylabel="Readings")
fig.tight_layout(); fig.savefig(os.path.join(M4_OUTPUT, "isolation_forest_counts.png"), dpi=180); plt.close(fig)
print(f"Isolation Forest complete: {int((labels == -1).sum())} anomalies; results: {M4_OUTPUT}")
