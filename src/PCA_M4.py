"""Principal component analysis of GlucoBench numeric features."""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from glucotrend_m4_utils import M4_OUTPUT, load_scaled_features

_, features, scaled = load_scaled_features()
pca = PCA(random_state=42).fit(scaled)
coordinates = pca.transform(scaled)
result = pd.DataFrame(coordinates, columns=[f"PC{i + 1}" for i in range(coordinates.shape[1])])
result.to_csv(os.path.join(M4_OUTPUT, "pca_coordinates.csv"), index=False)
variance = pd.DataFrame({"component": range(1, len(pca.explained_variance_ratio_) + 1),
                         "explained_variance_ratio": pca.explained_variance_ratio_,
                         "cumulative_variance": pca.explained_variance_ratio_.cumsum()})
variance.to_csv(os.path.join(M4_OUTPUT, "pca_explained_variance.csv"), index=False)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(variance.component, variance.cumulative_variance, marker="o", color="#0f9488")
ax.axhline(.95, color="#d97a1f", linestyle="--", label="95% variance")
ax.set(xlabel="Principal components", ylabel="Cumulative explained variance", ylim=(0, 1.02), title="GlucoBench PCA")
ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(M4_OUTPUT, "pca_cumulative_variance.png"), dpi=180); plt.close(fig)
print(f"PCA complete: first two components explain {variance.cumulative_variance.iloc[min(1, len(variance)-1)]:.1%}; results: {M4_OUTPUT}")
