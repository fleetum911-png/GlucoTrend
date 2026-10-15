"""Create a sampled t-SNE embedding of GlucoBench numeric features."""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from glucotrend_m4_utils import M4_OUTPUT, load_scaled_features

_, features, scaled = load_scaled_features(sample_size=2500)
perplexity = min(30, max(5, (len(features) - 1) // 3))
embedding = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42).fit_transform(scaled)
result = pd.DataFrame({"tsne_1": embedding[:, 0], "tsne_2": embedding[:, 1]})
result = pd.concat([result, features.reset_index(drop=True)], axis=1)
result.to_csv(os.path.join(M4_OUTPUT, "tsne_embedding.csv"), index=False)
fig, ax = plt.subplots(figsize=(8, 6))
points = ax.scatter(result.tsne_1, result.tsne_2, c=result.glucose, cmap="viridis", s=10, alpha=.7)
fig.colorbar(points, ax=ax, label="Glucose (mg/dL)")
ax.set(title="GlucoBench t-SNE (sampled)", xlabel="t-SNE 1", ylabel="t-SNE 2")
fig.tight_layout(); fig.savefig(os.path.join(M4_OUTPUT, "tsne_glucose_embedding.png"), dpi=180); plt.close(fig)
print(f"t-SNE complete on {len(result):,} sampled readings; results: {M4_OUTPUT}")
