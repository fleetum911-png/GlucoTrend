"""Create a visual snapshot of the GlucoTrend M6 serving lifecycle."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "ML_Engineering_M6_Outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

stages = [
    ("DATA", "Raw CGM readings", "dataset/GlucoBench…RAW.csv", "dataset/GlucoBench_benchmark_dataset_RAW.csv"),
    ("FEATURES", "Prepared features", "M2 targets + preprocessing", "dataset/target_engineered_M2.csv"),
    ("PACKAGE", "Trained model", "Decision Tree · pickle", "outputs/Decision_Tree_Classifier_M3_Outputs/model/decision_tree_model.pkl"),
    ("SERVE", "Prediction endpoint", "Flask · /prediction", "app.py"),
    ("OBSERVE", "Evaluation view", "Dashboard · model scores", "outputs/Decision_Tree_Classifier_M3_Outputs/metrics/decision_tree_metrics.csv"),
]

fig, ax = plt.subplots(figsize=(15, 6.6))
fig.patch.set_facecolor("#101c20")
ax.set_facecolor("#101c20")
ax.set_xlim(0, 15)
ax.set_ylim(0, 6.6)
ax.axis("off")
ax.text(0.4, 6.05, "M6 · ML ENGINEERING", color="#75d5bc", fontsize=11, weight="bold", family="DejaVu Sans")
ax.text(0.4, 5.47, "From dataset to a served prediction", color="#e7f2ef", fontsize=23, weight="bold", family="DejaVu Sans")
ax.text(0.4, 5.05, "A snapshot of the GlucoTrend app's current training and serving path", color="#aabbb7", fontsize=11)

box_width, box_height, gap, start_x, y = 2.45, 2.05, 0.48, 0.4, 2.35
for index, (label, title, detail, relative_path) in enumerate(stages):
    x = start_x + index * (box_width + gap)
    available = os.path.exists(os.path.join(BASE_DIR, relative_path))
    edge = "#438d7c" if available else "#806e56"
    face = "#17272c" if available else "#29251f"
    box = FancyBboxPatch((x, y), box_width, box_height,
                         boxstyle="round,pad=0.06,rounding_size=0.12",
                         linewidth=1.4, edgecolor=edge, facecolor=face)
    ax.add_patch(box)
    ax.text(x + 0.18, y + 1.72, label, color="#75d5bc" if available else "#e6bd82", fontsize=9, weight="bold")
    ax.text(x + 0.18, y + 1.25, title, color="#e7f2ef", fontsize=12, weight="bold", wrap=True)
    ax.text(x + 0.18, y + 0.72, detail, color="#b7c6c2", fontsize=9, wrap=True)
    ax.text(x + 0.18, y + 0.25, "AVAILABLE" if available else "NOT FOUND", color="#75d5bc" if available else "#e6bd82", fontsize=8, weight="bold")
    if index < len(stages) - 1:
        arrow = FancyArrowPatch((x + box_width + 0.03, y + box_height / 2),
                                (x + box_width + gap - 0.05, y + box_height / 2),
                                arrowstyle="-|>", mutation_scale=13, linewidth=1.4, color="#64837d")
        ax.add_patch(arrow)

ax.text(0.4, 1.35, "APP ROUTES", color="#75d5bc", fontsize=9, weight="bold")
ax.text(0.4, 0.95, "/prediction  ·  /dashboard  ·  /visualization", color="#e7f2ef", fontsize=12)
ax.text(0.4, 0.38, "Local educational service. Automated drift monitoring and retraining are not connected in this project.", color="#91a5a0", fontsize=9)
fig.savefig(os.path.join(OUTPUT_DIR, "ml_engineering_lifecycle_snapshot.png"), dpi=180,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"M6 lifecycle snapshot saved to {OUTPUT_DIR}")
