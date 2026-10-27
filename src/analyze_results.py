"""Rebuild summary tables and plots from included project result CSVs.

Run from repository root:
    python src/analyze_results.py

This script does not load images, train models, or evaluate test data.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

baseline = pd.read_csv(RESULTS / "final_baseline_test_results.csv")
tuned = pd.read_csv(RESULTS / "final_tuned_test_results.csv")
metrics = ["balanced_accuracy", "precision", "recall", "specificity", "f1"]

merged = baseline[["benchmark", *metrics, "TP", "FP", "FN", "TN"]].merge(
    tuned[["benchmark", *metrics, "TP", "FP", "FN", "TN"]],
    on="benchmark", suffixes=("_baseline", "_tuned"), validate="one_to_one"
)

for metric in metrics:
    merged[f"{metric}_delta"] = merged[f"{metric}_tuned"] - merged[f"{metric}_baseline"]
merged.to_csv(RESULTS / "final_test_baseline_vs_tuned.csv", index=False)

summary = pd.DataFrame([{
    "metric": metric,
    "baseline_macro_mean": merged[f"{metric}_baseline"].mean(),
    "tuned_macro_mean": merged[f"{metric}_tuned"].mean(),
    "change": merged[f"{metric}_delta"].mean(),
} for metric in metrics])
summary.to_csv(RESULTS / "final_test_comparison_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(merged))
w = 0.36
ax.bar([i - w/2 for i in x], merged["balanced_accuracy_baseline"], width=w, label="Baseline")
ax.bar([i + w/2 for i in x], merged["balanced_accuracy_tuned"], width=w, label="Tuned CNN")
ax.set_xticks(list(x), merged["benchmark"])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Test balanced accuracy")
ax.set_title("Baseline vs Tuned CNN (threshold 0.20)")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "final_test_balanced_accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Updated final_test_baseline_vs_tuned.csv")
print("Updated final_test_comparison_summary.csv")
print("Saved final_test_balanced_accuracy_comparison.png")
print("\nMacro summary:")
print(summary.round(4).to_string(index=False))
