"""
Generate a two-panel clarity chart:
  Left  — heatmap: recall rate per category × question difficulty tier
  Right — horizontal bar: faithfulness per category, sorted worst→best

Saves to eval/category_chart.png
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── Load per-question results ──────────────────────────────────────────────────
with open("eval_results.json") as f:
    data = json.load(f)

results = data["results"]

# ── Tag each question with its difficulty tier ─────────────────────────────────
def get_tier(qid: str) -> str:
    if qid.startswith("hard-"):
        return "Hard"
    if qid.startswith("implicit-"):
        return "Implicit"
    return "Explicit"

TIERS = ["Explicit", "Implicit", "Hard"]
CATEGORIES = ["compute", "storage", "database", "networking", "security", "cross-provider"]
CAT_LABELS  = ["Compute", "Storage", "Database", "Networking", "Security", "Cross-provider"]

# ── Build recall matrix [category × tier] ─────────────────────────────────────
recall_matrix = np.full((len(CATEGORIES), len(TIERS)), np.nan)
count_matrix  = np.zeros((len(CATEGORIES), len(TIERS)), dtype=int)   # denominator

for r in results:
    tier = get_tier(r["id"])
    cat  = r["category"]
    if cat not in CATEGORIES or tier not in TIERS:
        continue
    ci = CATEGORIES.index(cat)
    ti = TIERS.index(tier)
    if np.isnan(recall_matrix[ci, ti]):
        recall_matrix[ci, ti] = 0.0
    recall_matrix[ci, ti] = (recall_matrix[ci, ti] * count_matrix[ci, ti] + r["hit"]) / (count_matrix[ci, ti] + 1)
    count_matrix[ci, ti] += 1

# ── Faithfulness per category (from summary) ─────────────────────────────────
faith_raw = data["per_category"]
faith_vals = []
for cat in CATEGORIES:
    v = faith_raw.get(cat, {}).get("avg_faithfulness")
    faith_vals.append(v if v is not None else float("nan"))

# Sort faithfulness low→high for bar chart readability
faith_order = sorted(range(len(CATEGORIES)), key=lambda i: faith_vals[i] if not np.isnan(faith_vals[i]) else -1)
faith_cats  = [CAT_LABELS[i] for i in faith_order]
faith_sorted = [faith_vals[i] for i in faith_order]
faith_colors = [
    "#2ca02c" if v >= 4.3 else "#ff7f0e" if v >= 3.9 else "#d62728"
    for v in faith_sorted
]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(13, 5.5),
                                       gridspec_kw={"width_ratios": [1.3, 1]})
fig.suptitle("Where does the system work — and where does it break?",
             fontsize=13, fontweight="bold", y=1.01)

# ── Left: Heatmap ─────────────────────────────────────────────────────────────
cmap = mcolors.LinearSegmentedColormap.from_list(
    "rg", ["#d62728", "#ffdd57", "#2ca02c"]
)
im = ax_heat.imshow(recall_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

ax_heat.set_xticks(range(len(TIERS)))
ax_heat.set_xticklabels(TIERS, fontsize=11)
ax_heat.set_yticks(range(len(CATEGORIES)))
ax_heat.set_yticklabels(CAT_LABELS, fontsize=11)
ax_heat.set_xlabel("Question difficulty →   harder →", fontsize=10, labelpad=8)
ax_heat.set_title("Recall@3 by category and difficulty", fontsize=11, pad=10)

# Cell annotations
for ci in range(len(CATEGORIES)):
    for ti in range(len(TIERS)):
        v = recall_matrix[ci, ti]
        n = count_matrix[ci, ti]
        if np.isnan(v) or n == 0:
            continue
        hits = round(v * n)
        cell_text = f"{v:.0%}\n({hits}/{n})"
        text_color = "white" if v < 0.55 else "black"
        ax_heat.text(ti, ci, cell_text, ha="center", va="center",
                     fontsize=9, color=text_color, fontweight="bold")

# Column header shading bands
for ti, (tier, alpha) in enumerate(zip(TIERS, [0.0, 0.0, 0.0])):
    pass  # color already in imshow

# Colorbar
cbar = fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.03)
cbar.set_label("Recall rate", fontsize=9)
cbar.set_ticks([0, 0.5, 1.0])
cbar.set_ticklabels(["0%", "50%", "100%"])

# ── Right: Faithfulness bar ───────────────────────────────────────────────────
bars = ax_bar.barh(faith_cats, faith_sorted, color=faith_colors,
                   height=0.55, edgecolor="white", linewidth=0.5)

ax_bar.set_xlim(1, 5.4)
ax_bar.set_xlabel("Faithfulness score (1–5)", fontsize=10)
ax_bar.set_title("Answer faithfulness by category\n(LLM-as-judge, higher = more grounded)", fontsize=11, pad=10)
ax_bar.axvline(x=4.0, color="#888", linestyle="--", linewidth=1, alpha=0.7)
ax_bar.text(4.02, -0.6, "4.0 threshold", fontsize=8, color="#888", va="bottom")

for bar, val in zip(bars, faith_sorted):
    ax_bar.text(val + 0.04, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=10, fontweight="bold")

# Legend for bar colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(color="#2ca02c", label="≥ 4.3  Well grounded"),
    Patch(color="#ff7f0e", label="3.9–4.2  Mostly grounded"),
    Patch(color="#d62728", label="< 3.9  Occasional drift"),
]
ax_bar.legend(handles=legend_elements, fontsize=8.5, loc="lower right",
              framealpha=0.85)
ax_bar.grid(axis="x", alpha=0.3)

plt.tight_layout()
os.makedirs("eval", exist_ok=True)
plt.savefig("eval/category_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved eval/category_chart.png")
