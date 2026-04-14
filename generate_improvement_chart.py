"""
Generate a comparison chart showing baseline vs three improvement approaches.
Saves to eval/improvement_chart.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

methods = ["Baseline\n(dense only)", "Re-ranking\nonly", "HyDE +\nRe-ranking", "HyDE\nonly"]
recall  = [0.919, 0.919, 0.919, 0.952]
mrr     = [0.812, 0.828, 0.841, 0.860]

# How many of the 5 original misses each method fixes (net, after new misses)
original_misses_fixed = [0, 1, 1, 2]   # net fixed (fixed - new_misses)

x      = np.arange(len(methods))
width  = 0.35
colors_recall = ["#aaaaaa", "#ff9f40", "#ff9f40", "#2ca02c"]
colors_mrr    = ["#cccccc", "#ffcc90", "#ffcc90", "#98df8a"]

fig, (ax_main, ax_miss) = plt.subplots(1, 2, figsize=(13, 5.5),
                                        gridspec_kw={"width_ratios": [2, 1]})
fig.suptitle("Retrieval improvement: HyDE vs re-ranking vs combined",
             fontsize=13, fontweight="bold")

# ── Left: Recall and MRR bars ─────────────────────────────────────────────────
bars_r = ax_main.bar(x - width/2, recall, width, color=colors_recall,
                     edgecolor="white", linewidth=0.8, label="Recall@3")
bars_m = ax_main.bar(x + width/2, mrr,    width, color=colors_mrr,
                     edgecolor="white", linewidth=0.8, label="MRR@3")

ax_main.set_ylim(0.75, 1.02)
ax_main.set_xticks(x)
ax_main.set_xticklabels(methods, fontsize=10)
ax_main.set_ylabel("Score", fontsize=11)
ax_main.set_title("Recall@3 and MRR@3 across methods", fontsize=11)
ax_main.axhline(y=0.919, color="#aaaaaa", linestyle="--", linewidth=1, alpha=0.8)
ax_main.text(3.6, 0.922, "baseline recall", fontsize=8, color="#888")
ax_main.legend(fontsize=9)
ax_main.grid(axis="y", alpha=0.3)

for bar, val in zip(bars_r, recall):
    ax_main.text(bar.get_x() + bar.get_width()/2, val + 0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold", color="#333")
for bar, val in zip(bars_m, mrr):
    ax_main.text(bar.get_x() + bar.get_width()/2, val + 0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#555")

# ── Right: Net misses fixed ───────────────────────────────────────────────────
miss_colors = ["#dddddd", "#ffcc90", "#ffcc90", "#2ca02c"]
bars_miss = ax_miss.bar(x, original_misses_fixed, color=miss_colors,
                        edgecolor="white", linewidth=0.8, width=0.5)

ax_miss.set_ylim(0, 3)
ax_miss.set_xticks(x)
ax_miss.set_xticklabels(methods, fontsize=10)
ax_miss.set_ylabel("Net misses fixed (of 5)", fontsize=11)
ax_miss.set_title("How many of the 5\nbaseline misses were recovered", fontsize=11)
ax_miss.grid(axis="y", alpha=0.3)
ax_miss.set_yticks([0, 1, 2, 3])

for bar, val in zip(bars_miss, original_misses_fixed):
    label = str(val) if val > 0 else "0"
    ax_miss.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                 label, ha="center", va="bottom", fontsize=12, fontweight="bold")

# Add note about re-ranking trade-off
ax_miss.text(1, 1.35, "introduced\n1 new miss", ha="center", fontsize=8,
             color="#cc6600", style="italic")
ax_miss.text(2, 1.35, "introduced\n2 new misses", ha="center", fontsize=8,
             color="#cc6600", style="italic")

# Winner annotation
ax_miss.annotate("Best overall", xy=(3, 2), xytext=(2.3, 2.6),
                 fontsize=9, color="#2ca02c", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5))

plt.tight_layout()
os.makedirs("eval", exist_ok=True)
plt.savefig("eval/improvement_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved eval/improvement_chart.png")
