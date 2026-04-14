"""
Generate a single progression chart showing the full development journey:
  - every tuning experiment (BGE, BM25, provider detection, corpus fix)
  - every eval set difficulty tier (easy → implicit → hard)
  - retrieval method experiments (re-ranking, HyDE, multi-query)

Saves to eval/progression_chart.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ───────────────────────────────────────────────────────────────────────
stages = [
    "Broken corpus\n(baseline)",
    "BGE\nembedding",
    "BM25\nhybrid",
    "Provider\ndetection",
    "Fixed\ncorpus",
    "Provider detect\n+ fixed corpus",
    "47-question\neval (implicit)",
    "62-question\neval (hard)",
    "Cross-encoder\nre-ranking",
    "HyDE +\nRe-ranking",
    "HyDE\nonly",
    "Multi-query\nreform.",
]

recall = [0.630, 0.630, 0.518, 0.630, 1.000, 1.000, 1.000, 0.919,
          0.919, 0.919, 0.952, 0.903]
mrr    = [0.611, 0.611, 0.340, 0.611, 0.975, 1.000, 0.922, 0.812,
          0.828, 0.841, 0.860, 0.833]

# Faithfulness was measured at stages 0, 6, 7 (index)
faith_x    = [0, 6, 7]
faith_vals = [3.70, 4.13, 4.24]

# Eval set size at each stage
eval_sizes = [27, 27, 27, 27, 27, 27, 47, 62, 62, 62, 62, 62]

# Phase background bands (start_x, end_x, label, color)
phases = [
    (-.5,  3.5, "Algorithm experiments\n(corpus still broken)", "#fff3cd"),
    (3.5,  5.5, "Data fix",                                      "#d4edda"),
    (5.5,  7.5, "Eval hardening",                                "#cce5ff"),
    (7.5, 11.5, "Retrieval method experiments",                  "#f3e5f5"),
]

x = np.arange(len(stages))

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 1, figsize=(18, 9),
    gridspec_kw={"height_ratios": [3, 1.2]},
    sharex=True
)
fig.suptitle(
    "CloudDocs RAG — Full Development Progression",
    fontsize=14, fontweight="bold", y=0.98
)

# ── Top panel: Recall & MRR ────────────────────────────────────────────────────
ax = axes[0]

for x0, x1, label, color in phases:
    ax.axvspan(x0, x1, color=color, alpha=0.55, zorder=0)
    ax.text(
        (x0 + x1) / 2, 0.06, label,
        ha="center", va="bottom", fontsize=7.5,
        color="#555", style="italic"
    )

# Draw lines with per-segment coloring to highlight improvements/regressions
def draw_segmented_line(ax, xdata, ydata, baseline, color_up, color_down, color_flat,
                        linewidth=2.2, zorder=3):
    for i in range(len(xdata) - 1):
        diff = ydata[i + 1] - ydata[i]
        if diff > 0.01:
            c = color_up
        elif diff < -0.01:
            c = color_down
        else:
            c = color_flat
        ax.plot(xdata[i:i+2], ydata[i:i+2], color=c, linewidth=linewidth, zorder=zorder)

draw_segmented_line(ax, x, recall,
                    baseline=0.63,
                    color_up="#2ca02c",
                    color_down="#d62728",
                    color_flat="#aaaaaa")
draw_segmented_line(ax, x, mrr,
                    baseline=0.611,
                    color_up="#1f77b4",
                    color_down="#ff7f0e",
                    color_flat="#aaaaaa")

# Markers
ax.scatter(x, recall, color="#2ca02c", s=80, zorder=5, label="Recall@3")
ax.scatter(x, mrr,    color="#1f77b4", s=80, zorder=5, label="MRR@3",    marker="D")

# Star marker for best result
best_idx = 10  # HyDE only
ax.scatter([best_idx], [recall[best_idx]], color="#ff7f0e", s=200, zorder=6,
           marker="*", label="Best result")

# Value labels — offset alternately to avoid overlap
for i, (r, m) in enumerate(zip(recall, mrr)):
    ax.text(i, r + 0.025, f"{r:.3f}", ha="center", va="bottom",
            fontsize=7.5, color="#2ca02c", fontweight="bold")
    ax.text(i, m - 0.045, f"{m:.3f}", ha="center", va="top",
            fontsize=7.5, color="#1f77b4")

# Annotate key events
annotations = {
    2:  ("BM25 hurts\nranking",   -0.10),
    4:  ("URL fix\n+93 chunks",   +0.04),
    7:  ("First honest\nrecall",  -0.10),
    10: ("Best:\nHyDE only",      +0.04),
    11: ("Honest\nregression",    -0.10),
}
for idx, (note, dy) in annotations.items():
    ax.annotate(
        note,
        xy=(idx, recall[idx]),
        xytext=(idx, recall[idx] + dy + 0.04),
        ha="center", fontsize=7.5, color="#333",
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.2),
    )

ax.set_ylim(0.18, 1.22)
ax.set_ylabel("Score", fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_xlim(-0.5, len(stages) - 0.5)

# ── Bottom panel: Faithfulness + eval set size ─────────────────────────────────
ax2 = axes[1]

for x0, x1, _, color in phases:
    ax2.axvspan(x0, x1, color=color, alpha=0.55, zorder=0)

# Faithfulness (left axis)
ax2.plot(faith_x, faith_vals, "o--", color="#9467bd", linewidth=2,
         markersize=8, label="Faithfulness (1-5)", zorder=4)
for xi, fv in zip(faith_x, faith_vals):
    ax2.text(xi, fv + 0.05, f"{fv:.2f}", ha="center", va="bottom",
             fontsize=8.5, color="#9467bd", fontweight="bold")
ax2.set_ylim(1, 5.6)
ax2.set_ylabel("Faithfulness\n(1-5)", fontsize=10, color="#9467bd")
ax2.tick_params(axis="y", labelcolor="#9467bd")

# Eval set size (right axis)
ax2b = ax2.twinx()
ax2b.bar(x, eval_sizes, color="#bbbbbb", alpha=0.35, width=0.5, zorder=1,
         label="Eval set size")
ax2b.set_ylim(0, 110)
ax2b.set_ylabel("Eval set size\n(# questions)", fontsize=10, color="#888")
ax2b.tick_params(axis="y", labelcolor="#888")
for i, s in enumerate(eval_sizes):
    ax2b.text(i, s + 1.5, str(s), ha="center", va="bottom",
              fontsize=7, color="#666")

ax2.set_xticks(x)
ax2.set_xticklabels(stages, fontsize=7.5)
ax2.grid(axis="y", alpha=0.3)

# Combined legend
faith_patch = mpatches.Patch(color="#9467bd", label="Faithfulness (1-5)")
size_patch  = mpatches.Patch(color="#bbbbbb", alpha=0.6, label="Eval set size")
ax2.legend(handles=[faith_patch, size_patch], loc="lower right", fontsize=8)

plt.tight_layout()
os.makedirs("eval", exist_ok=True)
plt.savefig("eval/progression_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved eval/progression_chart.png")
