"""
plot_combined.py
================
Builds the two space-saving combined figures used by report-shortened.tex:

  outputs/exploration_combined.png   the three exploration sweeps as one
                                     3-panel figure instead of three figures
  outputs/quality_combined.png       edit distance and n-gram overlap side by
                                     side instead of two separate figures

Run from project root:  python src/plot_combined.py
"""

import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plot_exploration import (GROUPS, PALETTE, PROJECT_PATH,
                              find_train_loss_col, get_x)
from plot_final_results import CONFIGS, COLORS, LABELS, load_round

warnings.filterwarnings("ignore")
OUT = "outputs"

STYLE = {
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 6.6,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
}

PANEL_TITLES = [
    "(a) Capacity sweep (BPE-on-raw-bits era)",
    "(b) Input sequence-length experiments",
    "(c) Learning-rate / epoch-budget sweep",
]


def exploration_combined():
    try:
        import wandb
        api = wandb.Api(timeout=60)
        runs = {r.name: r for r in api.runs(PROJECT_PATH)}
    except Exception as exc:
        print(f"[skip] exploration_combined (W&B unavailable: {exc})")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1))
    for ax, group, title in zip(axes, GROUPS, PANEL_TITLES):
        drawn = 0
        for i, (name, label, ls) in enumerate(group["runs"]):
            if name not in runs:
                continue
            df = runs[name].history(samples=2000, pandas=True)
            col = find_train_loss_col(df)
            if col is None:
                continue
            # get_x returns a Series (step, back-filled from _step), not a
            # column name, so build the frame explicitly.
            import pandas as pd
            d = pd.DataFrame({"x": get_x(df), "y": df[col]}).dropna()
            if d.empty:
                continue
            d = d.sort_values("x")
            ax.plot(d["x"], d["y"], label=label, color=PALETTE[i % len(PALETTE)],
                    linestyle=ls, linewidth=1.6)
            drawn += 1
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        if drawn:
            ax.legend(loc="upper right", framealpha=0.9)
    axes[0].set_ylabel("Training loss")
    plt.tight_layout()
    p = f"{OUT}/exploration_combined.png"
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"[Plot] {p}")


def quality_combined():
    res = load_round()
    keys = [c for c in CONFIGS if res.get(c)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    # (a) edit distance, log scale: values span 0.64 to 140.
    vals = [res[c]["levenshtein"] for c in keys]
    bars = ax1.bar([LABELS[c] for c in keys], vals,
                   color=[COLORS[c] for c in keys], edgecolor="white", linewidth=1.1)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.13, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax1.set_yscale("log")
    ax1.set_ylim(0.3, max(vals) * 3.2)
    ax1.set_ylabel("Mean Levenshtein distance (log scale)")
    ax1.set_title("(a) Edit distance to reference (lower is better)", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, which="both")
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="x", labelrotation=12)

    # (b) grouped n-gram metrics.
    metrics = [("bleu", "BLEU"), ("rouge1", "ROUGE-1"),
               ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L")]
    ng = [c for c in keys if "bleu" in res[c]]
    x = np.arange(len(metrics))
    w = 0.8 / max(len(ng), 1)
    for i, c in enumerate(ng):
        v = [res[c][m] for m, _ in metrics]
        ax2.bar(x + (i - (len(ng) - 1) / 2) * w, v, w, label=LABELS[c],
                color=COLORS[c], edgecolor="white", linewidth=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([n for _, n in metrics])
    ax2.set_ylim(0, 1.16)
    ax2.set_ylabel("Score")
    ax2.set_title("(b) N-gram overlap with reference (higher is better)", fontweight="bold")
    ax2.legend(ncol=3, loc="upper center", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    p = f"{OUT}/quality_combined.png"
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"[Plot] {p}")


if __name__ == "__main__":
    plt.rcParams.update(STYLE)
    os.makedirs(OUT, exist_ok=True)
    quality_combined()
    exploration_combined()
