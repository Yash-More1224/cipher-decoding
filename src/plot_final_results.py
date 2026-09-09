"""
plot_final_results.py
=====================
Regenerates the final results figures for the C1-C5 ablation from the
*windowed BPE* round, which is the round reported in the paper.

Inputs (all local, no GPU needed):
    outputs/results_windowed/final_eval_C{1..4}.json   windowed BPE round
    outputs/results_c5/final_eval_C5.json              C5 (BLT), stitched
    outputs/results_fullline/final_eval_C{1..4}.json   superseded round, for
                                                       the before/after figure

Validation-loss curves are pulled from W&B (run-C{1..4}-bpew). If W&B is
unreachable that single figure is skipped and the rest still render.

Run from project root:
    python src/plot_final_results.py
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs"
PLOT_DIR = "outputs"
PROJECT_PATH = "yash-more1224-international-institute-of-information-tec/anlp-assignment-1"

CONFIGS = ["C1", "C2", "C3", "C4", "C5"]
LABELS = {
    "C1": "C1 (Base)",
    "C2": "C2 (RoPE)",
    "C3": "C3 (GQA)",
    "C4": "C4 (RMSNorm)",
    "C5": "C5 (BLT)",
}
COLORS = {
    "C1": "#4C72B0",
    "C2": "#DD8452",
    "C3": "#55A868",
    "C4": "#C44E52",
    "C5": "#8172B2",
}
WANDB_RUN = {c: f"run-{c}-bpew-long" for c in ["C1", "C2", "C3", "C4"]}

STYLE = {
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 110,
}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_round():
    """Windowed round for C1-C4, plus C5 from its own stitched evaluation."""
    r = {}
    for c in ["C1", "C2", "C3", "C4"]:
        r[c] = load(f"{OUTPUT_DIR}/results_windowed/final_eval_{c}.json")
    r["C5"] = load(f"{OUTPUT_DIR}/results_c5/final_eval_C5.json")
    return r


def annotate(ax, bars, vals, fmt, pad):
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + pad,
                fmt.format(val), ha="center", va="bottom",
                fontsize=9, fontweight="bold")


def fig_accuracy(res):
    """Character and sequence accuracy.

    Character accuracy is position-wise (see evaluate_final.char_accuracy), so
    a single inserted or deleted character shifts every later position and
    depresses the score even when the text is nearly right. C2 is the clear
    case: 37.9% here against a Levenshtein distance of only 10.4. The caption
    says so, because the bar alone is misleading.
    """
    keys = [c for c in CONFIGS if res.get(c)]
    char = [res[c]["char_accuracy"] * 100 for c in keys]
    seq = [res[c]["seq_accuracy"] * 100 for c in keys]

    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.2))
    b1 = ax.bar(x - w / 2, char, w, label="Character accuracy (position-wise)",
                color=[COLORS[c] for c in keys], edgecolor="white", linewidth=1.2)
    b2 = ax.bar(x + w / 2, seq, w, label="Sequence accuracy (exact line match)",
                color=[COLORS[c] for c in keys], edgecolor="white",
                linewidth=1.2, alpha=0.45, hatch="//")
    annotate(ax, b1, char, "{:.1f}%", 1.2)
    annotate(ax, b2, seq, "{:.1f}%", 1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in keys])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Decryption Accuracy per Configuration (500 held-out test lines)",
                 fontweight="bold")
    ax.legend(loc="upper left")
    fig.text(0.5, 0.008,
             "Position-wise character accuracy penalises length drift: C2 scores 37.9% "
             "despite a Levenshtein distance of only 10.4.",
             ha="center", fontsize=8, style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = f"{PLOT_DIR}/accuracy_comparison.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Plot] {out}")


def fig_levenshtein(res):
    """Log scale: the values span 0.64 (C5) to 140 (C3), over two decades."""
    keys = [c for c in CONFIGS if res.get(c)]
    vals = [res[c]["levenshtein"] for c in keys]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar([LABELS[c] for c in keys], vals,
                  color=[COLORS[c] for c in keys], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.12,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("Mean Levenshtein distance (lower is better, log scale)")
    ax.set_ylim(0.3, max(vals) * 3)
    ax.set_title("Edit Distance to Ground-Truth Plaintext", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = f"{PLOT_DIR}/levenshtein_comparison.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Plot] {out}")


def fig_bleu_rouge(res):
    keys = [c for c in CONFIGS if res.get(c) and "bleu" in res[c]]
    metrics = [("bleu", "BLEU"), ("rouge1", "ROUGE-1"),
               ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L")]

    x = np.arange(len(metrics))
    w = 0.8 / max(len(keys), 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i, c in enumerate(keys):
        vals = [res[c][m] for m, _ in metrics]
        off = (i - (len(keys) - 1) / 2) * w
        bars = ax.bar(x + off, vals, w, label=LABELS[c],
                      color=COLORS[c], edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title("N-gram Overlap with Ground-Truth Plaintext", fontweight="bold")
    ax.legend(ncol=2, loc="upper left")
    missing = [c for c in CONFIGS if res.get(c) and "bleu" not in res[c]]
    if missing:
        fig.text(0.5, 0.008,
                 f"{', '.join(missing)} omitted: n-gram metrics were not computed in "
                 f"that evaluation run.", ha="center", fontsize=8, style="italic")
        plt.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        plt.tight_layout()
    out = f"{PLOT_DIR}/bleu_rouge_comparison.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Plot] {out}")


def fig_windowing_effect():
    """The single most important methodological result.

    Same architectures, same tokenizer, same budget: the only change is
    training on 64-character windows instead of whole lines. Every config
    improves by roughly an order of magnitude, which is why the windowed
    round is the one reported.
    """
    keys = ["C1", "C2", "C3", "C4"]
    full, win = [], []
    for c in keys:
        a = load(f"{OUTPUT_DIR}/results_fullline/final_eval_{c}.json")
        b = load(f"{OUTPUT_DIR}/results_windowed/final_eval_{c}.json")
        if not (a and b):
            print("[Plot] skipping windowing-effect figure (missing inputs)")
            return
        full.append(a["levenshtein"])
        win.append(b["levenshtein"])

    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.2))
    b1 = ax.bar(x - w / 2, full, w, label="Full-line training (superseded)",
                color="#B0B0B0", edgecolor="white", linewidth=1.2)
    b2 = ax.bar(x + w / 2, win, w, label="64-character windowed training (reported)",
                color=[COLORS[c] for c in keys], edgecolor="white", linewidth=1.2)
    for bar, val in list(zip(b1, full)) + list(zip(b2, win)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in keys])
    ax.set_ylabel("Mean Levenshtein distance (lower is better, log scale)")
    ax.set_ylim(5, max(full) * 4)
    ax.set_title("Effect of Windowed Training on Decryption Quality", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = f"{PLOT_DIR}/windowing_effect.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Plot] {out}")


def fig_val_loss():
    try:
        import wandb
        api = wandb.Api(timeout=60)
        runs = {r.name: r for r in api.runs(PROJECT_PATH)}
    except Exception as exc:
        print(f"[Plot] skipping val-loss curves (W&B unavailable: {exc})")
        return

    fig, ax = plt.subplots(figsize=(9, 5.2))
    plotted = 0
    endpoints = []
    for c, name in WANDB_RUN.items():
        if name not in runs:
            continue
        h = runs[name].history(samples=5000, pandas=True)
        # train.py logs under namespaced keys ("val/val_loss"); older runs used
        # a bare "val_loss". Accept whichever this run actually has.
        col = next((k for k in ("val/val_loss", "val_loss") if k in h.columns), None)
        if col is None or "epoch" not in h.columns:
            continue
        # val loss is logged once per epoch, on a different step than the
        # per-batch train loss, so drop on these two columns only.
        d = h[["epoch", col]].dropna(subset=["epoch", col])
        if d.empty:
            continue
        d = d.groupby("epoch", as_index=False)[col].last().sort_values("epoch")
        ax.plot(d["epoch"], d[col], label=LABELS[c],
                color=COLORS[c], linewidth=2)
        endpoints.append((float(d["epoch"].iloc[-1]),
                          float(d[col].iloc[-1]), COLORS[c]))
        plotted += 1

    if not plotted:
        plt.close()
        print("[Plot] skipping val-loss curves (no history found)")
        return

    # C1, C3 and C4 converge to within 0.10 of each other, so their end-of-run
    # labels would overlap and become unreadable. Nudge them apart vertically,
    # keeping each label's colour so it stays attributable to its curve.
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    min_gap = span * 0.055
    endpoints.sort(key=lambda t: t[1])
    placed = []
    for x_end, y_end, colour in endpoints:
        y_lab = y_end
        if placed and y_lab - placed[-1] < min_gap:
            y_lab = placed[-1] + min_gap
        placed.append(y_lab)
        ax.annotate(f"{y_end:.3f}", xy=(x_end, y_end), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, color=colour,
                    fontweight="bold", va="center",
                    xycoords="data", annotation_clip=False)
        if abs(y_lab - y_end) > 1e-9:
            # Label was moved; redraw it at the shifted height instead.
            ax.texts[-1].remove()
            ax.annotate(f"{y_end:.3f}", xy=(x_end, y_lab), xytext=(6, 0),
                        textcoords="offset points", fontsize=8, color=colour,
                        fontweight="bold", va="center", annotation_clip=False)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss (cross-entropy per token)")
    ax.set_title("Validation Loss, Windowed BPE Round", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.06)
    plt.tight_layout()
    out = f"{PLOT_DIR}/val_loss_vs_epochs_final.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Plot] {out}")


def main():
    plt.rcParams.update(STYLE)
    os.makedirs(PLOT_DIR, exist_ok=True)
    res = load_round()

    missing = [c for c in CONFIGS if not res.get(c)]
    if missing:
        print(f"[WARN] no results found for: {missing}")

    print("\n=== Reported round (windowed BPE for C1-C4, stitched BLT for C5) ===")
    hdr = f"{'cfg':4s} {'char_acc':>9s} {'seq_acc':>8s} {'lev':>8s} {'bleu':>7s} {'rouge1':>7s}"
    print(hdr)
    for c in CONFIGS:
        r = res.get(c)
        if not r:
            continue
        b = f"{r['bleu']:7.4f}" if "bleu" in r else f"{'--':>7s}"
        g = f"{r['rouge1']:7.4f}" if "rouge1" in r else f"{'--':>7s}"
        print(f"{c:4s} {r['char_accuracy']:9.4f} {r['seq_accuracy']:8.4f} "
              f"{r['levenshtein']:8.3f} {b} {g}")
    print()

    fig_accuracy(res)
    fig_levenshtein(res)
    fig_bleu_rouge(res)
    fig_windowing_effect()
    fig_val_loss()


if __name__ == "__main__":
    main()
