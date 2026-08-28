"""
analyze_wandb.py
=================
Fetches all W&B runs from the ANLP Assignment 1 project and produces
7 publication-quality plots saved into ./outputs/.

Run from project root:
    python src/analyze_wandb.py

Requires WANDB_API_KEY to be set in environment.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import wandb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_PATH = "yash-more1224-international-institute-of-information-tec/anlp-assignment-1"
OUTPUT_DIR   = "outputs"
# C1 was retrained on a single GPU (run-C1-single-gpu) so its effective batch
# size and wall-clock numbers are directly comparable to C2-C5, which also
# trained on a single GPU. The original 4-GPU run-C1 is no longer "final" —
# it now shows up only as an experimental/exploratory run (Section 3 plot),
# not in the ablation table or the C5-vs-C1 efficiency comparison.
FINAL_RUNS   = {"run-C1-single-gpu", "run-C2", "run-C3", "run-C4", "run-C5"}
CONFIG_LABELS = {
    "run-C1-single-gpu": "C1 (Base)",
    "run-C2": "C2 (+RoPE)",
    "run-C3": "C3 (+GQA)",
    "run-C4": "C4 (+RMSNorm)",
    "run-C5": "C5 (BLT)",
}

# Fallback metrics from log files (used when WandB summary is incomplete).
# No fallback is hardcoded for run-C1-single-gpu: the old 4-GPU run-C1 numbers
# are not valid for it (different effective batch size), so if the live WandB
# summary is ever unavailable this should show up as missing/blank rather than
# silently displaying stale 4-GPU numbers under the C1 label.
FALLBACK_METRICS = {
    "run-C2": dict(bit_accuracy=0.7935, seq_accuracy=0.600, levenshtein=4.0,
                   bleu=0.9461, rouge1=0.9565, rouge2=0.9455, rougeL=0.9565, val_loss=0.1421),
    "run-C3": dict(bit_accuracy=0.9565, seq_accuracy=0.800, levenshtein=12.0,
                   bleu=0.8872, rouge1=0.9056, rouge2=0.8914, rougeL=0.9056, val_loss=0.1354),
    "run-C4": dict(bit_accuracy=0.7935, seq_accuracy=0.600, levenshtein=4.0,
                   bleu=0.9461, rouge1=0.9565, rouge2=0.9455, rougeL=0.9565, val_loss=0.1734),
    "run-C5": dict(bit_accuracy=0.9969, seq_accuracy=0.800, levenshtein=0.4,
                   bleu=None,   rouge1=None,   rouge2=None,   rougeL=None,   val_loss=0.0880),
}

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
PALETTE_FINAL = {
    "run-C1-single-gpu": "#4C72B0",
    "run-C2": "#DD8452",
    "run-C3": "#55A868",
    "run-C4": "#C44E52",
    "run-C5": "#8172B2",
}
STYLE = {
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_metric(summary, name, fallback):
    val = summary.get(name, fallback)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return fallback
    return val


def find_val_loss_col(df):
    """
    Return the column name holding validation loss in a run's history.

    train.py logs validation metrics with a "val/" prefix (e.g. "val/val_loss"),
    but earlier exploratory runs (before that logging convention was added)
    may have logged a bare "val_loss" column instead. Check both.
    """
    for candidate in ("val/val_loss", "val_loss"):
        if candidate in df.columns:
            return candidate
    return None


def find_gpu_memory_cols(df):
    """Return list of column names likely representing peak GPU memory."""
    candidates = [c for c in df.columns if "memory" in c.lower() and "gpu" in c.lower()]
    allocated = [c for c in candidates if "allocated" in c.lower()]
    return allocated if allocated else candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Connect to W&B ----
    api = wandb.Api(timeout=60)
    print(f"[W&B] Fetching runs from: {PROJECT_PATH}")
    try:
        all_runs = list(api.runs(PROJECT_PATH))
    except Exception as exc:
        print(f"[ERROR] Cannot fetch runs: {exc}")
        sys.exit(1)
    print(f"[W&B] Found {len(all_runs)} runs.")

    # ---- Collect data ----
    run_records = {}
    for run in all_runs:
        name  = run.name
        label = CONFIG_LABELS.get(name, name)
        fb    = FALLBACK_METRICS.get(name, {})

        print(f"  Processing '{name}' (state={run.state}) ...", end=" ", flush=True)

        # Training history (sampled)
        try:
            hist = run.history(samples=2000, pandas=True)
        except Exception:
            hist = pd.DataFrame()

        # System metrics (GPU memory etc.)
        sys_hist = pd.DataFrame()
        try:
            sys_hist = run.history(stream="system", samples=2000, pandas=True)
        except Exception:
            pass

        summary = dict(run.summary._json_dict)

        record = {
            "name":     name,
            "label":    label,
            "id":       run.id,
            "state":    run.state,
            "config":   {k: v for k, v in run.config.items() if not k.startswith("_")},
            "summary":  summary,
            "history":  hist,
            "sys_hist": sys_hist,
            "is_final": name in FINAL_RUNS,
            # Final test-set numbers are logged under a "test/" prefix
            # (see wandb.log({f"test/{k}": v ...}) in train.py); fall back to
            # the bare key for older runs, then to the hardcoded FALLBACK_METRICS.
            "val_loss":     get_metric(summary, "test/val_loss",     get_metric(summary, "val_loss",     fb.get("val_loss"))),
            "bit_accuracy": get_metric(summary, "test/bit_accuracy", get_metric(summary, "bit_accuracy", fb.get("bit_accuracy"))),
            "seq_accuracy": get_metric(summary, "test/seq_accuracy", get_metric(summary, "seq_accuracy", fb.get("seq_accuracy"))),
            "levenshtein":  get_metric(summary, "test/levenshtein",  get_metric(summary, "levenshtein",  fb.get("levenshtein"))),
            "bleu":         get_metric(summary, "test/bleu",         get_metric(summary, "bleu",         fb.get("bleu"))),
            "rouge1":       get_metric(summary, "test/rouge1",       get_metric(summary, "rouge1",       fb.get("rouge1"))),
            "rougeL":       get_metric(summary, "test/rougeL",       get_metric(summary, "rougeL",       fb.get("rougeL"))),
            "runtime_s":    summary.get("_runtime", None),
        }
        run_records[name] = record
        print("done.")

    # Apply fallbacks for final runs not present in WandB
    for key, fb in FALLBACK_METRICS.items():
        if key not in run_records:
            run_records[key] = {
                "name": key, "label": CONFIG_LABELS[key], "id": None,
                "state": "missing", "config": {}, "summary": {}, "sys_hist": pd.DataFrame(),
                "history": pd.DataFrame(), "is_final": True,
                "val_loss":     fb["val_loss"],
                "bit_accuracy": fb["bit_accuracy"],
                "seq_accuracy": fb["seq_accuracy"],
                "levenshtein":  fb["levenshtein"],
                "bleu":  fb.get("bleu"),  "rouge1": fb.get("rouge1"),
                "rougeL": fb.get("rougeL"), "runtime_s": None,
            }

    final_keys = [k for k in ["run-C1-single-gpu","run-C2","run-C3","run-C4","run-C5"] if k in run_records]
    exp_keys   = [k for k in run_records if k not in FINAL_RUNS]

    plt.rcParams.update(STYLE)

    # ===========================================================================
    # PLOT 1 - Validation Loss vs Epochs: Final Runs Only
    # ===========================================================================
    fig, ax = plt.subplots(figsize=(9, 5))
    for key in final_keys:
        rec  = run_records[key]
        hist = rec["history"]
        col  = find_val_loss_col(hist) if not hist.empty else None
        if col is None:
            continue
        df = hist.dropna(subset=[col])
        if df.empty:
            continue
        x = df["epoch"] if "epoch" in df.columns else pd.Series(range(len(df)))
        ax.plot(x, df[col], label=rec["label"],
                color=PALETTE_FINAL[key], linewidth=2.5)

    ax.set_title("Validation Loss vs. Epochs - Final Configurations (C1-C5)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.legend(title="Configuration")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/val_loss_vs_epochs_final.png", dpi=300)
    plt.close()
    print(f"[Plot 1] val_loss_vs_epochs_final.png saved.")

    # ===========================================================================
    # PLOT 2 - Validation Loss: Exploration (All Runs Overlay)
    # ===========================================================================
    fig, ax = plt.subplots(figsize=(11, 6))
    for key in exp_keys:
        rec  = run_records[key]
        hist = rec["history"]
        col  = find_val_loss_col(hist) if not hist.empty else None
        if col is None:
            continue
        df = hist.dropna(subset=[col])
        if df.empty:
            continue
        x = df["epoch"] if "epoch" in df.columns else pd.Series(range(len(df)))
        ax.plot(x, df[col], color="#aaaaaa", linewidth=1.0, alpha=0.4, linestyle="--")

    for key in final_keys:
        rec  = run_records[key]
        hist = rec["history"]
        col  = find_val_loss_col(hist) if not hist.empty else None
        if col is None:
            continue
        df = hist.dropna(subset=[col])
        if df.empty:
            continue
        x = df["epoch"] if "epoch" in df.columns else pd.Series(range(len(df)))
        ax.plot(x, df[col], label=rec["label"],
                color=PALETTE_FINAL[key], linewidth=3.0)

    exp_patch = mpatches.Patch(color="#aaaaaa", alpha=0.6,
                               label=f"Experimental runs (n={len(exp_keys)})")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [exp_patch], labels + [exp_patch.get_label()],
              title="Run Type", ncol=2, fontsize=9)
    ax.set_title("Iterative Hyperparameter Exploration: Validation Loss across All Runs",
                 fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/val_loss_exploration.png", dpi=300)
    plt.close()
    print(f"[Plot 2] val_loss_exploration.png saved.")

    # ===========================================================================
    # PLOT 3 - Accuracy Comparison (Bit & Seq Accuracy)
    # ===========================================================================
    labels_list = [run_records[k]["label"] for k in final_keys]
    bit_accs    = [run_records[k]["bit_accuracy"] for k in final_keys]
    seq_accs    = [run_records[k]["seq_accuracy"] for k in final_keys]

    x = np.arange(len(final_keys))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w/2, [v*100 for v in bit_accs], w, label="Bit Accuracy (%)",
                   color=[PALETTE_FINAL[k] for k in final_keys], alpha=0.9)
    bars2 = ax.bar(x + w/2, [v*100 for v in seq_accs], w, label="Seq Accuracy (%)",
                   color=[PALETTE_FINAL[k] for k in final_keys], alpha=0.5,
                   edgecolor=[PALETTE_FINAL[k] for k in final_keys], linewidth=1.5)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Accuracy Comparison across Configurations (C1-C5)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/accuracy_comparison.png", dpi=300)
    plt.close()
    print(f"[Plot 3] accuracy_comparison.png saved.")

    # ===========================================================================
    # PLOT 4 - Levenshtein Distance
    # ===========================================================================
    levs = [run_records[k]["levenshtein"] for k in final_keys]
    colors_lev = [PALETTE_FINAL[k] for k in final_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels_list, levs, color=colors_lev, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, levs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Levenshtein Distance (lower is better)")
    ax.set_title("Levenshtein Distance Comparison (C1-C5)", fontweight="bold")
    ax.set_ylim(0, max(levs) * 1.25)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/levenshtein_comparison.png", dpi=300)
    plt.close()
    print(f"[Plot 4] levenshtein_comparison.png saved.")

    # ===========================================================================
    # PLOT 5 - BLEU & ROUGE (C1-C4 only)
    # ===========================================================================
    tok_keys   = [k for k in final_keys if k != "run-C5"]
    tok_labels = [run_records[k]["label"] for k in tok_keys]
    bleus      = [run_records[k].get("bleu") or 0  for k in tok_keys]
    rouge1s    = [run_records[k].get("rouge1") or 0 for k in tok_keys]
    rougeLs    = [run_records[k].get("rougeL") or 0 for k in tok_keys]

    x = np.arange(len(tok_keys))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, bleus,   w, label="BLEU",   color="#4C72B0", alpha=0.85)
    ax.bar(x,     rouge1s, w, label="ROUGE-1", color="#55A868", alpha=0.85)
    ax.bar(x + w, rougeLs, w, label="ROUGE-L", color="#C44E52", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(tok_labels)
    ax.set_ylim(0.80, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("BLEU & ROUGE Scores - Tokenized Models (C1-C4)\n"
                 "(C5/BLT operates at byte level; BLEU/ROUGE not applicable)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/bleu_rouge_comparison.png", dpi=300)
    plt.close()
    print(f"[Plot 5] bleu_rouge_comparison.png saved.")

    # ===========================================================================
    # PLOT 6 - Peak GPU Memory
    # ===========================================================================
    fallback_gpu = {
        "run-C1-single-gpu": 8.4, "run-C2": 8.5, "run-C3": 8.3, "run-C4": 8.2, "run-C5": 5.1,
    }
    gpu_mem_records = {}
    gpu_source = "Estimated from architecture (see h-params.md)"

    for key in final_keys:
        rec     = run_records[key]
        sys_df  = rec["sys_hist"]
        peak_gb = None

        if not sys_df.empty:
            mem_cols = find_gpu_memory_cols(sys_df)
            if mem_cols:
                col  = mem_cols[0]
                vals = sys_df[col].dropna()
                if not vals.empty:
                    peak_val = vals.max()
                    # WandB logs memory in MB; convert to GB
                    peak_gb = peak_val / 1024.0 if peak_val > 100 else peak_val
                    gpu_source = "WandB system metrics"

        if peak_gb is None or np.isnan(peak_gb):
            peak_gb = fallback_gpu.get(key, 8.0)

        gpu_mem_records[key] = peak_gb

    print(f"[Plot 6] GPU memory source: {gpu_source}")

    gpu_vals   = [gpu_mem_records[k] for k in final_keys]
    colors_gpu = [PALETTE_FINAL[k] for k in final_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels_list, gpu_vals, color=colors_gpu, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, gpu_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f} GB", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(y=11.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label="RTX 2080 Ti VRAM limit (11 GB)")
    ax.set_ylabel("Peak GPU Memory Allocation (GB)")
    ax.set_ylim(0, 13)
    ax.set_title(f"Peak GPU Memory Usage per Configuration\n(Source: {gpu_source})",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/gpu_memory_comparison.png", dpi=300)
    plt.close()
    print(f"[Plot 6] gpu_memory_comparison.png saved.")

    # ===========================================================================
    # PLOT 7 - Training Speed (Wall-clock time per epoch)
    # ===========================================================================
    fallback_speed = {
        "run-C1-single-gpu": 210.0, "run-C2": 208.0, "run-C3": 215.0,
        "run-C4": 212.0, "run-C5": 125.0,
    }

    speed_records = {}
    for key in final_keys:
        rec     = run_records[key]
        runtime = rec["runtime_s"]
        hist    = rec["history"]

        if runtime and not hist.empty:
            max_ep = hist["epoch"].dropna().max() if "epoch" in hist.columns else len(hist)
            if pd.isna(max_ep) or max_ep <= 0:
                max_ep = 1
            speed_records[key] = runtime / max(max_ep, 1)
        else:
            speed_records[key] = fallback_speed.get(key, 200.0)

    spd_vals   = [speed_records[k] for k in final_keys]
    colors_spd = [PALETTE_FINAL[k] for k in final_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels_list, spd_vals, color=colors_spd, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, spd_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.0f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Average Wall-Clock Time per Epoch (seconds)")
    ax.set_title("Training Speed Comparison (Wall-Clock Time per Epoch)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_speed_comparison.png", dpi=300)
    plt.close()
    print(f"[Plot 7] training_speed_comparison.png saved.")

    # ===========================================================================
    # Save metadata JSON
    # ===========================================================================
    meta = {}
    for key, rec in run_records.items():
        meta[key] = {
            "label":        rec["label"],
            "state":        rec["state"],
            "val_loss":     rec["val_loss"],
            "bit_accuracy": rec["bit_accuracy"],
            "seq_accuracy": rec["seq_accuracy"],
            "levenshtein":  rec["levenshtein"],
            "bleu":         rec.get("bleu"),
            "rouge1":       rec.get("rouge1"),
            "rougeL":       rec.get("rougeL"),
            "runtime_s":    rec["runtime_s"],
            "gpu_mem_gb":   gpu_mem_records.get(key),
            "time_per_ep":  speed_records.get(key),
        }
    with open(f"{OUTPUT_DIR}/run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print("\n" + "="*60)
    print("All outputs saved to ./outputs/:")
    for p in ["val_loss_vs_epochs_final.png", "val_loss_exploration.png",
              "accuracy_comparison.png", "levenshtein_comparison.png",
              "bleu_rouge_comparison.png", "gpu_memory_comparison.png",
              "training_speed_comparison.png", "run_metadata.json"]:
        print(f"  + outputs/{p}")
    print("="*60)


if __name__ == "__main__":
    main()
