"""
check_gpu_stats.py
===================
Standalone diagnostic + plot regenerator for GPU memory/speed, from WandB
*system* metrics only. Does not touch training/eval logs or checkpoints
(per-config train/val loss and test-set numbers are unaffected by anything
here and come from the existing evaluation logs, not this script).

Background: C1's original single-GPU run (run-C1-single-gpu) logged a
suspicious peak GPU memory reading because another process was briefly
co-resident on the same physical GPU. run-C1-single-gpu-again is an
identical rerun (same code, same config -> same train/val loss and
checkpoints) done specifically to get a trustworthy memory/speed reading
on an otherwise-idle GPU.

While investigating, two bugs in the original memory pipeline surfaced:

  1. The field previously read as "GPU memory in GB" (system.gpu.N.memory
     Allocated) is actually a PERCENTAGE of that GPU's total VRAM, not GB.
     The real value is system.gpu.N.memoryAllocatedBytes.
  2. WandB's system-metrics stream logs EVERY GPU on the physical node
     (via a whole-machine nvidia-smi poll), not just the one a given job
     is actually using. C2, C3, C4, and C5 ran concurrently on the same
     4-GPU node (each pinned to its own index), so each run's own log
     also contains its three siblings' memory readings. Picking "the
     first GPU-memory column found" (as the original script did) silently
     grabbed whichever sibling happened to sort first, not the run's own
     GPU. This script fixes that by reading the specific index each
     config actually trained on (RUN_OWN_GPU_INDEX below, confirmed with
     the user).

Run from project root:
    python src/check_gpu_stats.py

Requires WANDB_API_KEY to be set in environment (or `wandb login` already done).
"""

import os
import sys
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

warnings.filterwarnings("ignore")

PROJECT_PATH = "yash-more1224-international-institute-of-information-tec/anlp-assignment-1"
OUTPUT_DIR   = "outputs"

# The physical GPU index each final run actually trained on, confirmed
# with the user. C1's two single-GPU runs are not in this dict: both were
# alone on their node (every other GPU index sat flat near 0 for their
# whole duration), so gpu.0 is unambiguously "self" for them without
# needing an explicit mapping.
RUN_OWN_GPU_INDEX = {
    "run-C2": 0,
    "run-C3": 1,
    "run-C4": 2,
    "run-C5": 3,
}
C1_RUNS = ["run-C1-single-gpu-again", "run-C1-single-gpu"]

CONFIG_LABELS = {
    "run-C1-single-gpu-again": "C1 (Base)",
    "run-C2": "C2 (+RoPE)",
    "run-C3": "C3 (+GQA)",
    "run-C4": "C4 (+RMSNorm)",
    "run-C5": "C5 (BLT)",
}
PALETTE = {
    "run-C1-single-gpu-again": "#4C72B0",
    "run-C2": "#DD8452",
    "run-C3": "#55A868",
    "run-C4": "#C44E52",
    "run-C5": "#8172B2",
}
FINAL_ORDER = ["run-C1-single-gpu-again", "run-C2", "run-C3", "run-C4", "run-C5"]

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


def own_gpu_bytes(run, gpu_idx):
    sys_df = run.history(stream="system", samples=10000, pandas=True)
    col = f"system.gpu.{gpu_idx}.memoryAllocatedBytes"
    if col not in sys_df.columns:
        return None
    vals = sys_df[col].dropna()
    return vals.max() if not vals.empty else None


def time_per_epoch(run):
    summary = dict(run.summary._json_dict)
    runtime_s = summary.get("_runtime")
    hist = run.history(samples=2000, pandas=True)
    max_epoch = hist["epoch"].dropna().max() if (not hist.empty and "epoch" in hist.columns) else None
    if runtime_s and max_epoch and max_epoch > 0:
        return runtime_s / max_epoch
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    api = wandb.Api(timeout=60)
    print(f"[W&B] Fetching runs from: {PROJECT_PATH}")
    try:
        all_runs = list(api.runs(PROJECT_PATH))
    except Exception as exc:
        print(f"[ERROR] Cannot fetch runs: {exc}")
        sys.exit(1)
    runs_by_name = {r.name: r for r in all_runs}

    needed = list(RUN_OWN_GPU_INDEX) + C1_RUNS
    missing = [n for n in needed if n not in runs_by_name]
    if missing:
        print(f"[ERROR] Missing runs: {missing}")
        sys.exit(1)

    rows = []
    for name, idx in RUN_OWN_GPU_INDEX.items():
        run = runs_by_name[name]
        peak_bytes = own_gpu_bytes(run, idx)
        tpe = time_per_epoch(run)
        rows.append({"run": name, "own_gpu_idx": idx,
                      "peak_gpu_mem_gb": peak_bytes / 1024**3 if peak_bytes else None,
                      "time_per_epoch_s": tpe})

    for name in C1_RUNS:
        run = runs_by_name[name]
        peak_bytes = own_gpu_bytes(run, 0)
        tpe = time_per_epoch(run)
        rows.append({"run": name, "own_gpu_idx": 0,
                      "peak_gpu_mem_gb": peak_bytes / 1024**3 if peak_bytes else None,
                      "time_per_epoch_s": tpe})

    df = pd.DataFrame(rows)
    print()
    with pd.option_context("display.max_columns", None, "display.width", 160,
                            "display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))
    print()

    mem = {r["run"]: r["peak_gpu_mem_gb"] for r in rows}
    spd = {r["run"]: r["time_per_epoch_s"] for r in rows}

    print(f"[Result] run-C1-single-gpu-again (clean, isolated): "
          f"{mem['run-C1-single-gpu-again']:.3f} GB, {spd['run-C1-single-gpu-again']:.1f} s/epoch")
    print(f"[Result] run-C1-single-gpu (old, another process co-resident): "
          f"{mem['run-C1-single-gpu']:.3f} GB, {spd['run-C1-single-gpu']:.1f} s/epoch")
    for name in RUN_OWN_GPU_INDEX:
        print(f"[Result] {name} (own gpu.{RUN_OWN_GPU_INDEX[name]}): "
              f"{mem[name]:.3f} GB, {spd[name]:.1f} s/epoch")

    # ------------------------------------------------------------------
    # Regenerate outputs/gpu_memory_comparison.png and
    # outputs/training_speed_comparison.png with the corrected numbers.
    # ------------------------------------------------------------------
    plt.rcParams.update(STYLE)
    labels = [CONFIG_LABELS[k] for k in FINAL_ORDER]
    colors = [PALETTE[k] for k in FINAL_ORDER]

    gpu_vals = [mem[k] for k in FINAL_ORDER]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, gpu_vals, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, gpu_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{val:.2f} GB", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Peak GPU Memory Allocation (GB)")
    ax.set_ylim(0, max(gpu_vals) * 1.35)
    ax.set_title("Peak GPU Memory Usage per Configuration\n"
                 "(system.gpu.N.memoryAllocatedBytes, each config's own GPU index)",
                 fontweight="bold")
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/gpu_memory_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] {out_path} saved.")

    spd_vals = [spd[k] for k in FINAL_ORDER]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, spd_vals, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, spd_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.0f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Average Wall-Clock Time per Epoch (seconds)")
    ax.set_title("Training Speed Comparison (Wall-Clock Time per Epoch)", fontweight="bold")
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/training_speed_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] {out_path} saved.")


if __name__ == "__main__":
    main()
