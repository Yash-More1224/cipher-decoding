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

# The physical GPU index each final run actually trained on. run_all.sh pins
# one config per GPU in this order, via CUDA_VISIBLE_DEVICES.
#
# This mapping is NOT optional. Verified 2026-08-28: every windowed run's
# system stream contains all four indices (gpu.0-gpu.3) carrying the *same*
# four values, because W&B polls the whole machine with nvidia-smi while all
# four jobs are co-resident. Reading "the first GPU column found" therefore
# returns whichever sibling sorts first, not the run's own GPU. Only the
# index below is genuinely this config's memory.
RUN_OWN_GPU_INDEX = {
    "run-C1-bpew-long": 0,
    "run-C2-bpew-long": 1,
    "run-C3-bpew-long": 2,
    "run-C4-bpew-long": 3,
}

# C5 was retrained (2026-08-29) to match C1-C4 exactly: 4 global blocks, the
# same offset-grouped sampler capped at 64 windows per batch, and therefore the
# same 648 batches/epoch over the same 39,683 windows. Every column below is now
# a like-for-like comparison, so C5 no longer needs a caveat.
#
# C5's 120-epoch schedule was interrupted when the gnode047 allocation expired at
# epoch 28 and was resumed on gnode081 from checkpoint_C5_epoch27.pt. wandb.init
# starts a fresh run on resume, so TWO runs share the name "run-C5-bpew-long".
# Selecting by name alone is ambiguous and would silently pick whichever sorts
# last, so C5 is pinned by run ID. The resumed segment (i9rtm63c) is the one
# used: it covers epochs 28-41, which includes the reported best checkpoint
# (epoch 29, val_loss 0.0745), and it ran alone on gpu.3 after C4 was stopped.
C5_RUN, C5_GPU_INDEX = "run-C5-bpew-long", 3
C5_RUN_ID = "i9rtm63c"

CONFIG_LABELS = {
    "run-C1-bpew-long": "C1 (Base)",
    "run-C2-bpew-long": "C2 (RoPE)",
    "run-C3-bpew-long": "C3 (GQA)",
    "run-C4-bpew-long": "C4 (RMSNorm)",
    "run-C5-bpew-long": "C5 (BLT)",
}
PALETTE = {
    "run-C1-bpew-long": "#4C72B0",
    "run-C2-bpew-long": "#DD8452",
    "run-C3-bpew-long": "#55A868",
    "run-C4-bpew-long": "#C44E52",
    "run-C5-bpew-long": "#8172B2",
}
COMPARABLE = ["run-C1-bpew-long", "run-C2-bpew-long", "run-C3-bpew-long",
              "run-C4-bpew-long", C5_RUN]
FINAL_ORDER = list(COMPARABLE)

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
    """Mean seconds per epoch.

    Prefers perf/epoch_time_s, which train.py logs per epoch by timing the
    epoch directly. That is the honest number: the _runtime fallback divides
    total wall-clock by epoch count, so it also absorbs process startup,
    dataset construction, tokenizer loading, and end-of-run test evaluation.
    The fallback is only used for runs predating that logging (C5).
    """
    hist = run.history(samples=5000, pandas=True)
    if not hist.empty and "perf/epoch_time_s" in hist.columns:
        vals = hist["perf/epoch_time_s"].dropna()
        if not vals.empty:
            return float(vals.mean()), "measured"

    summary = dict(run.summary._json_dict)
    runtime_s = summary.get("_runtime")
    max_epoch = hist["epoch"].dropna().max() if (not hist.empty and "epoch" in hist.columns) else None
    if runtime_s and max_epoch and max_epoch > 0:
        return runtime_s / max_epoch, "wall-clock/epochs"
    return None, "unavailable"


def n_params(run):
    return dict(run.summary._json_dict).get("perf/n_params")


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
    # C5's name is shared by the pre-interruption and resumed segments, so
    # resolve it by ID instead of letting the dict comprehension above pick one.
    runs_by_id = {r.id: r for r in all_runs}
    if C5_RUN_ID not in runs_by_id:
        print(f"[ERROR] C5 run id {C5_RUN_ID} not found in project")
        sys.exit(1)
    runs_by_name[C5_RUN] = runs_by_id[C5_RUN_ID]

    index_of = dict(RUN_OWN_GPU_INDEX)
    index_of[C5_RUN] = C5_GPU_INDEX

    missing = [n for n in FINAL_ORDER if n not in runs_by_name]
    if missing:
        print(f"[ERROR] Missing runs: {missing}")
        sys.exit(1)

    rows = []
    for name in FINAL_ORDER:
        run = runs_by_name[name]
        idx = index_of[name]
        peak_bytes = own_gpu_bytes(run, idx)
        tpe, source = time_per_epoch(run)
        rows.append({"run": name,
                     "own_gpu_idx": idx,
                     "params": n_params(run),
                     "peak_gpu_mem_gb": peak_bytes / 1024**3 if peak_bytes else None,
                     "time_per_epoch_s": tpe,
                     "time_source": source,
                     "comparable": name in COMPARABLE})

    df = pd.DataFrame(rows)
    print()
    with pd.option_context("display.max_columns", None, "display.width", 200,
                            "display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))
    print()

    mem = {r["run"]: r["peak_gpu_mem_gb"] for r in rows}
    spd = {r["run"]: r["time_per_epoch_s"] for r in rows}

    for name in FINAL_ORDER:
        tag = "" if name in COMPARABLE else "   [not comparable]"
        print(f"[Result] {CONFIG_LABELS[name]:15s} own gpu.{index_of[name]}  "
              f"{mem[name]:.3f} GB  {spd[name]:6.1f} s/epoch{tag}")

    # ------------------------------------------------------------------
    # Regenerate outputs/gpu_memory_comparison.png and
    # outputs/training_speed_comparison.png with the corrected numbers.
    # ------------------------------------------------------------------
    plt.rcParams.update(STYLE)
    labels = [CONFIG_LABELS[k] for k in FINAL_ORDER]
    colors = [PALETTE[k] for k in FINAL_ORDER]
    # Memory is comparable for all five (same data, own pinned GPU), so that plot
    # gets no hatching. Speed is not, because C5 batched differently, so only the
    # speed plot marks C5 out.
    hatches_flat = ["" for _ in FINAL_ORDER]
    hatches_speed = list(hatches_flat)
    NOTE = ("All five configurations train on the same 39,683 windows, 64 per "
            "batch, 648 batches per epoch; each bar is that run's own pinned GPU index.")
    MEM_NOTE = NOTE
    SPD_NOTE = NOTE

    def bar_plot(vals, ylabel, title, fmt, pad_frac, out_name,
                 hatches=hatches_flat, NOTE=""):
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.2)
        for bar, val, h in zip(bars, vals, hatches):
            if h:
                bar.set_hatch(h)
                bar.set_alpha(0.65)
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals) * pad_frac,
                    fmt.format(val), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(vals) * 1.35)
        ax.set_title(title, fontweight="bold")
        fig.text(0.5, 0.008, NOTE, ha="center", fontsize=8, style="italic")
        plt.tight_layout(rect=[0, 0.035, 1, 1])
        out_path = f"{OUTPUT_DIR}/{out_name}"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[Plot] {out_path} saved.")

    bar_plot([mem[k] for k in FINAL_ORDER],
             "Peak GPU Memory Allocation (GB)",
             "Peak GPU Memory Usage per Configuration\n"
             "(system.gpu.N.memoryAllocatedBytes, each config's own GPU index)",
             "{:.2f} GB", 0.02, "gpu_memory_comparison.png",
             hatches=hatches_flat, NOTE=MEM_NOTE)

    bar_plot([spd[k] for k in FINAL_ORDER],
             "Average Wall-Clock Time per Epoch (seconds)",
             "Training Speed Comparison (Mean Wall-Clock Time per Epoch)",
             "{:.1f}s", 0.02, "training_speed_comparison.png",
             hatches=hatches_speed, NOTE=SPD_NOTE)

    params = [r["params"] for r in rows]
    if all(params):
        vals = [p / 1e6 for p in params]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, vals, color=colors,
                      edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{val:.2f}M", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_ylabel("Trainable Parameters (millions)")
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_title("Parameter Count per Configuration", fontweight="bold")
        plt.tight_layout()
        out_path = f"{OUTPUT_DIR}/param_count_comparison.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[Plot] {out_path} saved.")


if __name__ == "__main__":
    main()
