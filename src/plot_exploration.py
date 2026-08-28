"""
plot_exploration.py
====================
Generates the exploratory-run figures for report.tex (Section 3). We
visualize the 12 exploratory runs relevant to the report's narrative,
split into 3 PNGs grouped by what each cluster of runs was investigating:

  1. outputs/exploration_capacity_sweep.png  - model-size sweep (BPE-on-cipher era);
                                                4 runs (51M/512d, 37M/512d, 14.9M/256d,
                                                13M-8bit/256d)
  2. outputs/exploration_seqlen_sweep.png    - input sequence-length experiments; 3 runs
  3. outputs/exploration_lr_epoch_sweep.png  - LR / epoch-budget sweep at seq_len=512,
                                                plus the no-BPE-cipher transition; 5 runs

Explicitly excluded:
  - The 4-run "13M, no-bpe-cipher, warmup-LR-1.5e-3, 50-epoch, chunking-64Bytes"
    family (base run + 2 resumes + the final resume). These are all the SAME
    training run, logged as separate W&B runs only because training.py was
    resumed from checkpoint multiple times, not 4 distinct experiments.
  - The original 4-GPU run-C1 and the single-GPU run-C1 retrain: the final
    single-GPU C1 already appears in Figure 2 (val_loss_vs_epochs_final.png)
    alongside C2-C5, so it does not need its own exploratory plot.

Run from project root:
    python src/plot_exploration.py

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

# Runs that must never be fetched or plotted here: the 4-way resumed
# chunking run (same underlying training run, not 4 experiments).
EXCLUDE_RUNS = {
    "13M, no-bpe-cipher, warmup-LR-1.5e-3, 50-epoch, chunking-64Bytes",
    "resume: 13M, no-bpe-cipher, warmup-LR-1.5e-3, 50-epoch, chunking-64Bytes",
    "resume2: 13M, no-bpe-cipher, warmup-LR-1.5e-3, 50-epoch, chunking-64Bytes",
    "FINAL: resume3: 13M, no-bpe-cipher, warmup-LR-1.5e-3, 50-epoch, chunking-64Bytes",
}

# Tableau-10-ish palette, one distinct color per run within its group.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

# Each run entry is (wandb_run_name, legend_label, linestyle). linestyle
# defaults to solid; it's overridden for runs whose loss trajectory nearly
# coincides with another run in the same plot (e.g. a run and a
# longer-epoch-budget rerun of the same config), so both stay visible
# instead of one silently drawing on top of the other.
GROUPS = [
    {
        "file": "exploration_capacity_sweep.png",
        "title": "Exploration: Capacity Sweep (Pre-Fix, BPE-on-Cipher Era)",
        "runs": [
            ("51M-bpe, 6-layer, 512-d_model, 2048-d_ff", "51M params (6L, d_model=512)", "-"),
            ("37M-bpe, 4-layers, 512-d_model, 2048-d_ff", "37M params (4L, d_model=512)", "-"),
            ("14.9M-bpe, 6-layer, 256-d_model, 1024-d_ff", "14.9M params (6L, d_model=256)", "-"),
            ("13M-8-bit, 6-layers, 256-d_model, 1024-d_ff", "13M params, 8-bit repr. (6L, d_model=256)", "-"),
        ],
    },
    {
        "file": "exploration_seqlen_sweep.png",
        "title": "Exploration: Input Sequence-Length Experiments",
        "runs": [
            ("14M, 64 ip-seq", "seq_len=64", "-"),
            ("14M, 256 ip-seq", "seq_len=256", "-"),
            ("14M, lr=3e-4, 1336-ip-len", "seq_len=1336, LR=3e-4", "-"),
        ],
    },
    {
        "file": "exploration_lr_epoch_sweep.png",
        "title": "Exploration: Learning-Rate / Epoch-Budget Sweep (seq_len=512)\nand the No-BPE-Cipher Transition",
        "runs": [
            # LR=1e-3/30-epochs and LR=1e-3/50-epochs are the same run to a
            # fraction of a percent for their shared first 30 epochs (the
            # 50-epoch run just kept going) - dashed vs. solid keeps both
            # visible instead of one perfectly overdrawing the other.
            ("14M, 512 seq, LR:le-3, 50-epochs", "seq_len=512, LR=1e-3, 50 epochs", "-"),
            ("14M, 512 seq, LR:le-3, 30-epochs", "seq_len=512, LR=1e-3, 30 epochs", "--"),
            ("14M, 512 seq, LR:le-2, 30-epochs", "seq_len=512, LR=1e-2, 30 epochs", "-"),
            ("13M, no-bpe-cipher, warmup-LR-3e-3, 50-epoch", "no-BPE cipher, warmup LR=3e-3", "-"),
            ("13M, no-bpe-cipher", "no-BPE cipher, baseline LR", "--"),
        ],
    },
]

STYLE = {
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}


def find_train_loss_col(df):
    for candidate in ("train/avg_loss", "train/loss"):
        if candidate in df.columns:
            return candidate
    return None


def get_x(df):
    """
    "step" is blank on rows where the logged metric wasn't part of the same
    wandb.log() call as "step" itself; "_step" is W&B's own always-populated
    log-call counter, used to fill those gaps.
    """
    if "step" in df.columns and "_step" in df.columns:
        return df["step"].fillna(df["_step"])
    if "step" in df.columns:
        return df["step"]
    if "_step" in df.columns:
        return df["_step"]
    return pd.Series(range(len(df)))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    needed_names = {name for group in GROUPS for name, _, _ in group["runs"]}

    api = wandb.Api(timeout=60)
    print(f"[W&B] Fetching runs from: {PROJECT_PATH}")
    try:
        all_runs = list(api.runs(PROJECT_PATH))
    except Exception as exc:
        print(f"[ERROR] Cannot fetch runs: {exc}")
        sys.exit(1)

    runs_by_name = {r.name: r for r in all_runs if r.name in needed_names}
    missing = needed_names - set(runs_by_name)
    if missing:
        print(f"[WARN] Could not find these runs in the project: {missing}")
    excluded_present = [r.name for r in all_runs if r.name in EXCLUDE_RUNS]
    print(f"[Info] Confirmed excluded (not fetched further): {excluded_present}")

    plt.rcParams.update(STYLE)

    for group in GROUPS:
        fig, ax = plt.subplots(figsize=(8, 5))
        n_plotted = 0
        for idx, (run_name, label, linestyle) in enumerate(group["runs"]):
            run = runs_by_name.get(run_name)
            if run is None:
                print(f"[WARN] Skipping missing run '{run_name}'")
                continue
            try:
                hist = run.history(samples=2000, pandas=True)
            except Exception as exc:
                print(f"[WARN] Could not fetch history for '{run_name}': {exc}")
                continue
            col = find_train_loss_col(hist) if not hist.empty else None
            if col is None:
                print(f"[WARN] No training loss column for '{run_name}'")
                continue
            df = hist.dropna(subset=[col])
            if df.empty:
                print(f"[WARN] Empty training loss history for '{run_name}'")
                continue
            x = get_x(df)
            color = PALETTE[idx % len(PALETTE)]
            ax.plot(x, df[col], label=label, color=color, linewidth=2.2,
                    linestyle=linestyle, alpha=0.9)
            n_plotted += 1

        ax.set_title(group["title"], fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Training Loss")
        ax.legend(title="Run")
        plt.tight_layout()
        out_path = f"{OUTPUT_DIR}/{group['file']}"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[Plot] {out_path} saved ({n_plotted}/{len(group['runs'])} runs plotted).")


if __name__ == "__main__":
    main()
