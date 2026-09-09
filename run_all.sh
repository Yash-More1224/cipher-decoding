#!/usr/bin/env bash
# run_all.sh - launch the reported ablation, one config per GPU.
#
# Usage:
#   bash run_all.sh timing        # 2 epochs on C1 only, to measure epoch time
#   bash run_all.sh <epochs>      # launch C1-C4 in parallel for <epochs> epochs
#   bash run_all.sh <epochs> c5   # launch C5 only, on cuda:0
#
# Each config gets its own GPU rather than sharing four via DDP. That keeps the
# effective batch size identical across configs (a GPU-count mismatch between
# C1 and the rest already invalidated one round of efficiency numbers) and lets
# all four finish in the wall-clock time of one.
#
# GPU assignment is recorded deliberately: W&B's system metrics log EVERY gpu on
# the node, not just the one a job uses, so without this mapping the per-config
# memory readings cannot be attributed correctly afterwards.
#
#   C1 -> cuda:0    C2 -> cuda:1    C3 -> cuda:2    C4 -> cuda:3
#
# IMPORTANT, on keeping the ablation controlled: C1-C4 batch by a token budget
# (--max_tokens 4096, the default), which on 64-character windows works out to a
# fixed 64 sequences per batch and 648 optimizer steps per epoch. C5's sequences
# are roughly 15x longer, so the same token budget would give it about 7
# sequences per batch and 8.7x more steps over the same data. C5 is therefore
# launched with an explicit --batch_size 64 so it matches on sequences per batch
# and hence on steps per epoch. Everything else is identical across all five.

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-timing}"
WHICH="${2:-c1c4}"
TAG="bpew"
mkdir -p outputs

# ── Guard against stale checkpoints ───────────────────────────────────────────
# Training writes outputs/checkpoint_C{N}_epoch*.pt. Eval auto-detection globs
# that pattern and takes the highest epoch, so a leftover high-epoch file from an
# earlier round would silently be evaluated instead of the new one. That exact
# bug already cost one round of results.
OLD=$(ls outputs/checkpoint_C[1-5]_epoch*.pt 2>/dev/null || true)
if [[ -n "$OLD" ]]; then
    STAMP="outputs/archive_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$STAMP"
    echo "[run_all] Archiving $(echo "$OLD" | wc -l) pre-existing checkpoint(s) -> $STAMP/"
    mv $OLD "$STAMP"/
fi

# Tokenizers must exist before any training job starts. Parallel jobs would
# otherwise each retrain BPE and race to write the same JSON files.
if [[ ! -f outputs/plain_bpe.json || ! -f outputs/cipher_bpe.json ]]; then
    echo "[run_all] Building tokenizers (one time)..."
    python src/build_tokenizers.py
fi

COMMON="--grad_accum_steps 1 --lr 7e-4 --num_layers 4 --max_seq_len 1408 --chunk_size 64"

if [[ "$MODE" == "timing" ]]; then
    echo "[run_all] Timing run: C1, 2 epochs on cuda:0."
    CUDA_VISIBLE_DEVICES=0 python -m src.train --config C1 --epochs 2 \
        --batch_size 16 $COMMON \
        --run_name "run-C1-${TAG}-timing" 2>&1 | tee "outputs/timing_C1_${TAG}.log"
    exit 0
fi

EPOCHS="$MODE"

if [[ "$WHICH" == "c5" ]]; then
    echo "[run_all] Launching C5 for $EPOCHS epochs on cuda:0 (batch_size 64, offset-grouped)."
    CUDA_VISIBLE_DEVICES=0 nohup python -m src.train \
        --config C5 --epochs "$EPOCHS" --batch_size 64 $COMMON \
        --patience 12 --run_name "run-C5-${TAG}-long" \
        > "outputs/C5_long.log" 2>&1 &
    echo "  C5 -> cuda:0  (pid $!)  log: outputs/C5_long.log"
    wait
    echo "[run_all] C5 finished. Evaluate with:"
    echo "  python -m src.evaluate_final --config C5 --checkpoint_path outputs/checkpoint_C5_epoch<N>.pt --blt_offset true"
    exit 0
fi

echo "[run_all] Launching C1-C4 for $EPOCHS epochs, one per GPU."
for i in 0 1 2 3; do
    CFG="C$((i + 1))"
    CUDA_VISIBLE_DEVICES=$i nohup python -m src.train \
        --config "$CFG" --epochs "$EPOCHS" --batch_size 16 $COMMON \
        --patience 15 --run_name "run-${CFG}-${TAG}-long" \
        > "outputs/${CFG}_long.log" 2>&1 &
    echo "  $CFG -> cuda:$i  (pid $!)  log: outputs/${CFG}_long.log"
done

wait
echo "[run_all] All four finished. Evaluate (--chunk_size MUST match training):"
echo "  for c in C1 C2 C3 C4; do"
echo "    python -m src.evaluate_final --config \$c --checkpoint_path outputs/checkpoint_\${c}_epoch<N>.pt --chunk_size 64"
echo "  done"
