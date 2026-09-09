# ANLP Assignment 1: Transformers from Scratch, Architectural Variants, and BLT

Character-level decryption framed as sequence-to-sequence translation, with a
controlled five-way ablation (C1 to C5) over positional encoding, attention,
normalization and tokenization.

All transformer components are written from scratch with basic PyTorch
operations. No `nn.Transformer` or `nn.MultiheadAttention` is used anywhere.
Both BPE tokenizers are also written from scratch, with no tokenizer library
used for training, encoding, decoding or serialization.

## Links

- **Weights & Biases project:** <https://wandb.ai/yash-more1224-international-institute-of-information-tec/anlp-assignment-1>
- **Hugging Face checkpoints:** <https://huggingface.co/yashmore1224/ANLP-A1>

The reported runs are `run-C1-bpew-long` through `run-C5-bpew-long`. The runs
named `run-C1-bpe` to `run-C4-bpe` are the superseded full-line round reported
as a negative result in the report. All other runs are exploratory.

## Directory layout

```
2024114004_assignment1/
|-- src/
|   |-- models/
|   |   |-- attention.py        # Scaled dot-product, MHA and GQA
|   |   |-- positional.py       # Sinusoidal absolute PE and RoPE
|   |   |-- norm.py             # Pre-LayerNorm and RMSNorm
|   |   `-- blt.py              # Local byte encoder/decoder for BLT (C5)
|   |-- dataset.py              # Tokenized (C1-C4) and token-free (C5) loaders
|   |-- train.py                # Main training loop with WandB logging
|   |-- utils.py                # Metrics and plotting helpers
|   |-- tokenizer.py            # From-scratch BPE (incremental merge algorithm)
|   |-- build_tokenizers.py     # Trains and serializes both BPE vocabularies
|   |-- evaluate_final.py       # Test-set decoding, stitching and scoring
|   |-- check_gpu_stats.py      # Pulls WandB system metrics, memory/speed plots
|   |-- plot_combined.py        # Combined exploration and quality figures
|   |-- plot_exploration.py     # Exploratory-run loss curves
|   `-- plot_final_results.py   # Accuracy, edit distance, BLEU/ROUGE figures
|-- outputs/                    # Plots, logs and per-config evaluation results
|-- Dataset_A1/                 # Provided data
|-- run_all.sh                  # Launches C1-C4, one per GPU
|-- README.md
`-- Report.pdf
```

Files beyond the required set (`tokenizer.py`, `build_tokenizers.py`,
`evaluate_final.py`, `check_gpu_stats.py` and the three `plot_*.py` scripts)
are the supporting scripts used to build the tokenizers, produce the reported
test numbers and generate every figure in the report.

### What is in `outputs/`

- `results_windowed/` and `results_c5/` hold the reported test metrics
  (`final_eval_C*.json`) and the full 500-line prediction dumps.
- `results_fullline/` holds the superseded full-line round, kept as the
  documented negative result.
- `logs/reported_runs/` holds the training and evaluation logs for the exact
  runs the report numbers come from.
- The `*.png` files are the figures used in the report.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy pandas matplotlib tqdm wandb \
            python-Levenshtein nltk rouge-score
wandb login          # or export WANDB_API_KEY=...
```

Trained on a single NVIDIA RTX 2080 Ti (11 GB) per configuration, four in
parallel on one node, with FP16 autocast.

## Reproducing the results

### 1. Build the tokenizers

```bash
python src/build_tokenizers.py
```

Writes `outputs/plain_bpe.json` (5,000 merges over characters) and
`outputs/cipher_bpe.json` (4,096 merges over the 8-bit byte-symbol stream).
Both are trained on the training split only. This takes well under a minute;
the cipher vocabulary trains in about 21 seconds.

### 2. Train

All five configurations use identical depth, width, head count, learning rate,
schedule and number of optimizer steps per epoch, and differ only in the single
ablated component. Keeping these matched is what makes the ablation controlled,
so change them together or not at all.

C1 to C4, one per GPU:

```bash
for i in 0 1 2 3; do
  CFG="C$((i + 1))"
  CUDA_VISIBLE_DEVICES=$i python -m src.train \
    --config "$CFG" --epochs 120 --chunk_size 64 \
    --batch_size 16 --grad_accum_steps 1 --lr 7e-4 --num_layers 4 \
    --max_seq_len 1408 --patience 15 --run_name "run-${CFG}-bpew-long" \
    > "outputs/${CFG}_long.log" 2>&1 &
done
```

C5 uses `--batch_size 64` so that it matches C1-C4 on **sequences per batch**,
and therefore on optimizer steps per epoch (648). This is the setting to be
careful with: C1-C4 batch by a 4,096-token budget rather than a fixed count, and
because C5's sequences are roughly 15 times longer, giving it the same token
budget would yield about 7 sequences per batch and an 8.7x mismatch in steps.

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train \
  --config C5 --epochs 120 --chunk_size 64 \
  --batch_size 64 --grad_accum_steps 1 --lr 7e-4 --num_layers 4 \
  --max_seq_len 1408 --patience 12 --run_name run-C5-bpew-long \
  > outputs/C5_long.log 2>&1
```

Checkpoints are written to `outputs/checkpoint_<CFG>_epoch<N>.pt` and are saved
only when validation loss improves, so the highest-numbered checkpoint for a
configuration is also its best. Add `--resume` to continue from the latest
checkpoint for that configuration; the learning-rate schedule is advanced to
match. Use `--no_save` for probe runs that should not leave checkpoints behind.

The reported models were stopped for compute time rather than at convergence:
C1 to C4 at epoch 70, and C5 at its best checkpoint, epoch 29, after which a
further 12 epochs gave no improvement.

### 3. Evaluate

`--chunk_size 64` is required for C1-C4 and must match how the checkpoint was
trained. The default is 0, which builds the full-line dataset and will silently
produce meaningless numbers against a windowed checkpoint.

```bash
python -m src.evaluate_final --config C1 \
  --checkpoint_path outputs/checkpoint_C1_epoch70.pt --chunk_size 64
# ...the same for C2, C3, C4

python -m src.evaluate_final --config C5 \
  --checkpoint_path outputs/checkpoint_C5_epoch29.pt --blt_offset true
```

Each run writes `outputs/final_eval_<CFG>.json`, a 50-example readable dump and
`outputs/predictions_full_<CFG>.jsonl` with all 500 stitched lines. Every model
decodes 64-character windows which are concatenated back into full lines, so all
five are scored on identical 500-line references.

C5's evaluation takes roughly 30 minutes against about 40 seconds for C1-C4,
because it generates eight times as many tokens per character. Raising
`--batch_size` (which for evaluation affects only C5's chunk decoding) reduces
this considerably.

### 4. Regenerate the figures

```bash
python src/check_gpu_stats.py      # memory, speed and parameter-count plots
python src/plot_final_results.py   # accuracy, edit distance, BLEU/ROUGE
python src/plot_combined.py        # combined exploration and quality figures
```

These read from the WandB API, so `WANDB_API_KEY` must be set. Note that
`check_gpu_stats.py` reads `system.gpu.N.memoryAllocatedBytes` at the specific
GPU index each configuration was pinned to. That indexing is deliberate: WandB
polls every GPU on the machine, so a run's log also contains its co-resident
siblings' readings, and reading the first available GPU column returns the wrong
run's memory.

## Using a released checkpoint

C1-C4 additionally need both tokenizer JSONs, which are in the same Hugging Face
repository. Place them in `outputs/` before evaluating.

| Config | Checkpoint | Best val loss |
|---|---|---|
| C1 (Base) | `checkpoint_C1_epoch70.pt` | 0.3034 |
| C2 (RoPE) | `checkpoint_C2_epoch70.pt` | 0.1347 |
| C3 (GQA) | `checkpoint_C3_epoch70.pt` | 0.3073 |
| C4 (RMSNorm) | `checkpoint_C4_epoch70.pt` | 0.2850 |
| C5 (BLT) | `checkpoint_C5_epoch29.pt` | 0.0745 |

Validation loss is comparable within C1-C4 only, since C5 predicts over a
259-token byte vocabulary rather than a 5,000-token subword one.
