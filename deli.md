# Assignment 1 Deliverables

Based on the assignment instructions (`ANLP_M26_A1.pdf` and `tasks.md`), here is a comprehensive list of what must be included in `Report.pdf` and `README.md`.

## Deliverables for `Report.pdf`

### 1. Formatting & Guidelines
- **Page Limit:** Maximum of 6 pages (excluding references and links).
- **Font:** 12pt Times New Roman.
- **Margins:** 1-inch margins.

### 2. Required Links
- **WandB Logs:** Links to your Weights & Biases training runs for all configurations.
- **Hugging Face Checkpoints:** Links to the hosted pre-trained model checkpoints on Hugging Face.

### 3. Required Content & Analysis
- **Ablation Study Results (Tables & Plots):** Include tables and plots summarizing the evaluation metrics across all 5 configurations (C1 through C5).
- **Component Impact Discussion:** A detailed discussion on how each single-component change (RoPE in C2, GQA in C3, RMSNorm in C4, BLT in C5) improved or worsened performance compared to the base model (C1).
- **C5 (BLT) Benchmarking vs C1:**
  - Discuss the tradeoffs in **memory** (peak GPU memory usage during training) and **accuracy** (overall sequence reconstruction performance on the test set).
  - Compare the **training speed** and **computational overhead** between C5 and C1.
- **Evaluation Metrics:** Ensure the reported results for all configurations are generated using **greedy decoding** and include:
  - Bit-Level Accuracy
  - Sequence Accuracy
  - Levenshtein Distance
  - BLEU and ROUGE Scores *(Note: BLEU and ROUGE are only required for tokenized models C1-C4, not for the token-free C5).*

---

## Deliverables for `README.md`

- **Setup Instructions:** Clear instructions on how to install dependencies, set up the environment, and run the code.
- **Hugging Face Links:** Links to the hosted pre-trained model checkpoints on Hugging Face (can be identical to the ones provided in the report).

---

## "Nice to Haves" (Enhancements for an Impressive Report)

### 1. Iterative Hyperparameter & Exploration Journey
- **The Optimization Story:** Include a section documenting the engineering evolution (as detailed in `report.md`):
  1. *Model Capacity Exploration:* 51M (6-layer) vs scaled-down 14M models (underfitting vs over-parameterization).
  2. *Tokenization Flaw Discovery:* Transitioning from standard BPE on raw bit strings (which broke positional cipher alignment) to Byte-level integer encodings (0-255).
  3. *Chunking & Data Augmentation:* Moving from full sequence lengths (1370) to fixed sliding-window 64/256 chunks with absolute positional offset tracking, expanding dataset size from ~4k to ~50k.
  4. *Regularization Strategy:* Tuning LR from `1e-4` to `1e-3`, applying `0.3` dropout, `0.05` weight decay, and `0.2` label smoothing.
- **Failed Trial Loss Curves Plot:** Overlay loss curves from early failed runs (e.g., low LR vs high LR, raw bit BPE vs byte integer encoding) to visually illustrate the impact of each optimization choice.

### 2. High-Impact Plots & Visualizations
- **Validation Loss vs. Epochs (C1–C5 Overlay):** A multi-line plot comparing convergence rate and validation loss trajectory across all five main ablation models.
- **Early Stopping & Convergence Rate (C5 Focus):** Highlight C5's rapid convergence where Early Stopping triggered at Epoch 46 (val_loss `0.0880`), comparing it to full 50-epoch runs of C1–C4.
- **Evaluation Metrics Bar Chart:** A comparative grouped bar chart comparing:
  - *Bit-Level Accuracy:* C5 (99.69%) vs C3 (95.65%) vs C1 (80.43%).
  - *Sequence Accuracy:* C5 (80%) vs C3 (80%) vs C1/C2/C4 (60%).
  - *Levenshtein Distance:* C5 (0.4) vs C1 (3.4) vs C3 (12.0).
- **System Metrics & Efficiency Graphs (WandB Export):**
  - *Peak GPU Memory Allocation:* Bar chart or line graph showing memory footprint during training (C1 vs C5).
  - *Training Speed:* Wall-clock time per epoch (e.g., C5 taking ~125s vs C1/C2 taking ~200s).

### 3. Qualitative Output Inspection & Decryption Samples
- **Reconstruction Sample Comparison Table:** Present 2–3 qualitative target plaintexts side-by-side with predicted outputs from C1, C3, and C5 to visually prove why C5's byte-level representation eliminates character substitution errors.
