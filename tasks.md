# Assignment 1: Transformers from Scratch, Architectural Variants, and Byte Latent Transformers (BLT)

**Course**: Advanced Natural Language Processing
**Deadline**: September 2nd, 2026, 23:59 IST

## General Instructions
- [ ] Implement everything in Python using PyTorch.
- [ ] Build core transformer components from scratch. **DO NOT** use `nn.Transformer` or `nn.MultiheadAttention`.
- [ ] Use Weights & Biases (WandB) for logging training runs.
- [ ] Host pre-trained model checkpoints on Hugging Face.
- [ ] Include WandB and Hugging Face links in the final report.
- [ ] Format the submission as a single `.zip` file named `<rollnumber>_assignment1.zip` (No GitHub links or email submissions).

## Task 1: Transformer Architecture
Design and implement a full Encoder-Decoder Transformer using basic PyTorch operations. Write custom `nn.Module` classes for the following in the `src/models/` directory:

- [ ] **Scaled Dot-Product Attention**: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`
- [ ] **Multi-Head Attention (MHA)** and **Grouped-Query Attention (GQA)** blocks (`attention.py`)
- [ ] **Position-wise Feed-Forward Networks (FFN)**
- [ ] **Pre-Layer Normalization** and **RMS Normalization** modules (`norm.py`)
- [ ] **Sinusoidal Absolute Positional Encoding** and **RoPE** (Rotary Positional Embeddings) (`positional.py`)
- [ ] **Local Encoder/Decoder patch modules** for the Byte Latent Transformer (BLT) approach (`blt.py`)

## Task 2: Architectural Ablation Study
Run a controlled experiment across five configurations to evaluate specific components. Ensure consistent hyperparameters (depth, width, learning rate, batch size) across runs.

### Configurations
- [ ] **C1 (Base)**: Sinusoidal Absolute | Multi-Head Attention | LayerNorm | Standard Subword
- [ ] **C2 (Positional)**: **RoPE** | Multi-Head Attention | LayerNorm | Standard Subword
- [ ] **C3 (Attention)**: Sinusoidal Absolute | **Grouped-Query** | LayerNorm | Standard Subword
- [ ] **C4 (Normalization)**: Sinusoidal Absolute | Multi-Head Attention | **RMSNorm** | Standard Subword
- [ ] **C5 (Tokenization/BLT)**: Sinusoidal Absolute | Multi-Head Attention | LayerNorm | **BLT (Token-Free)**

### Benchmarking Configuration 5 (BLT)
For C5, replace standard subword tokenizer with the BLT architecture (feed raw bytes to local encoder -> global transformer -> local byte-level decoder). Compare C5 against C1 paying special attention to:
- [ ] Training speed and computational overhead
- [ ] Peak GPU memory usage during training
- [ ] Overall sequence reconstruction performance on the test set

## Evaluation Metrics
Generate all results using **greedy decoding**.
- [ ] **Bit-Level Accuracy**: Percentage of exact bit matches
- [ ] **Sequence Accuracy**: Percentage of perfectly reconstructed sequences
- [ ] **Levenshtein Distance**: Edit distance between target and predicted outputs
- [ ] **BLEU and ROUGE Scores**: Standard n-gram overlap metrics (**for tokenized models only**)

## Deliverables & Directory Structure
Ensure your final `.zip` structure perfectly matches the following:

- [ ] `src/models/attention.py` (MHA and GQA)
- [ ] `src/models/positional.py` (Sinusoidal and RoPE)
- [ ] `src/models/norm.py` (LayerNorm and RMSNorm)
- [ ] `src/models/blt.py` (Local Encoder/Decoder for BLT)
- [ ] `src/dataset.py` (Tokenized vs Token-Free Loaders)
- [ ] `src/train.py` (Main training loop with WandB)
- [ ] `src/utils.py` (Metrics and plots)
- [ ] `outputs/` (Saved plots and logs)
- [ ] `README.md` (Setup instructions and HF links)
- [ ] `Report.pdf` (Your final report)

## Report Guidelines
- [ ] Maximum 6 pages (excluding references and links)
- [ ] 12pt Times New Roman, 1-inch margins
- [ ] Include tables and plots discussing the ablation study (C1 through C5)
- [ ] Discuss what each single-component change improved/worsened compared to the base model
- [ ] Discuss tradeoffs in memory and accuracy introduced by the token-free BLT approach
