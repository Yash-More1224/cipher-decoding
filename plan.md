# Hyperparameter Selection Plan

Before writing the actual model training code, we need to empirically determine the optimal hyperparameters for the Base Transformer (C1). This document outlines the experimental procedures we will run to finalize these values.

## 1. Determining `max_seq_len`
**Goal**: Find the absolute maximum number of tokens generated for any single line, so our positional embeddings and padding logic don't truncate data.

**Experimental Procedure**:
1. Train or load a standard subword tokenizer (e.g., HuggingFace's BPE `tokenizers`) on `brown_plain.txt`.
2. Train or load a separate subword tokenizer on `brown_cipher.txt`.
3. Iterate through all 5001 lines of the plaintext and tokenize them. Record the max token length: `max_len_plain`.
4. Iterate through all 5001 lines of the ciphertext and tokenize them. Record the max token length: `max_len_cipher`.
5. **Decision**: Set `max_seq_len = max(max_len_plain, max_len_cipher) + 2`. The `+ 2` accounts for the `<BOS>` and `<EOS>` tokens. 

## 2. Determining `d_model` (Hidden Dimension Size)
**Goal**: Find a hidden size that is large enough to learn the complex cipher mapping, but small enough to fit in GPU memory and train quickly.

**Experimental Procedure**:
1. Take a small, representative subset of the data (e.g., the first 200 lines). This is our "overfitting test set."
2. Instantiate the C1 model with varying `d_model` sizes: **128**, **256**, and **512**. 
   * *(Note: Keep `n_heads` proportional, e.g., `d_model // 64`, and `d_ff = 4 * d_model` for this test).*
3. Train each model on the 200-line subset for 500-1000 steps and monitor the training loss.
4. **Decision**: Because mapping ciphertext to plaintext requires significant memorization and capacity, we must choose the smallest `d_model` that successfully converges (loss approaches zero) on this small subset. If `d_model=256` plateaus at a high loss but `d_model=512` converges, we choose `512`.

## 3. Determining Feed-Forward Network Size (`d_ff`)
**Goal**: The FFN serves as the "memory bank" of the transformer. Since the subword tokenizer scrambles the 8-bit boundaries, the FFN needs enough capacity to map irregular bit-chunks back to plaintext concepts.

**Experimental Procedure**:
1. The standard transformer convention is an expansion factor of 4 (`d_ff = 4 * d_model`).
2. We will test two configurations on a validation split (using the `d_model` chosen above):
   * Configuration A: `d_ff = 4 * d_model`
   * Configuration B: `d_ff = 8 * d_model`
3. **Decision**: If Configuration B significantly improves sequence reconstruction accuracy, we will use it. Otherwise, we stick to the standard `4 * d_model` to save computational overhead and memory.

## 4. Determining Number of Heads (`n_heads`)
**Goal**: Determine how many representation subspaces (heads) are needed to track the different cipher dependencies (e.g., absolute position, local block dependencies).

**Experimental Procedure**:
1. Using the established `d_model` (e.g., 512), test different numbers of heads: **4, 8, and 16**.
2. Note that the head dimension will scale inversely: `d_k = d_model / n_heads` (e.g., 128, 64, and 32 respectively).
3. Train each variant on a 10% subset of the data for 1 epoch.
4. Compare the validation loss and training speed.
5. **Decision**: Choose the `n_heads` that provides the lowest validation loss. If results are identical, prefer **8 heads** as it strikes a standard balance between representation splitting and computational efficiency.

## Next Steps
Once we execute the data script to find `max_seq_len`, we can begin building the architecture and running the subset sweeps for the other parameters!
