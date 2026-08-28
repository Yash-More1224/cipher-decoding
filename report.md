# Model C1 - Experimentation & Training Log

The following is a chronological record of the experiments, optimizations, and hyperparameter tuning performed to train the C1 model (standard BPE-based Transformer) on the encrypted dataset.

## 1. Initial Dataset Analysis
We first ran experiments to analyze the dataset, documenting our findings in `analysis.md`. Our analysis showed:
- The cipher uses a fixed-length **8-bit encoding** per character.
- The encryption is **position/context-dependent** (not a simple monoalphabetic substitution).
- The maximum length of a line in the dataset is 1366 characters. Based on this, we set the initial `max_seq_len` to **1370** (1366 characters + 4 special tokens). 
- Based on the fixed 8-bit encoding, we noted that preserving positional information and using byte-level or bit-level processing would be crucial.

## 2. First Iteration: Large Model (BPE on Raw Bits)
Based on the initial analysis, we ran our first model using standard BPE tokenization on both the cipher text (raw bits) and plain text.
- **Architecture**: 51M parameters, hidden dimension = 512, feed-forward dimension (`d_ff`) = 2048, layers = 6.
- **Result**: Due to the massive size of the model and the long sequence length, training took an excessively long time.

## 3. Second Iteration: Smaller Model
To reduce training time, we scaled down the model dimensions.
- **Architecture**: Hidden dimension reduced to 256, `d_ff` reduced to 1024.
- **Result**: Training was significantly faster, but the lowest validation loss achieved was higher than the 51M parameter variant, indicating underfitting or lack of capacity.

## 4. Third Iteration: Large Model with Fewer Layers
We attempted to find a middle ground by using the larger hidden dimensions but reducing depth.
- **Architecture**: Hidden dimension = 512, `d_ff` = 2048, but reduced to 4 layers instead of 6.
- **Result**: Training was faster than the 6-layer 51M model, but the validation loss was still not much lower.

## 5. Sequence Length Reduction & Fixing Learning Rate
We hypothesized that the long sequences (1370 tokens) were bottlenecking the model, so we reduced the input sequence length to 64 tokens, and then 256 tokens, while keeping the model size at ~14M parameters.
- **Result**: The loss stayed higher than when feeding the full sequence. This was unexpected, as shorter sequences should theoretically be easier to memorize.
- **Root Cause**: Upon closer inspection, we discovered the learning rate was too low (`1e-4`).

## 6. Learning Rate Optimization
We fixed the learning rate to `3e-4` and then to `1e-3`.
- **Result**: This immediately yielded lower loss values and lead to much faster convergence.

## 7. BPE on Byte-Values (Integers) vs Raw Bits
We reconsidered how BPE was applied to the ciphertext. Since each plaintext character maps to exactly 8 bits, applying BPE to a raw string of '0's and '1's was inefficient.
- **Change**: We converted the 8-bit strings into numbers (between 0 and 255) and then applied the BPE merges to these byte-values.
- **Result**: This representation significantly improved training efficiency and performance.

## 8. Hardware & System Optimizations
To further speed up training and maximize GPU utilization, we implemented several system-level optimizations:
- **Distributed Data Parallel (DDP)**: Enabled multi-GPU training across 4 GPUs using `torchrun`.
- **Automatic Mixed Precision (AMP)**: Used `torch.autocast` (FP16) to speed up tensor operations and reduce VRAM footprint.
- **Gradient Accumulation**: Allowed us to simulate a much larger effective batch size without running out of memory.
- **Data Loading**: Enabled `pin_memory=True` and optimized `num_workers` to prevent the dataloader from bottlenecking the GPUs.
- **Cosine Annealing Scheduler**: Implemented a linear warmup phase followed by cosine decay for optimal learning rate pacing.

## 9. Rethinking Tokenization on Ciphertext
We realized a fundamental flaw in applying BPE to the ciphertext: because the cipher is highly position-dependent (many-to-many mapping), applying BPE merges strips away critical positional alignment information. BPE merges frequently occurring sequences, but in this cipher, the frequency and meaning of an 8-bit chunk are heavily influenced by its absolute position. By merging chunks, we were providing a very poor signal for the model to learn from.

## 10. Byte-Level Integer Encoding (No BPE on Cipher)
Instead of BPE, we fed integers (0 to 255) directly mapped from the 8-bit chunks into the model.
- **Setup**: We initially tested this by treating every full sentence in the dataset as a single training example.
- **Result**: The shift away from BPE on the ciphertext provided excellent gains. The validation loss reached a new low of `5.36`, but the model began to overfit (test loss kept decreasing while validation loss increased).

## 11. Chunking for Data Augmentation
The overfitting was primarily due to the small size of the dataset (only ~4,000 training examples). 
- **Change**: To artificially expand the dataset and improve generalization, we applied **fixed-size chunking**. We segmented the training examples into sliding windows of 64 bytes (mapping exactly to 64 characters of plaintext). We carefully tracked the absolute byte-offset of each chunk to ensure the Sinusoidal Positional Encoding remained accurate.
- **Result**: This increased the number of training samples from ~4,000 to nearly 50,000, significantly delaying the onset of overfitting.

## 12. Aggressive Regularization
To further combat overfitting and push the validation loss down, we applied aggressive regularization to the hyperparameters:
- Increased **Dropout** from `0.1` to `0.3`.
- Increased **Weight Decay** from `0.01` to `0.05`.
- Increased **Label Smoothing** from `0.1` to `0.2` (preventing overconfident predictions).
- Reduced peak **Learning Rate** from `3e-3` to `1.5e-3` to stabilize the final descent.

## 13. Final Result
Combining the absolute byte-offset chunking mechanism with aggressive regularization and the optimized hyperparameter suite proved to be the optimal training setup for the C1 architecture. 
- **Final Result**: We achieved the lowest development loss yet: **val_loss = 0.1361**.

## 14. Final Model Comparison (C1 - C5)

We trained and evaluated all five configurations on the encrypted dataset. The final evaluation metrics across all models are summarized below. Note that C5 (BLT) operates natively at the byte/character level and does not compute word-level BLEU/ROUGE metrics in this pipeline.

| Metric | C1 | C2 | C3 | C4 | C5 (BLT) |
|---|---|---|---|---|---|
| **val_loss** | 0.1361 | 0.1421 | 0.1354 | 0.1734 | **0.0880** |
| **bit_accuracy** | 0.8043 | 0.7935 | 0.9565 | 0.7935 | **0.9969** |
| **seq_accuracy** | 0.6000 | 0.6000 | 0.8000 | 0.6000 | **0.8000** |
| **levenshtein** | 3.4000 | 4.0000 | 12.0000 | 4.0000 | **0.4000** |
| **bleu** | **0.9504** | 0.9461 | 0.8872 | 0.9461 | N/A |
| **rouge1** | **0.9652** | 0.9565 | 0.9056 | 0.9565 | N/A |
| **rouge2** | **0.9545** | 0.9455 | 0.8914 | 0.9455 | N/A |
| **rougeL** | **0.9652** | 0.9565 | 0.9056 | 0.9565 | N/A |

### Observations:
- **C5 (Byte-Level Transformer)** vastly outperformed the other models, achieving the lowest validation loss (`0.0880`), the highest bit-level accuracy (`99.69%`), and the best Levenshtein distance (`0.4000`). This proves that treating the encryption cipher at the byte/character level preserves the critical sequence positioning information necessary for decryption.
- **C3** showed strong sequence accuracy (`80%`) and bit accuracy (`95.65%`), but had a high Levenshtein distance (`12.0`) and lower BLEU/ROUGE scores compared to C1, C2, and C4, indicating that when it makes errors, the token-level structure diverges heavily.
- **C1, C2, and C4** performed similarly, with C1 holding a slight edge in generation metrics (BLEU/ROUGE) and C4 performing the worst with the highest validation loss.
