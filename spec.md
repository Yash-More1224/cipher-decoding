# Base Transformer (C1) Architectural Specification

This specification outlines the design choices for the **C1 Baseline Model** to maximize its ability to learn the complex, context-dependent encryption scheme identified in our analysis.

## 1. Sequence Processing Strategy (Handling Newlines)
* **The Challenge**: We found that newlines (`\n`) are preserved physically in both files rather than being encrypted into 8-bit blocks.
* **Architectural Decision (Line-by-Line)**: We will structure our PyTorch `Dataset` to process the text **line-by-line** rather than as one massive continuous string. 
* **Mechanics**: Each line of ciphertext will act as the Encoder's source sequence, and the corresponding plaintext line as the Decoder's target sequence. We will pad the lines to a maximum length and use `<BOS>` and `<EOS>` tokens. This entirely avoids the 8-bit alignment desync issue caused by the raw newlines.

## 2. Tokenization (Standard Subword)
* **The Challenge**: The assignment requires C1 to use a **Standard Subword** tokenizer (e.g., BPE via HuggingFace's `tokenizers`). While this is perfect for the English plaintext, applying BPE to the ciphertext string of `'0'`s and `'1'`s will group the bits into chunks of irregular lengths (e.g., `'010'`, `'110011'`, etc.), completely destroying the explicit 8-bit boundaries.
* **Architectural Decision**: Because the subword tokens will lack strict 8-bit alignment, the Transformer's Encoder has to do the heavy lifting of implicitly re-aligning the bits. We must ensure the **hidden dimension (`d_model`)** and **Feed-Forward network size (`d_ff`)** are sufficiently large (e.g., `d_model=512`, `d_ff=2048`) to give the network enough capacity to memorize and map these irregular bit-chunks into meaningful latent representations.

## 3. Positional Encoding (Sinusoidal Absolute)
* **The Challenge**: As discovered, the cipher is highly position/context-dependent (likely a stream cipher or polyalphabetic cipher). The model *must* know exactly where it is in the sequence to reverse the cipher.
* **Architectural Decision**: C1 uses Sinusoidal Absolute Positional Encodings. 
  * We must configure a large `max_seq_len`. A typical line is ~800 plaintext characters (~6400 bits). Even with subword compression, the Encoder will see very long sequences. We should set the positional encoding `max_seq_len` to at least **2048 or 4096**.
  * The absolute positions are actually beneficial here: if the cipher uses a repeating key (like Vigenère) or a position-based keystream, providing absolute sequence positions directly helps the model infer the current state of the cipher's key.

## 4. Multi-Head Attention (MHA)
* **The Challenge**: The cross-attention mechanism needs to map irregular subword ciphertext tokens to plaintext subwords. The Encoder self-attention needs to gather local context (if the cipher is a block cipher like CBC, decrypting block $i$ requires looking at block $i-1$).
* **Architectural Decision**: 
  * **Encoder Self-Attention**: Will be fully bidirectional (no causal mask) so the model can look at adjacent cipher tokens to resolve dependencies.
  * **Number of Heads**: We should use at least **8 heads**. Because the cipher scrambles information, the model needs multiple representation subspaces. For example, one head might learn to track bit-alignment, another might track the absolute position modulo the key length, and a third might look at the previous cipher block for CBC decryption.

## 5. Normalization (Pre-LayerNorm)
* **The Challenge**: Training a transformer from scratch on pseudo-random ciphertext is notoriously difficult because the gradients can be extremely noisy (the ciphertext has high entropy).
* **Architectural Decision**: C1 requires standard `LayerNorm`. We will implement this as **Pre-LayerNorm** (where the normalization is applied *before* the MHA and FFN blocks, rather than after). Pre-LayerNorm drastically improves gradient flow and training stability for deep transformers compared to the original Post-LayerNorm formulation in the "Attention is All You Need" paper.
