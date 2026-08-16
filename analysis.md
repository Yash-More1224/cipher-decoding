# Cipher Encoding Analysis

Based on our initial data analysis of the `brown_plain.txt` and `brown_cipher.txt` dataset, we have discovered several key properties of the encryption scheme. Understanding these properties will be crucial for designing a Sequence-to-Sequence Transformer that can effectively model the mapping.

## 1. Fixed-Length 8-Bit Encoding
* **Observation**: The total number of bits ('0' and '1' characters) in the ciphertext is exactly 8 times the number of non-newline characters in the plaintext.
* **Conclusion**: The cipher encodes each plaintext character into exactly **8 bits**. It is a fixed-length encoding, not a variable-length encoding (like Huffman coding).

## 2. Newline Preservation
* **Observation**: Both the plaintext and ciphertext files have exactly the same number of lines (5001 lines). 
* **Conclusion**: The newline character (`\n`) is **not** encrypted into an 8-bit sequence. It is preserved as a physical newline in the ciphertext. When loading the data into the Transformer, newlines will need to be handled separately or stripped out to avoid breaking the 8-bit alignment.

## 3. Not a Simple Substitution Cipher
* **Observation**: When analyzing the mapping, almost every character in the alphabet (e.g., 'A', 'e', ' ', 't') encrypts to multiple, completely different 8-bit sequences across the dataset.
* **Conclusion**: The encryption is **not a monoalphabetic substitution cipher**. The model cannot simply learn a static dictionary mapping of `Character -> 8-bits`.

## 4. Context and Position Dependency
* **Observation**: Furthermore, the analysis script revealed that **117 different 8-bit sequences map to multiple different characters**. For example, the same sequence `01011010` might decrypt to an 'e' in one place and an 'm' in another.
* **Conclusion**: The cipher is highly **context-dependent or position-dependent**. It is likely a polyalphabetic cipher (e.g., Vigenère), a stream cipher (XOR with a pseudorandom stream), or a block cipher operating in a chaining mode (e.g., AES-CBC). 

## Impact on Transformer Design
Because the encryption relies on context/position, our Transformer architecture must be designed to capture these dependencies:
* **Positional Encodings** will be absolutely critical. A sequence-to-sequence model must know exactly where a character is located in the sequence to reverse the positional shifts of the cipher. We should pay close attention to how **RoPE** compares to standard Sinusoidal encodings during the ablation study (C1 vs C2).
* **Attention Mechanisms** (Self-Attention and Cross-Attention) will need to be powerful enough to learn the contextual dependencies and potential repeating key lengths. 
* **Tokenization**: Since it is an 8-bit fixed encoding, byte-level processing makes perfect sense. Evaluating the **Token-Free (BLT)** architecture (C5) will be particularly interesting, as processing raw bytes might align naturally with the 8-bit structure of the ciphertext without relying on subword boundaries that are meaningless for encrypted data.
