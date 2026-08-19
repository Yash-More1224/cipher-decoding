"""
dataset.py — Dataset loaders for Assignment 1.

Two dataset classes:
  EncryptedDataset   — BPE-tokenized (plaintext) + Byte-Level (ciphertext) loader for C1–C4.
  TokenFreeDataset   — Raw-byte loader for config C5 (BLT).

Both classes expose a `split()` class-method that returns
(train_dataset, val_dataset, test_dataset) with an 80 / 10 / 10 split.
The split is deterministic (seeded shuffle) so results are reproducible.

Cipher Byte-Level Encoding
───────────────────────────
Each cipher line is a binary string (e.g. "01001000..."). We group bits into
8-bit chunks and interpret each chunk as an integer (0–255). These integers
become token IDs directly — no BPE merging. Special tokens are placed above
the byte range:
  PAD_ID_CIPHER = 256
  BOS_ID_CIPHER = 257
  EOS_ID_CIPHER = 258
Total cipher vocabulary size: 259
"""


import os
import random

import torch
from torch.utils.data import Dataset, Subset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split_indices(n: int, seed: int = 42):
    """
    Returns (train_idx, val_idx, test_idx) for an 80/10/10 split.
    The indices are shuffled deterministically using `seed`.
    """
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    # test gets the remainder so rounding never drops samples
    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


# ──────────────────────────────────────────────────────────────────────────────
# BPE-Tokenized Dataset  (C1 – C4)
# ──────────────────────────────────────────────────────────────────────────────

class EncryptedDataset(Dataset):
    """
    Loads brown_plain.txt + brown_cipher.txt, tokenizes each line and
    pads to max_seq_len.

    When chunk_size > 0, each line is segmented into non-overlapping
    chunks of `chunk_size` characters (plaintext) / bytes (ciphertext).
    This multiplies the training set by ~(line_length / chunk_size) and
    dramatically reduces sequence length, addressing overfitting on small
    datasets.

    Each chunk carries its absolute byte-offset in the original line so
    the encoder's sinusoidal positional encoding starts at the correct
    position (critical for position-dependent ciphers).

    Returns (cipher_tensor, plain_tensor, offset) triples where:
      cipher_tensor : (chunk_size+2,)  — encoder input (BOS + bytes + EOS)
      plain_tensor  : (max_seq_len,)   — decoder input/target (BOS + BPE + EOS)
      offset        : int              — byte offset of this chunk in the original line
    """

    def __init__(
        self,
        plain_file: str = "brown_plain.txt",
        cipher_file: str = "brown_cipher.txt",
        max_seq_len: int = 1366,
        chunk_size: int = 0,           # 0 = no chunking (original behaviour)
        _lines: tuple | None = None,   # internal use by split()
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        outputs_dir = os.path.join(base_dir, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        plain_tok_path  = os.path.join(outputs_dir, "plain_tokenizer.json")
        cipher_tok_path = os.path.join(outputs_dir, "cipher_tokenizer.json")

        plain_path  = os.path.join(base_dir, "Dataset_A1", plain_file)
        cipher_path = os.path.join(base_dir, "Dataset_A1", cipher_file)

        if _lines is not None:
            self.plain_lines, self.cipher_lines = _lines
        else:
            with open(plain_path,  "r", encoding="utf-8") as f:
                self.plain_lines = f.readlines()
            with open(cipher_path, "r", encoding="utf-8") as f:
                self.cipher_lines = f.readlines()

        assert len(self.plain_lines) == len(self.cipher_lines), \
            "Mismatched line counts between plaintext and ciphertext files."

        # ── Plaintext tokenizer (BPE) ──────────────────────────────────────────
        if not os.path.exists(plain_tok_path):
            print("Plain tokenizer not found in outputs/. Training now...")
            with open(plain_path, 'r', encoding='utf-8') as f:
                plain_full = f.readlines()
            plain_tok = Tokenizer(BPE(unk_token="[UNK]"))
            plain_tok.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"], vocab_size=5000)
            plain_tok.train_from_iterator(plain_full, trainer=trainer)
            plain_tok.save(plain_tok_path)
            print("Finished training and saving plain tokenizer to outputs/.")

        # ── Cipher byte-level encoding (no file needed) ───────────────────────
        # Each 8-bit chunk → integer 0-255; special tokens above byte range.
        self.PAD_ID_CIPHER = 256
        self.BOS_ID_CIPHER = 257
        self.EOS_ID_CIPHER = 258
        CIPHER_VOCAB_SIZE  = 259   # 256 bytes + PAD + BOS + EOS

        self.plain_tokenizer  = Tokenizer.from_file(plain_tok_path)
        self.max_seq_len = max_seq_len
        self.chunk_size  = chunk_size

        self.plain_vocab_size  = self.plain_tokenizer.get_vocab_size()
        self.cipher_vocab_size = CIPHER_VOCAB_SIZE

        self.bos_id_plain  = self.plain_tokenizer.token_to_id("[BOS]")
        self.eos_id_plain  = self.plain_tokenizer.token_to_id("[EOS]")
        self.pad_id_plain  = self.plain_tokenizer.token_to_id("[PAD]")
        self.bos_id_cipher = self.BOS_ID_CIPHER
        self.eos_id_cipher = self.EOS_ID_CIPHER
        self.pad_id_cipher = self.PAD_ID_CIPHER

        # ── Pre-compute all (cipher_bytes, plain_str, offset) chunks ──────────
        # Each element: (cipher_byte_list, plaintext_substring, byte_offset)
        self._samples: list[tuple[list[int], str, int]] = []
        for plain_line, cipher_line in zip(self.plain_lines, self.cipher_lines):
            plain_line  = plain_line.strip()
            cipher_line = cipher_line.strip()
            all_cipher_bytes = self._bits_to_byte_ids(cipher_line)
            # Verify 1:1 alignment (each char → 1 cipher byte)
            n = min(len(plain_line), len(all_cipher_bytes))
            if chunk_size > 0:
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    self._samples.append((
                        all_cipher_bytes[start:end],
                        plain_line[start:end],
                        start,                    # absolute byte offset
                    ))
            else:
                self._samples.append((all_cipher_bytes, plain_line, 0))

    # ------------------------------------------------------------------
    # 80 / 10 / 10 split  (class-method)
    # ------------------------------------------------------------------

    @classmethod
    def split(
        cls,
        plain_file: str = "brown_plain.txt",
        cipher_file: str = "brown_cipher.txt",
        max_seq_len: int = 1366,
        chunk_size: int = 0,
        seed: int = 42,
    ):
        """
        Returns (train_ds, val_ds, test_ds) — each is an EncryptedDataset
        sharing the same tokenizers but holding only its partition's lines.
        """
        # Load all lines once
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        with open(os.path.join(base_dir, "Dataset_A1", plain_file),  "r", encoding="utf-8") as f:
            all_plain  = f.readlines()
        with open(os.path.join(base_dir, "Dataset_A1", cipher_file), "r", encoding="utf-8") as f:
            all_cipher = f.readlines()

        assert len(all_plain) == len(all_cipher)
        train_idx, val_idx, test_idx = _split_indices(len(all_plain), seed)

        def _make(idx):
            plain_subset  = [all_plain[i]  for i in idx]
            cipher_subset = [all_cipher[i] for i in idx]
            return cls(max_seq_len=max_seq_len, chunk_size=chunk_size,
                       _lines=(plain_subset, cipher_subset))

        return _make(train_idx), _make(val_idx), _make(test_idx)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def _pad_sequence(
        self,
        token_ids: list,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        max_len: int | None = None,
    ) -> torch.Tensor:
        L = max_len if max_len is not None else self.max_seq_len
        token_ids = token_ids[: L - 2]                  # truncate if needed
        sequence  = [bos_id] + token_ids + [eos_id]
        sequence += [pad_id] * (L - len(sequence))
        return torch.tensor(sequence, dtype=torch.long)

    def _bits_to_byte_ids(self, bit_string: str) -> list:
        """Convert a binary string to a list of byte integer IDs (0-255)."""
        ids = []
        for i in range(0, len(bit_string), 8):
            chunk = bit_string[i:i+8]
            if len(chunk) == 8:
                ids.append(int(chunk, 2))
        return ids

    def __getitem__(self, idx: int):
        cipher_ids, plain_str, offset = self._samples[idx]

        # Encoder: byte-level cipher tokens. Pad to chunk_size+2 (BOS+EOS)
        # or max_seq_len when not chunking.
        cipher_max = (self.chunk_size + 2) if self.chunk_size > 0 else self.max_seq_len
        cipher_tensor = self._pad_sequence(
            cipher_ids, self.bos_id_cipher, self.eos_id_cipher, self.pad_id_cipher,
            max_len=cipher_max,
        )

        # Decoder: BPE-tokenize just this character substring.
        plain_ids     = self.plain_tokenizer.encode(plain_str).ids
        plain_tensor  = self._pad_sequence(
            plain_ids, self.bos_id_plain, self.eos_id_plain, self.pad_id_plain,
        )

        return cipher_tensor, plain_tensor, offset


# ──────────────────────────────────────────────────────────────────────────────
# Token-Free Dataset  (C5 — BLT)
# ──────────────────────────────────────────────────────────────────────────────

class TokenFreeDataset(Dataset):
    """
    Raw-byte loader for the Byte Latent Transformer (C5).

    Encoding:
      Special tokens:  PAD=0, BOS=1, EOS=2
      Plaintext bytes: UTF-8 byte value b → ID (b + 3)
      Ciphertext bytes: each '0'/'1' ASCII character → ord('0'/'1') + 3 = 51 / 52
                        (no tokenization — pure byte stream)

    Vocabulary size: 259  (256 raw bytes + 3 special tokens)

    Sequence lengths:
      Ciphertext: each line is ~6400 '0'/'1' characters → max_src_len = 6402
      Plaintext:  each line is ≤800 UTF-8 characters    → max_tgt_len = 802
    """

    # Special token IDs
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    VOCAB_SIZE = 259   # 256 byte values + 3 special tokens

    def __init__(
        self,
        plain_file: str = "brown_plain.txt",
        cipher_file: str = "brown_cipher.txt",
        max_src_len: int = 6402,   # cipher: ~6400 bits/chars + BOS/EOS
        max_tgt_len: int = 802,    # plain:  ≤800 UTF-8 bytes  + BOS/EOS
        _lines: tuple | None = None,
    ):
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # Expose these so train.py can reference them uniformly
        self.pad_id_plain  = self.PAD_ID
        self.pad_id_cipher = self.PAD_ID
        self.plain_vocab_size  = self.VOCAB_SIZE
        self.cipher_vocab_size = self.VOCAB_SIZE

        if _lines is not None:
            self.plain_lines, self.cipher_lines = _lines
        else:
            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            with open(os.path.join(base_dir, "Dataset_A1", plain_file),  "r", encoding="utf-8") as f:
                self.plain_lines = f.readlines()
            with open(os.path.join(base_dir, "Dataset_A1", cipher_file), "r", encoding="utf-8") as f:
                self.cipher_lines = f.readlines()

        assert len(self.plain_lines) == len(self.cipher_lines)

    # ------------------------------------------------------------------
    # 80 / 10 / 10 split  (class-method)
    # ------------------------------------------------------------------

    @classmethod
    def split(
        cls,
        plain_file: str = "brown_plain.txt",
        cipher_file: str = "brown_cipher.txt",
        max_src_len: int = 6402,
        max_tgt_len: int = 802,
        seed: int = 42,
    ):
        """Returns (train_ds, val_ds, test_ds)."""
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        with open(os.path.join(base_dir, "Dataset_A1", plain_file),  "r", encoding="utf-8") as f:
            all_plain  = f.readlines()
        with open(os.path.join(base_dir, "Dataset_A1", cipher_file), "r", encoding="utf-8") as f:
            all_cipher = f.readlines()

        assert len(all_plain) == len(all_cipher)
        train_idx, val_idx, test_idx = _split_indices(len(all_plain), seed)

        def _make(idx):
            return cls(
                max_src_len=max_src_len,
                max_tgt_len=max_tgt_len,
                _lines=([all_plain[i] for i in idx], [all_cipher[i] for i in idx]),
            )

        return _make(train_idx), _make(val_idx), _make(test_idx)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_plain(self, text: str) -> list[int]:
        """Encode plaintext as UTF-8 bytes, shifted by +3."""
        return [b + 3 for b in text.encode("utf-8")]

    def _encode_cipher(self, text: str) -> list[int]:
        """
        Encode ciphertext ('0'/'1' characters) as raw ASCII bytes shifted by +3.
        '0' → ord('0') + 3 = 51
        '1' → ord('1') + 3 = 52
        This preserves the 2-symbol byte stream without any tokenization.
        """
        return [ord(c) + 3 for c in text]

    def _pad(self, ids: list[int], max_len: int) -> torch.Tensor:
        ids = ids[: max_len - 2]                              # truncate
        sequence = [self.BOS_ID] + ids + [self.EOS_ID]
        sequence += [self.PAD_ID] * (max_len - len(sequence))
        return torch.tensor(sequence, dtype=torch.long)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.plain_lines)

    def __getitem__(self, idx: int):
        plain_line  = self.plain_lines[idx].strip()
        cipher_line = self.cipher_lines[idx].strip()

        plain_ids  = self._encode_plain(plain_line)
        cipher_ids = self._encode_cipher(cipher_line)

        plain_tensor  = self._pad(plain_ids,  self.max_tgt_len)
        cipher_tensor = self._pad(cipher_ids, self.max_src_len)

        return cipher_tensor, plain_tensor
