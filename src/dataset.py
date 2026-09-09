"""
dataset.py — Dataset loaders for Assignment 1.

Two dataset classes:
  EncryptedDataset   — BPE on BOTH sides (plaintext and ciphertext) for C1-C4.
  TokenFreeDataset   — Raw-byte loader for config C5 (BLT).

Both expose a `split()` class-method returning (train, val, test) under a
deterministic seeded 80/10/10 split, so results are reproducible.

Ciphertext tokenization (C1-C4)
───────────────────────────────
The ciphertext is tokenized with a from-scratch BPE tokenizer (src/tokenizer.py;
no tokenizer library is used). BPE is learned over a *byte-symbol* stream: each
8-bit block of the binary string becomes one symbol from a 256-symbol alphabet,
and merges are learned over those symbols. Running BPE on the raw '0'/'1'
string instead would let merges straddle 8-bit boundaries and destroy the byte
alignment the cipher depends on.

There is no manual chunking here. Each line is one training example, tokenized
end to end, and `collate_encrypted` pads only to the longest sequence in the
batch (see LengthGroupedSampler) rather than to a global maximum.

Note that BPE breaks the exact correspondence between token index and absolute
byte offset that byte-grouped tokenization had: one token now covers a variable
number of cipher bytes. Since this cipher is position-dependent, the model has
to infer byte position from content. C5 keeps one token per bit and so keeps
that alignment exactly.
"""

import os
import random

import torch
from torch.utils.data import Dataset, Sampler

try:                                    # imported as `dataset` (src/ on path)
    from tokenizer import BPETokenizer, PAD_ID, BOS_ID, EOS_ID
except ImportError:                     # imported as `src.dataset`
    from src.tokenizer import BPETokenizer, PAD_ID, BOS_ID, EOS_ID

# Tokenizer artefacts, produced once by src/build_tokenizers.py
_OUTPUTS_DIR    = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "outputs")
)
PLAIN_TOK_PATH  = os.path.join(_OUTPUTS_DIR, "plain_bpe.json")
CIPHER_TOK_PATH = os.path.join(_OUTPUTS_DIR, "cipher_bpe.json")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split_indices(n: int, seed: int = 42):
    """(train_idx, val_idx, test_idx) for a deterministic 80/10/10 split."""
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


# ──────────────────────────────────────────────────────────────────────────────
# BPE Dataset  (C1 - C4)
# ──────────────────────────────────────────────────────────────────────────────

class EncryptedDataset(Dataset):
    """
    Returns (cipher_tokens, plain_tokens, offset), unpadded; `collate_encrypted`
    pads each batch to its own longest member.

    Two modes:

    chunk_size == 0 (full-line)
        One example per line. BPE segments the whole line, so a token spans a
        variable number of cipher bytes and the model must infer absolute byte
        position by counting over ~300 tokens. `offset` is always 0.

    chunk_size > 0 (windowed)
        Each line is cut into non-overlapping windows of `chunk_size`
        characters, and BPE is applied to each window independently. `offset`
        is the window's absolute CHARACTER offset in the line, which is fed to
        the positional encoding, so the model is told where the window starts
        and only has to resolve position across the ~30 tokens inside it.
        Windowing is what makes the byte-offset problem tractable; see the
        report's Section 3.
    """

    def __init__(
        self,
        plain_file: str = "brown_plain.txt",
        cipher_file: str = "brown_cipher.txt",
        max_seq_len: int = 1408,
        chunk_size: int = 0,
        _lines: tuple | None = None,
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        for p in (PLAIN_TOK_PATH, CIPHER_TOK_PATH):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Tokenizer not found: {p}\n"
                    "Run `python src/build_tokenizers.py` once before training."
                )
        self.plain_tokenizer  = BPETokenizer.load(PLAIN_TOK_PATH)
        self.cipher_tokenizer = BPETokenizer.load(CIPHER_TOK_PATH)

        if _lines is not None:
            self.plain_lines, self.cipher_lines = _lines
        else:
            with open(os.path.join(base_dir, "Dataset_A1", plain_file), encoding="utf-8") as f:
                self.plain_lines = f.readlines()
            with open(os.path.join(base_dir, "Dataset_A1", cipher_file), encoding="utf-8") as f:
                self.cipher_lines = f.readlines()

        assert len(self.plain_lines) == len(self.cipher_lines), \
            "Mismatched line counts between plaintext and ciphertext files."

        self.max_seq_len = max_seq_len
        self.chunk_size  = chunk_size

        self.plain_vocab_size  = self.plain_tokenizer.get_vocab_size()
        self.cipher_vocab_size = self.cipher_tokenizer.get_vocab_size()
        self.bos_id_plain  = self.bos_id_cipher = BOS_ID
        self.eos_id_plain  = self.eos_id_cipher = EOS_ID
        self.pad_id_plain  = self.pad_id_cipher = PAD_ID

        # Encode ONCE here, not per __getitem__. The tokenizer is pure Python,
        # so re-encoding every epoch would dominate training time.
        body = max_seq_len - 2                       # room for BOS/EOS
        self._samples: list[tuple[list[int], list[int], int]] = []
        self.plain_texts: list[str] = []
        self.sample_line_id: list[int] = []
        self.sample_start:   list[int] = []

        for line_id, (plain_line, cipher_line) in enumerate(
            zip(self.plain_lines, self.cipher_lines)
        ):
            plain_line, cipher_line = plain_line.strip(), cipher_line.strip()
            self.plain_texts.append(plain_line)

            if chunk_size > 0:
                # One cipher byte per plaintext character, 8 bits per byte.
                n = min(len(plain_line), len(cipher_line) // 8)
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    cw = self.cipher_tokenizer.encode(cipher_line[start * 8 : end * 8])[:body]
                    pw = self.plain_tokenizer.encode(plain_line[start:end])[:body]
                    self._samples.append(
                        ([BOS_ID] + cw + [EOS_ID], [BOS_ID] + pw + [EOS_ID], start)
                    )
                    self.sample_line_id.append(line_id)
                    self.sample_start.append(start)
            else:
                cf = self.cipher_tokenizer.encode(cipher_line)[:body]
                pf = self.plain_tokenizer.encode(plain_line)[:body]
                self._samples.append(
                    ([BOS_ID] + cf + [EOS_ID], [BOS_ID] + pf + [EOS_ID], 0)
                )
                self.sample_line_id.append(line_id)
                self.sample_start.append(0)

    # ------------------------------------------------------------------
    @classmethod
    def split(
        cls,
        plain_file: str = "brown_plain.txt",
        cipher_file: str = "brown_cipher.txt",
        max_seq_len: int = 1408,
        chunk_size: int = 0,
        seed: int = 42,
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(base_dir, "Dataset_A1", plain_file), encoding="utf-8") as f:
            all_plain = f.readlines()
        with open(os.path.join(base_dir, "Dataset_A1", cipher_file), encoding="utf-8") as f:
            all_cipher = f.readlines()

        assert len(all_plain) == len(all_cipher)
        train_idx, val_idx, test_idx = _split_indices(len(all_plain), seed)

        def _make(idx):
            return cls(
                max_seq_len=max_seq_len, chunk_size=chunk_size,
                _lines=([all_plain[i] for i in idx], [all_cipher[i] for i in idx]),
            )

        return _make(train_idx), _make(val_idx), _make(test_idx)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    def source_length(self, idx: int) -> int:
        """Encoder length of example `idx`, used by the length-aware samplers."""
        return len(self._samples[idx][0])

    def sample_offset(self, idx: int) -> int:
        """Absolute character offset of this example's window (0 if unwindowed)."""
        return self._samples[idx][2]

    def cost_length(self, idx: int) -> int:
        """
        Length used for memory budgeting: the larger of the encoder and decoder
        sequence, since attention memory scales with the square of whichever
        dominates.
        """
        src, tgt, _ = self._samples[idx]
        return max(len(src), len(tgt))

    def __getitem__(self, idx: int):
        cipher_ids, plain_ids, offset = self._samples[idx]
        return (
            torch.tensor(cipher_ids, dtype=torch.long),
            torch.tensor(plain_ids,  dtype=torch.long),
            offset,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic padding + length bucketing
# ──────────────────────────────────────────────────────────────────────────────

def collate_encrypted(batch, pad_id: int = PAD_ID):
    """
    Pad a batch to its own longest source and target, not to a global maximum.

    Cipher sequences average ~300 tokens but reach ~1,400, so padding everything
    to the global maximum would waste roughly 4x the compute and memory. Paired
    with LengthGroupedSampler (which puts similar-length lines in the same
    batch) the padding overhead is small.
    """
    srcs, tgts, offsets = zip(*batch)
    max_src = max(s.size(0) for s in srcs)
    max_tgt = max(t.size(0) for t in tgts)

    src_out = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    tgt_out = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_out[i, : s.size(0)] = s
        tgt_out[i, : t.size(0)] = t

    return src_out, tgt_out, torch.tensor(offsets, dtype=torch.long)


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """
    Yields batches holding at most `max_tokens` padded positions, instead of a
    fixed number of sequences.

    A fixed sequence count cannot work here. Attention memory grows with
    batch * length^2, and grouping by length (which we want, to keep padding
    tight) guarantees that one batch collects all the longest lines together.
    At 16 sequences of ~1,383 tokens a single encoder attention matrix is about
    490 MB in FP16, and with four encoder layers plus decoder self- and
    cross-attention retained for backward that batch alone exhausts an 11 GB
    card, even though the average line is only ~296 tokens.

    Budgeting by tokens makes batch size adapt to length automatically: short
    lines batch wide, the rare long ones batch narrow, and peak memory stays
    roughly constant across batches.
    """

    def __init__(self, dataset, max_tokens: int = 4096, max_batch: int = 64,
                 shuffle: bool = True, seed: int = 42, pool_factor: int = 4096,
                 group_by_offset: bool = False):
        self.lengths    = [dataset.cost_length(i) for i in range(len(dataset))]
        # In windowed mode every example carries its own absolute offset, but
        # train.py feeds the model ONE scalar offset per batch (offsets.min()).
        # That is only correct if every example in a batch shares an offset, so
        # batches are partitioned by offset here. Doing it in the sampler avoids
        # rewriting the positional encoding to take per-sample offsets, and it
        # is exact rather than approximate.
        self.group_by_offset = group_by_offset
        self.offsets = ([dataset.sample_offset(i) for i in range(len(dataset))]
                        if group_by_offset else None)
        self.max_tokens = max_tokens
        self.max_batch  = max_batch
        self.shuffle    = shuffle
        self.seed       = seed
        self.epoch      = 0
        self._cached_batches: list[list[int]] | None = None
        self.pool_size  = max(pool_factor, max_batch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._cached_batches = None

    def _build(self) -> list[list[int]]:
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(idx)

        if self.group_by_offset:
            by_off: dict[int, list[int]] = {}
            for i in idx:
                by_off.setdefault(self.offsets[i], []).append(i)
            groups = list(by_off.values())
        else:
            groups = [idx]

        pools = []
        for g in groups:
            if self.shuffle:
                pools += [g[i : i + self.pool_size]
                          for i in range(0, len(g), self.pool_size)]
            else:
                pools.append(g)

        batches: list[list[int]] = []
        for pool in pools:
            pool.sort(key=lambda i: self.lengths[i])
            cur: list[int] = []
            cur_max = 0
            for i in pool:
                cand_max = max(cur_max, self.lengths[i])
                # Would adding this sequence exceed the budget?
                if cur and ((len(cur) + 1) * cand_max > self.max_tokens
                            or len(cur) + 1 > self.max_batch):
                    batches.append(cur)
                    cur, cur_max = [i], self.lengths[i]
                else:
                    cur.append(i)
                    cur_max = cand_max
            if cur:
                batches.append(cur)

        if self.shuffle:
            random.Random(self.seed + self.epoch + 1).shuffle(batches)
        return batches

    def __iter__(self):
        self._cached_batches = self._build()
        yield from self._cached_batches

    def __len__(self) -> int:
        if self._cached_batches is None:
            self._cached_batches = self._build()
        return len(self._cached_batches)


class LengthGroupedSampler(Sampler[int]):
    """
    Yields indices so that consecutive ones have similar source lengths, which
    keeps `collate_encrypted`'s padding tight.

    Each epoch: shuffle, cut into mega-batches of `batch_size * pool_factor`,
    sort each mega-batch by length, then shuffle the resulting batch order.
    Randomness is preserved (batch composition and order both change every
    epoch) while lengths within a batch stay close.
    """

    def __init__(self, dataset, batch_size: int, pool_factor: int = 50,
                 shuffle: bool = True, seed: int = 42):
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.pool_size   = batch_size * pool_factor
        self.shuffle     = shuffle
        self.seed        = seed
        self.epoch       = 0
        self.lengths     = [dataset.source_length(i) for i in range(len(dataset))]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        if not self.shuffle:
            idx.sort(key=lambda i: self.lengths[i])
            yield from idx
            return

        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(idx)

        pools = [idx[i : i + self.pool_size] for i in range(0, len(idx), self.pool_size)]
        batches = []
        for pool in pools:
            pool.sort(key=lambda i: self.lengths[i])
            batches += [pool[i : i + self.batch_size]
                        for i in range(0, len(pool), self.batch_size)]
        rng.shuffle(batches)
        for b in batches:
            yield from b


# ──────────────────────────────────────────────────────────────────────────────
# Token-Free Dataset  (C5 — BLT)
# ──────────────────────────────────────────────────────────────────────────────

class TokenFreeDataset(Dataset):
    """
    Raw-byte loader for the Byte Latent Transformer (C5).

    Encoding:
      Special tokens:  PAD=0, BOS=1, EOS=2
      Plaintext bytes: UTF-8 byte value b → ID (b + 3)
      Ciphertext:      each '0'/'1' ASCII character → ord(c) + 3  (51 / 52)

    Vocabulary size: 259  (256 raw bytes + 3 special tokens)

    Why C5 still chunks at 64 characters
    ────────────────────────────────────
    C5 reads the cipher as one token per bit, so a full line costs 8 tokens per
    plaintext character: ~4,800 tokens for an average line and 21,360 for the
    longest. The local encoder attends over every bit token (full O(S^2)
    attention), which at S=21,360 needs ~7.3 GB for a single attention matrix
    on one sample, before backward. That does not fit an 11 GB RTX 2080 Ti, so
    C5 is trained and decoded in 64-character windows.

    To keep evaluation comparable with C1-C4 (which run on whole lines), each
    sample records the line it came from and its position in that line, so
    predicted chunks can be concatenated back into full-line predictions before
    scoring. See `stitch_chunk_predictions` in train.py.
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    VOCAB_SIZE = 259

    def __init__(
        self,
        plain_file: str = "brown_plain.txt",
        cipher_file: str = "brown_cipher.txt",
        max_src_len: int = 6402,
        max_tgt_len: int = 802,
        chunk_size: int = 0,
        _lines: tuple | None = None,
    ):
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.chunk_size = chunk_size

        self.pad_id_plain  = self.PAD_ID
        self.pad_id_cipher = self.PAD_ID
        self.plain_vocab_size  = self.VOCAB_SIZE
        self.cipher_vocab_size = self.VOCAB_SIZE

        if _lines is not None:
            self.plain_lines, self.cipher_lines = _lines
        else:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            with open(os.path.join(base_dir, "Dataset_A1", plain_file), encoding="utf-8") as f:
                self.plain_lines = f.readlines()
            with open(os.path.join(base_dir, "Dataset_A1", cipher_file), encoding="utf-8") as f:
                self.cipher_lines = f.readlines()

        assert len(self.plain_lines) == len(self.cipher_lines)

        self._samples: list[tuple[list[int], list[int], int]] = []
        # Parallel bookkeeping so chunks can be reassembled into whole lines.
        self.sample_line_id: list[int] = []
        self.sample_start:   list[int] = []
        self.line_texts:     list[str] = [l.strip() for l in self.plain_lines]

        for line_id, (plain_line, cipher_line) in enumerate(
            zip(self.plain_lines, self.cipher_lines)
        ):
            plain_line  = plain_line.strip()
            cipher_line = cipher_line.strip()

            if chunk_size > 0:
                n = min(len(plain_line), len(cipher_line) // 8)
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    self._samples.append((
                        self._encode_cipher(cipher_line[start * 8 : end * 8]),
                        self._encode_plain(plain_line[start:end]),
                        start * 8,
                    ))
                    self.sample_line_id.append(line_id)
                    self.sample_start.append(start)
            else:
                self._samples.append((
                    self._encode_cipher(cipher_line),
                    self._encode_plain(plain_line),
                    0,
                ))
                self.sample_line_id.append(line_id)
                self.sample_start.append(0)

    # ------------------------------------------------------------------
    @classmethod
    def split(
        cls,
        plain_file: str = "brown_plain.txt",
        cipher_file: str = "brown_cipher.txt",
        max_src_len: int = 6402,
        max_tgt_len: int = 802,
        chunk_size: int = 0,
        seed: int = 42,
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(base_dir, "Dataset_A1", plain_file), encoding="utf-8") as f:
            all_plain = f.readlines()
        with open(os.path.join(base_dir, "Dataset_A1", cipher_file), encoding="utf-8") as f:
            all_cipher = f.readlines()

        assert len(all_plain) == len(all_cipher)
        train_idx, val_idx, test_idx = _split_indices(len(all_plain), seed)

        def _make(idx):
            return cls(
                max_src_len=max_src_len,
                max_tgt_len=max_tgt_len,
                chunk_size=chunk_size,
                _lines=([all_plain[i] for i in idx], [all_cipher[i] for i in idx]),
            )

        return _make(train_idx), _make(val_idx), _make(test_idx)

    # ------------------------------------------------------------------
    def _encode_plain(self, text: str) -> list[int]:
        return [b + 3 for b in text.encode("utf-8")]

    def _encode_cipher(self, text: str) -> list[int]:
        return [ord(c) + 3 for c in text]

    def _pad(self, ids: list[int], max_len: int) -> torch.Tensor:
        ids = ids[: max_len - 2]
        sequence = [self.BOS_ID] + ids + [self.EOS_ID]
        sequence += [self.PAD_ID] * (max_len - len(sequence))
        return torch.tensor(sequence, dtype=torch.long)

    def __len__(self) -> int:
        return len(self._samples)

    # The two hooks below let C5 share TokenBudgetBatchSampler with C1-C4, so
    # every configuration sees the same number of windows per optimizer step and
    # the same batches-per-epoch. Without them C5 fell back to a plain
    # DataLoader, which both broke step parity and produced offset-mixed batches.
    def sample_offset(self, idx: int) -> int:
        """Absolute character offset of this example's window (0 if unwindowed)."""
        return self._samples[idx][2]

    def cost_length(self, idx: int) -> int:
        """
        Padded encoder length, which is what drives attention memory.

        Note this is a constant for C5 in windowed mode: every window is exactly
        chunk_size bytes, so every sequence pads to chunk_size * 8 + 2. That is
        why C5 must be batched by sequence count rather than token budget. Its
        sequences are about eight times longer than a BPE window, so an equal
        token budget would hand it roughly an eighth of the batch and multiply
        its optimizer steps per epoch, which would break the ablation.
        """
        return (self.chunk_size * 8 + 2) if self.chunk_size > 0 else self.max_src_len

    def __getitem__(self, idx: int):
        cipher_ids, plain_ids, offset = self._samples[idx]

        tgt_max = (self.chunk_size + 2) if self.chunk_size > 0 else self.max_tgt_len
        src_max = (self.chunk_size * 8 + 2) if self.chunk_size > 0 else self.max_src_len

        return self._pad(cipher_ids, src_max), self._pad(plain_ids, tgt_max), offset
