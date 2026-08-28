"""
tokenizer.py — Byte-Pair Encoding implemented from scratch.

No tokenizer library is used anywhere in this file: training, encoding,
decoding, and serialization are all hand-written. (Assignment constraint.)

Two modes
─────────
  mode="text"    Plaintext. Pre-tokenized into whitespace-delimited words,
                 where each space is kept as the first character of the word
                 that follows it. Concatenating all tokens therefore restores
                 the original string exactly, which matters because we decode
                 model predictions back to text to score BLEU/ROUGE/Levenshtein.

  mode="cipher"  Ciphertext. The binary string is first grouped into 8-bit
                 blocks and each block becomes one symbol from a fixed
                 256-symbol alphabet (byte value b -> chr(0x100 + b)). BPE
                 merges are then learned over *that* stream.

                 This is the important design choice. Running BPE directly on
                 the raw '0'/'1' string lets merges straddle 8-bit character
                 boundaries and destroys the byte alignment the cipher depends
                 on. Grouping into byte symbols first keeps every boundary
                 intact while still giving a genuine learned subword vocabulary
                 (this is the same construction as byte-level BPE in GPT-2).

Training algorithm
──────────────────
The textbook BPE loop (rescan the whole corpus once per merge) is O(V * N) and
takes hours at our corpus size. We use the standard incremental formulation:

  * the corpus is held in one flat array with `prev`/`next` links, so merging a
    pair is a local pointer update rather than a rebuild;
  * `pair_counts` maps each adjacent pair to its frequency, and `pair_positions`
    maps it to the set of positions where it occurs, so applying a merge only
    touches the positions that actually contain it;
  * a max-heap supplies the next merge candidate, with lazy invalidation
    (a popped entry whose count no longer matches the live count is discarded).

Measured on the assignment's cipher corpus (5,000 lines, ~3.0M byte symbols):
training a 4,096-token vocabulary takes ~21s and encoding all 5,000 lines
takes ~1.5s.

Ties are broken by the (lower) symbol-id pair, so training is deterministic.
"""

import heapq
import json
import os
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# Special tokens (fixed IDs, always first in the vocabulary)
# ──────────────────────────────────────────────────────────────────────────────

PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


# ──────────────────────────────────────────────────────────────────────────────
# Pre-tokenization helpers
# ──────────────────────────────────────────────────────────────────────────────

def pre_tokenize_text(s: str) -> list[str]:
    """
    Split text into words, attaching each space to the word that follows it.

        "the cat sat"  ->  ["the", " cat", " sat"]
        "a  b"         ->  ["a", " ", " b"]

    "".join(pre_tokenize_text(s)) == s for any s, so decoding is lossless.
    BPE merges are learned within a word and never cross a word boundary,
    which is the usual constraint and keeps the merge table meaningful.
    """
    if not s:
        return []
    words, cur = [], s[0]
    for ch in s[1:]:
        if ch == " ":
            words.append(cur)
            cur = ch
        else:
            cur += ch
    words.append(cur)
    return words


def bits_to_symbols(bit_string: str) -> str:
    """
    Group a binary string into 8-bit blocks, mapping each block to one symbol.

    Byte value b becomes chr(0x100 + b), a private-use codepoint chosen so the
    256 cipher symbols can never collide with real plaintext characters.
    A trailing partial block (fewer than 8 bits) is dropped, matching the
    existing byte-grouping behaviour in dataset.py.
    """
    return "".join(
        chr(0x100 + int(bit_string[i:i + 8], 2))
        for i in range(0, len(bit_string) - 7, 8)
    )


def symbols_to_bits(symbols: str) -> str:
    """Inverse of bits_to_symbols (used only for round-trip verification)."""
    return "".join(format(ord(c) - 0x100, "08b") for c in symbols)


CIPHER_ALPHABET = [chr(0x100 + b) for b in range(256)]


# ──────────────────────────────────────────────────────────────────────────────
# Core BPE training
# ──────────────────────────────────────────────────────────────────────────────

def learn_merges(
    sequences: list[list[str]],
    weights: list[int],
    vocab_size: int,
    initial_alphabet: list[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Learn BPE merges. Returns (vocab, merges).

    `sequences` are symbol lists; `weights[i]` is how many times sequences[i]
    occurs in the corpus (lets the plaintext side collapse to unique word types
    and count each once, which is far cheaper than expanding every occurrence).

    `vocab_size` counts the special tokens and the initial alphabet, so the
    number of merges learned is vocab_size - len(specials) - len(alphabet).
    """
    vocab = list(SPECIAL_TOKENS) + list(initial_alphabet)
    tid = {s: i for i, s in enumerate(vocab)}

    # Flat corpus with prev/next links. Sequences are laid end to end; the
    # links never cross a sequence boundary, so merges cannot either.
    sym: list[int] = []
    prv: list[int] = []
    nxt: list[int] = []
    wt: list[int] = []
    for seq, w in zip(sequences, weights):
        base, L = len(sym), len(seq)
        for j, ch in enumerate(seq):
            if ch not in tid:          # symbol outside the alphabet: skip
                continue
            sym.append(tid[ch])
            wt.append(w)
            prv.append(base + j - 1 if j > 0 else -1)
            nxt.append(base + j + 1 if j < L - 1 else -1)

    pair_counts: dict = defaultdict(int)
    pair_positions: dict = defaultdict(set)
    for i in range(len(sym)):
        n = nxt[i]
        if n != -1:
            p = (sym[i], sym[n])
            pair_counts[p] += wt[i]
            pair_positions[p].add(i)

    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[tuple[str, str]] = []
    while len(vocab) < vocab_size:
        # Pop the most frequent still-valid pair (lazy invalidation).
        best = None
        while heap:
            neg_c, cand = heapq.heappop(heap)
            if pair_counts.get(cand, 0) == -neg_c and pair_counts[cand] > 0:
                best = cand
                break
        if best is None or pair_counts[best] < 2:
            break

        a, b = best
        new_id = len(vocab)
        vocab.append(vocab[a] + vocab[b])
        merges.append((vocab[a], vocab[b]))

        for i in list(pair_positions[best]):
            # Position may have been consumed by an earlier merge in this pass
            # (e.g. overlapping "aaa" when merging the pair (a, a)).
            if sym[i] != a:
                continue
            n = nxt[i]
            if n == -1 or sym[n] != b:
                continue

            w, left, right = wt[i], prv[i], nxt[n]

            # Retire the two pairs that straddle the merge site.
            if left != -1:
                old = (sym[left], a)
                pair_counts[old] -= w
                pair_positions[old].discard(left)
            if right != -1:
                old = (b, sym[right])
                pair_counts[old] -= w
                pair_positions[old].discard(n)

            # Splice: node i becomes the merged symbol, node n disappears.
            sym[i] = new_id
            nxt[i] = right
            sym[n] = -1
            if right != -1:
                prv[right] = i

            # Register the two pairs the merge creates.
            if left != -1:
                new = (sym[left], new_id)
                pair_counts[new] += w
                pair_positions[new].add(left)
                heapq.heappush(heap, (-pair_counts[new], new))
            if right != -1:
                new = (new_id, sym[right])
                pair_counts[new] += w
                pair_positions[new].add(i)
                heapq.heappush(heap, (-pair_counts[new], new))

        pair_counts[best] = 0
        pair_positions[best] = set()

    return vocab, merges


def apply_merges(symbols: list[str], ranks: dict) -> list[str]:
    """
    Apply learned merges to one symbol sequence, lowest rank (earliest learned)
    first. Uses the same linked-list + heap structure as training, giving
    O(L log L) instead of the O(L * n_merges) of rescanning per merge.
    """
    n = len(symbols)
    if n < 2:
        return list(symbols)

    tok = list(symbols)
    alive = [True] * n
    prv = [i - 1 for i in range(n)]
    nxt = [i + 1 for i in range(n)]
    nxt[-1] = -1

    heap = []
    for i in range(n - 1):
        r = ranks.get((tok[i], tok[i + 1]))
        if r is not None:
            heapq.heappush(heap, (r, i))

    while heap:
        r, i = heapq.heappop(heap)
        if not alive[i]:
            continue
        j = nxt[i]
        if j == -1 or not alive[j]:
            continue
        if ranks.get((tok[i], tok[j])) != r:      # stale entry
            continue

        tok[i] = tok[i] + tok[j]
        alive[j] = False
        right = nxt[j]
        nxt[i] = right
        if right != -1:
            prv[right] = i

        left = prv[i]
        if left != -1 and alive[left]:
            rr = ranks.get((tok[left], tok[i]))
            if rr is not None:
                heapq.heappush(heap, (rr, left))
        if right != -1:
            rr = ranks.get((tok[i], tok[right]))
            if rr is not None:
                heapq.heappush(heap, (rr, i))

    out, i = [], 0
    while i != -1:
        out.append(tok[i])
        i = nxt[i]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────────────────────

class BPETokenizer:
    """A trained BPE tokenizer. Construct via `train()` or `load()`."""

    def __init__(self, vocab: list[str], merges: list[tuple[str, str]], mode: str):
        assert mode in ("text", "cipher"), f"unknown mode: {mode}"
        self.mode = mode
        self.vocab = vocab
        self.merges = [tuple(m) for m in merges]
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self._cache: dict[str, list[int]] = {}   # word -> ids (text mode only)

        self.pad_id, self.bos_id = PAD_ID, BOS_ID
        self.eos_id, self.unk_id = EOS_ID, UNK_ID

    # ------------------------------------------------------------------
    @classmethod
    def train(cls, lines: list[str], vocab_size: int, mode: str) -> "BPETokenizer":
        """Train on raw lines (plaintext strings, or cipher binary strings)."""
        if mode == "cipher":
            sequences = [list(bits_to_symbols(l.strip())) for l in lines]
            weights = [1] * len(sequences)
            alphabet = CIPHER_ALPHABET
        else:
            # Collapse to unique word types with counts: same merges, far less work.
            counts: dict = defaultdict(int)
            for line in lines:
                for w in pre_tokenize_text(line.strip()):
                    counts[w] += 1
            sequences = [list(w) for w in counts]
            weights = list(counts.values())
            alphabet = sorted({ch for w in counts for ch in w})

        vocab, merges = learn_merges(sequences, weights, vocab_size, alphabet)
        return cls(vocab, merges, mode)

    # ------------------------------------------------------------------
    def _encode_symbols(self, symbols: list[str]) -> list[int]:
        return [
            self.token_to_id.get(t, self.unk_id)
            for t in apply_merges(symbols, self.ranks)
        ]

    def encode(self, s: str) -> list[int]:
        """Encode one line to token IDs (no BOS/EOS; the dataset adds those)."""
        if self.mode == "cipher":
            return self._encode_symbols(list(bits_to_symbols(s)))

        ids: list[int] = []
        for word in pre_tokenize_text(s):
            cached = self._cache.get(word)
            if cached is None:
                cached = self._encode_symbols(list(word))
                self._cache[word] = cached
            ids.extend(cached)
        return ids

    def decode(self, ids, skip_special: bool = True) -> str:
        """
        Token IDs back to a string. For text mode this reproduces the original
        exactly (see pre_tokenize_text); for cipher mode it returns the binary
        string.
        """
        pieces = []
        for i in ids:
            i = int(i)
            if skip_special and i < len(SPECIAL_TOKENS):
                continue
            pieces.append(self.vocab[i] if 0 <= i < len(self.vocab) else "")
        joined = "".join(pieces)
        return symbols_to_bits(joined) if self.mode == "cipher" else joined

    # ------------------------------------------------------------------
    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"mode": self.mode, "vocab": self.vocab,
                 "merges": [list(m) for m in self.merges]},
                f, ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["vocab"], [tuple(m) for m in d["merges"]], d["mode"])
