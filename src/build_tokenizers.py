"""
build_tokenizers.py — train the plaintext and ciphertext BPE tokenizers once.

Run this ONCE before launching any training job:

    python src/build_tokenizers.py

Why it is a separate step: the four C1-C4 runs are launched in parallel, one
per GPU. If each job trained its own tokenizer they would duplicate the work
and race to write the same JSON files, so training jobs only ever *load*.

Both tokenizers are trained on the TRAINING SPLIT ONLY (the same seeded 80/10/10
split dataset.py uses), so no validation or test line influences the merge table.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer import BPETokenizer                      # noqa: E402
from dataset import _split_indices, PLAIN_TOK_PATH, CIPHER_TOK_PATH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain_vocab",  type=int, default=5000)
    ap.add_argument("--cipher_vocab", type=int, default=4096)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--plain_file",   type=str, default="brown_plain.txt")
    ap.add_argument("--cipher_file",  type=str, default="brown_cipher.txt")
    ap.add_argument("--force", action="store_true",
                    help="Retrain even if tokenizer files already exist.")
    args = ap.parse_args()

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(base, "Dataset_A1", args.plain_file), encoding="utf-8") as f:
        all_plain = f.readlines()
    with open(os.path.join(base, "Dataset_A1", args.cipher_file), encoding="utf-8") as f:
        all_cipher = f.readlines()
    assert len(all_plain) == len(all_cipher)

    train_idx, _, _ = _split_indices(len(all_plain), args.seed)
    plain_train  = [all_plain[i]  for i in train_idx]
    cipher_train = [all_cipher[i] for i in train_idx]
    print(f"[Build] {len(all_plain)} lines total, training tokenizers on "
          f"{len(train_idx)} train-split lines only.")

    if os.path.exists(PLAIN_TOK_PATH) and os.path.exists(CIPHER_TOK_PATH) and not args.force:
        print("[Build] Tokenizers already exist. Use --force to retrain.")
        return

    t = time.time()
    plain_tok = BPETokenizer.train(plain_train, args.plain_vocab, mode="text")
    plain_tok.save(PLAIN_TOK_PATH)
    print(f"[Build] plain  vocab={plain_tok.get_vocab_size()} "
          f"merges={len(plain_tok.merges)}  ({time.time() - t:.1f}s)  -> {PLAIN_TOK_PATH}")

    t = time.time()
    cipher_tok = BPETokenizer.train(cipher_train, args.cipher_vocab, mode="cipher")
    cipher_tok.save(CIPHER_TOK_PATH)
    print(f"[Build] cipher vocab={cipher_tok.get_vocab_size()} "
          f"merges={len(cipher_tok.merges)}  ({time.time() - t:.1f}s)  -> {CIPHER_TOK_PATH}")

    # Report the length distribution so --max_seq_len can be sanity-checked.
    import numpy as np
    lp = np.array([len(plain_tok.encode(l.strip()))  for l in all_plain])
    lc = np.array([len(cipher_tok.encode(l.strip())) for l in all_cipher])
    for name, L in (("plain", lp), ("cipher", lc)):
        print(f"[Build] {name:6s} tokens/line: mean {L.mean():6.1f}  p95 {np.percentile(L, 95):5.0f}  "
              f"p99 {np.percentile(L, 99):5.0f}  max {L.max():5d}")
    need = int(max(lp.max(), lc.max())) + 2
    print(f"[Build] smallest --max_seq_len that truncates nothing: {need}")


if __name__ == "__main__":
    main()
