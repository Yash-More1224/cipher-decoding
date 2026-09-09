"""
evaluate_final.py — one comparable test-set evaluation for all five configs.

Why this exists separately from train.py's evaluate()
─────────────────────────────────────────────────────
Two things have to be equalized before C1-C5 numbers can be put in one table.

1. Metrics must be computed on TEXT, not token IDs. C1-C4 emit BPE tokens from
   a 5,000-token vocabulary while C5 emits raw bytes from a 259-token one, so
   token-level accuracy compares different units and means nothing across the
   two families. Everything here is scored after decoding to characters.

2. Predictions must cover the SAME span. C1-C4 read a whole line at once. C5
   cannot: it uses one token per bit, so a full line is ~4,800 tokens on
   average and 21,360 at worst, and its local encoder attends over every bit
   token, needing ~7.3 GB for a single attention matrix on the longest line.
   That does not fit an 11 GB RTX 2080 Ti, so C5 is trained and decoded in
   64-character windows.

   The fix is stitching: C5 decodes each window separately (each carrying its
   true byte offset so positional encoding stays correct), and the decoded
   windows for a line are concatenated back, in order, into one full-line
   prediction. All five models are then scored against the identical
   full-line references.

Usage:
    python src/evaluate_final.py --config C1 --checkpoint_path outputs/checkpoint_C1_epochNN.pt
"""

import argparse
import os
import sys
from collections import defaultdict

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import EncryptedDataset, TokenFreeDataset          # noqa: E402
from src.utils import greedy_decode, calculate_bleu, calculate_rouge  # noqa: E402
import Levenshtein                                                   # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Text-level metrics (identical definitions for every config)
# ──────────────────────────────────────────────────────────────────────────────

def char_accuracy(pred: str, ref: str) -> tuple[int, int]:
    """Position-wise character matches, and the reference length."""
    n = len(ref)
    correct = sum(1 for i in range(min(len(pred), n)) if pred[i] == ref[i])
    return correct, n


def score_all(preds: list[str], refs: list[str], with_ngram: bool) -> dict:
    correct = total = 0
    exact = 0
    lev_sum = 0.0
    bleus, r1s, r2s, rLs = [], [], [], []

    for p, r in zip(preds, refs):
        c, n = char_accuracy(p, r)
        correct += c
        total += n
        exact += int(p == r)
        lev_sum += Levenshtein.distance(p, r)
        if with_ngram:
            bleus.append(calculate_bleu(p.split(), r.split()))
            rg = calculate_rouge(p, r)
            r1s.append(rg["rouge1"]); r2s.append(rg["rouge2"]); rLs.append(rg["rougeL"])

    out = {
        "char_accuracy": correct / max(total, 1),
        "seq_accuracy":  exact / max(len(refs), 1),
        "levenshtein":   lev_sum / max(len(refs), 1),
    }
    if with_ngram:
        n = max(len(bleus), 1)
        out.update(bleu=sum(bleus) / n, rouge1=sum(r1s) / n,
                   rouge2=sum(r2s) / n, rougeL=sum(rLs) / n)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Decoding
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_tokenized(model, ds, device, max_tokens: int, cap: int) -> tuple[list[str], list[str]]:
    """
    C1-C4: one sequence per line, decoded then detokenized back to text.

    Batched by token budget rather than a fixed count, for the same reason
    training is: attention memory grows with batch * length^2, so a fixed batch
    of the longest lines will not fit. Lines are grouped by length so each batch
    also decodes only as far as it needs to.
    """
    chunked = getattr(ds, "chunk_size", 0) > 0

    # In windowed mode a batch must be offset-homogeneous, because the model is
    # given one scalar offset per batch (same constraint as training).
    order = sorted(range(len(ds)),
                   key=lambda i: (ds.sample_offset(i) if chunked else 0,
                                  ds.source_length(i)))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = ds.source_length(i)
        cand = max(cur_max, L)
        breaks_offset = chunked and cur and ds.sample_offset(cur[0]) != ds.sample_offset(i)
        if cur and (breaks_offset or (len(cur) + 1) * cand > max_tokens):
            batches.append(cur)
            cur, cur_max = [i], L
        else:
            cur.append(i)
            cur_max = cand
    if cur:
        batches.append(cur)

    out_by_idx: dict[int, str] = {}
    for idxs in tqdm(batches, desc="Decoding (C1-C4)"):
        seqs = [ds[i][0] for i in idxs]
        maxlen = max(s.size(0) for s in seqs)
        src = torch.full((len(seqs), maxlen), ds.pad_id_cipher, dtype=torch.long)
        for r, s in enumerate(seqs):
            src[r, : s.size(0)] = s
        src = src.to(device)

        # Targets never exceed 1.06x the source on this corpus, so 1.2x plus a
        # margin is a safe decode budget and far cheaper than always running to
        # max_seq_len.
        budget = min(cap, int(1.2 * maxlen) + 8)

        out = greedy_decode(
            model, src,
            src_pad_idx=ds.pad_id_cipher, tgt_pad_idx=ds.pad_id_plain,
            max_len=budget, is_blt=False,
            bos_id=ds.bos_id_plain, eos_id=ds.eos_id_plain, dataset=ds,
            src_offset=(ds.sample_offset(idxs[0]) if chunked else 0),
        )
        for r, gi in enumerate(idxs):
            ids = [int(t) for t in out[r, 1:]]           # drop BOS
            if ds.eos_id_plain in ids:
                ids = ids[: ids.index(ds.eos_id_plain)]
            out_by_idx[gi] = ds.plain_tokenizer.decode(ids)

    if not chunked:
        return ([out_by_idx[i] for i in range(len(ds))],
                [ds.plain_texts[i] for i in range(len(ds))])

    # Windowed: concatenate each line's windows back in order, exactly as C5 is
    # stitched, so both families are scored on the same full-line references.
    per_line: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for i in range(len(ds)):
        per_line[ds.sample_line_id[i]].append((ds.sample_start[i], out_by_idx[i]))
    preds, refs = [], []
    for line_id in sorted(per_line):
        preds.append("".join(t for _, t in sorted(per_line[line_id], key=lambda x: x[0])))
        refs.append(ds.plain_texts[line_id])
    return preds, refs


@torch.no_grad()
def decode_blt_stitched(model, ds, device, batch_size: int,
                        offset_mode: str = "zero") -> tuple[list[str], list[str]]:
    """
    C5: decode each 64-character window, then concatenate a line's windows back
    together in order so the prediction covers the same span as C1-C4's.
    """
    per_line: dict[int, list[tuple[int, str]]] = defaultdict(list)

    for start in tqdm(range(0, len(ds), batch_size), desc="Decoding (C5 chunks)"):
        idxs = list(range(start, min(start + batch_size, len(ds))))
        batch = [ds[i] for i in idxs]
        src = torch.stack([b[0] for b in batch]).to(device)

        # Every window in a batch may sit at a different byte offset; decode
        # groups sharing one offset together so positional encoding is exact.
        by_offset: dict[int, list[int]] = defaultdict(list)
        for pos, i in enumerate(idxs):
            by_offset[int(ds._samples[i][2])].append(pos)

        for offset, rows in by_offset.items():
            sub = src[rows]
            out = greedy_decode(
                model, sub,
                src_pad_idx=ds.pad_id_cipher, tgt_pad_idx=ds.pad_id_plain,
                max_len=ds.chunk_size + 2, is_blt=True,
                bos_id=TokenFreeDataset.BOS_ID, eos_id=TokenFreeDataset.EOS_ID,
                dataset=ds, src_offset=(offset if offset_mode == "true" else 0),
            )
            for r, row in enumerate(rows):
                gi = idxs[row]
                ids = [int(t) for t in out[r, 1:]]
                if TokenFreeDataset.EOS_ID in ids:
                    ids = ids[: ids.index(TokenFreeDataset.EOS_ID)]
                byts = bytes(max(0, min(255, t - 3)) for t in ids
                             if t >= 3)                  # undo the +3 shift
                text = byts.decode("utf-8", errors="replace")
                per_line[ds.sample_line_id[gi]].append((ds.sample_start[gi], text))

    preds, refs = [], []
    for line_id in sorted(per_line):
        chunks = sorted(per_line[line_id], key=lambda x: x[0])
        preds.append("".join(t for _, t in chunks))
        refs.append(ds.line_texts[line_id])
    return preds, refs


# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["C1", "C2", "C3", "C4", "C5"])
    ap.add_argument("--checkpoint_path", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=16, help="C5 chunk decoding only")
    ap.add_argument("--blt_offset", choices=["zero", "true"], default="true",
                    help="C5 only. 'true' feeds each window its real byte offset, which "
                         "matches how C5 is now trained: it shares C1-C4's offset-grouped "
                         "sampler, so its batches are offset-homogeneous and the true "
                         "offset is exact. 'zero' reproduces the older behaviour, when C5 "
                         "used a plain DataLoader whose mixed-offset batches collapsed to "
                         "~0.")
    ap.add_argument("--max_tokens", type=int, default=4096,
                    help="C1-C4: padded positions per decode batch (see train.py).")
    ap.add_argument("--max_seq_len", type=int, default=1408)
    ap.add_argument("--chunk_size",  type=int, default=0,
                    help="Window size for C1-C4. MUST match how the checkpoint was trained: "
                         "0 for the full-line runs (default), 64 for the windowed runs. "
                         "Ignored for C5, which is always 64.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--n_local_layers", type=int, default=1)
    ap.add_argument("--n_global_layers", type=int, default=4)
    args = ap.parse_args()

    from src.train import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_blt = args.config == "C5"

    if is_blt:
        # C5 is always windowed at 64; its checkpoint was trained that way.
        _, _, test_ds = TokenFreeDataset.split(
            max_src_len=args.max_seq_len * 8 + 4, max_tgt_len=args.max_seq_len + 4,
            chunk_size=(args.chunk_size or 64), seed=args.seed,
        )
        n_lines = len(set(test_ds.sample_line_id))
        print(f"[Data] C5 test: {len(test_ds)} windows spanning {n_lines} lines")
    else:
        _, _, test_ds = EncryptedDataset.split(
            max_seq_len=args.max_seq_len, chunk_size=args.chunk_size, seed=args.seed)
        if args.chunk_size > 0:
            n_lines = len(set(test_ds.sample_line_id))
            print(f"[Data] {args.config} test: {len(test_ds)} windows spanning {n_lines} lines")
        else:
            print(f"[Data] {args.config} test: {len(test_ds)} lines")

    ckpt = args.checkpoint_path
    if not ckpt:
        import glob
        found = sorted(glob.glob(f"outputs/checkpoint_{args.config}_epoch*.pt"))
        if not found:
            sys.exit(f"No checkpoint found for {args.config}; pass --checkpoint_path.")
        ckpt = found[-1]
        print(f"[Eval] Auto-detected checkpoint: {ckpt}")

    model = build_model(args, test_ds).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
    for k in [k for k in state if k.endswith(".pe") or k == "pe"]:
        state.pop(k)
    model.load_state_dict(state, strict=False)
    model.eval()

    if is_blt:
        preds, refs = decode_blt_stitched(model, test_ds, device, args.batch_size,
                                          offset_mode=args.blt_offset)
    else:
        preds, refs = decode_tokenized(model, test_ds, device,
                                       max_tokens=args.max_tokens, cap=args.max_seq_len)

    # BLEU/ROUGE are scored for EVERY config including C5. They were previously
    # skipped for C5 back when it was scored at the token level, where its
    # 259-token byte vocabulary made n-gram overlap meaningless against C1-C4's
    # 5,000-token subword vocabulary. That no longer applies: stitching makes
    # C5 emit ordinary full-line text, scored against the same references by
    # the same functions, so the numbers are directly comparable.
    m = score_all(preds, refs, with_ngram=True)
    print(f"\n=== {args.config}: full-line test metrics (n={len(refs)}) ===")
    for k, v in m.items():
        print(f"  {k:14s} {v:.4f}")

    os.makedirs("outputs", exist_ok=True)
    import json
    with open(f"outputs/final_eval_{args.config}.json", "w") as f:
        json.dump({"config": args.config, "checkpoint": ckpt,
                   "n_lines": len(refs), **m}, f, indent=2)
    with open(f"outputs/predictions_{args.config}.txt", "w", encoding="utf-8") as f:
        for p, r in zip(preds[:50], refs[:50]):
            f.write(f"REF : {r}\nPRED: {p}\n\n")
    # Full prediction/reference dump. Decoding needs a GPU, but rescoring does
    # not: keeping all pairs means any later metric change can be applied
    # offline instead of requiring a fresh node allocation.
    with open(f"outputs/predictions_full_{args.config}.jsonl", "w", encoding="utf-8") as f:
        for p, r in zip(preds, refs):
            f.write(json.dumps({"pred": p, "ref": r}, ensure_ascii=False) + "\n")
    print(f"[Eval] Wrote outputs/final_eval_{args.config}.json, a 50-example dump, "
          f"and outputs/predictions_full_{args.config}.jsonl ({len(refs)} pairs).")


if __name__ == "__main__":
    main()
