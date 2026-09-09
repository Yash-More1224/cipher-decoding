"""
utils.py — Evaluation metrics and inference utilities for Assignment 1.

Metrics (all computed using greedy decoding, as required):
  bit_level_accuracy   — % of exact token/bit matches (ignoring PAD)
  sequence_accuracy    — % of perfectly reconstructed sequences in a batch
  calculate_levenshtein — edit distance between predicted and target sequences
  calculate_bleu        — BLEU score (tokenized models C1–C4 only)
  calculate_rouge       — ROUGE-1/2/L scores (tokenized models C1–C4 only)
"""

import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import Levenshtein


# ──────────────────────────────────────────────────────────────────────────────
# Greedy Decoding
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(
    model,
    src: torch.Tensor,
    src_pad_idx: int,
    tgt_pad_idx: int,
    max_len: int,
    is_blt: bool = False,
    bos_id: int = 1,
    eos_id: int = 2,
    dataset=None,
    src_offset: int = 0,
) -> torch.Tensor:
    """
    Autoregressive greedy decoding.

    For standard Transformer (C1–C4):
      - Feeds cipher token IDs as src.
      - Starts decoder sequence with BOS, appends argmax token at each step.
      - Stops when all sequences have generated EOS or max_len is reached.

    For BLT (C5):
      - Same procedure but src is raw bytes and the model is BLTTransformer.
      - The BLTTransformer.forward() handles variable-length src automatically.
      - Note: the BLT local decoder outputs next-byte logits for ALL positions
        simultaneously given the current tgt prefix (teacher-forced cross-attn
        over global patches). We take the last position's logit at each step,
        matching the standard autoregressive decoding protocol.

    Args:
        model:        Raw (non-DDP-wrapped) model.
        src:          (B, S)  source token/byte IDs.
        src_pad_idx:  PAD id for the source vocabulary.
        tgt_pad_idx:  PAD id for the target vocabulary.
        max_len:      Maximum number of tokens to generate (including BOS).
        is_blt:       True if model is BLTTransformer.
        bos_id:       BOS token id in the target vocabulary.
        eos_id:       EOS token id in the target vocabulary.
        dataset:      Dataset instance (used to retrieve pad/bos/eos if needed).

    Returns:
        tgt: (B, T)  generated token sequences starting with BOS.
    """
    model.eval()
    device     = src.device
    batch_size = src.size(0)

    # Initialise decoder input with BOS
    tgt = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    # Track which sequences have finished (generated EOS)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        # Forward pass — get logits for the full current prefix
        if is_blt:
            logits = model(src, tgt, pad_idx_src=src_pad_idx, pad_idx_tgt=tgt_pad_idx,
                           src_offset=src_offset)
        else:
            logits = model(src, tgt, src_pad_idx, tgt_pad_idx, src_offset=src_offset)
        # logits: (B, T_so_far, vocab)

        # Greedy: take argmax at the last position
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)

        # For finished sequences, append PAD instead of a real token
        next_token[finished] = tgt_pad_idx
        tgt = torch.cat([tgt, next_token], dim=1)

        # Mark newly finished sequences
        finished = finished | (next_token.squeeze(1) == eos_id)
        if finished.all():
            break

    return tgt   # (B, T)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Metrics
# ──────────────────────────────────────────────────────────────────────────────

def bit_level_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    pad_idx: int,
) -> float:
    """
    Percentage of exact token/bit matches, ignoring PAD positions.

    Args:
        pred:   (B, T)  predicted token IDs
        target: (B, T)  ground-truth token IDs (same length as pred)
        pad_idx: PAD token ID to ignore.

    Returns:
        Scalar accuracy in [0, 1].
    """
    mask    = (target != pad_idx)
    correct = (pred == target) & mask
    return (correct.sum().float() / mask.sum().clamp(min=1)).item()


def sequence_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    pad_idx: int,
) -> float:
    """
    Percentage of perfectly reconstructed sequences in the batch.

    A sequence is "perfect" if every non-PAD token matches exactly.

    Returns:
        Scalar accuracy in [0, 1].
    """
    batch_size = pred.size(0)
    perfect    = 0
    for i in range(batch_size):
        t = target[i]
        p = pred[i]
        valid_mask = (t != pad_idx)
        valid_t = t[valid_mask]
        valid_p = p[valid_mask]
        if valid_p.shape == valid_t.shape and torch.equal(valid_p, valid_t):
            perfect += 1
    return perfect / batch_size


def calculate_levenshtein(
    pred_ids: list,
    target_ids: list,
) -> int:
    """
    Character-level Levenshtein (edit) distance between two token-ID sequences.

    Converts both sequences to space-joined strings of token IDs and uses the
    Levenshtein library for efficient computation.

    Args:
        pred_ids:   list of int  — predicted token IDs (PAD stripped by caller)
        target_ids: list of int  — ground-truth token IDs

    Returns:
        Integer edit distance.
    """
    pred_str   = " ".join(map(str, pred_ids))
    target_str = " ".join(map(str, target_ids))
    return Levenshtein.distance(pred_str, target_str)


def calculate_bleu(
    pred_tokens: list,
    target_tokens: list,
) -> float:
    """
    Sentence-level BLEU score (for tokenized models C1–C4 only).

    Uses NLTK's smoothing method 1 to avoid zero n-gram precision issues
    on short sequences.

    Args:
        pred_tokens:   list of token IDs (hypothesis)
        target_tokens: list of token IDs (reference)

    Returns:
        BLEU score in [0, 1].
    """
    smoothing = SmoothingFunction().method1
    return sentence_bleu(
        [target_tokens],
        pred_tokens,
        smoothing_function=smoothing,
    )


def calculate_rouge(
    pred_str: str,
    target_str: str,
) -> dict:
    """
    ROUGE-1, ROUGE-2, and ROUGE-L F1 scores (for tokenized models C1–C4 only).

    Args:
        pred_str:   hypothesis string (space-joined token IDs or decoded text)
        target_str: reference string

    Returns:
        dict with keys 'rouge1', 'rouge2', 'rougeL' — each an F1 float in [0,1].
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )
    scores = scorer.score(target_str, pred_str)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }
