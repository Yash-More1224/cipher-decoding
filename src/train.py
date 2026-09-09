"""
train.py — Main training script for Assignment 1.

Usage (single GPU):
    python -m src.train --config C1 --epochs 10

Usage (4-GPU DDP via torchrun):
    torchrun --nproc_per_node=4 -m src.train --config C1 --epochs 10

Configuration → model mapping:
    C1  Base:           Sinusoidal | MHA | LayerNorm | BPE
    C2  Positional:     RoPE       | MHA | LayerNorm | BPE
    C3  Attention:      Sinusoidal | GQA | LayerNorm | BPE
    C4  Normalization:  Sinusoidal | MHA | RMSNorm   | BPE
    C5  BLT:            Sinusoidal | MHA | LayerNorm | Token-Free (raw bytes)

Model classes defined here (to avoid an extra transformer.py file):
    PositionwiseFeedForward  — position-wise FFN
    EncoderBlock             — pre-LN encoder block (MHA/GQA + LayerNorm/RMSNorm)
    DecoderBlock             — pre-LN decoder block with cross-attention
    Transformer              — full encoder-decoder model for C1–C4
"""

import os
import argparse
import contextlib
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

import wandb
from huggingface_hub import HfApi, upload_folder
from tqdm import tqdm

# Project imports
from src.dataset import (
    EncryptedDataset, TokenFreeDataset, collate_encrypted,
    LengthGroupedSampler, TokenBudgetBatchSampler,
)
from src.models.attention import MultiHeadAttention, GroupedQueryAttention
from src.models.norm import LayerNorm, RMSNorm
from src.models.positional import SinusoidalPositionalEncoding, get_rotary_matrix
from src.models.blt import BLTTransformer
from src.utils import (
    greedy_decode,
    bit_level_accuracy,
    sequence_accuracy,
    calculate_levenshtein,
    calculate_bleu,
    calculate_rouge,
)


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Building Blocks  (C1 – C4)
# ──────────────────────────────────────────────────────────────────────────────

class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network: FFN(x) = W2 · ReLU(W1 · x)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class EncoderBlock(nn.Module):
    """
    Single Transformer Encoder block with Pre-LayerNorm residuals.
    Supports MHA or GQA attention, and LayerNorm or RMSNorm.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_type: str = "mha",
        norm_type: str = "layernorm",
        n_kv_heads: int | None = None,
    ):
        super().__init__()
        if attention_type == "mha":
            self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        elif attention_type == "gqa":
            kv_heads = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 4)
            self.self_attn = GroupedQueryAttention(d_model, n_heads, kv_heads, dropout)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        NormClass = LayerNorm if norm_type == "layernorm" else RMSNorm
        self.norm1 = NormClass(d_model)
        self.norm2 = NormClass(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, x_norm, x_norm, mask=mask, sin=sin, cos=cos)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    """
    Single Transformer Decoder block with Pre-LayerNorm residuals.
    RoPE (sin/cos) is applied only to self-attention, not cross-attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_type: str = "mha",
        norm_type: str = "layernorm",
        n_kv_heads: int | None = None,
    ):
        super().__init__()
        kv_heads = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 4)
        if attention_type == "mha":
            self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout)
            self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        elif attention_type == "gqa":
            self.self_attn  = GroupedQueryAttention(d_model, n_heads, kv_heads, dropout)
            self.cross_attn = GroupedQueryAttention(d_model, n_heads, kv_heads, dropout)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        NormClass = LayerNorm if norm_type == "layernorm" else RMSNorm
        self.norm1 = NormClass(d_model)
        self.norm2 = NormClass(d_model)
        self.norm3 = NormClass(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Masked self-attention (with RoPE if provided)
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, x_norm, x_norm, mask=tgt_mask, sin=sin, cos=cos)
        x = x + self.dropout(attn_out)
        # Cross-attention (no RoPE — standard practice)
        x_norm = self.norm2(x)
        attn_out = self.cross_attn(x_norm, enc_output, enc_output, mask=src_mask)
        x = x + self.dropout(attn_out)
        # Feed-forward
        x = x + self.dropout(self.feed_forward(self.norm3(x)))
        return x


class Transformer(nn.Module):
    """
    Encoder-Decoder Transformer supporting configs C1–C4.

    Config → component mapping
    --------------------------
    C1 (Base):           Sinusoidal | MHA | LayerNorm
    C2 (Positional):     RoPE       | MHA | LayerNorm
    C3 (Attention):      Sinusoidal | GQA | LayerNorm   (n_kv_heads = n_heads // 4)
    C4 (Normalization):  Sinusoidal | MHA | RMSNorm
    """

    def __init__(
        self,
        vocab_size_plain: int,
        vocab_size_cipher: int,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        max_seq_len: int = 1366,
        dropout: float = 0.1,
        config: str = "C1",
        n_kv_heads: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.n_heads = n_heads

        attention_type = "gqa" if config == "C3" else "mha"
        norm_type      = "rmsnorm" if config == "C4" else "layernorm"
        self.pos_enc_type = "rope" if config == "C2" else "sinusoidal"
        kv_heads = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 4)

        self.src_emb = nn.Embedding(vocab_size_cipher, d_model)
        self.tgt_emb = nn.Embedding(vocab_size_plain,  d_model)

        if self.pos_enc_type == "sinusoidal":
            # Use a large fixed max_len (4096) instead of max_seq_len so the PE
            # buffer can cover any absolute byte-offset produced by chunked training.
            # PE is deterministic and costs ~4MB — no reason to be stingy here.
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=4096)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout, attention_type, norm_type, kv_heads)
            for _ in range(num_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout, attention_type, norm_type, kv_heads)
            for _ in range(num_layers)
        ])

        NormClass = LayerNorm if norm_type == "layernorm" else RMSNorm
        self.final_norm = NormClass(d_model)
        self.fc_out  = nn.Linear(d_model, vocab_size_plain)
        self.dropout = nn.Dropout(dropout)

    def _make_masks(self, src, tgt, pad_idx_src, pad_idx_tgt):
        src_mask = (src == pad_idx_src).unsqueeze(1).unsqueeze(2)
        tgt_len  = tgt.size(1)
        tgt_pad_mask = (tgt == pad_idx_tgt).unsqueeze(1).unsqueeze(2)
        tgt_sub_mask = ~torch.tril(
            torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(1)
        return src_mask, tgt_pad_mask | tgt_sub_mask

    def _get_rope(self, src_len, tgt_len, src_offset, device):
        d_k = self.d_model // self.n_heads
        sin, cos = get_rotary_matrix(max(src_offset + src_len, tgt_len), d_k, device)
        return sin, cos

    def forward(self, src, tgt, pad_idx_src, pad_idx_tgt, src_offset: int = 0):
        src_mask, tgt_mask = self._make_masks(src, tgt, pad_idx_src, pad_idx_tgt)

        enc_out = self.src_emb(src)
        dec_out = self.tgt_emb(tgt)

        sin, cos = None, None
        if self.pos_enc_type == "sinusoidal":
            enc_out = self.pos_enc(enc_out, offset=src_offset)  # absolute offset!
            dec_out = self.pos_enc(dec_out)  # decoder always starts at 0
        else:
            sin, cos = self._get_rope(src.size(1), tgt.size(1), src_offset, src.device)

        enc_out = self.dropout(enc_out)
        for block in self.encoder_blocks:
            enc_out = block(enc_out, mask=src_mask,
                            sin=sin[src_offset:src_offset+src.size(1)] if sin is not None else None,
                            cos=cos[src_offset:src_offset+src.size(1)] if cos is not None else None)
        enc_out = self.final_norm(enc_out)

        dec_out = self.dropout(dec_out)
        for block in self.decoder_blocks:
            dec_out = block(dec_out, enc_out, src_mask=src_mask, tgt_mask=tgt_mask,
                            sin=sin[:tgt.size(1)] if sin is not None else None,
                            cos=cos[:tgt.size(1)] if cos is not None else None)

        dec_out = self.final_norm(dec_out)
        return self.fc_out(dec_out)

# ──────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ──────────────────────────────────────────────────────────────────────────────

def setup_ddp() -> int:
    """Initialise NCCL process group if running under torchrun.  Returns local_rank."""
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return local_rank
    return 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main() -> bool:
    """True on the rank-0 / single-GPU process."""
    return not dist.is_initialized() or dist.get_rank() == 0


# ──────────────────────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(args, dataset) -> nn.Module:
    """Instantiate the correct model for the chosen config."""
    if args.config == "C5":
        return BLTTransformer(
            vocab_size=dataset.plain_vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            n_local_layers=args.n_local_layers,
            patch_size=args.patch_size,
            dropout=args.dropout,
        )
    else:
        return Transformer(
            vocab_size_plain=dataset.plain_vocab_size,
            vocab_size_cipher=dataset.cipher_vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            config=args.config,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Forward pass (unified for tokenized and token-free)
# ──────────────────────────────────────────────────────────────────────────────

def forward_pass(model, src, tgt, dataset, is_blt: bool, src_offset: int = 0) -> torch.Tensor:
    """
    Run one forward pass.  Returns logits (B, T-1, V).
    tgt is the full target sequence; we feed tgt[:,:-1] as decoder input
    and compare against tgt[:,1:] as labels.
    """
    dec_input = tgt[:, :-1]    # shift right: remove last token
    if is_blt:
        logits = model(src, dec_input, pad_idx_src=dataset.pad_id_cipher, pad_idx_tgt=dataset.pad_id_plain, src_offset=src_offset)
    else:
        logits = model(src, dec_input, dataset.pad_id_cipher, dataset.pad_id_plain,
                       src_offset=src_offset)
    return logits


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation  (val / test)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, dataset, device, is_blt: bool, max_decode_batches: int = 2):
    """
    Compute validation/test loss and evaluation metrics.

    Metrics are computed on the first `max_decode_batches` batches to keep
    evaluation time tractable (greedy decoding is O(T²)).

    Returns dict of metric values.
    """
    model.eval()
    raw_model = model.module if isinstance(model, DDP) else model

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_id_plain)
    total_loss   = 0.0
    n_batches    = 0

    # Accumulators for metrics across evaluated batches
    all_bit_acc  = []
    all_seq_acc  = []
    all_lev      = []
    all_bleu     = []
    all_rouge1   = []
    all_rouge2   = []
    all_rougeL   = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
            src, tgt, offsets = batch[0].to(device), batch[1].to(device), batch[2]
            src_offset = int(offsets.min().item())

            logits  = forward_pass(raw_model, src, tgt, dataset, is_blt, src_offset=src_offset)
            targets = tgt[:, 1:].contiguous()

            loss = criterion(logits.contiguous().view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
            n_batches  += 1

            # Compute metrics on first N batches only
            if batch_idx < max_decode_batches:
                # Cap decode length to 128 tokens during training validation
                # (full decode is O(T²) — too slow for every epoch).
                # Full-length decode is only done at final test evaluation.
                decode_max_len = min(128, tgt.size(1))
                pred = greedy_decode(
                    raw_model, src,
                    src_pad_idx=dataset.pad_id_cipher,
                    tgt_pad_idx=dataset.pad_id_plain,
                    max_len=decode_max_len,
                    is_blt=is_blt,
                    bos_id=dataset.bos_id_plain if not is_blt else TokenFreeDataset.BOS_ID,
                    eos_id=dataset.eos_id_plain if not is_blt else TokenFreeDataset.EOS_ID,
                    dataset=dataset,
                )
                # Align lengths: pred starts with BOS, compare against tgt[:,1:]
                pred_trimmed = pred[:, 1:]                    # remove BOS
                min_len = min(pred_trimmed.size(1), targets.size(1))
                pred_aligned = pred_trimmed[:, :min_len]
                tgt_aligned  = targets[:, :min_len]

                all_bit_acc.append(bit_level_accuracy(pred_aligned, tgt_aligned, dataset.pad_id_plain))
                all_seq_acc.append(sequence_accuracy(pred_aligned,  tgt_aligned, dataset.pad_id_plain))

                # Per-sample Levenshtein (average over batch)
                batch_lev = []
                for i in range(pred.size(0)):
                    p = pred_aligned[i].tolist()
                    t = tgt_aligned[i].tolist()
                    batch_lev.append(calculate_levenshtein(p, t))
                all_lev.append(sum(batch_lev) / len(batch_lev))

                # BLEU / ROUGE only for tokenized models (C1–C4)
                if not is_blt:
                    batch_bleu   = []
                    batch_rouge1 = []
                    batch_rouge2 = []
                    batch_rougeL = []
                    for i in range(pred.size(0)):
                        p_tok = pred_aligned[i].tolist()
                        t_tok = tgt_aligned[i].tolist()
                        batch_bleu.append(calculate_bleu(p_tok, t_tok))
                        r = calculate_rouge(
                            " ".join(map(str, p_tok)),
                            " ".join(map(str, t_tok)),
                        )
                        batch_rouge1.append(r["rouge1"])
                        batch_rouge2.append(r["rouge2"])
                        batch_rougeL.append(r["rougeL"])
                    all_bleu.append(sum(batch_bleu) / len(batch_bleu))
                    all_rouge1.append(sum(batch_rouge1) / len(batch_rouge1))
                    all_rouge2.append(sum(batch_rouge2) / len(batch_rouge2))
                    all_rougeL.append(sum(batch_rougeL) / len(batch_rougeL))

    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0

    metrics = {
        "val_loss":      total_loss / n_batches,
        "bit_accuracy":  avg(all_bit_acc),
        "seq_accuracy":  avg(all_seq_acc),
        "levenshtein":   avg(all_lev),
    }
    if not is_blt:
        metrics["bleu"]   = avg(all_bleu)
        metrics["rouge1"] = avg(all_rouge1)
        metrics["rouge2"] = avg(all_rouge2)
        metrics["rougeL"] = avg(all_rougeL)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Transformer Variants — Assignment 1")

    # Config
    parser.add_argument("--config", type=str, default="C1",
                        choices=["C1", "C2", "C3", "C4", "C5"])

    # Training
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--batch_size", type=int,   default=16,
                        help="Per-GPU micro-batch size (total = batch_size × n_GPUs)")
    parser.add_argument("--lr",              type=float, default=7e-4,
                        help="Tuned for effective batch 16 (sqrt-scaled down from the "
                             "1.5e-3 that was tuned at effective batch 240).")
    parser.add_argument("--warmup_ratio",    type=float, default=0.1,
                        help="Fraction of total steps used for linear LR warmup.")
    parser.add_argument("--grad_accum_steps", type=int,   default=1,
                        help="Accumulate gradients over N mini-batches before stepping optimizer. "
                             "Effective batch = batch_size * n_GPUs * grad_accum_steps. "
                             "Kept at 1: without chunking there are only 4k training lines, so "
                             "optimizer steps are scarce and accumulation spends them.")
    parser.add_argument("--seed",            type=int,   default=42)

    # Model
    parser.add_argument("--d_model",      type=int,   default=256)
    parser.add_argument("--n_heads",      type=int,   default=8)
    parser.add_argument("--d_ff",         type=int,   default=1024)
    parser.add_argument("--num_layers",   type=int,   default=4,
                        help="4 encoder + 4 decoder blocks (~11M params with the BPE "
                             "cipher vocabulary; 6+6 would be ~14.6M).")
    parser.add_argument("--max_seq_len",  type=int,   default=1408,
                        help="Safety cap for C1-C4. Dynamic padding means batches pad to "
                             "their own longest member (~300), so a high cap is nearly free. "
                             "Longest observed line needs 1385.")
    parser.add_argument("--dropout",      type=float, default=0.3)
    # These were hardcoded at 0.3/0.05/0.2, tuned when the model OVERFIT ~40k
    # chunks. Without chunking there are 4k lines and the model underfits, so
    # they need to be tunable rather than fixed.
    parser.add_argument("--weight_decay",    type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.2)

    # BLT-specific
    parser.add_argument("--zero_offset", action="store_true",
                        help="Withhold the absolute window offset from the positional "
                             "encoding (encode every window as starting at 0). See the "
                             "note at the training loop for why it is uninformative here.")
    parser.add_argument("--patch_size",      type=int, default=8)
    parser.add_argument("--n_local_layers",  type=int, default=1)
    parser.add_argument("--n_global_layers", type=int, default=4)

    # Chunking. C5 (BLT) only: C1-C4 are segmented by the learned BPE tokenizer
    # and ignore this flag entirely (see the DataLoader section).
    parser.add_argument("--chunk_size", type=int, default=64,
                        help="C5 only: segment each line into non-overlapping windows of this "
                             "many characters. Ignored for C1-C4, which never chunk.")

    # WandB
    parser.add_argument("--no_save", action="store_true",
                        help="Do not write checkpoints. For short profiling runs whose only "
                             "purpose is measuring speed/memory, so they cannot disturb the "
                             "trained checkpoints eval auto-detection relies on.")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early-stopping patience in epochs. Raised from 4: with only "
                             "4k training lines and a harder (BPE-tokenized) cipher, short "
                             "plateaus are common and a premature stop wastes the run.")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="C1-C4: max padded positions per batch. Batch size adapts to "
                             "sequence length so the longest lines cannot OOM. Raise if "
                             "GPU memory allows; lower if you still hit OOM.")
    parser.add_argument("--run_name", type=str, default="",
                        help="W&B run name. Defaults to run-<config>. Set this so a new "
                             "round of runs does not collide with earlier ones of the "
                             "same config in the same project.")
    parser.add_argument("--wandb_project", type=str, default="anlp-assignment-1")
    parser.add_argument("--wandb_entity",  type=str, default="")

    # Hugging Face Hub
    parser.add_argument("--push_to_hub",  action="store_true")
    parser.add_argument("--hf_token",     type=str,  default="")
    parser.add_argument("--hf_repo_id",   type=str,  default="",
                        help="e.g. your-username/anlp-a1-C1")

    # Evaluation / Resuming
    parser.add_argument("--eval_only",      action="store_true", help="Run evaluation only")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to checkpoint to load for eval")
    parser.add_argument("--resume",          action="store_true",
                        help="Resume training from the latest checkpoint in outputs/ for this config")

    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    local_rank = setup_ddp()
    device     = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_blt     = (args.config == "C5")

    os.makedirs("outputs", exist_ok=True)

    # ── WandB (rank 0 only) ────────────────────────────────────────────────
    if is_main():
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.run_name or f"run-{args.config}",
            config=vars(args),
        )

    # ── Dataset & DataLoaders ──────────────────────────────────────────────
    # C5 (BLT) still chunks: one token per bit means a full line runs to ~21k
    # positions, which does not fit quadratic attention on an 11 GB card.
    # C1-C4 never chunk — the learned BPE tokenizer does all the segmenting.
    if is_blt:
        train_ds, val_ds, test_ds = TokenFreeDataset.split(
            max_src_len=args.max_seq_len * 8 + 4,
            max_tgt_len=args.max_seq_len + 4,
            chunk_size=args.chunk_size,
            seed=args.seed
        )
    else:
        train_ds, val_ds, test_ds = EncryptedDataset.split(
            max_seq_len=args.max_seq_len, chunk_size=args.chunk_size, seed=args.seed
        )

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    val_sampler   = DistributedSampler(val_ds,  shuffle=False) if dist.is_initialized() else None

    # C1-C4 pad per batch instead of to a global max, and batch by TOKEN budget
    # rather than sequence count. Attention memory scales with batch * len^2, so
    # a fixed batch size OOMs on the rare ~1,383-token lines even though the mean
    # is ~296; budgeting tokens lets long lines batch narrow automatically.
    if is_blt and args.chunk_size > 0:
        # C5 shares C1-C4's sampler so the ablation stays controlled: same
        # windows per batch, same batches per epoch, same offset grouping.
        #
        # max_tokens is deliberately disabled here. C5's encoder sequence is a
        # fixed chunk_size*8+2 positions, roughly 8x a BPE window, so an equal
        # token budget would give C5 ~7 windows per batch against C1-C4's 64 and
        # multiply its optimizer steps per epoch by ~8.7. Sequence count is the
        # quantity that has to match, so batching is capped by max_batch alone.
        blt_batch = dict(max_tokens=10**9, max_batch=args.batch_size,
                         group_by_offset=True)
        train_sampler = TokenBudgetBatchSampler(
            train_ds, shuffle=True, seed=args.seed, **blt_batch)
        train_loader = DataLoader(
            train_ds, batch_sampler=train_sampler, num_workers=2, pin_memory=True)
        val_loader = DataLoader(
            val_ds, batch_sampler=TokenBudgetBatchSampler(
                val_ds, shuffle=False, **blt_batch),
            num_workers=2, pin_memory=True)
        test_loader = DataLoader(
            test_ds, batch_sampler=TokenBudgetBatchSampler(
                test_ds, shuffle=False, **blt_batch),
            num_workers=2, pin_memory=True)
        if is_main():
            print(f"[Batching] C5 fixed batch {args.batch_size}, offset-grouped -> "
                  f"{len(train_sampler)} train batches/epoch")
    elif is_blt:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            sampler=train_sampler, shuffle=(train_sampler is None),
            num_workers=2, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, sampler=val_sampler,
            shuffle=False, num_workers=2, pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )
    else:
        # Windowed mode: batches must be offset-homogeneous, because the model
        # is given one scalar offset per batch.
        grp = args.chunk_size > 0
        train_sampler = TokenBudgetBatchSampler(
            train_ds, max_tokens=args.max_tokens, shuffle=True, seed=args.seed,
            group_by_offset=grp,
        )
        train_loader = DataLoader(
            train_ds, batch_sampler=train_sampler,
            num_workers=2, pin_memory=True, collate_fn=collate_encrypted,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=TokenBudgetBatchSampler(
                val_ds, max_tokens=args.max_tokens, shuffle=False, group_by_offset=grp),
            num_workers=2, pin_memory=True, collate_fn=collate_encrypted,
        )
        test_loader = DataLoader(
            test_ds,
            batch_sampler=TokenBudgetBatchSampler(
                test_ds, max_tokens=args.max_tokens, shuffle=False, group_by_offset=grp),
            num_workers=2, pin_memory=True, collate_fn=collate_encrypted,
        )
        if is_main():
            print(f"[Batching] token budget {args.max_tokens}/batch -> "
                  f"{len(train_sampler)} train batches/epoch")

    if is_main():
        print(f"[Dataset] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(args, train_ds).to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    checkpoint_path = args.checkpoint_path
    if not checkpoint_path and args.eval_only:
        # Auto-detect the best (highest-epoch) checkpoint for this config,
        # same glob pattern used by --resume. Checkpoints are only saved when
        # val_loss improves, so the highest epoch number is the best one.
        import glob
        pattern = f"outputs/checkpoint_{args.config}_epoch*.pt"
        ckpts = sorted(glob.glob(pattern))
        if ckpts:
            checkpoint_path = ckpts[-1]
            if is_main():
                print(f"[Eval] Auto-detected checkpoint: {checkpoint_path}")
        elif is_main():
            print(f"[Eval] No checkpoint found matching '{pattern}'. Pass --checkpoint_path explicitly.")

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        raw_model = model.module if isinstance(model, DDP) else model
        # Same fix as --resume: drop the old positional-encoding buffer so it
        # doesn't clash with the current (larger, fixed-size) buffer shape.
        if "pos_enc.pe" in checkpoint["state_dict"]:
            del checkpoint["state_dict"]["pos_enc.pe"]
        raw_model.load_state_dict(checkpoint["state_dict"], strict=False)
        if is_main():
            print(f"[Checkpoint] Loaded from {checkpoint_path}")

    if args.eval_only:
        if is_main():
            raw_model = model.module if isinstance(model, DDP) else model
            test_metrics = evaluate(
                raw_model, test_loader, test_ds, device, is_blt,
                max_decode_batches=len(test_loader),
            )
            print(f"\n=== EVALUATION RESULTS ({args.config}, n={len(test_ds)}) ===")
            for k, v in test_metrics.items():
                print(f"  {k}: {v:.4f}")
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
            wandb.finish()
        cleanup_ddp()
        return

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main():
        print(f"[Model] Config={args.config} | Parameters={n_params:,}")

    # ── Optimizer & Loss ──────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
    
    # Calculate total training steps for the scheduler
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    
    # Warmup + Cosine Annealing schedule
    # Phase 1: LinearLR ramps from 0 → peak LR over `warmup_steps` steps
    # Phase 2: CosineAnnealingLR decays from peak LR → 0 over remaining steps
    warmup_steps  = max(1, int(args.warmup_ratio * total_steps))
    cosine_steps  = max(1, total_steps - warmup_steps)
    warmup_sched  = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_sched  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps)
    scheduler     = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
    )
    
    # Label smoothing prevents the model from becoming overconfident (helps generalization)
    criterion = nn.CrossEntropyLoss(ignore_index=train_ds.pad_id_plain, label_smoothing=args.label_smoothing)
    
    # Automatic Mixed Precision (AMP) scaler for fp16 training
    scaler = torch.amp.GradScaler("cuda")

    # ── Resume (auto-detect latest checkpoint) ────────────────────────────
    start_epoch = 0
    global_step = 0
    
    # Early Stopping state
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = args.patience
    last_4_checkpoints = []

    if args.resume:
        import glob
        pattern = f"outputs/checkpoint_{args.config}_epoch*.pt"
        ckpts = sorted(glob.glob(pattern))
        if ckpts:
            latest_ckpt = ckpts[-1]   # sorted lexicographically; epoch00 < epoch01 < ...
            checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
            raw_model  = model.module if isinstance(model, DDP) else model
            # Remove pe from the checkpoint if it exists, so PyTorch doesn't 
            # silently overwrite our new 4096-size buffer with the old 850-size one.
            if "pos_enc.pe" in checkpoint["state_dict"]:
                del checkpoint["state_dict"]["pos_enc.pe"]
            raw_model.load_state_dict(checkpoint["state_dict"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"] + 1          # resume from NEXT epoch
            global_step = start_epoch * steps_per_epoch
            # Advance the scheduler to match where we left off
            for _ in range(global_step):
                scheduler.step()
            if is_main():
                print(f"[Resume] Loaded checkpoint '{latest_ckpt}' → resuming from epoch {start_epoch}")
                
            last_4_checkpoints = ckpts[-4:]
            
            # Clean up any older stranded checkpoints immediately on resume
            if len(ckpts) > 4:
                for old_ckpt in ckpts[:-4]:
                    if os.path.exists(old_ckpt):
                        try:
                            os.remove(old_ckpt)
                        except OSError:
                            pass

            if checkpoint.get("val_loss") is not None:
                best_val_loss = checkpoint["val_loss"]
        else:
            if is_main():
                print(f"[Resume] No checkpoints found matching '{pattern}' — starting from scratch.")

    # ── Training Loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            src, tgt, offsets = batch[0].to(device), batch[1].to(device), batch[2]
            # The model takes ONE scalar offset per batch, so a batch carrying
            # several different window offsets cannot be represented exactly.
            #
            # C1-C4 solve this properly: TokenBudgetBatchSampler is built with
            # group_by_offset=True, so every batch is offset-homogeneous and the
            # scalar below is the true offset for every sample in it.
            #
            # Every configuration is batched with group_by_offset=True, so this
            # scalar is the exact offset shared by every sample in the batch
            # rather than an approximation.
            if offsets.numel() and int(offsets.max()) != int(offsets.min()):
                raise RuntimeError(
                    f"Mixed offsets in one batch ({int(offsets.min())}..{int(offsets.max())}). "
                    "Batches must be offset-homogeneous; use group_by_offset=True."
                )
            # --zero_offset withholds the absolute window offset from the
            # positional encoding, so every window is encoded as if it began at
            # position 0. The offset cannot change which key byte applies (window
            # starts are multiples of the chunk size, itself a multiple of the
            # 8-bit block), so it is uninformative for the mapping. With absolute
            # positional encoding it is worse than uninformative: the same content
            # appears under as many distinct encodings as there are offsets, and
            # the model must learn to be invariant to a shift that never mattered.
            src_offset = 0 if args.zero_offset else int(offsets.min().item())

            is_accumulating = (step + 1) % args.grad_accum_steps != 0

            ctx = model.no_sync() if (isinstance(model, DDP) and is_accumulating) else contextlib.nullcontext()
            with ctx:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = forward_pass(model, src, tgt, train_ds, is_blt, src_offset=src_offset)
                    targets = tgt[:, 1:].contiguous().view(-1)
                    # Divide loss by accum steps so the sum equals the mean over the effective batch
                    loss = criterion(logits.contiguous().view(-1, logits.size(-1)), targets)
                    loss = loss / args.grad_accum_steps

                # Backward pass with scaled gradients
                scaler.scale(loss).backward()

            # Only step the optimizer after accumulating enough gradients
            if not is_accumulating:
                # Unscale before clipping so max_norm=1.0 is accurate
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Step optimizer and update scaler
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                scheduler.step()
                global_step += 1

            # Undo the loss scaling for logging (show the un-divided loss)
            epoch_loss += loss.item() * args.grad_accum_steps

            if is_main() and step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {loss.item() * args.grad_accum_steps:.4f}")
                wandb.log({"train/loss": loss.item() * args.grad_accum_steps, "epoch": epoch, "step": global_step})

        # ── Validation ────────────────────────────────────────────────────────────
        
        # Evaluate loss every epoch. Run full decoding evaluation every 5 epochs.
        decode_batches = 2 if (epoch % 5 == 0 or epoch == args.epochs - 1) else 0
        val_metrics = evaluate(model, val_loader, train_ds, device, is_blt, max_decode_batches=decode_batches)
        
        if is_main():
            epoch_time = time.time() - t0
            avg_train_loss = epoch_loss / len(train_loader)
            
            val_str = f" | val_loss={val_metrics['val_loss']:.4f}" if "val_loss" in val_metrics else ""
            print(
                f"Epoch {epoch} done | "
                f"train_loss={avg_train_loss:.4f}{val_str} | "
                f"time={epoch_time:.1f}s"
            )
            
            log_dict = {
                "epoch":          epoch,
                "step":           global_step,
                "train/avg_loss": avg_train_loss,
                # Logged explicitly rather than derived from _runtime/n_epochs,
                # which folds in startup, eval and checkpoint-saving overhead.
                "perf/epoch_time_s": epoch_time,
                "perf/n_params":     n_params,
            }
            if val_metrics:
                for k, v in val_metrics.items():
                    log_dict[f"val/{k}"] = v
            wandb.log(log_dict)

            # Save checkpoint & Early stopping
            current_val_loss = val_metrics.get("val_loss", float('inf')) if val_metrics else avg_train_loss
            
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0

                if args.no_save:
                    print(f"[Checkpoint] Skipped (--no_save), val_loss={current_val_loss:.4f}")
                    continue

                ckpt_path = f"outputs/checkpoint_{args.config}_epoch{epoch:02d}.pt"
                raw_model = model.module if isinstance(model, DDP) else model
                torch.save(
                    {
                        "epoch":      epoch,
                        "config":     args.config,
                        "state_dict": raw_model.state_dict(),
                        "optimizer":  optimizer.state_dict(),
                        "val_loss":   val_metrics.get("val_loss", None),
                    },
                    ckpt_path,
                )
                print(f"[Checkpoint] Saved → {ckpt_path}")
                
                # Keep only last 4 checkpoints
                last_4_checkpoints.append(ckpt_path)
                if len(last_4_checkpoints) > 4:
                    old_ckpt = last_4_checkpoints.pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)
            else:
                patience_counter += 1
                print(f"[Early Stopping] val_loss didn't improve ({current_val_loss:.4f} >= {best_val_loss:.4f}). Patience: {patience_counter}/{patience_limit}")

        if patience_counter >= patience_limit:
            if is_main():
                print(f"Stopping early at epoch {epoch} due to no improvement in val_loss.")
            break

    # ── Final test evaluation (rank 0 only) ───────────────────────────────
    if is_main():
        raw_model = model.module if isinstance(model, DDP) else model
        test_metrics = evaluate(
            raw_model, test_loader, test_ds, device, is_blt,
            max_decode_batches=len(test_loader),   # full test set
        )
        print("\n=== TEST RESULTS ===")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

    # ── Hugging Face Hub upload ────────────────────────────────────────────
    if is_main() and args.push_to_hub and args.hf_token and args.hf_repo_id:
        api = HfApi(token=args.hf_token)
        # Create repo if it doesn't exist
        api.create_repo(repo_id=args.hf_repo_id, exist_ok=True, repo_type="model")
        # Upload the entire outputs/ directory
        upload_folder(
            folder_path="outputs",
            repo_id=args.hf_repo_id,
            repo_type="model",
            token=args.hf_token,
            commit_message=f"Upload checkpoints for config {args.config}",
        )
        print(f"[HF Hub] Uploaded outputs/ → https://huggingface.co/{args.hf_repo_id}")

    if is_main():
        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
