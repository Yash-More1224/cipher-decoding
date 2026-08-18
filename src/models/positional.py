"""
positional.py — Positional encoding modules for Assignment 1.

Implements:
  SinusoidalPositionalEncoding  — standard sinusoidal absolute PE (C1, C3, C4)
  get_rotary_matrix             — pre-computes RoPE sin/cos (C2)
  apply_rotary_emb              — applies RoPE to a query/key tensor

RoPE convention used here (matching Llama / GPT-NeoX)
──────────────────────────────────────────────────────
For a head vector x of dimension d_k, pair adjacent dimensions:
  (x₀, x₁), (x₂, x₃), ..., (x_{d_k-2}, x_{d_k-1})

  rotate(x₂ᵢ,  x₂ᵢ₊₁) = (x₂ᵢ cos θᵢ - x₂ᵢ₊₁ sin θᵢ,
                            x₂ᵢ sin θᵢ + x₂ᵢ₊₁ cos θᵢ)

Implementation:
  rotated_x[..., 0::2] = -x[..., 1::2]   # -odd  (the '-sin' part)
  rotated_x[..., 1::2] =  x[..., 0::2]   #  even  (the '+cos' part)

  output = x * cos + rotated_x * sin

  where sin/cos are shape (seq_len, d_k) with the SAME θᵢ broadcast to
  both even and odd positions of each pair:
  sin/cos[t, 2i]   = sin/cos(t · θᵢ)
  sin/cos[t, 2i+1] = sin/cos(t · θᵢ)   ← same value, not θᵢ₊₁

  This is achieved by: torch.cat([freqs, freqs], dim=-1) after sorting by i,
  which gives the layout [θ₀,θ₁,...,θ_{d_k/2-1}, θ₀,θ₁,...,θ_{d_k/2-1}],
  but we actually want [θ₀,θ₀,θ₁,θ₁,...] to align with 0::2/1::2 indexing.
  We therefore use torch.repeat_interleave, not torch.cat.
"""

import math

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Sinusoidal Absolute Positional Encoding
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.

      PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position  = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term  = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_len, d_model) — batch dim for broadcasting
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)  →  x + pe[:, :T, :]"""
        return x + self.pe[:, : x.size(1), :].requires_grad_(False)


# ──────────────────────────────────────────────────────────────────────────────
# Rotary Positional Embeddings (RoPE)
# ──────────────────────────────────────────────────────────────────────────────

def get_rotary_matrix(
    seq_len: int,
    d_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute RoPE sin/cos tables.

    Returns:
      sin, cos — each of shape (seq_len, d_k)

    Layout: for pair index i (0 … d_k/2-1):
      out[:, 2i]   = sin/cos(position · θᵢ)
      out[:, 2i+1] = sin/cos(position · θᵢ)   ← interleaved duplicate

    This interleaved layout aligns with the apply_rotary_emb implementation
    which reads even (0::2) and odd (1::2) features as the two members of
    each rotation pair.
    """
    assert d_k % 2 == 0, "d_k must be even for RoPE"

    # θᵢ = 1 / 10000^(2i / d_k),  i = 0 … d_k/2-1
    half = d_k // 2
    theta = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float) * 2.0 / d_k)
    )                                           # (half,)

    # position × θᵢ  →  (seq_len, half)
    pos    = torch.arange(seq_len, device=device, dtype=torch.float).unsqueeze(1)
    freqs  = pos * theta.unsqueeze(0)           # (seq_len, half)

    # Interleave: [θ₀,θ₀,θ₁,θ₁,...] along the d_k axis
    # torch.repeat_interleave repeats each element: [a,b] → [a,a,b,b]
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)   # (seq_len, d_k)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)   # (seq_len, d_k)

    return sin, cos


def apply_rotary_emb(
    x: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE to tensor x.

    Args:
        x:   (B, n_heads, T, d_k)
        sin: (T, d_k)  — from get_rotary_matrix
        cos: (T, d_k)  — from get_rotary_matrix

    Returns:
        Rotated tensor of same shape as x.

    Rotation formula per pair (x₂ᵢ, x₂ᵢ₊₁):
      out₂ᵢ   =  x₂ᵢ   · cos(θ) - x₂ᵢ₊₁ · sin(θ)
      out₂ᵢ₊₁ =  x₂ᵢ₊₁ · cos(θ) + x₂ᵢ   · sin(θ)

    Implemented as:
      rotated_x[even] = -x[odd]
      rotated_x[odd]  =  x[even]
      output = x * cos + rotated_x * sin
    """
    # Build the "rotated" counterpart: swap and negate pairs
    rotated_x = torch.empty_like(x)
    rotated_x[..., 0::2] = -x[..., 1::2]   # -x₂ᵢ₊₁
    rotated_x[..., 1::2] =  x[..., 0::2]   #  x₂ᵢ

    # Broadcast sin/cos from (T, d_k) → (1, 1, T, d_k)
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = cos.unsqueeze(0).unsqueeze(0)

    return x * cos + rotated_x * sin
