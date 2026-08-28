"""
blt.py — Byte Latent Transformer (Configuration C5).

Architecture based on:
  Pagnoni et al. (2025) "Byte Latent Transformer: Patches Scale Better Than Tokens"
  ACL 2025, https://aclanthology.org/2025.acl-long.489

Design (simplified for this assignment):
─────────────────────────────────────────
Local Encoder
  1. Byte embedding layer (vocab = 256 raw bytes + 3 special tokens = 259)
  2. Small byte-level Transformer (n_enc_local_layers light encoder blocks)
  3. Cross-Attention pooling: patch queries attend over byte hidden states
     → outputs one patch representation per patch

Latent (Global) Transformer
  • Full-size EncoderBlock stack operating on patch sequence

Local Decoder
  1. Cross-Attention unrolling: byte queries attend over global patch outputs
  2. Small byte-level Transformer (n_dec_local_layers light decoder blocks)
  3. Linear projection → next-byte logits (vocab = 259)

Patch strategy
  • Fixed-stride patching (patch_size bytes → 1 patch). The full BLT paper
    uses entropy-based dynamic patching; we use the simpler fixed-stride
    variant which is standard for controlled ablation comparisons.

Seq-to-seq (cipher → plain) variant
  • Cipher bytes enter the Local Encoder.
  • Plain bytes are the decoder target (teacher-forced during training,
    autoregressive during inference).
  • The decoder cross-attends over the Latent Transformer's output patches.
"""

import torch
import torch.nn as nn

from src.models.attention import MultiHeadAttention
from src.models.norm import LayerNorm
from src.models.positional import SinusoidalPositionalEncoding



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ByteLevelTransformerLayer(nn.Module):
    """
    Lightweight Transformer layer used inside the Local Encoder/Decoder.
    Identical to EncoderBlock with MHA + LayerNorm (pre-norm), but kept
    separate so its hyperparameters can differ from the global stack.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff_w1 = nn.Linear(d_model, d_ff)
        self.ff_w2 = nn.Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-LN self-attention
        x_n = self.norm1(x)
        x = x + self.dropout(self.self_attn(x_n, x_n, x_n, mask=mask))
        # Pre-LN FFN
        x_n = self.norm2(x)
        x = x + self.dropout(self.ff_w2(self.dropout(torch.relu(self.ff_w1(x_n)))))
        return x


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer (pre-norm).
    Used for both encoder-side patch pooling and decoder-side patch unrolling.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_q = LayerNorm(d_model)
        self.norm_kv = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        query:     (B, Q, D)
        key_value: (B, KV, D)
        """
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)
        attn_out = self.cross_attn(q_norm, kv_norm, kv_norm, mask=mask)
        return query + self.dropout(attn_out)


# ---------------------------------------------------------------------------
# Local Encoder
# ---------------------------------------------------------------------------

class LocalEncoder(nn.Module):
    """
    BLT Local Encoder: bytes → patches.

    Steps (following Fig. 5 of the paper):
      1. Embed each input byte.
      2. Run n_local_layers of byte-level Transformer.
      3. Cross-attention pooling: patch queries (learnable) attend over
         byte hidden states → one patch vector per patch.
    """

    def __init__(
        self,
        vocab_size: int = 259,
        d_model: int = 512,
        patch_size: int = 8,
        n_local_layers: int = 1,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size

        # 1. Byte embedding
        self.byte_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # 2. Lightweight byte-level Transformer
        self.byte_layers = nn.ModuleList([
            ByteLevelTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_local_layers)
        ])

        # 3. Cross-attention pooling
        #    Patch queries are initialised from a learnable prototype and then
        #    projected (gives the model flexibility to learn patch pooling).
        self.patch_query_proj = nn.Linear(d_model, d_model)
        self.cross_attn = CrossAttentionLayer(d_model, n_heads, dropout)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=8192)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                           # (B, S)  raw byte IDs
        padding_mask: torch.Tensor | None = None,  # (B, 1, 1, S) True=pad
        src_offset: int = 0,                       # Absolute byte offset
    ) -> torch.Tensor:                             # (B, n_patches, D)
        B, S = x.shape
        device = x.device

        # 1. Byte embeddings + positional encoding
        # Use relative positions (offset=0) for the local encoder — the absolute
        # context is provided to the global transformer via patch_offset instead.
        # Using raw src_offset here would exceed pe's max_len for large chunks.
        h = self.byte_emb(x)           # (B, S, D)
        h = self.pos_enc(h, offset=0)
        h = self.dropout(h)

        # 2. Byte-level Transformer layers
        for layer in self.byte_layers:
            h = layer(h, mask=padding_mask)

        # 3. Build patch queries by average-pooling byte embeddings per patch
        #    and projecting — this gives a good initialisation for cross-attn.
        pad_len = (self.patch_size - S % self.patch_size) % self.patch_size
        if pad_len > 0:
            # pad h so its length is divisible by patch_size
            h_pad = torch.zeros(B, pad_len, h.size(-1), device=device, dtype=h.dtype)
            h_padded = torch.cat([h, h_pad], dim=1)   # (B, S+pad, D)
        else:
            h_padded = h

        n_patches = h_padded.size(1) // self.patch_size
        # Average pool each patch window → (B, n_patches, D)
        patch_queries = h_padded.view(B, n_patches, self.patch_size, -1).mean(dim=2)
        patch_queries = self.patch_query_proj(patch_queries)

        # 4. Cross-attention: patch queries attend over byte hidden states
        patches = self.cross_attn(patch_queries, h, mask=padding_mask)
        return patches   # (B, n_patches, D)


# ---------------------------------------------------------------------------
# Local Decoder
# ---------------------------------------------------------------------------

class LocalDecoder(nn.Module):
    """
    BLT Local Decoder: patches + byte context → byte logits.

    Steps (following Fig. 5 of the paper, right side):
      1. Embed target bytes.
      2. Cross-attention unrolling: byte queries attend over global patch outputs
         (+ byte encoder hidden states for information flow, as in the paper).
      3. n_local_layers of byte-level Transformer.
      4. Linear projection → byte logits.
    """

    def __init__(
        self,
        vocab_size: int = 259,
        d_model: int = 512,
        patch_size: int = 8,
        n_local_layers: int = 1,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Byte embedding for decoder input
        self.byte_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=8192)

        # Cross-attention: byte queries ← global patch KV
        self.cross_attn = CrossAttentionLayer(d_model, n_heads, dropout)

        # Byte-level Transformer after cross-attention
        self.byte_layers = nn.ModuleList([
            ByteLevelTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_local_layers)
        ])

        # Final projection to byte vocabulary
        self.final_norm = LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt_bytes: torch.Tensor,        # (B, T)  target byte IDs (shifted right)
        global_patches: torch.Tensor,   # (B, n_patches, D)  from Latent Transformer
        tgt_mask: torch.Tensor | None = None,   # causal + padding mask (B, 1, T, T)
    ) -> torch.Tensor:                  # (B, T, vocab_size)

        # 1. Target byte embeddings + positional encoding
        h = self.byte_emb(tgt_bytes)   # (B, T, D)
        h = self.pos_enc(h)
        h = self.dropout(h)

        # 2. Cross-attention: each target byte attends over the global patches
        #    No causal masking on the cross-attn side (standard seq2seq)
        h = self.cross_attn(h, global_patches, mask=None)

        # 3. Byte-level Transformer with causal masking
        for layer in self.byte_layers:
            h = layer(h, mask=tgt_mask)

        # 4. Project to vocab
        h = self.final_norm(h)
        return self.out_proj(h)   # (B, T, vocab_size)


# ---------------------------------------------------------------------------
# BLT Transformer  (C5)
# ---------------------------------------------------------------------------

class BLTTransformer(nn.Module):
    """
    Configuration C5: Byte Latent Transformer (Token-Free seq-to-seq).

    Input  (encoder side):  cipher byte sequence  — vocab 259
    Output (decoder side):  plain  byte sequence  — vocab 259

    Pipeline:
        cipher bytes
          → LocalEncoder        → patch representations
          → EncoderBlock stack  → global contextualised patches
        plain bytes (shifted)
          → LocalDecoder        → byte logits for next-byte prediction
    """

    def __init__(
        self,
        vocab_size: int = 259,          # 256 bytes + PAD/BOS/EOS
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,            # global Transformer depth
        n_local_layers: int = 1,        # local encoder/decoder depth
        patch_size: int = 8,
        max_seq_len: int = 8192,        # max raw byte length (src)
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Local Encoder (cipher bytes → patches)
        self.local_encoder = LocalEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            patch_size=patch_size,
            n_local_layers=n_local_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Latent (Global) Transformer — operates on patch sequence
        # Uses ByteLevelTransformerLayer (pre-LN MHA + FFN), same architecture
        # as a standard encoder block. Defined locally to keep blt.py self-contained.
        self.global_pos_enc = SinusoidalPositionalEncoding(d_model, max_len=8192 // patch_size + 1)
        self.global_blocks = nn.ModuleList([
            ByteLevelTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.global_norm = LayerNorm(d_model)

        # Local Decoder (global patches + plain bytes → byte logits)
        self.local_decoder = LocalDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            patch_size=patch_size,
            n_local_layers=n_local_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_causal_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Returns (1, 1, T, T) causal mask; True = mask out (future)."""
        mask = ~torch.tril(
            torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)
        )
        return mask.unsqueeze(0).unsqueeze(0)

    def _make_padding_mask(
        self, x: torch.Tensor, pad_idx: int
    ) -> torch.Tensor:
        """Returns (B, 1, 1, S) padding mask; True = PAD."""
        return (x == pad_idx).unsqueeze(1).unsqueeze(2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, 
        src: torch.Tensor,          # (B, S_src) cipher bytes
        tgt: torch.Tensor,          # (B, S_tgt) plain bytes
        pad_idx_src: int,
        pad_idx_tgt: int,
        src_offset: int = 0         # Absolute byte offset for chunking
    ) -> torch.Tensor:          # (B, T, vocab_size)
        """
        Teacher-forced forward pass used during training.
        src: full cipher byte sequence (encoder input).
        tgt: plain byte sequence shifted right (decoder input, without last token).
        """
        # --- Encoder side ---
        src_padding_mask = self._make_padding_mask(src, pad_idx_src)
        # 1. Local Encoder (cipher bytes → patches)
        # patch_reprs: (B, num_patches, D)
        patches = self.local_encoder(src, padding_mask=src_padding_mask, src_offset=src_offset)  # (B, n_patches, D)

        # 2. Global Transformer (contextualise patches)
        patch_offset = src_offset // self.patch_size
        patches = self.global_pos_enc(patches, offset=patch_offset)
        patches = self.dropout(patches)
        for block in self.global_blocks:
            patches = block(patches)               # no mask: patches are fully dense
        patches = self.global_norm(patches)        # (B, n_patches, D)

        # --- Decoder side ---
        tgt_len = tgt.size(1)
        tgt_causal_mask = self._make_causal_mask(tgt_len, tgt.device)  # (1, 1, T, T)

        logits = self.local_decoder(tgt, patches, tgt_mask=tgt_causal_mask)
        return logits   # (B, T, vocab_size)
