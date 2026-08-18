import torch
import torch.nn as nn
from src.models.attention import MultiHeadAttention
from src.models.norm import LayerNorm
from src.models.positional import SinusoidalPositionalEncoding

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Pre-LayerNorm configuration (as decided in spec.md for C1)
        # x: (batch_size, seq_len, d_model)
        attn_out = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Pre-LayerNorm
        # Self Attention (masked)
        attn_out = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask)
        x = x + self.dropout(attn_out)
        
        # Cross Attention
        # Queries come from decoder (x), Keys/Values come from encoder (enc_output)
        attn_out = self.cross_attn(self.norm2(x), enc_output, enc_output, src_mask)
        x = x + self.dropout(attn_out)
        
        # Feed Forward
        ff_out = self.feed_forward(self.norm3(x))
        x = x + self.dropout(ff_out)
        return x

class TransformerC1(nn.Module):
    """
    The Base Transformer (C1) implementation.
    """
    def __init__(self, vocab_size_plain, vocab_size_cipher, d_model=256, n_heads=8, d_ff=1024, num_layers=6, max_seq_len=1366, dropout=0.1):
        super().__init__()
        
        self.src_emb = nn.Embedding(vocab_size_cipher, d_model)
        self.tgt_emb = nn.Embedding(vocab_size_plain, d_model)
        
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        
        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.final_norm = LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size_plain)
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, src, tgt, pad_idx_src, pad_idx_tgt):
        """
        Creates padding masks and look-ahead masks for the decoder.
        """
        # src_mask: (batch_size, 1, 1, src_len)
        src_mask = (src == pad_idx_src).unsqueeze(1).unsqueeze(2)
        
        # tgt_mask: (batch_size, 1, tgt_len, tgt_len)
        tgt_len = tgt.size(1)
        tgt_pad_mask = (tgt == pad_idx_tgt).unsqueeze(1).unsqueeze(2)
        
        # Create look-ahead mask for decoder self-attention
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        # Invert tril mask so True means 'mask out' (future tokens)
        tgt_sub_mask = ~tgt_sub_mask.unsqueeze(0).unsqueeze(1)
        
        tgt_mask = tgt_pad_mask | tgt_sub_mask
        
        return src_mask, tgt_mask
        
    def forward(self, src, tgt, pad_idx_src, pad_idx_tgt):
        # Create masks
        src_mask, tgt_mask = self.generate_mask(src, tgt, pad_idx_src, pad_idx_tgt)
        
        # Encoder Forward
        enc_out = self.dropout(self.pos_enc(self.src_emb(src)))
        # We need to apply normalization to enc_output before it goes to cross-attention
        # Because we use Pre-LayerNorm, the final encoder output isn't normalized by the block itself.
        for block in self.encoder_blocks:
            enc_out = block(enc_out, src_mask)
        enc_out = self.final_norm(enc_out) 
            
        # Decoder Forward
        dec_out = self.dropout(self.pos_enc(self.tgt_emb(tgt)))
        for block in self.decoder_blocks:
            dec_out = block(dec_out, enc_out, src_mask, tgt_mask)
            
        # Final projection
        dec_out = self.final_norm(dec_out)
        out = self.fc_out(dec_out)
        
        return out
