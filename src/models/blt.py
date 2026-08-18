import torch
import torch.nn as nn
from src.models.attention import MultiHeadAttention
from src.models.norm import LayerNorm
from src.models.positional import SinusoidalPositionalEncoding
from src.train import EncoderBlock, DecoderBlock

class LocalEncoder(nn.Module):
    """
    Byte Latent Transformer Local Encoder.
    Takes raw byte sequence and encodes them into patches.
    """
    def __init__(self, vocab_size=256, d_model=512, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.byte_emb = nn.Embedding(vocab_size, d_model)
        # We can use a 1D convolution with stride = patch_size to compress bytes into patches
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, 
                              kernel_size=patch_size, stride=patch_size)
                              
    def forward(self, x):
        # x: (batch_size, seq_len)
        emb = self.byte_emb(x) # (batch_size, seq_len, d_model)
        # Conv1d expects (batch_size, channels, length)
        emb = emb.transpose(1, 2)
        
        # padding to ensure divisible by patch_size
        rem = emb.size(-1) % self.patch_size
        if rem > 0:
            pad = self.patch_size - rem
            emb = nn.functional.pad(emb, (0, pad))
            
        patches = self.conv(emb) # (batch_size, d_model, seq_len // patch_size)
        return patches.transpose(1, 2) # (batch_size, seq_len // patch_size, d_model)


class LocalDecoder(nn.Module):
    """
    Byte Latent Transformer Local Decoder.
    Takes global latent patches and decodes them back to raw bytes.
    """
    def __init__(self, vocab_size=256, d_model=512, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.fc = nn.Linear(d_model, d_model * patch_size)
        self.out_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # x: (batch_size, patch_seq_len, d_model)
        batch_size, patch_seq_len, d_model = x.size()
        
        # Unroll patches
        unrolled = self.fc(x) # (batch_size, patch_seq_len, d_model * patch_size)
        unrolled = unrolled.view(batch_size, patch_seq_len * self.patch_size, d_model)
        
        logits = self.out_proj(unrolled) # (batch_size, seq_len, vocab_size)
        return logits


class BLTTransformer(nn.Module):
    """
    Configuration C5: Byte Latent Transformer (Token-Free)
    """
    def __init__(self, vocab_size_plain=256, d_model=512, n_heads=8, d_ff=2048, 
                 num_layers=6, max_seq_len=2048, patch_size=4, dropout=0.1):
        super().__init__()
        
        self.patch_size = patch_size
        self.local_encoder = LocalEncoder(vocab_size=256, d_model=d_model, patch_size=patch_size)
        self.local_decoder = LocalDecoder(vocab_size=vocab_size_plain, d_model=d_model, patch_size=patch_size)
        
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout, 'mha', 'layernorm') for _ in range(num_layers)
        ])
        
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout, 'mha', 'layernorm') for _ in range(num_layers)
        ])
        
        self.final_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, src_len, tgt_len, device):
        # We assume no padding for simplicity in BLT, or minimal padding.
        # tgt_sub_mask
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
        tgt_sub_mask = ~tgt_sub_mask.unsqueeze(0).unsqueeze(1)
        return None, tgt_sub_mask
        
    def forward(self, src, tgt, pad_idx_plain):
        # src: (batch_size, src_seq_len) raw bytes
        # tgt: (batch_size, tgt_seq_len) raw bytes
        
        # Local Encoding
        enc_out = self.local_encoder(src) # (batch_size, src_len // patch, d_model)
        dec_out = self.local_encoder(tgt) # (batch_size, tgt_len // patch, d_model)
        
        src_mask, tgt_mask = self.generate_mask(enc_out.size(1), dec_out.size(1), src.device)
        
        # Positional Encoding
        enc_out = self.pos_enc(enc_out)
        dec_out = self.pos_enc(dec_out)
        
        enc_out = self.dropout(enc_out)
        for block in self.encoder_blocks:
            enc_out = block(enc_out, mask=src_mask)
        enc_out = self.final_norm(enc_out)
        
        dec_out = self.dropout(dec_out)
        for block in self.decoder_blocks:
            dec_out = block(dec_out, enc_out, src_mask, tgt_mask)
        dec_out = self.final_norm(dec_out)
        
        # Local Decoding
        out = self.local_decoder(dec_out)
        return out
