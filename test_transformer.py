import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.models.transformer import TransformerC1

def test_model():
    batch_size = 2
    vocab_size_plain = 100
    vocab_size_cipher = 50
    max_seq_len = 1366
    
    model = TransformerC1(
        vocab_size_plain=vocab_size_plain,
        vocab_size_cipher=vocab_size_cipher,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        num_layers=2,
        max_seq_len=max_seq_len
    )
    
    src = torch.randint(0, vocab_size_cipher, (batch_size, max_seq_len))
    tgt = torch.randint(0, vocab_size_plain, (batch_size, max_seq_len - 1))
    
    pad_idx_src = 0
    pad_idx_tgt = 0
    
    out = model(src, tgt, pad_idx_src, pad_idx_tgt)
    print(f"Output shape: {out.shape}")
    assert out.shape == (batch_size, max_seq_len - 1, vocab_size_plain)
    print("Transformer forward pass successful!")

if __name__ == "__main__":
    test_model()
