import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard Absolute Positional Encoding using sine and cosine functions.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register pe as a buffer so it's not considered a model parameter
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        """
        # Add the positional encodings to the input embeddings
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return x

def get_rotary_matrix(seq_len, d_k, device):
    """
    Generates the rotary matrix for Rotary Positional Embeddings (RoPE).
    Used for configuration C2.
    """
    # (seq_len, 1)
    position = torch.arange(seq_len, device=device, dtype=torch.float).unsqueeze(1)
    
    # (d_k // 2)
    div_term = torch.exp(torch.arange(0, d_k, 2, device=device).float() * (-math.log(10000.0) / d_k))
    
    # (seq_len, d_k // 2)
    freqs = position * div_term
    
    # Duplicate to match (seq_len, d_k) like: [theta_0, theta_0, theta_1, theta_1, ...]
    emb = torch.stack((freqs, freqs), dim=-1).view(seq_len, d_k)
    
    sin = torch.sin(emb)
    cos = torch.cos(emb)
    
    return sin, cos

def apply_rotary_emb(x, sin, cos):
    """
    Applies RoPE to a tensor x of shape (batch_size, n_heads, seq_len, d_k).
    """
    # Reshape x to apply the rotation: pair adjacent features
    # x1: even features, x2: odd features
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    
    # Rotate: [-x2, x1]
    rotated_x = torch.empty_like(x)
    rotated_x[..., 0::2] = -x2
    rotated_x[..., 1::2] = x1
    
    # Apply sin and cos (sin and cos are shape (seq_len, d_k))
    # We unsqueeze to broadcast across batch_size and n_heads: (1, 1, seq_len, d_k)
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = cos.unsqueeze(0).unsqueeze(0)
    
    return (x * cos) + (rotated_x * sin)
