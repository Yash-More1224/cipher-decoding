import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        """
        q: (batch_size, n_heads, seq_len, d_k)
        k: (batch_size, n_heads, seq_len, d_k)
        v: (batch_size, n_heads, seq_len, d_v)
        mask: (batch_size, 1, 1, seq_len) or similar broadcastable shape
        """
        d_k = q.size(-1)
        # scores: (batch_size, n_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # Mask should be boolean (True for padding/future, False for valid)
            scores = scores.masked_fill(mask, -1e9)
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # output: (batch_size, n_heads, seq_len_q, d_v)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None, sin=None, cos=None):
        batch_size = q.size(0)
        
        # Linear projections & reshape for multi-head
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k) -> (batch_size, n_heads, seq_len, d_k)
        q = self.W_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply RoPE if provided
        if sin is not None and cos is not None:
            from src.models.positional import apply_rotary_emb
            q = apply_rotary_emb(q, sin, cos)
            k = apply_rotary_emb(k, sin, cos)
        
        # Apply attention
        x, attn_weights = self.attention(q, k, v, mask)
        
        # Concatenate heads
        # (batch_size, n_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(x)
        return self.dropout(output)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_query_heads, num_kv_heads, dropout=0.1):
        """
        Grouped-Query Attention (GQA) for configuration C3.
        num_query_heads > num_kv_heads, and num_query_heads must be divisible by num_kv_heads.
        """
        super().__init__()
        assert d_model % num_query_heads == 0, "d_model must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be a multiple of num_kv_heads"
        
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_query_heads // num_kv_heads
        self.d_k = d_model // num_query_heads
        
        # Note: W_k and W_v project to a smaller dimension (num_kv_heads * d_k)
        self.W_q = nn.Linear(d_model, num_query_heads * self.d_k)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_o = nn.Linear(num_query_heads * self.d_k, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None, sin=None, cos=None):
        batch_size = q.size(0)
        
        # 1. Linear projections
        # Q: (batch_size, seq_len, num_query_heads, d_k) -> (batch_size, num_query_heads, seq_len, d_k)
        q = self.W_q(q).view(batch_size, -1, self.num_query_heads, self.d_k).transpose(1, 2)
        
        # K/V: (batch_size, seq_len, num_kv_heads, d_k) -> (batch_size, num_kv_heads, seq_len, d_k)
        k = self.W_k(k).view(batch_size, -1, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # Apply RoPE if provided
        if sin is not None and cos is not None:
            from src.models.positional import apply_rotary_emb
            q = apply_rotary_emb(q, sin, cos)
            k = apply_rotary_emb(k, sin, cos)
            
        # 2. Repeat K/V to match Q's heads (Broadcasting)
        # We expand K and V from num_kv_heads to num_query_heads by repeating them across the group dimension.
        # k: (batch_size, num_kv_heads, 1, seq_len, d_k) -> (batch_size, num_kv_heads, num_groups, seq_len, d_k)
        k = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).contiguous()
        v = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).contiguous()
        
        # Reshape to (batch_size, num_query_heads, seq_len, d_k)
        k = k.view(batch_size, self.num_query_heads, -1, self.d_k)
        v = v.view(batch_size, self.num_query_heads, -1, self.d_k)
        
        # 3. Apply attention
        x, attn_weights = self.attention(q, k, v, mask)
        
        # 4. Concatenate heads and final projection
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_query_heads * self.d_k)
        output = self.W_o(x)
        return self.dropout(output)
