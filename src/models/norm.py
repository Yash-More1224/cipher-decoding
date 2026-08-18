import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Standard Layer Normalization module.
    """
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        # Normalize
        x_norm = (x - mean) / (std + self.eps)
        # Scale and shift
        return self.gamma * x_norm + self.beta

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) module.
    Used for configuration C4.
    """
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.eps = eps
        
    def forward(self, x):
        # Calculate root mean square
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return self.gamma * (x / rms)
