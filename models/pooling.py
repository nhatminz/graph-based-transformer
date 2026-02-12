#pooling.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class AttentionPooling(nn.Module):
    """
    Attention-based pooling layer
    Learns importance weights for each node
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        """
        Args:
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input (batch_size, num_nodes, hidden_dim)
            
        Returns:
            Pooled output (batch_size, hidden_dim)
        """
        # Compute attention scores and squeeze for clearer softmax
        attn_scores = self.attention_weights(x).squeeze(-1)  # (B, N)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, N, 1)
        
        # Weighted sum
        pooled = (x * attn_weights).sum(dim=1)  # (B, hidden_dim)
        
        return pooled

class CLSTokenPooling(nn.Module):
    """
    CLS token-based pooling (similar to BERT)
    Prepends a learnable [CLS] token and returns its representation
    """
    
    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        # Learnable CLS token (init with zeros then trunc_normal for cleaner init)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def prepend_cls(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepend CLS token to input
        
        Args:
            x: Input (batch_size, num_nodes, hidden_dim)
            
        Returns:
            Output (batch_size, num_nodes + 1, hidden_dim)
        """
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_tokens, x], dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - extract CLS token representation
        
        Args:
            x: Input with CLS token (batch_size, num_nodes + 1, hidden_dim)
            
        Returns:
            CLS token representation (batch_size, hidden_dim)
        """
        return x[:, 0, :]  # Return only CLS token


class MeanPooling(nn.Module):
    """Simple mean pooling"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input (batch_size, num_nodes, hidden_dim)
            
        Returns:
            Pooled output (batch_size, hidden_dim)
        """
        return torch.mean(x, dim=1)


class MaxPooling(nn.Module):
    """Simple max pooling"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input (batch_size, num_nodes, hidden_dim)
            
        Returns:
            Pooled output (batch_size, hidden_dim)
        """
        return torch.max(x, dim=1)[0]


class GraphReadout(nn.Module):
    """
    Graph readout module
    Aggregates node features into a single graph-level representation
    """
    
    def __init__(
        self,
        hidden_dim: int,
        method: str = 'attention',
        dropout: float = 0.2
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            method: Pooling method ('mean', 'max', 'attention', 'cls')
            dropout: Dropout rate
        """
        super().__init__()
        
        self.method = method
        
        if method == 'attention':
            self.pool = AttentionPooling(hidden_dim, dropout)
        elif method == 'cls':
            self.pool = CLSTokenPooling(hidden_dim)
        elif method == 'mean':
            self.pool = MeanPooling()
        elif method == 'max':
            self.pool = MaxPooling()
        else:
            raise ValueError(f"Unknown pooling method: {method}")
    
    def prepend_cls_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        """Prepend CLS token if using CLS pooling"""
        if self.method == 'cls':
            return self.pool.prepend_cls(x)
        return x
    
    def forward(self, x: torch.Tensor, has_cls_token: bool = False) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input (batch_size, num_nodes, hidden_dim)
               or (batch_size, num_nodes + 1, hidden_dim) if CLS token prepended
            has_cls_token: Whether input has CLS token prepended (explicit control)
            
        Returns:
            Pooled output (batch_size, hidden_dim)
        """
        # Strip CLS token if present and method != 'cls'
        # This prevents mean/attention pooling from accidentally including the CLS token
        if self.method != 'cls' and has_cls_token:
            x = x[:, 1:, :]  # Remove first token (CLS)
        
        return self.pool(x)