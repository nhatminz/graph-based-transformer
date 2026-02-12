#temporal_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal sequences
    """
    
    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch_size, seq_len, d_model)
            
        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TemporalTransformerLayer(nn.Module):
    """
    Temporal Transformer layer
    Standard transformer encoder layer for processing temporal sequences
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        ffn_hidden_dim: int = 256,
        dropout: float = 0.2,
        activation: str = 'gelu'
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            ffn_hidden_dim: FFN hidden dimension
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def _get_activation(self, name: str):
        """Get activation function"""
        if name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input (batch_size, seq_len, hidden_dim)
            mask: Attention mask (optional)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention with Pre-LN (modern transformer architecture)
        # x = x + attn(LN(x))
        normed = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            normed, normed, normed,
            attn_mask=mask,
            need_weights=return_attention
        )
        x = x + self.dropout(attn_out)
        
        # FFN with Pre-LN
        # x = x + ffn(LN(x))
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        
        if return_attention:
            return x, attn_weights
        return x, None


class TemporalTransformer(nn.Module):
    """
    Complete Temporal Transformer
    Processes temporal sequences with multiple transformer layers
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 3,
        num_heads: int = 4,
        ffn_hidden_dim: int = 256,
        dropout: float = 0.2,
        activation: str = 'gelu',
        use_positional_encoding: bool = True,
        max_seq_len: int = 128
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ffn_hidden_dim: FFN hidden dimension
            dropout: Dropout rate
            activation: Activation function
            use_positional_encoding: Whether to use positional encoding
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.use_positional_encoding = use_positional_encoding
        
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                d_model=hidden_dim,
                max_len=max_seq_len,
                dropout=dropout
            )
        
        # Stack of transformer layers
        self.layers = nn.ModuleList([
            TemporalTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass
        
        Args:
            x: Input (batch_size, seq_len, hidden_dim)
            mask: Attention mask (optional)
            return_attention: Whether to return attention weights from all layers
            
        Returns:
            Tuple of (output, list of attention_weights)
            output: (batch_size, seq_len, hidden_dim)
            attention_weights: List of attention weights from each layer (if return_attention)
        """
        # Add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoder(x)
        
        attention_weights_list = [] if return_attention else None
        
        # Pass through transformer layers
        for layer in self.layers:
            x, attn_weights = layer(x, mask, return_attention)
            if return_attention:
                attention_weights_list.append(attn_weights)
        
        # Final normalization
        x = self.norm(x)
        
        return x, attention_weights_list