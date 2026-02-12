#graph_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class GraphAttentionLayer(nn.Module):
    #  Attention(Q, K, V) = Softmax((QK^T / sqrt(d)) + Bias_pos).V where Bias_pos encodes spatial relationships between electrodes
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_bias: bool = True, # Whether to use bias in linear layers (q,k,v)
        use_positional_bias: bool = True, # Whether to use positional bias
        bias_type: str = 'distance',
        concat_heads: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.use_positional_bias = use_positional_bias
        self.bias_type = bias_type

        # Ensure output dimension is divisible by num_heads
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        self.head_dim = out_features // num_heads
        # Linear transformations for Q, K, V
        self.query = nn.Linear(in_features, out_features, bias=use_bias)
        self.key = nn.Linear(in_features, out_features, bias=use_bias)
        self.value = nn.Linear(in_features, out_features, bias=use_bias)
        # Output projection
        if concat_heads:
            self.out_proj = nn.Linear(out_features, out_features, bias=use_bias)
        else:
            self.out_proj = nn.Linear(self.head_dim, out_features, bias=use_bias)

        # Positional bias
        if use_positional_bias:
            if bias_type == 'learned':
                # Learnable bias for each head
                # Will be initialized later based on number of nodes
                self.pos_bias = None  # Use regular attribute, not buffer
                self.bias_initialized = False  # Python bool, not tensor
            else:
                # Static bias (will be set externally)
                self.pos_bias = None  # Use regular attribute
        else:
            self.pos_bias = None

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def initialize_positional_bias(self, num_nodes: int):
        """Initialize learnable positional bias"""
        if self.bias_type == 'learned' and self.pos_bias is None:
            # Initialize one bias matrix per head with small values
            # Small init is critical: large bias can dominate QK^T in early training
            self.pos_bias = nn.Parameter(
                torch.zeros(self.num_heads, num_nodes, num_nodes)
            )
            nn.init.normal_(self.pos_bias, mean=0.0, std=0.01)  # Small std for attention logits
            self.bias_initialized = True  # Python bool

    #use when bias is distance or correlation
    def set_positional_bias(self, bias: torch.Tensor):
        """
        Set external positional bias (e.g., from distance matrix)
        
        Args:
            bias: Tensor of shape (num_nodes, num_nodes) or (num_heads, num_nodes, num_nodes)
        """
        if len(bias.shape) == 2:
            # Broadcast to all heads
            bias = bias.unsqueeze(0).repeat(self.num_heads, 1, 1)

        # If pos_bias already exists as attribute, delete it first
        if hasattr(self, "pos_bias"):
            del self.pos_bias
        # Register as buffer so it moves with model.to(device)
        self.register_buffer('pos_bias', bias, persistent=False)

    def _pad_bias_for_cls(self, bias: torch.Tensor, num_nodes_with_cls: int) -> torch.Tensor:
        """
        Pad bias matrix to accommodate CLS token
        
        Args:
            bias: Original bias (num_heads, N, N)
            num_nodes_with_cls: N+1 (after adding CLS)
            
        Returns:
            Padded bias (num_heads, N+1, N+1) with zeros for CLS row/col
        """
        num_heads = bias.shape[0]
        num_nodes = bias.shape[1]
        
        # Create padded bias with zeros
        padded_bias = torch.zeros(
            num_heads, num_nodes_with_cls, num_nodes_with_cls,
            dtype=bias.dtype, device=bias.device
        )
        
        # Copy original bias to bottom-right (skip first row/col for CLS)
        padded_bias[:, 1:, 1:] = bias
        
        # CLS row and column remain zero (no positional bias with/from CLS)
        return padded_bias

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, num_nodes, _ = x.shape
        # initialize learnable bias if needed
        if self.use_positional_bias and self.bias_type == 'learned':
            if self.pos_bias is None or not self.bias_initialized:
                self.initialize_positional_bias(num_nodes)

        # Linear transformations
        Q = self.query(x)  # (B, N, out_features)
        K = self.key(x)    # (B, N, out_features)
        V = self.value(x)  # (B, N, out_features)
        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        # Compute attention scores: Q @ K^T / sqrt(d)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, N, N)
        # Add positional bias if enabled
        if self.use_positional_bias and self.pos_bias is not None:
            # pos_bias shape: (H, N_bias, N_bias) where N_bias might be N or N+1
            bias = self.pos_bias
            
            # If input has CLS token (num_nodes = N+1) but bias is (H, N, N), pad it
            if bias.shape[1] < num_nodes:
                bias = self._pad_bias_for_cls(bias, num_nodes)
            
            # Broadcast to batch: (1, H, N, N)
            attn_scores = attn_scores + bias.unsqueeze(0)

        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (B, H, N, d)
        # Concatenate or average heads
        if self.concat_heads:
            out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)  # (B, N, out_features)
        else:
            out = out.mean(dim=1)  # (B, N, d)
        
        # Output projection
        out = self.out_proj(out)
        
        if return_attention:
            return out, attn_weights
        return out, None

class SpatialGraphTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        ffn_hidden_dim: int = 256,
        dropout: float = 0.2,
        activation: str = 'gelu',
        use_positional_bias: bool = True,
        bias_type: str = 'distance'
    ):
        super().__init__()
        # Multi-head graph attention
        self.graph_attn = GraphAttentionLayer(
            in_features=hidden_dim,
            out_features=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_positional_bias=use_positional_bias,
            bias_type=bias_type,
            concat_heads=True
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
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x = x + attn(LN(x))
        normed = self.norm1(x)
        attn_out, attn_weights = self.graph_attn(normed, return_attention=return_attention)
        x = x + self.dropout(attn_out)
        # x = x + ffn(LN(x))
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        
        return x, attn_weights