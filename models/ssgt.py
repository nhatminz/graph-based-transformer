#ssgt.py
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple

class SpatialSpectralGraphTransformer(nn.Module):
    def __init__(
        self,
        #input, output
        num_channels: int = 62,
        num_bands: int = 5,
        num_classes: int = 3,

        # Embedding
        hidden_dim: int = 64,

        # Spatial Graph Transformer
        num_spatial_layers: int = 3,
        spatial_heads: int = 4,
        spatial_dropout: float = 0.2,
        use_positional_bias: bool = True,
        bias_type: str = 'distance',

        # Readout/Pooling
        pooling_method: str = 'attention', # ('attention', 'mean', 'max', 'cls')
        use_temporal_cls: bool = False,  # Whether to use CLS token for temporal pooling

        # Temporal Transformer
        num_temporal_layers: int = 3,
        temporal_heads: int = 4,
        temporal_dropout: float = 0.2,
        temporal_sequence_length: int = 2,

        # FFN
        ffn_hidden_dim: int = 256,
        ffn_dropout: float = 0.3,

        # Architecture options
        activation: str = 'gelu',
        use_layer_norm: bool = True,

        # Graph structure
        adjacency_matrix: Optional[np.ndarray] = None, #Pre-computed adjacency matrix (num_channels, num_channels)
        distance_matrix: Optional[np.ndarray] = None #Pre-computed distance matrix (num_channels, num_channels)
    ):
        super().__init__()
        
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.pooling_method = pooling_method
        self.use_positional_bias = use_positional_bias
        self.bias_type = bias_type
        self.use_temporal_cls = use_temporal_cls
        # Temporal CLS token (if enabled)
        if use_temporal_cls:
            self.temporal_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.trunc_normal_(self.temporal_cls_token, std=0.02)
        else:
            self.temporal_cls_token = None

        # input embedding 
        # Project from num_bands to hidden_dim for each channel
        self.input_embedding = nn.Sequential(
            nn.Linear(num_bands, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(spatial_dropout)
        )
        #Spatial Graph Transformer
        self.spatial_layers = nn.ModuleList([
            SpatialGraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=spatial_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=spatial_dropout,
                activation=activation,
                use_positional_bias=use_positional_bias,
                bias_type=bias_type
            )
            for _ in range(num_spatial_layers)
        ])
        # Set positional bias for spatial layers
        if use_positional_bias and bias_type != 'learned':
            self._initialize_positional_bias(distance_matrix, adjacency_matrix)

        # Graph Readout 
        self.readout = GraphReadout(
            hidden_dim=hidden_dim,
            method=pooling_method,
            dropout=spatial_dropout
        )
        # If using CLS token, we need to account for it in spatial layers
        self.use_cls = (pooling_method == 'cls')
        # temporal transformer
        self.temporal_transformer = TemporalTransformer(
            hidden_dim=hidden_dim,
            num_layers=num_temporal_layers,
            num_heads=temporal_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=temporal_dropout,
            activation=activation,
            use_positional_encoding=True,
            max_seq_len=128  # Large enough for most sequences
        )
        #classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _initialize_positional_bias(
        self,
        distance_matrix: Optional[np.ndarray],
        adjacency_matrix: Optional[np.ndarray]
    ):
        if self.bias_type == 'distance' and distance_matrix is not None:
            # Convert distance to negative bias (closer = less negative = higher attention)
            # Using negative normalized distance with learnable scale
            dist_normalized = distance_matrix / (np.max(distance_matrix) + 1e-8)
            bias = -dist_normalized  # Range: [-1, 0], closer nodes have values near 0
            bias = torch.FloatTensor(bias)
            
            # Add learnable bias scale (will be small, ~0.1-0.5)
            self.bias_scale = nn.Parameter(torch.tensor(0.1))
            
        elif self.bias_type == 'correlation' and adjacency_matrix is not None:
            # Use adjacency/correlation as bias
            # Assuming adjacency_matrix contains correlation values or topology weights
            bias = torch.FloatTensor(adjacency_matrix)
            self.bias_scale = nn.Parameter(torch.tensor(1.0))
            
        else:
            # Default: no bias
            bias = torch.zeros(self.num_channels, self.num_channels)
            self.bias_scale = None
        
        # Set bias for all spatial layers
        for layer in self.spatial_layers:
            layer.graph_attn.set_positional_bias(bias)

    def forward_spatial(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        
        attention_weights = [] if return_attention else None
        
        # Add CLS token if needed
        if self.use_cls:
            x = self.readout.prepend_cls_if_needed(x)
        
        # Pass through spatial layers
        for layer in self.spatial_layers:
            x, attn = layer(x, return_attention=return_attention)
            if return_attention:
                attention_weights.append(attn)
        
        return x, attention_weights

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, num_channels, num_bands)
               OR (batch_size, num_channels, num_bands) for single time step
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - 'logits': Classification logits (batch_size, num_classes)
                - 'probabilities': Class probabilities (batch_size, num_classes)
                - 'spatial_attention': Spatial attention weights (if return_attention)
                - 'temporal_attention': Temporal attention weights (if return_attention)
        """
        # Handle input shape
        if x.dim() == 3:
            # Single time step: (B, N, F) -> (B, 1, N, F)
            x = x.unsqueeze(1)
        
        batch_size, time_steps, num_channels, num_bands = x.shape
        
        # ============ Batch Process Spatial Layers ============
        # Reshape: (B, T, N, F) -> (B*T, N, F) for efficient batch processing
        x_flat = x.view(batch_size * time_steps, num_channels, num_bands)
        
        # Embedding: (B*T, N, F) -> (B*T, N, H)
        embedded = self.input_embedding(x_flat)
        
        # Spatial Graph Transformer: (B*T, N, H) -> (B*T, N, H) or (B*T, N+1, H) if CLS
        spatial_out, spatial_attn = self.forward_spatial(embedded, return_attention)
        
        # Reshape spatial attention if needed: split B*T back to B, T
        if return_attention and spatial_attn is not None:
            # spatial_attn is list of attention weights from each layer
            # Each element shape: (B*T, num_heads, N, N)
            spatial_attention_all = []
            for layer_attn in spatial_attn:
                if layer_attn is not None:
                    # Reshape: (B*T, H, N, N) -> (B, T, H, N, N)
                    reshaped = layer_attn.view(batch_size, time_steps, *layer_attn.shape[1:])
                    spatial_attention_all.append(reshaped)
                else:
                    spatial_attention_all.append(None)
        else:
            spatial_attention_all = None
        
        # Graph Readout: (B*T, N, H) or (B*T, N+1, H) -> (B*T, H)
        # Pass has_cls_token flag so readout can strip CLS if method != 'cls'
        graph_repr = self.readout(spatial_out, has_cls_token=self.use_cls)
        
        # Reshape back: (B*T, H) -> (B, T, H)
        temporal_features = graph_repr.view(batch_size, time_steps, self.hidden_dim)
        
        # ============ Temporal Transformer ============
        # Optionally prepend temporal CLS token
        if self.use_temporal_cls:
            batch_size = temporal_features.shape[0]
            cls_tokens = self.temporal_cls_token.expand(batch_size, -1, -1)
            temporal_features = torch.cat([cls_tokens, temporal_features], dim=1)  # (B, T+1, H)
        
        # Process temporal sequence: (B, T, H) or (B, T+1, H) -> same shape
        temporal_out, temporal_attn = self.temporal_transformer(
            temporal_features,
            return_attention=return_attention
        )
        
        # Extract temporal representation
        if self.use_temporal_cls:
            # Use CLS token representation
            temporal_repr = temporal_out[:, 0, :]  # (B, H)
        else:
            # Use mean pooling over time
            temporal_repr = temporal_out.mean(dim=1)  # (B, H)
        
        # ============ Classification ============
        logits = self.classifier(temporal_repr)  # (B, num_classes)
        probabilities = torch.softmax(logits, dim=-1)
        
        # Prepare output
        output = {
            'logits': logits,
            'probabilities': probabilities,
        }
        
        if return_attention:
            output['spatial_attention'] = spatial_attention_all
            output['temporal_attention'] = temporal_attn
        
        return output
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)