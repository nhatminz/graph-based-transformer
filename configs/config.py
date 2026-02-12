from dataclasses import dataclass, field
from typing import Literal, List

@dataclass
class DataConfig:
    seed_eeg_path: str = "/kaggle/input/seed-dataset/SEED/SEED_EEG/ExtractedFeatures_1s" # 15 people, .mat
    label_mat_path: str = (
        "/kaggle/input/seed-dataset/SEED/SEED_EEG/ExtractedFeatures_1s/label.mat"
    )
    feature_key_prefix: str = "de_LDS"   # alternative: "de_movingAve"
    
    seed_multimodal_path: str = "/kaggle/input/seed-dataset/SEED/SEED_Multimodal/Chinese/02-EEG-DE-feature/eeg_used_1s" # 12 people, .npz

    use_mat: bool = False  # False: use 12 people, True: 15 people

    # EEG properties
    num_channels: int = 62
    num_bands: int = 5
    window_size: int = 1
    stride: int = 1 
    # Preprocessing
    normalize: bool = True
    normalization_method: Literal['zscore', 'minmax'] = 'zscore' # Limit the valid method

@dataclass
class SplitConfig:
    """
    mode='within'  →  single-subject, clip-level split  (standard SEED protocol)
        train = clips 1..9 minus val_clip
        val   = val_clip
        test  = clips 10..15
    mode='cross'   →  cross-subject split using the three subject lists below.
    """
    mode: Literal['within', 'cross'] = 'within'

    # within-subject only
    val_clip: int = 1   # which clip (1-9) to hold out as val

    # cross-subject only  (ignored when mode='within')
    train_subjects: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 8, 9, 10]
    )
    val_subjects:   List[int] = field(
        default_factory=lambda: [11, 12]
    )
    test_subjects:  List[int] = field(
        default_factory=lambda: [13, 14]
    )


@dataclass
class GraphConfig:
    graph_type: Literal['distance', 'correlation', 'learned', 'hybrid'] = 'hybrid'

    #distance-based graph
    distance_threshold: float = 0.3
    use_physical_distance: bool = True

    #correlation-based graph
    correlation_window: int = 200
    correlation_threshold: float = 0.5

    #learned graph
    learned_adjacency: bool = True
    adjacency_init: Literal['distance', 'correlation', 'identity'] = 'distance'

    #graph properties
    use_self_loops: bool = True
    use_edge_weights:       bool  = True
    symmetric_adjacency:    bool  = True

@dataclass
class ModelConfig:
    # embedding
    input_dim:  int = 5
    hidden_dim: int = 64

    # spatial graph transformer
    num_spatial_layers: int = 3
    num_attention_heads: int = 4
    spatial_dropout: float = 0.2
    use_positional_bias: bool = True
    bias_type: Literal['distance','correlation','learned'] = 'distance'
    # readout
    pooling_method: Literal['mean','max','attention','sagpool','cls'] = 'attention'
    pooling_ratio:  float = 0.5

    # temporal transformer
    num_temporal_layers: int = 3
    temporal_heads: int = 4
    temporal_dropout: float = 0.2
    # NOTE: set automatically to DataConfig.window_size in ExperimentConfig.__post_init__
    temporal_sequence_length: int   = 1 # window size
    # FFN
    ffn_hidden_dim: int   = 256
    ffn_dropout: float = 0.3
    # output
    num_classes: int = 3   # SEED: {negative, neutral, positive}
    # regularization
    layer_norm: bool = True
    use_residual: bool = True
    activation: Literal['relu','gelu','elu']  = 'gelu'

@dataclass
class TrainingConfig:
    # basics
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    # optimiser / scheduler
    optimizer: Literal['adam','adamw','sgd'] = 'adamw'
    scheduler: Literal['cosine','step','plateau','none'] = 'cosine'
    warmup_epochs: int = 5
    # loss
    loss_type: Literal['ce','focal','label_smoothing'] = 'ce'
    label_smoothing: float = 0.1
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    # early stopping
    early_stopping: bool  = False
    patience: int = 15
    min_delta: float = 1e-4
    # gradient clipping
    grad_clip_norm: float = 1.0  # max norm for gradient clipping
    # cross-subject evaluation
    cross_subject_sessions: Literal['first', 'all'] = 'all'  # Use session 1 only or all 3
    # checkpointing
    save_best_only: bool = True
    checkpoint_dir: str  = "/kaggle/working/checkpoints"
    # logging
    log_interval: int  = 10
    use_tensorboard: bool = True
    # dataloader  (memory-pinning is decided automatically by SeedLoaderFactory)
    num_workers: int = 2
    device: str = "cuda"
    seed: int = 42

@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_name: str = "ssgt_seed"
    description:     str = "Spatial-Spectral Graph Transformer – EEG Emotion Recognition"
    def __post_init__(self):
        # 1) hidden_dim must be divisible by num_attention_heads
        if self.model.hidden_dim % self.model.num_attention_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.model.hidden_dim}) must be divisible by "
                f"num_attention_heads ({self.model.num_attention_heads})"
            )
        # 2) keep temporal_sequence_length in sync with window_size
        self.model.temporal_sequence_length = self.data.window_size


def get_default_config() -> ExperimentConfig:
    return ExperimentConfig()

def get_seed_within_config() -> ExperimentConfig:
    """Standard SEED within-subject, clip-level split."""
    cfg = ExperimentConfig()
    cfg.experiment_name  = "ssgt_seed_within"
    cfg.model.num_classes = 3
    cfg.split.mode = "within"
    return cfg

def get_seed_cross_config() -> ExperimentConfig:
    """SEED cross-subject split (default 9/3/3 subjects)."""
    cfg = ExperimentConfig()
    cfg.experiment_name  = "ssgt_seed_cross"
    cfg.model.num_classes = 3
    cfg.split.mode = "cross"
    return cfg