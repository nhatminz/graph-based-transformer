# # EEG Emotion Recognition using Spatial-Spectral Graph Transformer (SSGT)

## Introduction

This is an implementation of the **Spatial-Spectral Graph Transformer (SSGT)** method for emotion recognition from EEG signals. The model combines:

- **Graph Transformer** with positional bias to learn spatial features  
- **Temporal Transformer** to learn temporal features  
- **Attention-based Pooling** to aggregate information  
## Pipeline

```
Input (B, T, N, F) 
    ↓
Embedding (Linear + LayerNorm + GELU)
    ↓
Spatial Graph Transformer (N layers)
    - Multi-Head Graph Attention với Positional Bias
    - Feed-Forward Network
    - Residual Connections + LayerNorm
    ↓
Graph Readout/Pooling
    - Attention Pooling / Mean / Max / CLS Token
    ↓
Temporal Transformer (N layers)
    - Multi-Head Self-Attention
    - Positional Encoding
    - Feed-Forward Network
    ↓
Classifier (Linear + GELU + Dropout + Linear)
    ↓
Output (B, num_classes)
```

## Project structures

```
eeg_emotion_recognition/
├── configs/
│   ├── __init__.py
│   └── config.py                 # configurations
├── models/
│   ├── __init__.py
│   ├── graph_attention.py        # Graph Attention Layer
│   ├── pooling.py                # Pooling/Readout layers
│   ├── temporal_transformer.py   # Temporal Transformer
│   └── ssgt.py                   # Main SSGT model
├── data/
│   ├── __init__.py
│   └── dataset.py                # Dataset loaders 
├── utils/
│   ├── __init__.py
│   ├── eeg_positions.py          
│   └── training_utils.py         # Loss functions, metrics, early stopping
├── checkpoints/                  # Model checkpoints
├── train.py                      # Main training script
└── README.md
```

## Requirements

```bash
pip install torch torchvision
pip install numpy scipy scikit-learn
pip install tensorboard tqdm
pip install matplotlib seaborn
```

## Usage

### 1. Configuration

Edit the file configs/config.py to change hyperparameters:

```python
from configs.config import get_seed_config

config = get_seed_config()

config.model.hidden_dim = 128
config.model.num_spatial_layers = 4
config.training.batch_size = 64
config.training.learning_rate = 1e-3
```

### 2. Training

#### Train all 15 subjects with 10-fold cross-validation:

```python
from train import Trainer
from configs.config import get_seed_config

config = get_seed_config()
config.data.seed_eeg_path = "/path/to/SEED/SEED_EEG/ExtractedFeatures_1s"

trainer = Trainer(config)
results = trainer.train_all_subjects(session=1)
```

#### Train a specific subject:

```python
trainer = Trainer(config)
metrics = trainer.train_subject(subject_id=1, session=1, fold_idx=0)
```

#### Run from command line:

```bash
python train.py
```

### 3. Load and use a trained model

```python
import torch
from models.ssgt import SpatialSpectralGraphTransformer
from utils.eeg_positions import DISTANCE_MATRIX, NORMALIZED_ADJACENCY

# Load model
model = SpatialSpectralGraphTransformer(
    num_channels=62,
    num_bands=5,
    num_classes=3,
    hidden_dim=64,
    # ... other parameters
)

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    # data shape: (batch_size, time_steps, num_channels, num_bands)
    output = model(data)
    predictions = output['logits'].argmax(dim=1)
    probabilities = output['probabilities']
```

## Details of Main Components

### 1. Graph Construction

File: `utils/eeg_positions.py`

Supports multiple graph construction methods:
- **Distance-based**: Based on physical distances between electrodes
- **Correlation-based**: Based on correlation between channels
- **Hybrid**: Combination of both
- **Learned**: Adjacency matrix learned during training

```python
from utils.eeg_positions import (
    DISTANCE_MATRIX,
    compute_correlation_matrix,
    create_hybrid_adjacency
)

# Distance-based adjacency
dist_adj = distance_to_adjacency(DISTANCE_MATRIX, method='gaussian')

# Correlation-based
corr_matrix = compute_correlation_matrix(features)
corr_adj = distance_to_adjacency(corr_matrix, threshold=0.5)

# Hybrid
hybrid_adj = create_hybrid_adjacency(dist_adj, corr_adj, distance_weight=0.5)
```

### 2. Positional Bias

In the Graph Attention Layer, positional bias helps the model understand spatial structure:

```
Attention(Q, K, V) = Softmax((QK^T / √d) + Bias_pos) V
```

Có 3 loại bias:
- **distance**: From physical distance (Gaussian kernel)
- **correlation**: From correlation matrix
- **learned**: Learnable parameters

### 3. Pooling Methods

File: `models/pooling.py`

- **Attention Pooling**: Learns weights for each node
- **Mean/Max Pooling**: Simple and fast
- **CLS Token**: Similar to BERT, adds a special token

### 4. Loss Functions

File: `utils/training_utils.py`

- **CrossEntropy**
- **Focal Loss**
- **Label Smoothing**


## Monitoring training

```bash
# View tensorboard
tensorboard --logdir=runs/

# Check checkpoints
ls checkpoints/
```
## Contact
If you have any questions, please contact the author or open an issue on GitHub.
