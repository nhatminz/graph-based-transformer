
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="None")
        p_t = torch.exp(-ce_loss)
        #FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        focal_loss = self.alpha * (1 - p_t)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)

        num_classes = input.size(-1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1
        )

        targets_smooth = targets_one_hot * (1 - self.smoothing) + \
                        self.smoothing / num_classes
        loss = -(targets_smooth * log_probs).sum(dim=-1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=kwargs.get('focal_alpha', 0.25),
            gamma=kwargs.get('focal_gamma', 2.0)
        )
    elif loss_type == 'label_smoothing':
        return LabelSmoothingCrossEntropy(
            smoothing=kwargs.get('label_smoothing', 0.1)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

class MetricsCalculator:
    @staticmethod
    def calculate_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        num_classes: int
    ) -> Dict[str, float]:
        labels = list(range(num_classes))  # [0,1,2]
        # Accuracy
        accuracy = accuracy_score(targets, predictions)
        
        # Precision, Recall, F1 (macro and weighted)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            targets, predictions, labels=labels, average='macro', zero_division=0
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            targets, predictions, labels=labels, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            targets, predictions, labels=labels, average=None, zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(targets, predictions, labels=list(range(num_classes)))
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': conf_matrix,
        }
        
        # Add per-class metrics
        for i in range(num_classes):
            metrics[f'precision_class_{i}'] = precision_per_class[i]
            metrics[f'recall_class_{i}'] = recall_per_class[i]
            metrics[f'f1_class_{i}'] = f1_per_class[i]
            metrics[f'support_class_{i}'] = support[i]
        
        return metrics

class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' - whether lower or higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.is_better = lambda new, best: new < best - min_delta
        else:
            self.is_better = lambda new, best: new > best + min_delta
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current score (loss or metric)
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(
    state: dict,
    filename: str,
    is_best: bool = False,
    best_filename: str = None
):
    """
    Save model checkpoint
    
    Args:
        state: Dictionary containing model state and other info
        filename: Path to save checkpoint
        is_best: Whether this is the best model so far
        best_filename: Path to save best model
    """
    torch.save(state, filename)
    if is_best and best_filename:
        torch.save(state, best_filename)

def load_checkpoint(
    filename: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = 'cuda'
) -> Tuple[int, float]:
    """
    Load model checkpoint
    
    Args:
        filename: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into
        device: Device to load model onto
        
    Returns:
        Tuple of (epoch, best_metric)
    """
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', 0.0)
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return epoch, best_metric