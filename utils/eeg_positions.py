
import numpy as np
import torch
from typing import Dict, Tuple, Optional
from scipy.spatial.distance import pdist, squareform

#62 channel names
CHANNEL_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz',
    'O2', 'CB2'
]

#electrode positions (angle, radius)
CHANNEL_POSITIONS_POLAR = np.array([
    [-18, 0.51111],   # Fp1
    [0, 0.51111],     # Fpz
    [18, 0.51111],    # Fp2
    [-23, 0.41111],   # AF3
    [23, 0.41111],    # AF4
    [-54, 0.51111],   # F7
    [-49, 0.41667],   # F5
    [-39, 0.33333],   # F3
    [-22, 0.27778],   # F1
    [0, 0.25556],     # Fz
    [22, 0.27778],    # F2
    [39, 0.33333],    # F4
    [49, 0.41667],    # F6
    [54, 0.51111],    # F8
    [-72, 0.51111],   # FT7
    [-69, 0.39444],   # FC5
    [-62, 0.27778],   # FC3
    [-45, 0.17778],   # FC1
    [0, 0.12778],     # FCz
    [45, 0.17778],    # FC2
    [62, 0.27778],    # FC4
    [69, 0.39444],    # FC6
    [72, 0.51111],    # FT8
    [-90, 0.51111],   # T7
    [-90, 0.38333],   # C5
    [-90, 0.25556],   # C3
    [-90, 0.12778],   # C1
    [90, 0.0],        # Cz (at center)
    [90, 0.12778],    # C2
    [90, 0.25556],    # C4
    [90, 0.38333],    # C6
    [90, 0.51111],    # T8
    [-105, 0.51111],  # TP7
    [-111, 0.39444],  # CP5
    [-118, 0.27778],  # CP3
    [-135, 0.17778],  # CP1
    [180, 0.12778],   # CPz
    [135, 0.17778],   # CP2
    [118, 0.27778],   # CP4
    [111, 0.39444],   # CP6
    [105, 0.51111],   # TP8
    [-120, 0.51111],  # P7
    [-131, 0.41667],  # P5
    [-141, 0.33333],  # P3
    [-158, 0.27778],  # P1
    [180, 0.25556],   # Pz
    [158, 0.27778],   # P2
    [141, 0.33333],   # P4
    [131, 0.41667],   # P6
    [120, 0.51111],   # P8
    [-135, 0.51111],  # PO7
    [-147, 0.46838],  # PO5
    [-157, 0.41111],  # PO3
    [180, 0.38333],   # POz
    [157, 0.41111],   # PO4
    [147, 0.46838],   # PO6
    [135, 0.51111],   # PO8
    [-150, 0.51111],  # CB1
    [-165, 0.51111],  # O1
    [180, 0.51111],   # Oz
    [165, 0.51111],   # O2
    [150, 0.51111],   # CB2
])

def polar_to_cartesian_2d(angles: np.ndarray, radii: np.ndarray) -> np.ndarray:
    #degrees to radians
    angles_rad = np.deg2rad(angles)
    # x = rcos(theta), y = rsin(theta)
    x = radii * np.sin(angles_rad)
    y = radii * np.cos(angles_rad)

    return np.column_stack([x, y])

def polar_to_cartesian_3d(angles: np.ndarray, radii: np.ndarray) -> np.ndarray:
    azimuth = np.deg2rad(angles)
    polar = radii * np.pi
    x = np.sin(polar) * np.sin(azimuth)
    y = np.sin(polar) * np.cos(azimuth)
    z = np.cos(polar)
    return np.column_stack([x,y,z])

#generate 2d
CHANNEL_POSITIONS_2D = polar_to_cartesian_2d(
    CHANNEL_POSITIONS_POLAR[:, 0],
    CHANNEL_POSITIONS_POLAR[:, 1]
)

#generate 3d 
CHANNEL_POSITIONS_3D = polar_to_cartesian_3d(
    CHANNEL_POSITIONS_POLAR[:, 0],
    CHANNEL_POSITIONS_POLAR[:, 1]
)

def compute_distance_matrix(positions: np.ndarray) -> np.ndarray:
    # compute base on 3d positions
    distances = squareform(pdist(positions, metric = 'euclidean'))
    return distances

def distance_to_adjacency(
    distance_matrix: np.ndarray,
    threshold: Optional[float] = None,
    sigma: Optional[float] = None,
    method: str = 'threshold' #threshold or gaussian
) -> np.ndarray:
    if method == 'threshold':
        if threshold is None:
            threshold = np.median(distance_matrix)
        adj = (distance_matrix <= threshold).astype(np.float32)
    elif method == 'gaussian':
        if sigma is None:
            sigma = np.std(distance_matrix)
        adj = np.exp(-distance_matrix**2/ (2 * sigma**2))

    else:
        raise ValueError(f"Unknown method: {method}")

    return adj

def compute_correlation_matrix(
    features: np.ndarray,
    method: str='pearson'
) -> np.ndarray:
    """
    Compute correlation matrix between channels
    
    Args:
        features: Array of shape (num_samples, num_channels, num_features)
        method: 'pearson' or 'spearman'
        
    Returns:
        Correlation matrix of shape (num_channels, num_channels)
    """
    num_channels = features.shape[1]
    corr_matrix = np.zeros((num_channels, num_channels))

    for i in range(num_channels):
        for j in range(num_channels):
            if method == 'pearson':
                corr_matrix[i,j] = np.corrcoef(
                    features[:, i].flatten(),
                    features[:, j].flatten()
                )[0,1]
            else:
                from scipy.stats import spearmanr
                corr_matrix[i,j] = spearmanr(
                    features[:, i].flatten(),
                    features[:, j].flatten() 
                )[0]
    #Handle nan values
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    return corr_matrix

def create_hybrid_adjacency(
    distance_adj: np.ndarray,
    correlation_adj: np.ndarray,
    distance_weight: float = 0.5,
) -> np.ndarray:
    """
    Create hybrid adjacency matrix combining distance and correlation
    
    Args:
        distance_adj: Distance-based adjacency
        correlation_adj: Correlation-based adjacency
        distance_weight: Weight for distance component
        
    Returns:
        Hybrid adjacency matrix
    """
    return distance_weight * distance_adj + (1 - distance_weight) * correlation_adj

def normalize_adjacency(adj: np.ndarray, method: str = "symmetric") -> np.ndarray:
    # set diagonal elements to 1
    adj_with_self_loops = adj + np.eye(adj.shape[0]) 
    # degree matrix
    degree = np.sum(adj_with_self_loops, axis = 1)

    if method == 'symmetric':
        # D^{-1/2} A D^{-1/2}
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.
        degree_mat_inv_sqrt = np.diag(degree_inv_sqrt) # Transform a vector into a diagonal matrix
        normalized_adj = degree_mat_inv_sqrt @ adj_with_self_loops @ degree_mat_inv_sqrt

    elif method == 'random_walk':
        # D^{-1} A
        degree_inv = np.power(degree, -1.0)
        degree_inv[np.isinf(degree_inv)] = 0.
        degree_mat_inv = np.diag(degree_inv)
        normalized_adj = degree_mat_inv @ adj_with_self_loops
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_adj

def get_channel_regions() -> Dict[str, list]:
    regions = {
        'frontal': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],  # FP, AF, F
        'frontocentral': [14, 15, 16, 17, 18, 19, 20, 21, 22],  # FT, FC
        'central': [23, 24, 25, 26, 27, 28, 29, 30, 31],  # T, C
        'centroparietal': [32, 33, 34, 35, 36, 37, 38, 39, 40],  # TP, CP
        'parietal': [41, 42, 43, 44, 45, 46, 47, 48, 49],  # P
        'occipital': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61],  # PO, O, CB
    }
    return regions

# Pre-compute commonly used matrices
DISTANCE_MATRIX = compute_distance_matrix(CHANNEL_POSITIONS_3D)
DEFAULT_ADJACENCY = distance_to_adjacency(DISTANCE_MATRIX, method='gaussian')
NORMALIZED_ADJACENCY = normalize_adjacency(DEFAULT_ADJACENCY, method='symmetric')