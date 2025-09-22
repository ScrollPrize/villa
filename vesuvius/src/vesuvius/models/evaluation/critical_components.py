"""
Critical components metric for 3D segmentation evaluation.
Adapted from the segmentation evaluation metrics.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Set, List
from collections import deque
from scipy.ndimage import label
from numba import njit, prange
import numba as nb
from .base_metric import BaseMetric


# Pre-compute 26-connected neighbor offsets
NEIGHBOR_OFFSETS = np.array([
    [dx, dy, dz] 
    for dx in [-1, 0, 1] 
    for dy in [-1, 0, 1] 
    for dz in [-1, 0, 1]
    if not (dx == 0 and dy == 0 and dz == 0)
])


class CriticalComponentsMetric(BaseMetric):
    def __init__(self):
        super().__init__("critical_components")
        self._visited_mask_cache = None
        self._cache_shape = None
    
    def compute(self, pred: torch.Tensor, gt: torch.Tensor, **kwargs) -> Dict[str, float]:
        """
        Compute positive and negative critical components.
        
        Parameters
        ----------
        pred : torch.Tensor
            Predicted segmentation tensor
        gt : torch.Tensor
            Ground truth segmentation tensor
        
        Returns
        -------
        Dict[str, float]
            Dictionary with positive and negative critical components
        """
        # Convert to numpy and handle different tensor shapes
        # Convert BFloat16 to Float32 before numpy conversion
        if pred.dtype == torch.bfloat16:
            pred = pred.float()
        if gt.dtype == torch.bfloat16:
            gt = gt.float()
        
        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        
        # Handle different input shapes
        if pred_np.ndim == 5:  # (batch, channels, depth, height, width)
            if pred_np.shape[1] > 1:
                pred_np = np.argmax(pred_np, axis=1)
            else:
                pred_np = pred_np.squeeze(1)
        elif pred_np.ndim == 4:  # (batch, depth, height, width) or (batch, channels, height, width)
            if pred_np.shape[1] <= 10 and pred_np.shape[1] > 1:  # Likely channels
                pred_np = np.argmax(pred_np, axis=1)
            elif pred_np.shape[1] == 1:
                pred_np = pred_np.squeeze(1)
        
        # Handle ground truth shapes
        if gt_np.ndim == 5:  # (batch, channels, depth, height, width)
            if gt_np.shape[1] == 1:
                gt_np = gt_np.squeeze(1)
            else:
                gt_np = np.argmax(gt_np, axis=1)
        elif gt_np.ndim == 4 and gt_np.shape[1] == 1:
            gt_np = gt_np.squeeze(1)
        
        # Process each batch item
        batch_size = pred_np.shape[0]
        
        # Check if we need to allocate or resize the visited mask cache
        volume_shape = pred_np.shape[1:] if pred_np.ndim == 4 else pred_np.shape[2:]
        if self._visited_mask_cache is None or self._cache_shape != volume_shape:
            self._visited_mask_cache = np.zeros(volume_shape, dtype=np.uint8)
            self._cache_shape = volume_shape
        
        # Process batch in parallel
        pos_criticals = np.zeros(batch_size, dtype=np.int32)
        neg_criticals = np.zeros(batch_size, dtype=np.int32)
        
        for i in range(batch_size):
            # Binarize the predictions and ground truth
            pred_binary = (pred_np[i] > 0.5).astype(int) if pred_np[i].max() <= 1 else (pred_np[i] > 0).astype(int)
            gt_binary = (gt_np[i] > 0.5).astype(int) if gt_np[i].max() <= 1 else (gt_np[i] > 0).astype(int)
            
            # Compute critical components with cached mask
            pos_criticals[i] = detect_critical_3d(gt_binary, pred_binary, self._visited_mask_cache)
            neg_criticals[i] = detect_critical_3d(pred_binary, gt_binary, self._visited_mask_cache)
        
        # Average over batch
        return {
            "critical components negative": float(np.mean(neg_criticals)),
            "critical components positive": float(np.mean(pos_criticals)),
            "critical components total": float(np.mean(pos_criticals + neg_criticals)),
        }


@njit(parallel=True)
def false_negative_mask(y_target: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Computes the false negative mask."""
    return (y_target > 0) & (y_pred == 0)


@njit
def extract_component_jit(
    y_target: np.ndarray,
    y_mistakes: np.ndarray,
    y_minus_mistakes: np.ndarray,
    start_x: int,
    start_y: int,
    start_z: int,
    visited_mask: np.ndarray  # Now expects uint8
) -> Tuple[bool, int]:
    """JIT-compiled BFS for extracting connected components."""
    shape = y_target.shape
    target_val = y_target[start_x, start_y, start_z]
    
    # Initialize queue with pre-allocated size
    queue_size = 10000
    queue_x = np.empty(queue_size, dtype=np.int32)
    queue_y = np.empty(queue_size, dtype=np.int32)
    queue_z = np.empty(queue_size, dtype=np.int32)
    
    queue_x[0] = start_x
    queue_y[0] = start_y
    queue_z[0] = start_z
    queue_start = 0
    queue_end = 1
    
    visited_mask[start_x, start_y, start_z] = 1
    visited_count = 1
    
    # Track collisions using arrays (since numba doesn't support dicts well)
    max_labels = 1000
    collision_keys = np.full(max_labels, -1, dtype=np.int32)
    collision_vals = np.full(max_labels, -1, dtype=np.int32)
    n_collisions = 0
    is_critical = False
    
    # 26-connected offsets
    offsets = np.array([
        [-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
        [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
        [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
        [0, -1, -1], [0, -1, 0], [0, -1, 1],
        [0, 0, -1], [0, 0, 1],
        [0, 1, -1], [0, 1, 0], [0, 1, 1],
        [1, -1, -1], [1, -1, 0], [1, -1, 1],
        [1, 0, -1], [1, 0, 0], [1, 0, 1],
        [1, 1, -1], [1, 1, 0], [1, 1, 1]
    ])
    
    while queue_start < queue_end:
        # Dequeue
        x = queue_x[queue_start]
        y = queue_y[queue_start]
        z = queue_z[queue_start]
        queue_start += 1
        
        # Check neighbors
        for i in range(26):
            nx = x + offsets[i, 0]
            ny = y + offsets[i, 1]
            nz = z + offsets[i, 2]
            
            # Bounds check
            if nx >= 0 and nx < shape[0] and ny >= 0 and ny < shape[1] and nz >= 0 and nz < shape[2]:
                if visited_mask[nx, ny, nz] == 0 and y_target[nx, ny, nz] == target_val:
                    visited_mask[nx, ny, nz] = 1
                    visited_count += 1
                    
                    if y_mistakes[nx, ny, nz] == 1:
                        # Enqueue
                        if queue_end >= queue_size:
                            # Resize queue
                            new_size = queue_size * 2
                            new_queue_x = np.empty(new_size, dtype=np.int32)
                            new_queue_y = np.empty(new_size, dtype=np.int32)
                            new_queue_z = np.empty(new_size, dtype=np.int32)
                            new_queue_x[:queue_size] = queue_x
                            new_queue_y[:queue_size] = queue_y
                            new_queue_z[:queue_size] = queue_z
                            queue_x = new_queue_x
                            queue_y = new_queue_y
                            queue_z = new_queue_z
                            queue_size = new_size
                        
                        queue_x[queue_end] = nx
                        queue_y[queue_end] = ny
                        queue_z[queue_end] = nz
                        queue_end += 1
                    elif not is_critical:
                        key = int(y_target[nx, ny, nz])
                        val = int(y_minus_mistakes[nx, ny, nz])
                        
                        # Check collisions
                        found = False
                        for j in range(n_collisions):
                            if collision_keys[j] == key:
                                if collision_vals[j] != val:
                                    is_critical = True
                                found = True
                                break
                        
                        if not found and n_collisions < max_labels:
                            collision_keys[n_collisions] = key
                            collision_vals[n_collisions] = val
                            n_collisions += 1
    
    # Check if target_val is in collisions
    found_target = False
    for j in range(n_collisions):
        if collision_keys[j] == int(target_val):
            found_target = True
            break
    
    if not found_target:
        is_critical = True
    
    return is_critical, visited_count


def get_foreground(img: np.ndarray) -> np.ndarray:
    """Retrieves the foreground voxel coordinates as a numpy array."""
    return np.column_stack(np.nonzero(img))


def detect_critical_3d(y_target: np.ndarray, y_pred: np.ndarray, visited_mask: np.ndarray = None) -> int:
    """Detects critical components in a 3D segmentation using JIT-compiled BFS."""
    y_mistakes = false_negative_mask(y_target, y_pred)
    y_target_minus_mistakes, _ = label(y_target * (1 - y_mistakes))
    n_criticals = 0
    
    # Get foreground coordinates as numpy array
    foreground_coords = get_foreground(y_mistakes)
    if len(foreground_coords) == 0:
        return 0
    
    # Use provided mask or create new one
    if visited_mask is None:
        visited_global = np.zeros(y_target.shape, dtype=np.uint8)
    else:
        visited_global = visited_mask
        visited_global.fill(0)  # Reset the mask
    
    # Process each connected component
    for i in range(len(foreground_coords)):
        x, y, z = foreground_coords[i]
        if visited_global[x, y, z] == 0:
            is_critical, _ = extract_component_jit(
                y_target, y_mistakes, y_target_minus_mistakes,
                x, y, z, visited_global
            )
            if is_critical:
                n_criticals += 1
    
    return n_criticals
