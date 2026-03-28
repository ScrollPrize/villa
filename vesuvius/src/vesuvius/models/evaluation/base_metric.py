from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import numpy as np


def _sigmoid_if_needed(array: np.ndarray) -> np.ndarray:
    arr = array.astype(np.float32, copy=False)
    if arr.size == 0:
        return arr
    if np.nanmin(arr) < 0.0 or np.nanmax(arr) > 1.0:
        return 1.0 / (1.0 + np.exp(-arr))
    return arr


def prediction_to_discrete_labels(pred_np: np.ndarray) -> np.ndarray:
    """Convert model outputs into discrete label maps for evaluation metrics."""
    if pred_np.ndim == 5:
        if pred_np.shape[1] > 1:
            return np.argmax(pred_np, axis=1).astype(np.int32)
        probs = _sigmoid_if_needed(np.squeeze(pred_np, axis=1))
        return (probs >= 0.5).astype(np.int32)

    if pred_np.ndim == 4:
        if pred_np.shape[1] <= 10:
            if pred_np.shape[1] > 1:
                return np.argmax(pred_np, axis=1).astype(np.int32)
            probs = _sigmoid_if_needed(np.squeeze(pred_np, axis=1))
            return (probs >= 0.5).astype(np.int32)
        return pred_np

    if pred_np.ndim == 3 and pred_np.shape[0] <= 10:
        if pred_np.shape[0] > 1:
            return np.argmax(pred_np, axis=0).astype(np.int32)
        probs = _sigmoid_if_needed(np.squeeze(pred_np, axis=0))
        return (probs >= 0.5).astype(np.int32)

    return pred_np


class BaseMetric(ABC):
    def __init__(self, name: str):
        self.name = name
        self.results = []
    
    @abstractmethod
    def compute(self, pred: torch.Tensor, gt: torch.Tensor, **kwargs) -> Dict[str, float]:
        pass
    
    def update(self, pred: torch.Tensor, gt: torch.Tensor, **kwargs):
        result = self.compute(pred, gt, **kwargs)
        self.results.append(result)
        return result
    
    def aggregate(self) -> Dict[str, float]:
        if not self.results:
            return {}
        
        aggregated = {}
        all_keys = set()
        for result in self.results:
            all_keys.update(result.keys())
        
        for key in all_keys:
            values = [r[key] for r in self.results if key in r]
            if values:
                aggregated[key] = np.mean(values)
        
        return aggregated
    
    def reset(self):
        self.results = []
