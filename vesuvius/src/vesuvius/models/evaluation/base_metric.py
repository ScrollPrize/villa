from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import numpy as np


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


def binarize_scores(scores):
    """Turn a single-channel score volume into class ids.

    Single-channel heads emit a score per voxel rather than a class id, so
    comparing one to the class ids 0/1 is essentially never true. Threshold
    instead: at 0 for logits (the shipped configs use ``activation: "none"``),
    at 0.5 once an activation has been applied.
    """
    import numpy as np

    cutoff = 0.0 if (scores.min() < 0.0 or scores.max() > 1.0) else 0.5
    return (scores > cutoff).astype(np.int64)
