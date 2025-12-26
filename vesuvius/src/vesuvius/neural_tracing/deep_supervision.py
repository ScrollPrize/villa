import numpy as np
import torch.nn.functional as F

def _compute_ds_weights(n):
    if n <= 0:
        return None
    weights = np.array([1 / (2 ** i) for i in range(n)], dtype=np.float32)
    weights[-1] = 0.0  # discard the lowest-res prediction
    s = weights.sum()
    if s > 0:
        weights = weights / s
    return weights.tolist()


def _resize_for_ds(tensor, size, *, mode, align_corners=None):
    if tensor.shape[2:] == size:
        return tensor
    if align_corners is None:
        return F.interpolate(tensor.float(), size=size, mode=mode).to(tensor.dtype)
    return F.interpolate(tensor.float(), size=size, mode=mode, align_corners=align_corners).to(tensor.dtype)
