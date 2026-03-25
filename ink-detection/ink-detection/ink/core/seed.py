from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int | None, *, cudnn_deterministic: bool = True) -> int | None:
    """Seed process-wide Python, NumPy, and Torch RNG state."""
    if seed is None:
        return None

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    torch.backends.cudnn.benchmark = False
    return seed
