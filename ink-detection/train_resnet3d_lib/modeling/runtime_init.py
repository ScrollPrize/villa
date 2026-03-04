"""Compatibility exports for model runtime initialization helpers.

Primary implementation now lives in `train_resnet3d_lib.model`.
"""

from train_resnet3d_lib.model import initialize_regression_state, save_regression_hyperparameters

__all__ = [
    "initialize_regression_state",
    "save_regression_hyperparameters",
]
