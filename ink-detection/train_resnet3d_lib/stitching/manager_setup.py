"""Compatibility exports for StitchManager setup helpers.

Primary implementation now lives in `train_resnet3d_lib.stitch_manager`.
"""

from train_resnet3d_lib.stitch_manager import initialize_manager_state, register_initial_segments

__all__ = [
    "initialize_manager_state",
    "register_initial_segments",
]
