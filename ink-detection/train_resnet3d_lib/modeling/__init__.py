"""Modeling helpers for train_resnet3d.

- `group_dro`: GroupDRO objective helper
- `losses`: reusable loss utilities
- `architecture`: decoder/norm helpers
- `runtime_init`: compatibility re-export of init helpers from `train_resnet3d_lib.model`
- `train_val_runtime`: compatibility re-export of train/val helpers from `train_resnet3d_lib.model`
- `optimizers_runtime`: optimizer/scheduler builders
"""

__all__ = [
    "group_dro",
    "losses",
    "architecture",
    "runtime_init",
    "train_val_runtime",
    "optimizers_runtime",
]
