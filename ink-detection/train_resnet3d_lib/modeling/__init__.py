"""Modeling helpers for train_resnet3d.

- `model_config`: grouped config dataclasses + builders
- `group_dro`: GroupDRO objective helper
- `losses`: reusable loss utilities
- `architecture`: decoder/norm helpers
- `runtime_init`: RegressionPLModel initialization wiring
- `train_val_runtime`: training/validation loop logic helpers
- `optimizers_runtime`: optimizer/scheduler builders
"""

__all__ = [
    "model_config",
    "group_dro",
    "losses",
    "architecture",
    "runtime_init",
    "train_val_runtime",
    "optimizers_runtime",
]
