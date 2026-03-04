"""Runtime orchestration and experiment lifecycle modules.

- `orchestration`: CLI/runtime state assembly
- `wandb_runtime`: W&B init/sync + summary definition
- `wandb_local_metrics`: local-safe W&B logger wrapper
- `checkpointing`: checkpoint path/state helpers
- `run_naming`: run slug generation
- `metadata_config`: top-level stitch metadata wiring
"""

__all__ = [
    "orchestration",
    "wandb_runtime",
    "wandb_local_metrics",
    "checkpointing",
    "run_naming",
    "metadata_config",
]
