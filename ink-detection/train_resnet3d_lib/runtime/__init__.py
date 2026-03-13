"""Runtime orchestration and experiment lifecycle modules.

- `orchestration`: CLI/runtime state assembly
- `wandb_runtime`: W&B init/sync + summary definition
- `wandb_local_metrics`: local-safe W&B logger wrapper
- `checkpointing`: checkpoint path/state helpers
- `ensemble`: checkpoint selection/weighting/averaging helpers
- `run_naming`: run slug generation
"""

__all__ = [
    "orchestration",
    "wandb_runtime",
    "wandb_local_metrics",
    "checkpointing",
    "ensemble",
    "run_naming",
]
