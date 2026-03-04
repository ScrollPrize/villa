"""Compatibility exports for StitchManager runtime helpers.

Primary implementation now lives in `train_resnet3d_lib.stitch_manager`.
"""

from train_resnet3d_lib.stitch_manager import (
    build_validation_epoch_context,
    collect_val_segments_for_logging,
    distributed_world_size,
    maybe_run_log_only_stitch,
    maybe_run_train_stitch,
    precision_context,
    reduce_sum_distributed,
    reset_epoch_end_buffers,
    reset_split_buffers,
    sync_val_buffers_and_maybe_exit_nonzero_worker,
    sync_val_buffers_distributed,
)

__all__ = [
    "build_validation_epoch_context",
    "collect_val_segments_for_logging",
    "distributed_world_size",
    "maybe_run_log_only_stitch",
    "maybe_run_train_stitch",
    "precision_context",
    "reduce_sum_distributed",
    "reset_epoch_end_buffers",
    "reset_split_buffers",
    "sync_val_buffers_and_maybe_exit_nonzero_worker",
    "sync_val_buffers_distributed",
]
