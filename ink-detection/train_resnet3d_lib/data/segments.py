"""Public segment assembly surface used by datasets_builder.

This module stays stable for imports while implementation is split by concern:
- `segment_groups`: group metadata/context
- `segment_trainval`: train/val segment loading
- `segment_stitching`: stitch loader/output builders
"""

from train_resnet3d_lib.data.segment_groups import (
    build_group_metadata,
    segment_group_context,
)
from train_resnet3d_lib.data.segment_trainval import (
    load_train_segment,
    load_train_segment_lazy,
    load_val_segment,
    load_val_segment_lazy,
)
from train_resnet3d_lib.data.segment_stitching import (
    build_log_only_outputs,
    build_log_only_stitch_loaders,
    build_log_only_stitch_loaders_lazy,
    build_train_stitch_loaders,
    build_train_stitch_loaders_lazy,
    build_train_stitch_outputs,
)


__all__ = [
    "build_group_metadata",
    "segment_group_context",
    "load_train_segment",
    "load_train_segment_lazy",
    "load_val_segment",
    "load_val_segment_lazy",
    "build_train_stitch_loaders",
    "build_train_stitch_loaders_lazy",
    "build_train_stitch_outputs",
    "build_log_only_stitch_loaders",
    "build_log_only_stitch_loaders_lazy",
    "build_log_only_outputs",
]
