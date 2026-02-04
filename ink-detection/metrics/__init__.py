"""Evaluation metrics for ink-detection experiments."""

from .binary_segmentation import StreamingBinarySegmentationMetrics
from .textseg_metrics import compute_stitched_metrics

__all__ = ["StreamingBinarySegmentationMetrics", "compute_stitched_metrics"]
