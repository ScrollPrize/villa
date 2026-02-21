"""Evaluation metrics for ink-detection experiments."""

__all__ = ["StreamingBinarySegmentationMetrics", "compute_stitched_metrics"]


def __getattr__(name):
    if name == "StreamingBinarySegmentationMetrics":
        from .binary_segmentation import StreamingBinarySegmentationMetrics

        return StreamingBinarySegmentationMetrics
    if name == "compute_stitched_metrics":
        from .stitched_metrics import compute_stitched_metrics

        return compute_stitched_metrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
