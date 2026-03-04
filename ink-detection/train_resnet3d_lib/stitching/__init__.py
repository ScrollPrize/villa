"""Stitching internals split by concern.

- `roi_layout`: ROI resolution + buffer allocation
- `epoch_passes`: train/log-only stitch passes
- `manager_setup`: StitchManager initialization + segment registration
- `manager_runtime`: StitchManager runtime helpers (sync/context/epoch flow)
- `wandb_media`: stitched image logging
- `metrics_runtime`: stitched metric computation/logging
- `metrics_runtime_config`: stitched metrics schedule + option parsing
- `metrics_runtime_aggregation`: stitched metrics group/global aggregation logging
- `buffer_ops`: accumulation + probability-map composition
"""

__all__ = [
    "buffer_ops",
    "roi_layout",
    "epoch_passes",
    "manager_setup",
    "manager_runtime",
    "wandb_media",
    "metrics_runtime",
    "metrics_runtime_config",
    "metrics_runtime_aggregation",
]
