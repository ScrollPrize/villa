# Coordinate-Space Geometric TTA Log

Current task: plan deletion of all image-space geometric augmentation functions
from `fiber_trace_2d` and Trace2CP TTA conversion to coordinate-space volume
sampling from oversized coordinate strips.

Plan notes:

- The plan makes the spec rule explicit: geometric augmentations must never be
  image-space operations; they must be coordinate manipulations before
  sampling/slicing.
- The plan requires geometric image-space augmentation functions to be deleted,
  not just unused.
- The plan replaces Trace2CP's current image-warp TTA helper path with
  coordinate-sampled TTA strips.
- The plan adds `--vis-tta` debug output for per-TTA slices with transformed
  base-strip corner outlines.

Implementation notes:

- Pending.

Validation:

- Pending; this step only created the plan.
