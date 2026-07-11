# Coordinate-Space Geometric TTA For Trace2CP

Remove all image-space geometric augmentation functions from 2D fiber tracing.
They must not exist anywhere in `fiber_trace_2d`. Geometric augmentations must
always be coordinate manipulations applied before volume sampling/slicing the
patch, never image-space operations applied after a patch image has already
been sampled.

For Trace2CP median-TTA tracing, build TTA variants from oversized coordinate
patches. Each augmented output strip should be large enough to contain the
current base strip footprint after the requested augmentation, accepting
invalid/black areas where the expanded coordinate strip has no valid samples.

Add a `--vis-tta` debug flag. When TTA is active, it should export all
individual TTA slices with the transformed base-strip corners marked by drawing
lines between those corners, so the TTA geometry can be inspected directly.
