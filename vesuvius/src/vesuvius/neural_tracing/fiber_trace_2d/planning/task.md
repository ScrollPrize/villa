# Task: Truly Rolling Shared 3D Tiled Inference

Replace the current full-Z-stack sparse mmap accumulator with a genuinely
rolling, fixed-depth circular mmap accumulator. Its backing storage must scale
with the active Z window, Y/X dimensions, and channel count, but never with the
full output Z depth. The operating system should manage mmap page residency;
the application must not impose an artificial RAM budget.

Flush completed regions to output one small Zarr-aligned chunk at a time so
normalization and product finalization never allocate a full-XY slab or a
full-band multichannel temporary. Reuse circular slots only after their data
has been finalized and written. Infer every globally anchored model tile once;
do not partition the volume in a way that repeats model inference.

There must be one shared tiled-inference implementation for Lasagna and Fiber.
The callers may provide model/product adapters and narrowly scoped derived
output behavior, but must not independently implement tile traversal,
blending, scaling, accumulation, resume scheduling, flushing, or progress.
Audit and consolidate all current behavioral divergences while preserving the
documented output, scale, crop, resume, atomic-write, and numerical semantics.
