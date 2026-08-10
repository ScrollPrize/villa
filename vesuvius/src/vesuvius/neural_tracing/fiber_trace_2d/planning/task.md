# Task: preload and parallelize fiberlet DP

Accelerate the experimental C++ `vc_fiberlets paths` stage for current small
test crops without changing its candidate set, objective, accepted paths, or
deterministic artifacts.

For now:

1. Load the complete scoring region once before any DP searches. Every stored
   prediction voxel needed by any accepted candidate corridor must have its
   fiber direction/presence and Lasagna normal sampled exactly once into an
   immutable in-memory volume.
2. Solve independent candidate paths concurrently with the requested fixed
   worker count. Sampling may use that count during the one preload, but there
   must be no nested per-candidate sampling/thread teams.
3. Print monotonic path-search progress from the CLI, including completed and
   total candidate searches, percentage, elapsed time, processing rate, and ETA.
   Rate-limit updates while guaranteeing a final completed update.

The complete scoring region may be represented densely because current test
crops are small. Preserve paths that leave the anchor selection box by using
the union of actual candidate-corridor bounds, clipped to the prediction grid.
