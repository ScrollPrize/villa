# Task: anchor-seeded fiberlet over-segmentation

Build a staged C++ pipeline that over-segments a stored 3D fiber-prediction
volume into short, reliable fiberlets.

## Overall approach

1. Partition the stored prediction grid into coarse 3D cells, initially
   configurable from 2 to 8 stored-prediction voxels per side. Choose the
   largest cell size for which two distinct, approximately parallel fibers
   cannot occupy one cell. This represents the minimum sustained fiber/sheet
   separation, not an implementation of conceptual 4D Lasagna data. All
   low-level prediction arrays and coordinates remain 3D.
2. Extract zero, one, or two supported anchors per cell. Each anchor has a
   sub-voxel position and an unoriented principal fiber axis. Use Gaussian
   spatial weighting, prediction presence, and sign-invariant directional
   agreement. Jointly fit two arbitrary, potentially non-orthogonal directions
   and then discard either result if it has insufficient aligned presence
   support.
3. In a later stage, consider nearby anchor pairs up to the maximum prediction
   hole/confusion distance that the system is intended to bridge. Straighten
   each local problem with the endpoint axes and solve a directed path problem
   with no backtracking. The eventual path cost should retain prediction,
   direction, tangent curvature, and normal curvature terms. Lasagna
   tangent/normal geometry may constrain or prune the search volume. Batched
   CUDA dynamic programming is a possible acceleration, not a requirement of
   the anchor stage.
4. Reject low-quality paths, deduplicate overlapping paths, and extend accepted
   paths to produce the final fiberlet over-segmentation.

## Current stage: cell anchors only

Plan and then implement only cell construction and extraction/refinement of at
most two anchors per cell from an existing local or remote fiber Lasagna
manifest. Reuse the existing manifest, remote cache, compact-axis decoder, and
`FiberPredictionField` sampling implementation.

The current stage must:

- define cell size in stored prediction voxels and convert positions explicitly
  between prediction-grid and base-volume coordinates;
- treat every decoded direction as an unoriented axis;
- use deterministic two-component Gaussian/presence-weighted directional
  clustering with a weighted PCA update for each independently fitted line;
- modulate presence support by squared unoriented direction alignment and
  independently retain zero, one, or two fitted anchors by support;
- produce a machine-readable anchor artifact and an OBJ visualization of the
  accepted anchor axes;
- expose enough diagnostics to choose the cell size and thresholds on a real
  crop before the connection stage is designed.

NML input, anchor connection, path search, CUDA, Lasagna-normal path costs,
path-quality filtering, fiberlet deduplication, and path extension are out of
scope for this stage.
