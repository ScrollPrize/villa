# Tracing Documentation
(a starting point)

## apps/src/vc_grow_seg_from_seed.cpp

- starting point for patch tracing - the seeding logic is here (and might need improvements/debugging)
- calls space_tracing_quad_phys from surface_helpers.cpp to run actual patch tracer

## space_tracing_quad_phys() (surface_helpers.cpp)

- general process: optimize a surface from a thresholded surface prediction (using CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp(proc_tensor))
- cv::Mat_<uint8_t> state(size,0) - maintain a state of the current surface corners 
- general tracing loop:
    - outer loop:
        - loop: add corners greedily (for several iterations)
        - optimize globally / optimzed windowed (large "active" edge area of the trace)
- we use a bunch of heuristics to decided when to accept some solution and go on and when to skip

## How losses operate
    - loss generation functions are somewhat "region aware" - functions get supplied with global state array as well as the corner idxs and global corner array and operate on that. 
    - check out emptytrace_create_missing_centered_losses - recurses into various losses
    - unconditional losses: e.g. gen_straight_loss() -> generates a straightness loss for o1,o2,o3 three points, based on the supplied data and state
    - conditiona loss: conditional_straight_loss() -> generates the straightness loss only if the loss position is not marked as in-use already - and marks the location as used

## Where next

- look at the code and comments in surface_helpers.cpp
- ask in https://discord.com/channels/1079907749569237093/1243576621722767412

## Gen Neighbor Mode

- vc_grow_seg_from_seed supports a mode to generate a neighbor surface by raycasting along per-vertex normals from an existing tifxyz.
- Usage: set in params.json: `{"mode":"gen_neighbor", "neighbor_dir":"out", "neighbor_step":1.0, "neighbor_max_distance":250.0, "neighbor_threshold":1.0}` and run:
  `vc_grow_seg_from_seed --volume <volpkg> --target-dir <paths_dir> --params params.json --resume <existing_tifxyz_dir>`
- For each valid vertex, casts a ray either "out" (along normal) or "in" (opposite), stepping by `neighbor_step` until a sampled voxel in the volume meets or exceeds `neighbor_threshold`. Misses stay marked as invalid (`-1,-1,-1`) unless interpolation is enabled, in which case row/column scanlines are blended from nearby hits.
- Output is saved to `<paths_dir>/neighbor_<in|out>_<timestamp>` as tifxyz with meta.
- Optional controls:
  - `neighbor_exit_threshold` (default `neighbor_threshold * 0.5`): intensity the ray must drop below before searching for a new surface, ensuring we leave the source layer.
  - `neighbor_exit_count` (default 1): number of consecutive samples below `neighbor_exit_threshold` required before we consider the source layer exited.
  - `neighbor_min_clearance` (default 0.0, voxels) and `neighbor_min_clearance_steps` (default 0): enforce a minimum travel distance/step count before scanning for the next surface, useful for thick predictions.
  - `neighbor_fill` (default `true`): when enabled, gaps are filled by combining row/column scanline interpolation from neighboring hits; when disabled, unmatched vertices remain invalid.
  - `neighbor_interp_window` (default `5`): number of valid samples to pull from each side of the row/column scan during interpolation.
