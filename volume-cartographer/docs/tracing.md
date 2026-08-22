# Tracing Documentation
(a starting point)

## Reproducibility

By default a trace is not reproducible: run-to-run variance can exceed the
effect of most tracer changes being benchmarked (see issue #1317). Two levers
together make `vc_grow_seg_from_seed` deterministic per platform:

```bash
OMP_NUM_THREADS=1 VC_GROWPATCH_RNG_SEED=42 vc_grow_seg_from_seed ...
```

- `VC_GROWPATCH_RNG_SEED` fixes the base seed of the tracer's RNGs. Unset, a
  base seed is drawn once per process instead.
- A single thread is required because work distribution across OpenMP threads
  is scheduling-dependent. `OMP_NUM_THREADS=1` or `"thread_limit": 1` in the
  params JSON both work. Expect roughly 2x wall time versus unbounded threads
  (measured in #1317), which is why this is a debugging/benchmarking mode
  rather than the default.

Every trace records what it used in its `meta.json`:

- `vc_gsfs_rng_seed` — the effective base seed, whether it came from the
  environment or was drawn at startup. Re-running with
  `VC_GROWPATCH_RNG_SEED=<that value>` and one thread reproduces the trace on
  the same platform/build.
- `vc_gsfs_omp_max_threads` — the OpenMP thread count in effect.

Scope: this covers `seed` mode with an explicit seed point. The seed-location
picking in `random_seed` and `expansion` modes uses `rand()` seeded from the
clock and is not covered by `VC_GROWPATCH_RNG_SEED`.

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
