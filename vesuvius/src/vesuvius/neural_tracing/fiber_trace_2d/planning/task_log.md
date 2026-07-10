# Task Log: Parallel And Batched Loader I/O

## Planning Notes

- Warm profiling shows `load` and remaining cache/sample orchestration are now
  the dominant loader costs.
- The next step is sampler-level batch loading for strip-z patch stacks plus
  CP-level parallelism in `load_batch`.
- Flattening a patch stack into one larger coordinate image is acceptable for
  correctness because each sampled pixel is driven by explicit coordinates.

## Implementation Notes

- Added `loader_workers` with a logical-core default and `1` as serial debug
  mode.
- Added `CoordinateSampler.sample_coord_batch(...)`.
- Implemented the default sampler batch path by flattening
  `[patches,H,W,3] -> [patches*H,W,3]`, calling `sample_coords(...)` once, and
  reshaping image/valid outputs back.
- Routed `FiberStrip2DLoader.build_sample(...)` through the batch sampler for
  the whole strip-z stack.
- Parallelized `load_batch(...)` over CP sample candidates with deterministic
  raw-sample-index consumption, so accepted output order and skip behavior stay
  serial-equivalent.
- Worker tasks collect local profile dicts; the main thread merges numeric
  timings/stats after deterministic consumption.
- Added profile-only loader overlap counters:
  - `load_batch_wall`: real wall time for the whole `load_batch` call;
  - `load_batch_worker`: summed elapsed time inside accepted/skipped worker
    candidate builds;
  - profile table `tf`: `load_batch_worker / load_batch_wall`.

## Deviations

- No native VC3D multi-dimensional batch API was used. The implemented runtime
  path is the planned flattened single-call batch path, with stats exposing
  `batch_mode_flattened`.
- The stage columns under `loader_workers > 1` are intentionally summed worker
  timings. They are useful for CPU/work accounting but are not wall timings;
  the new `wall/work/tf` columns should be used to judge actual overlap.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/sampling.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 91 tests.
- After adding wall/work/threading-factor profiling, the same compile command
  passed again and the focused pytest command passed again: 91 tests.
