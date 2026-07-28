# Task Log: Fiber Scale-2 Output, Sparse Accumulator Activity, and 64³ Chunks

## Planning findings

- The requested setting is whole-volume inference's
  `--inference-scaledown-power`, not tracer/model config `scaledown`. Power 2
  converts to the literal source-relative runner factor 4. The config setting
  must remain untouched.
- Power-based inference output scaling includes the shared repeated 5-tap
  blur-plus-2x-decimation operation. It is not naive stride subsampling; power
  2 performs two filtered pyramid steps for raw products and blend weights.
- User confirmed Fiber must use the historical Lasagna weighted-pyrdown and
  border interaction exactly, through the same shared runner implementation.
- Lasagna's `cos_scaledown=2` and `scaledown=4` are literal factors and must not
  be reinterpreted.
- Fiber and Lasagna public inference paths default to 32³ OME chunks; the Fiber
  adapter's isolated default of 64 is overridden by its caller.
- The shared flush walks the complete XY chunk grid, performs an expensive
  support rescan, and unconditionally zeroes every raw/weight region. This
  materializes untouched sparse mmap pages and caused the observed stall in
  `_CircularZBand.clear`.
- The planned fix is contribution-driven: lazily cache exact output-chunk
  source support, record dirty chunks/generations during actual accumulation,
  flush and clear dirty chunks only, and release untouched regions without any
  mmap assignment or output write.

## Implementation

- Fiber now exposes `inference_scaledown_power` in its Python entry point and
  `--inference-scaledown-power` in its CLI. The default is 2 and is converted
  once to literal factor 4 for the shared runner. Input/base scale validation
  now checks all three axes against exact ceil-divided power-of-two geometry.
- The shared runner caches direct output-chunk source support and records only
  chunks that receive positive shared weight plus product contributions.
  Flushes iterate that ledger, write/clear only dirty products, and release Z
  generations without assigning zeros to untouched XY regions.
- Shared flush logs now include dirty progress, products written, unique
  unsupported/resume chunks, cumulative bytes touched/cleared, and elapsed
  time. Skipped tiles update the shared progress dictionary.
- Fiber and Lasagna public OME chunk defaults are now 64 cubed. Explicit
  overrides are unchanged.
- Specs, code-structure documentation, and the changelog describe the scale,
  blur/border, sparse-output, and chunk-default contracts.

## Validation

- `/home/hendrik/.venv_las/bin/python -m py_compile
  lasagna/tiled_predict3d.py
  vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/infer.py
  lasagna/preprocess_cos_omezarr.py` passed.
- Focused shared multi-scale runner and circular wrap/reuse unit tests passed
  (2 tests, 0.031 seconds).
- Added regressions for Fiber scale/chunk defaults, exact three-axis scale
  validation, and unsupported-XY output/clear suppression.
- The project venv has no `pytest`, so the Vesuvius pytest suite was not run.
  The local Zarr-backed unittest fixtures stalled in this environment before
  reaching inference and were interrupted/time-limited; the NumPy-backed
  shared runner regressions completed normally.
- Representative masked-volume performance/allocation measurement was not run
  because it would execute the user's large inference workload. This remains
  the outstanding acceptance validation.

## Planning review record

- Independent review caught and corrected an erroneous conflation of
  `--inference-scaledown-power` with tracer/model config `scaledown`.
- Review also required the plan to state that direct-footprint sparsity is an
  intentional masked-boundary output-policy change; add exact three-axis scale
  validation; track dirty products separately from shared weight; test
  per-generation XY reuse and physical sparse allocation; document local
  Zarr-v2 absent-key limitations; deduplicate progress counters; and validate
  interruption cleanup.
