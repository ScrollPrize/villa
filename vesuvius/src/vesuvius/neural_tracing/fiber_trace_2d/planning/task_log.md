# Trace2CP Short Z-Search Task Log

## Implementation Notes

- Added `--trace2cp-z-search`, `--trace2cp-z-step-voxels`, and
  `--trace2cp-z-max-layer` to the Trace2CP runner CLI.
- Z-search requires `--trace2cp-combined`; it is rejected with `--med-tta` in
  this first implementation because the candidate scorer needs per-layer
  embedding fields.
- Extended `build_trace2cp_segment_patch` with an optional explicit
  selected-scale `strip_z_offset` argument. Existing callers keep the center
  strip-z offset default.
- Added a runner-side lazy z-plane prediction cache. It starts from the already
  inferred center layer, samples/inferes neighbor layers on demand, and keeps
  layer images, masks, directions, and embeddings.
- Added z-aware combined tracing. Each 2D angular candidate is evaluated in
  neighbor layers `current-1/current/current+1`; selected trace points carry
  `x`, `y`, and selected-scale `z_voxels`.
- Added z-aware closest/fusion diagnostics using
  `abs(dy) + abs(dz_voxels)` with the existing center-distance magnification.
  Fused z is linearly corrected toward the closest-approach midpoint with both
  CP z coordinates fixed at zero.
- The public `trace2cp_error`, training test metric, and best-checkpoint
  semantics remain target-column y error per horizontal CP span.
- Single-pair `trace2cp_vis.jpg` now appends a z column when z-search is
  enabled. It shows separate forward/reverse z-corrected images. The images are
  assembled per column from already inferred layer images; there is no
  reconstruction-time volume re-sampling and no inter-layer image
  interpolation.

## Deviations / Limits

- The first z-search implementation skips the existing y-only optimized
  refinement solve and passes through the z-fused line as the optimized row.
- The whole-fiber Trace2CP summary records z-search diagnostics, but the
  long-strip whole-fiber JPG remains the existing four-row pair-composition
  view rather than adding per-pair z-corrected columns.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: `165 passed in 7.14s`.
