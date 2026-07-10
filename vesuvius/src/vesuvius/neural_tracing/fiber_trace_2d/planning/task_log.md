# Task Log: Batched Strip Augmentation And Line Mapping

## Planning Notes

- Current profiling indicates sparse line/CP mapping is dominated by many tiny
  PyTorch operations and tiny `grid_sample` calls, not by the amount of point
  data.
- The next implementation should batch work across offsets/patches and replace
  sparse point `grid_sample` with direct bilinear gather against
  `forward_map_xy`.
- VC3D volume sampling remains the likely per-patch external boundary unless an
  existing batched sampler API is found.

## Implementation Notes

- Added `sample_xy_maps_bilinear(...)` for batched sparse bilinear lookup
  against concrete fused map tensors.
- Rewired single-transform point lookup to use the same sparse gather helper
  rather than tiny `grid_sample`.
- Updated `FiberStrip2DLoader.build_sample(...)` to:
  - build all offset augmentation params/transforms first;
  - batch line/CP lookup across all unique offset transforms;
  - batch coordinate augmentation across strip-z offsets through stacked
    `backward_map_xy` tensors;
  - keep VC3D `sample_coords(...)` as the explicit per-patch I/O boundary;
  - batch post-load value augmentation over the loaded image stack.
- Added profile keys for `map_build`, `line_lookup`, `line_filter`,
  `coord_aug_batch`, and `value_aug_batch` while preserving existing aggregate
  keys for compatibility.
- Added tests for batched sparse map lookup, out-of-bounds sparse points,
  batched value augmentation equivalence, and `build_sample` batched line
  lookup.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/augmentation.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: `87 passed in 3.95s`.
