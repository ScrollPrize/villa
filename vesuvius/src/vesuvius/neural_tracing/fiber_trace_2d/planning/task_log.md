# Task Log: Cache Source Line Coordinates And Split Coord Profiling

## Implementation Notes

- Bumped strip-coordinate cache version from `fiber_strip_2d_source_v1` to
  `fiber_strip_2d_source_v2`, so old cache entries are ignored.
- Added source-space `line_xy` and source-space `control_point_xy` tensors to
  `_StripSource` and persisted them in the `.npz` cache.
- Larger cached source entries still satisfy smaller requests; source line/CP
  coordinates are center-cropped and shifted into the smaller source coordinate
  system.
- Unaugmented patches reuse cached source line/CP tensors. Augmented patches
  still compute transformed output line/CP coordinates per patch because those
  depend on the augmentation parameters.
- Split training profile output into aggregate `coord`, plus `desc`, `cache`,
  `source`, and `line` columns. The summary now reports the same split as
  `descriptor`, `coord_cache`, `source_geom`, and `line`.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 75 tests.
