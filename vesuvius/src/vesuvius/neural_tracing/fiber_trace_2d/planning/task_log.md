# Task Log: Strip Coordinate Cache

## Implementation Notes

- Added top-level `strip_coord_cache_dir` parsing and included it in the
  example config.
- Added a disk-backed `.npz` cache around the shared `build_strip_source` path.
- Cache entries store CP-local source coords, validity, frame vectors, and
  strip offset axes before strip-z offsets, coordinate augmentation, image
  sampling, or value augmentation.
- Cache identity includes volume path/scale/spacing, strip offset metadata, CP
  coordinate, and a fiber-line identity. The fiber-line identity is stricter
  than CP coordinate alone so overlapping fibers cannot share incompatible
  source geometry.
- Larger cached source grids satisfy smaller requests by center-cropping.
- Cache writes use unique temporary files and atomic rename.
- Training, runner center loads, augment-vis, line/dir visualizations, and
  prefetch all use the cache through the existing shared source path.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 74 tests.
