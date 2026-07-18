# Task Log: Shared Trace2CP Strip Builder

## Implementation Notes

- Added a shared Trace2CP source builder in `fiber_trace_2d.loader` that takes
  an explicit `FiberStripLineWindow`, target-local index, shape, anchor column,
  strip offset, and Lasagna normals, then calls the existing
  `build_side_strip_patch_grid_tensor_from_line_window(...)`.
- Original CP-pair `build_trace2cp_segment_source(...)` now computes the
  sample/record/window metadata and delegates the common source construction to
  the shared helper.
- Added `build_trace2cp_volume_trace_segment_source(...)` for native 3D
  regenerated/fused strips. It accepts base-volume XYZ trace points, filters
  only non-finite/consecutive-duplicate points, samples fresh Lasagna normals
  at the traced points, sign-aligns them along the traced line, and delegates
  to the shared helper.
- Native 3D single-pair fused-strip rendering and whole-fiber regenerated rows
  now call `build_trace2cp_volume_trace_segment_source(...)` with actual traced
  volume coordinates instead of converting through the original source-strip
  grid.
- Lowered the native whole-fiber error threshold default from 100 to 10
  selected-scale voxels in both `NativeTrace2CpConfig` and the CLI parser.
- Updated planning specs, code-structure docs, and changelog.

## Deviations / Boundaries

- `build_trace2cp_refined_segment_source(...)` remains as the existing 2D
  refinement compatibility adapter for source-strip `(x,y,z)` traces because
  `fiber_trace_2d.runner` still uses that API. Its final side-strip
  construction now goes through the shared helper, but the adapter still
  converts old 2D source-strip trace coordinates into volume points before
  delegating. Native 3D regenerated/fused strips bypass this adapter entirely.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k 'trace2cp_refined or trace2cp_segment_source or volume_trace_segment_source'`
  passed: 7 passed, 270 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k 'trace2cp_defaults or whole_fiber_span_panels or whole_fiber_span_source or converts_volume_trace'`
  passed: 4 passed, 102 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 277 passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: 106 passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: 383 passed.
