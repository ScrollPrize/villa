# Native 3D Trace2CP Render Brightness And Blocking Guard Task Log

## Findings

- Native 3D Trace2CP strip panels used per-panel `1..99` percentile scaling in
  `_image_to_u8(...)`. That changes brightness and can hide raw loading
  problems.
- 3D training model input uses `fiber_trace_3d.loader._normalize_image(...)`
  with the configured `image_normalization`. The current fast 3D config uses
  `image_normalization: "zscore"`, meaning mean/std over valid voxels before
  inference.
- Native strip visualization delegates image sampling to
  `FiberStrip2DLoader.sample_trace2cp_segment_source(...)` and
  `sample_trace2cp_top_strip_source(...)`.
- Those functions already call the coordinate sampler, but they did not reject
  non-blocking samplers or VC3D sampler chunk errors.
- The VC3D binding's `sample_coords` uses `sampleCoordsFineToCoarse` after
  blocking dependency prefetch. If fine chunks fail, the call can produce a
  coarser fallback image unless Python rejects the error stats.

## Implementation

- Changed native 3D Trace2CP `_image_to_u8(...)` to raw clipped rendering:
  valid finite values are rounded/clipped to `0..255`; invalid pixels stay
  black.
- Added `FiberStrip2DLoader._sample_trace2cp_coords_blocking(...)`.
- Routed Trace2CP side strip, top strip, traced top strip, and side-z strip
  image sampling through that helper.
- The helper rejects samplers with `blocking=False` and raises on
  `error_chunks > 0` to avoid silently rendering fine-to-coarse fallback data.
- Added tests for raw native render brightness and the Trace2CP render sampler
  guard.
- Updated `planning/specs.md`, `docs/code_structure.md`, and
  `planning/changelog.md`.

## Deviations Or Deferrals

- The strip geometry/orientation issues are intentionally not fixed in this
  task per the user note. This only addresses raw brightness and strict
  full-resolution/blocking render sampling behavior.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k native_3d_trace2cp`
  passed: 10 passed, 39 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k "trace2cp_render_sampling or trace2cp_segment_source_offsets_each_pixel_axis or trace2cp_traced_top_strip_samples_from_fused_trace"`
  passed: 4 passed, 266 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 319 passed.
