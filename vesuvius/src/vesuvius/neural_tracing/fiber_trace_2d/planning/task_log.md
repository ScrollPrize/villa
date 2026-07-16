# Native 3D Trace2CP Tool Task Log

## Implementation Notes

- Added `vesuvius.neural_tracing.fiber_trace_3d.trace2cp_tool` as a separate
  native 3D Trace2CP inspection CLI.
- Implemented deterministic cone candidate generation around the current
  inferred 3D direction.
- Implemented `NativeTraceFieldCache`, which lazily infers overlapped 3D model
  output blocks and routes point/candidate lookups through trusted cropped
  cores.
- Implemented bidirectional native tracing in selected-level ZYX coordinates
  with target-plane crossing and simple forward/reverse trace fusion.
- Implemented tool-local stdout/JSON metrics:
  `native_trace2cp_plane_error` and
  `native_trace2cp_closest_target_error`.
- Implemented visualization export by projecting the fused native 3D trace into
  the existing Trace2CP source frame and rebuilding side/top strip images with
  the existing refined 2D strip-source path.
- Updated `planning/specs.md` and `docs/code_structure.md`.
- Added focused synthetic tests in `test_fiber_trace_3d.py`.

## Deviations Or Deferrals

- The native tool is an inspection/debug tool only. It does not replace the
  existing projected `test/trace2cp_error` metric or best-checkpoint selection.
- The visualization adapter projects the fused native 3D trace into the
  existing Trace2CP source frame before calling the refined strip builder. It
  reuses the existing strip geometry path, but it is not yet exposed as a
  standalone public loader API for arbitrary volume-space polylines.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: 311 passed in 12.36s.
