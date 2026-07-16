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
- Implemented visualization export by converting the fused native 3D trace to
  base XYZ, sampling Lasagna normals at those traced coordinates, and building
  a fresh Trace2CP-style side/top strip source from the traced 3D polyline.
- Added live forward/backward native trace progress bars and changed the native
  inference patch default to `64,64,64`, matching the fast 3D training config.
- Updated `planning/specs.md` and `docs/code_structure.md`.
- Added focused synthetic tests in `test_fiber_trace_3d.py`.

## Deviations Or Deferrals

- The native tool is an inspection/debug tool only. It does not replace the
  existing projected `test/trace2cp_error` metric or best-checkpoint selection.
- The native visualization helper is still local to
  `fiber_trace_3d.trace2cp_tool`; it is not yet exposed as a standalone public
  loader API for arbitrary volume-space polylines.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py`
  - Result: passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: 312 passed in 13.32s.
