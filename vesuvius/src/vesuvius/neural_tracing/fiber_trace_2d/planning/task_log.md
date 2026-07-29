# Native 3D Trace2CP Target Plane Normals

## Implementation Notes

- Replaced implicit CP-to-CP chord target-plane fallback in Python native 3D
  Trace2CP with explicit target-plane objects.
- Python whole-fiber and single-pair runs now build target-local planes from
  target CP line-neighbor directions plus the model-inferred direction sampled
  at the target CP.
- Python greedy and beam tracing now track which target-local planes have
  crossed and only report success after all configured target planes are
  crossed.
- Python selected endpoint error is now the minimum in-plane CP error among
  crossed target-local planes; the selected plane name/crossing is stored in
  trace and whole-fiber summaries.
- Native C++ `vc_fiber_tracer` now takes explicit target plane sets, derives
  segment/whole-fiber target planes internally from the reference line and
  persisted prediction field, and reports selected endpoint plane metadata.
- VC3D GUI segment optimization no longer passes a caller-supplied
  target-plane normal; the native tracer derives the target-local planes from
  the selected segment.
- Progress now uses distance-to-target progress instead of signed progress
  along a CP chord plane normal.

## Deviations / Deferred

- No requirement was simplified or deferred.
- Low-level Python `trace_native_3d_one_way` still accepts an explicitly
  supplied single plane for synthetic tests/callers, but it no longer creates a
  CP-to-CP chord fallback when no target plane is provided.

## Validation

- `PYTHONPATH=vesuvius/src:. python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - Result: 174 passed, 2 skipped.
- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target test_fiber_trace3d`
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_fiber_trace3d`
  - Result: 10 test case(s) passed.
