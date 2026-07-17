# Native 3D Whole-Fiber Trace2CP Eight-Row Visualization Log

## Implementation Notes

- Task started from user request: full-fiber native 3D Trace2CP visualization
  should have eight rows and include the regenerated strip.
- Updated whole-fiber span rendering to return eight panels: initial side
  volume/presence, initial top volume/presence, regenerated side
  volume/presence, and regenerated top volume/presence.
- The regenerated rows are built by converting each span's traced 3D segment
  into Trace2CP source-strip coordinates and passing that trace through
  `build_trace2cp_refined_segment_source`, matching the single-pair rendering
  path.
- Failed-segment regenerated traces use the same target-overlap trimming
  convention as failed overlays before the refined source is built.
- Updated specs/changelog to state the eight-row whole-fiber visualization
  contract.
- Follow-up request: draw the full-fiber span control points in the
  visualization as well.
- Added optional control-point markers to native Trace2CP panel drawing.
- Whole-fiber span rendering now projects the span's actual fiber control
  points into the initial and regenerated side/top strip sources, then draws
  those markers on all eight rows. This uses volume-coordinate projection so
  regenerated strips do not depend on original line-index metadata.
- Follow-up request: overlay each CP's Trace2CP distance to the trace at the
  CP plane.
- Added CP-plane distance labels beside whole-fiber CP markers. Labels use
  `d=0.0` for each span's start CP, `d=<in_plane_error_voxels>` for reached
  target planes, and `miss` when a segment did not reach the CP plane.
- The same label tuple is drawn on all eight initial/regenerated side/top
  volume/presence rows.
- CP projection for whole-fiber markers preserves one slot per CP with `NaN`
  for unprojectable markers, so labels cannot shift onto the wrong CP if one
  marker is outside a regenerated strip.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed after eight-row implementation: `104 passed in 3.42s`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed after CP marker implementation: `104 passed in 4.24s`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed after CP-plane distance label implementation: `104 passed in 3.78s`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed after preserving CP label slots: `104 passed in 3.46s`.
- `git diff --check -- <touched files>` passed.

## Deviations

- No intentional simplifications or deferred requirements.
