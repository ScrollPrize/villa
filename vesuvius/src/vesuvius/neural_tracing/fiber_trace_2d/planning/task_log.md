# Task Log: Python Native 3D Trace2CP Continuous CP Handling

## Notes

- User clarified that crossings are only for the current segment target CP.
  Previous segment planes and CP-to-CP chord planes must not be considered.
- User requested focusing on Python first; C++ parity is intentionally deferred
  until the Python behavior is validated.
- Updated Python target-plane crossing storage to keep the closest in-plane
  crossing seen so far per target plane instead of keeping the first crossing.
- Added optional one-way `target_plane_accept_threshold_voxels` and wired the
  whole-fiber Python tracer to pass its existing restart threshold.
- In thresholded whole-fiber tracing, all target planes crossed is no longer
  enough to terminate successfully; the best selected crossing error must be at
  or below the threshold. Above-threshold all-crossed states keep tracing until
  accepted or until budget/failure ends the segment.
- Above-threshold all-crossed failures preserve the selected crossing/error so
  whole-fiber reporting can show an in-plane error instead of `inf`.
- User then identified the remaining visible bug: successful CP crossings were
  still acting like a handoff/reinitialization point. The next segment could
  reselect the ambiguous direction or inherit a terminal direction unrelated to
  the selected crossing.
- Added terminal live trace state to `NativeTraceResult`: actual stepped point,
  previous step direction, smoothing-history direction, and sampled-current
  direction when available.
- Added continuation arguments to the Python one-way tracer. Whole-fiber success
  now passes those fields into the next CP target instead of CP-start
  resampling. The selected plane crossing remains metric-only.
- In whole-fiber mode, accepted target crossings no longer snap the stored trace
  endpoint to the selected crossing. The trace/visualization follows the live
  stepped path through the CP; failures still restart from the failed CP.
- Added Python regression tests for replacing farther plane crossings, for
  beam-mode continuation after early far crossings, and for whole-fiber
  continuation from the live endpoint rather than the selected crossing.

## Deviations / Deferred

- C++ native tracer parity is deferred by user request.
- The full user-provided S3/cache visualization command was attempted inside
  the sandbox and failed at VC3D remote open with `HTTP 0 fetching .zattrs`.
  Retrying with escalation was interrupted by the user, who will run the command
  locally.

## Validation

- `PYTHONPATH=vesuvius/src:. python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py` passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py` passed: 177 passed, 2 skipped.
- `git diff --check` passed.
