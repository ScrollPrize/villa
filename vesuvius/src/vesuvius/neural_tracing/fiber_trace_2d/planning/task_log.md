# Native 3D Trace2CP Beam Lookahead Log

## Implemented

- Added `beam_lookahead_steps` to `NativeTrace2CpConfig` and CLI flag
  `--beam-lookahead-steps`, defaulting to `3`.
- Changed the native 3D beam tracer so it expands up to
  `beam_lookahead_steps` future steps before pruning back to `beam_width`.
- Kept `beam_width <= 1` on the greedy compatibility path, so lookahead is not
  used for explicit greedy runs.
- Added `beam_lookahead_steps` to native Trace2CP summary JSON and progress
  text.
- Added a fake-cache regression where one-step beam pruning drops the needed
  branch but three-step lookahead reaches the target plane.
- Pinned older constant-field native Trace2CP tests to `beam_width=1` because
  they validate trace guards/geometry, not default beam runtime.
- Updated specs, code-structure docs, changelog, and status.

## Validation

- First focused test run was interrupted after it became slow: legacy
  constant-field tests were accidentally exercising default beam lookahead.
- After pinning those compatibility tests to greedy:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - Result: `83 passed in 8.36s`

## Deviations / Deferred

- No planned requirement was intentionally skipped.
- Runtime cost is intentionally higher for default beam tracing: with
  `beam_width=8`, 81 default candidates, and lookahead `3`, the frontier can be
  much larger before pruning. The depth is configurable with
  `--beam-lookahead-steps`.
