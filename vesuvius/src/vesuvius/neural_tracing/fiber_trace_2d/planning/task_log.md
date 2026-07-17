# Native 3D Trace2CP Beam Search Log

## Implemented

- Added `cone_angle_step_degrees`, `beam_width`, and
  `beam_prune_distance_voxels` to `NativeTrace2CpConfig` and the native 3D
  Trace2CP CLI.
- Added deterministic tangent-plane angular candidate generation. The default
  `25` degree cone with `5` degree steps produces 81 unit candidates including
  the center direction.
- Kept the legacy `cone_grid_size` square-grid generator as an explicit
  fallback when `cone_angle_step_degrees <= 0`.
- Refactored candidate scoring so greedy and beam tracing share the same
  branch-aware candidate-by-branch loss tensor.
- Added beam-state native one-way tracing. `beam_width <= 1` keeps the existing
  greedy control flow; wider beams accumulate local candidate losses, prune
  near-duplicate live states, and select the lowest-loss target-plane-reaching
  state.
- Updated native Trace2CP summary JSON with beam and cone-step settings.
- Updated specs, code-structure docs, changelog, and status.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - Result: `83 passed in 7.49s`

## Deviations / Deferred

- No planned requirement was intentionally skipped.
- The legacy `--cone-grid-size` CLI remains accepted for compatibility, but it
  is ignored by default while `--cone-angle-step-degrees` is positive. Set
  `--cone-angle-step-degrees 0` to use the old grid path.
