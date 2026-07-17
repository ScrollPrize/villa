# Native 3D Trace2CP Default Tuning Log

## Implementation Notes

- Updated native 3D Trace2CP code and CLI defaults to:
  `--beam-lookahead-steps 1`, `--beam-width 8`,
  `--smoothness-normal-weight 0.1`, `--smoothness-tangent-weight 10.0`, and
  `--core-margin-voxels 20`.
- Kept `args.sample_index` as `None` in the parser so bare `--fiber-json`
  still triggers whole-fiber mode; ordinary single-sample resolution now falls
  back to sample index 13.
- Changed no-normal-sampler paths to fall back to isotropic smoothness instead
  of rejecting the default split smoothness weights.
- Updated specs, code-structure docs, and changelog.

## Deviations / Deferred

None.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - Result: `102 passed in 3.47s`.
