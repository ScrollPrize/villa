# Joint Top-View Direction And Distance-Transform Model Task Log

## Plan

- Replaced the active task and task plan with the requested jointly trained
  top-view model.
- Plan keeps the side-strip model unchanged and adds an optional top-view
  direction+DT model enabled by config.

## Implementation Notes

- Added `training.top_view_enabled`, top-view loss weights, and
  `training.top_view_dt_radius_px`.
- Added top-view batch loading through the existing top-strip coordinate
  construction. Top batches contain one top patch per CP sample and carry the
  same deterministic augmentation params as the CP's center side-strip patch.
- Added top-view dependency inclusion for prefetch.
- Added top-view DT supervision along the rounded cross-fiber line through the
  transformed CP, including explicit zero targets beyond the configured radius.
- Added joint training with a second `FiberStripDirectionNet` whose scalar
  sigmoid channel is interpreted as DT. Snapshots include
  `top_model_state_dict` when enabled.
- Added TensorBoard scalars/images for top-view direction and DT outputs.
- Enabled top-view joint training in `configs/loader_example.json`.
- Added focused tests for DT targets, top model channels, top batch loading,
  and top-view prefetch forwarding.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: `197 passed in 7.84s`.
