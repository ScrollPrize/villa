# 3D Test Visualization And Raw Index Cleanup Task Log

## Implementation Notes

- Removed the public alternate augmentation-index loader arguments. Public 3D
  `load_sample`, `load_batch`, and prefetch dependency generation now treat
  `sample_index` as the raw/global deterministic stream index, derive data
  sample identity through `sample_index_limit`, and seed augmentation from the
  raw index.
- Added a display-only line-presence raster to the 3D TensorBoard
  target/context presence panel. It is built from `target_segment_*` metadata
  and composited with max-pooled `presence_target`, so CP-only JSON/test fibers
  show the full carried fiber context without changing loss materialization.
- Added `training.sample_vis_count` / `training.train_sample_vis_count`
  defaulting to `4`, plus `training.test_sample_vis_count` for test override.
  Train and dense-test TensorBoard sheets concatenate up to that many batch
  samples side by side.
- Removed the stale dense-test visualization slice that forced the test sheet
  to one sample before writing.
- Changed omitted dense 3D `training.test_control_points` to resolve like the
  explicit `0` sentinel: all held-out CPs in flat order from zero. Positive
  values remain an explicit debugging cap.
- Updated specs, code-structure docs, task plan/status, and changelog for the
  corrected visualization and index semantics.

## Deviations Or Deferrals

- None.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/targets.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - Result: `39 passed in 3.59s`
- `git diff --check`
