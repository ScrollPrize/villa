# 3D Branch Choice Grid Routing Update Task Log

## Implementation Notes

- Replaced ambiguous 3D batch index storage with explicit
  `stream_index`/`stream_indices` and `data_index`/`data_indices`.
- `FiberTrace3DLoader.load_sample(...)` still accepts `sample_index` at the
  compatibility boundary, but immediately treats it as `stream_index`.
- Dataset/CP lookup and prefetch dependency lookup apply
  `sample_index_limit` only to derive `data_index`.
- Augmentation parameters, smooth displacement, value noise, and branch-grid
  offsets remain keyed by the unbounded `stream_index`.
- Replaced the old boolean branch-repair flag with explicit loss routing modes:
  `eval_voxel` and `train_offset_grid_min_fraction`.
- Training now groups positive two-branch routing by deterministic
  per-sample-offset `2x2x2` chunks and keeps the 10% underrepresented-branch
  repair.
- Test/eval loss keeps independent per-positive-voxel detached argmax branch
  selection and does not apply grouping, offsets, or 10% repair.

## Deviations / Deferred Items

- Public loader methods still use the name `sample_index` for compatibility,
  as planned. Internally the value is normalized to `stream_index`.
- No other requirements were simplified or deferred.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/loader.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k 'branch or compute_losses or sample_index_limit or augmentation_index'`
  - Result: 14 passed, 97 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - Result: 111 passed.
- `git diff --check`
  - Result: passed.
