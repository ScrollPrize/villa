# Task Log: Fix Loader Threading Bottlenecks

## Implementation Notes

- Added `FiberStrip2DLoader.close()` plus context-manager cleanup for the
  loader-owned persistent CP worker executor.
- `load_batch` now reuses a lazy `ThreadPoolExecutor` across batches when
  `loader_workers > 1`; `loader_workers=1` uses a direct serial path without
  futures.
- The parallel scheduler now keeps only enough candidates in flight to fill the
  remaining deterministic batch, so slow earlier samples do not cause
  speculative stale futures to spill into the next batch.
- Random deterministic dataset-pass orders are now cached through
  `_ensure_random_pass_order`, and `load_batch` prewarms pass orders for the
  attempted batch window before submitting CP workers.
- `_random_flat_index` now uses the cached order fast path, avoiding a lock on
  warm descriptor lookups.
- `random_order` setup time is folded into the profile descriptor total so cold
  first-batch setup remains visible.

## Deviations

- No behavior changes were made to coordinate generation, augmentation,
  sampling, labels, model, or prefetch.
- No local VC3D/data profile run was performed in this environment; validation
  used compile checks and the focused test suite.

## Validation

```bash
python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
```

Result:

```text
93 passed in 4.02s
```
