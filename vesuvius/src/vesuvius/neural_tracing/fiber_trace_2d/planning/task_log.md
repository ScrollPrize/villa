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
- Strip-coordinate cache payload loading now avoids redundant cached xyz arrays:
  supported older entries are still accepted, but the loader reads zyx
  coordinates/offset axes and derives xyz tensors from them when needed.

## Deviations

- No behavior changes were made to coordinate generation, augmentation,
  sampling, labels, model, or prefetch.
- A trial that batched all accepted CP volume sampling into one VC3D call per
  batch reduced summed worker sampling time but made warm wall time slower
  because it serialized sampling after coordinate preparation. That change was
  reverted; the final code keeps per-worker VC3D sampling.

## Validation

```bash
python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_2d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --benchmark --load-only --profile
```

Result:

```text
93 passed in 3.82s
benchmark: 6400 patches, 13742.6 ms, 465.71 patches/s
```

Reference measurements while debugging on the same command/data:

```text
baseline reproduced before the final cache/read cleanup: 14546.5 ms, 439.97 patches/s
final: 13742.6 ms, 465.71 patches/s
```
