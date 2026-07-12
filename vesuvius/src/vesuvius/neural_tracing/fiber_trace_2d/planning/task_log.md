# Prefetch Download Priority Task Log

## Plan

- Replaced the active task and task plan with the requested prefetch scheduler
  change.
- Plan keeps dependency generation parallel and globally deduplicated, but
  keeps missing downloads in an explicit priority queue keyed by earliest raw
  deterministic sample index.

## Implementation Notes

- `FiberStrip2DLoader.prefetch` no longer submits every missing chunk directly
  into the download executor's FIFO.
- Completed dependency producer results are buffered and consumed in raw
  deterministic sample-index order before chunk classification/download
  enqueueing.
- Missing chunks are stored in a lazy heap keyed by `(raw_sample_index,
  sequence)`.
- Only up to `prefetch_workers` active transfers are submitted at a time.
- If a pending chunk was first discovered by a later producer and then an
  earlier raw sample requests it before submission, the pending priority is
  lowered.
- Active transfers are allowed to finish; they are not cancelled or restarted
  for reprioritization.
- Prefetch temporarily sets PyTorch CPU intra-op threads to `1` and restores
  the previous value afterward, preventing each producer worker from expanding
  over the full CPU pool.
- Added a regression test where sample 1 computes dependency requests before
  sample 0, but download submission still starts with sample 0 once the earlier
  raw sample is available.
- Added a regression test that prefetch sets/restores the PyTorch CPU thread
  count.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: `199 passed in 6.53s`.
- `git diff --check` passed.
