# Prefetch Download Priority Plan

## Scope

- Keep prefetch dependency generation, cache classification, `.empty` marker
  handling, and global chunk deduplication unchanged.
- Change only the scheduling of chunks that still require download/missing
  resolution.
- Preserve deterministic sample order and the `idx` meaning.

## Implementation

- Add a pending-download priority queue keyed by the earliest raw sample index
  that requested each chunk, with a tie-break sequence for stable ordering.
- Buffer completed dependency-producer results and consume them in raw
  deterministic sample-index order before classifying/enqueuing chunks.
- Only submit up to `prefetch_workers` active download futures. Keep additional
  downloads in the priority queue instead of in the executor's opaque FIFO.
- When a chunk was first discovered by a later producer but an earlier sample
  later requests it before submission, lower its pending priority.
- Do not attempt to cancel/reorder transfers that are already active.
- Keep `idx` as the completed contiguous bounded deterministic prefix.
- Keep progress output shape stable; `queued` may include both pending queued
  downloads and active futures.
- During prefetch, temporarily force PyTorch CPU intra-op threads to `1` and
  restore the previous value at the end. This prevents each dependency producer
  from using the full global CPU thread pool, so `prefetch_sampler_workers` is
  the practical generation concurrency knob.

## Spec Update

- Clarify that prefetch schedules unsent downloads by earliest requesting raw
  sample index, while active transfers are not cancelled for reprioritization.
- Clarify that prefetch CPU generation limits PyTorch intra-op fanout so
  `prefetch_sampler_workers` bounds source/dependency generation.

## Docs Updates

- Update `docs/code_structure.md` prefetch description with the priority queue
  behavior.
- Update `planning/status.md`, `planning/task_log.md`, and changelog.

## Tests

- Add focused tests for:
  - later-sample dependency generation finishing first still submits downloads
    in raw sample order once the earlier sample is available;
  - prefetch temporarily sets and restores PyTorch intra-op thread count;
  - existing prefetch cache-marker, `idx`, sampler-worker, and skip behavior
    continues to pass.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
