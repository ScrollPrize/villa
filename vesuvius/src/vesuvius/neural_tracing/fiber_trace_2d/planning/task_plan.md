# Task Plan: Fix Loader Threading Bottlenecks

## Implementation

1. Reduce deterministic random-order locking.
   - Split random pass construction into `_ensure_random_pass_order`.
   - Make `_random_flat_index` use an unlocked fast path when the pass order is
     already cached.
   - Before submitting parallel batch workers, precompute/cache the pass orders
     needed for the attempted sample-index window so workers do not all block on
     first-pass construction.

2. Reuse the CP-level thread pool.
   - Add a loader-owned lazy `ThreadPoolExecutor`.
   - Reuse it across `load_batch` calls when `loader_workers > 1`.
   - Recreate it only when the requested worker count changes.
   - Add `close()` and context-manager cleanup hooks so tests and callers can
     shut it down explicitly.
   - Keep the serial `loader_workers=1` path free of thread-pool overhead.

3. Preserve deterministic output behavior.
   - Continue consuming worker results by raw sample index.
   - Keep invalid `ValueError` samples as skips.
   - Keep non-`ValueError` failures fatal.
   - Keep returned image/coord/valid tensor order unchanged.

4. Keep profile semantics.
   - Keep `load_batch_wall`, `load_batch_worker`, and `loader_thread_factor`.
   - Keep existing per-stage worker-time columns.
   - Do not add noisy per-sample output.

## Spec Update

Update `planning/specs.md` to state:

- Parallel loader workers must not serialize on deterministic random-order
  locks during the warm path.
- The CP-level executor is reused across training batches and can be closed
  explicitly with `FiberStrip2DLoader.close()`.

## Docs Updates

Update `docs/code_structure.md` to describe:

- Lazy persistent loader executor lifecycle.
- Random pass-order prewarming before worker submission.
- `loader_workers=1` as the serial no-thread debugging path.

Update `planning/status.md`, `planning/task_log.md`, and
`planning/changelog.md`.

## Tests

- Unit-test that random pass construction is cached before/while loading and
  warm descriptor lookup remains deterministic.
- Unit-test that `load_batch` reuses one persistent executor across calls when
  `loader_workers > 1`.
- Keep existing deterministic ordering and skip tests.

## Validation

- Compile touched Python files.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Compare a local `--profile --load-only` warm run before/after where the local
  VC3D/data environment is available.
