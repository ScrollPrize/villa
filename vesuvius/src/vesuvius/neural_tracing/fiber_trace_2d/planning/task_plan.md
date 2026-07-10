# Task Plan: Parallel And Batched Loader I/O

## Implementation

1. Add loader worker configuration.
   - Add `loader_workers` to `FiberStrip2DConfig`.
   - Parse optional JSON key `loader_workers`.
   - Default value: logical core count from `os.cpu_count()`; clamp to at least
     `1`.
   - Keep `loader_workers=1` as the deterministic single-thread debugging mode.
   - Document that this is CP-sample parallelism, not GPU/model parallelism.

2. Add sampler batch API.
   - Extend `CoordinateSampler` with:
     `sample_coord_batch(coords_zyx_base, valid_mask)`.
   - Expected input shape: `[B, H, W, 3]` and `[B, H, W]`.
   - Expected output shape: image `[B, H, W]`, valid `[B, H, W]`, merged stats.
   - Default implementation should call a shared flattening helper first if
     the concrete sampler supports ordinary `sample_coords`.
   - Stats should include enough information to preserve existing profile
     counters by summing numeric per-call values.

3. Implement flattened single-call batch loading.
   - Add helper:
     - flatten coords `[B,H,W,3]` to `[B*H,W,3]` or `[B,H*W,3]`;
     - flatten valid `[B,H,W]` accordingly;
     - call `sample_coords(...)` once;
     - reshape returned image/valid back to `[B,H,W]`.
   - Prefer `[B*H, W, 3]` because it preserves row-like layout and avoids one
     very long row.
   - Validate returned shape and fail clearly if a sampler returns incompatible
     output.
   - This is allowed because coordinate sampling is value-by-coordinate; the
     request image shape should not change values, only performance/chunk
     traversal behavior.

4. Use native VC3D support if available.
   - Check whether current `Volume.sample_coords(...)` accepts `[B,H,W,3]` or
     other native batched shape.
   - If supported, use native batch mode.
   - If not supported, use the flattened single-call path.
   - Keep a final explicit loop fallback only for test/fake samplers that cannot
     support flattening, and make that fallback visible in stats.

5. Wire `build_sample` to sampler batch loading.
   - After batched coordinate augmentation, call
     `source.record.sampler.sample_coord_batch(coords_zyx_np, valid_mask_np)`.
   - Remove the per-strip-z `sample_coords(...)` loop for normal samplers.
   - Split returned images/valids back while constructing `FiberStripSample`
     metadata.
   - Preserve per-patch value augmentation and final sample order.

6. Parallelize `load_batch` across CP samples.
   - Use `ThreadPoolExecutor(max_workers=config.loader_workers)`.
   - Submit one `build_sample(...)` task per requested CP sample slot.
   - Store results by original slot index so output order is exactly the same
     as serial loading.
   - Preserve invalid-sample skipping semantics:
     - each slot should keep advancing through deterministic sample indices
       until it finds a valid sample;
     - skipped invalid data remains a skip, not a fatal error;
     - programming/infrastructure exceptions remain fatal.
   - Make sure profiling aggregates stats from worker results deterministically
     after futures complete.

7. Keep cache loading parallel through CP workers.
   - Because `build_sample(...)` includes descriptor lookup, strip-coordinate
     cache load, cache miss source construction, coordinate augmentation, and
     image sampling, CP-level parallelism also parallelizes cache loading.
   - Do not introduce separate cache-specific threads unless profiling shows
     CP-level parallelism is insufficient.

8. Profile output updates.
   - Add/record `batch_sample_parallel` or equivalent aggregate if useful.
   - Preserve existing profile table columns.
   - Add sampler stats indicating whether batch loading used:
     - native batch;
     - flattened single-call batch;
     - loop fallback.

## Spec Update

Update `planning/specs.md` to state:

- Loader image sampling should use sampler-level batch loading for strip-z
  patch stacks.
- Flattening `[B,H,W]` patch stacks into one larger coordinate image is
  functionally valid because sampling is explicit-coordinate based; only
  performance/chunk traversal may change.
- `load_batch` may parallelize CP samples with `loader_workers`, defaulting to
  logical CPU count, while preserving deterministic output order and skipping
  behavior.

## Docs Updates

Update `docs/code_structure.md` to describe:

- `CoordinateSampler.sample_coord_batch`.
- VC3D native/flattened/fallback batch modes.
- `loader_workers` CP-sample parallelism and deterministic ordering.

Update `planning/changelog.md`, `planning/status.md`, and
`planning/task_log.md` for this current task only.

## Tests

- Unit-test flattened sampler batch loading against repeated per-patch
  `sample_coords` on the fake NumPy sampler.
- Unit-test `build_sample` calls sampler batch API once for the strip-z stack.
- Unit-test `load_batch` preserves deterministic output order with
  `loader_workers > 1`.
- Unit-test `loader_workers` default is logical CPU count and explicit
  `loader_workers=1` is accepted.
- Unit-test invalid-sample skip behavior remains deterministic under parallel
  loading, if existing test harness makes this practical.
- Keep existing loader, batching, augment, cache, and deterministic-order tests.

## Validation

- Run compile checks for touched Python files.
- Run focused pytest:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Run local warm-profile comparison when available:
  `PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_2d.train $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --profile`
- Compare at minimum `load`, `cache`, `coord`, and total ms/patch for warm
  cached steps.
