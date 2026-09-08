# Plan: shared Fiber3D sample-worker pool

## Design and implementation

- Keep `OrderedTwoStageSamplePipeline` as the shared reusable implementation
  and add a shared-pool mode rather than copying its scheduler.
- Spawn one bounded process population. Workers claim eligible load indices
  atomically and consume ready augmentation descriptors directly, without a
  parent scheduling round trip. Soft load/augmentation affinities preserve hot
  stage-local state, but every worker has the same entrypoint and lazily creates
  the other stage when it must help: a loader augments its just-completed sample
  instead of blocking on a full raw queue, and loaders help drain augmentation
  once all loads have been dispatched. No future or dependency-blocked
  augmentation is queued.
- Keep task ownership explicit. Every admitted sample has exactly one load and,
  when enabled, exactly one augmentation. No stage-two task may be enqueued
  before its stage-one payload exists. Data/results continue to use bounded
  queues and the existing shared-slot transport and generation validation.
- Replace eager full-window admission in shared mode with explicit gradual
  admission state: `target_frontier`, `next_load_to_dispatch`, and
  `outstanding`. Initialize the target frontier with a positive configurable
  window. Each accepted load completion advances it by exactly two indices,
  capped by the stream end. Dispatch remains capped by
  `outstanding <= max_in_flight`; ordered delivery or explicit lease release
  retires one outstanding credit. This allows the active lookahead to grow but
  never beyond the hard bound.
- Ensure scheduling cannot deadlock or priority-invert: workers pull the next
  runnable task rather than receiving permanent stage assignments; stage-two
  work is never queued ahead of its dependency; queue publication is bounded;
  process death, worker exceptions, duplicate completion, and no progress are
  explicit failures.
- Preserve the existing fixed-pool mode for compatibility and the stage-one-only
  benchmark. Expose shared mode and initial lookahead in the benchmark CLI and
  stable launcher, with one total worker count. Report task counts per worker,
  pool utilization, admission high-water, and stage service distributions.
- A shared transport slot is owned by its stream index from load encode through
  augmentation and final consumer decode. It cannot be reused until ordered
  delivery or explicit `release(index)`. Validate that the transport exposes at
  least `max_in_flight` slots and preserve generation checks.
- Attribute worker errors to stage and stream index. Detect either factory's
  startup failure and any premature process exit, unblock dispatcher/consumer
  waits during teardown, close pipes and queue resources, and clear leases.

## Validation and performance

- Add synthetic regression tests for exact ordered output, one execution per
  stage, gradual admission bounds (including initial window one, short streams,
  and a partial final growth), pool-level work stealing, slow-load/no-ready-
  augmentation progress, persistent augmentation backlog followed by resumed
  loading, saturated queues, both-stage/startup/hard-exit failures, deferred
  leased slots with wraparound, and clean shutdown.
- Compare shared-pool values tensor-for-tensor with synchronous and fixed-pool
  output for the same stream and repeat with deliberately different completion
  timing. RNG remains keyed only by stream index, never worker identity/order.
- Run focused pipeline/Fiber3D tests and `git diff --check`.
- Run the approved CPU-only benchmark launcher on the saved S1a workload with
  128 shared workers, 4096 measured samples and 128 warmup samples. Compare
  command, input, build, process count, production/full-span throughput and
  mean/p50/p95 stage times against commit `2e1714967` (229.75 samples/s
  production; 223.95 samples/s full start/drain).
- If shared initialization materially increases memory, measure and report it;
  do not silently increase the configured 512-slot shared tensor bound.
- Record exact cache/build/commit metadata plus peak observed process/thread/RSS
  usage. Run repeated performance trials when practical; otherwise explicitly
  record that the long 4096-sample comparison is a single-run limitation.

## Spec update

Document shared-pool scheduling, dependency eligibility, gradual admission,
the global 128-process bound, deterministic delivery, and fixed-pool
compatibility.

## Docs updates

Update the sample-pipeline benchmark document with the shared scheduler,
launcher arguments, before/after results, lifecycle, and limitations.

## Changelog update

Add one concise entry once the shared-pool implementation and benchmark are
complete.
