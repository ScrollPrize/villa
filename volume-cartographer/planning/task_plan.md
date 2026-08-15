# Task plan

## Problem

`Volume::setCacheBudget()` still follows the pre-service design: it invalidates
the source and drops the handle because decoded capacity used to be immutable
cache-constructor state. The shared service now retains that source, so the
operation cancels work and clears warm data while reacquisition still uses the
old capacity. Source state also redundantly applies a local decoded-byte
ceiling in addition to the global decoded budget.

## Implementation

1. Make `DecodedChunkCacheBudget` capacity atomically mutable and provide an
   in-place setter that enforces reductions through its existing global LRU
   participant callbacks.
2. Add `ChunkCacheService::configureDecodedByteCapacity()` as the sole runtime
   RAM-capacity API. It updates the existing budget object without replacing
   schedulers or source states.
3. Remove the service default copied into each source and remove the
   source-local decoded-byte capacity check. Keep per-source decoded usage,
   touches, and eviction callbacks because entries remain physically owned by
   their source.
4. Keep `ChunkCache::Stats::decodedByteCapacity` as the global budget capacity
   for status/UI compatibility.
5. Change `Volume::setCacheBudget()` to update stored defaults and configure the
   attached service without invalidating or resetting its source handle. Reject
   attempts to attach a different decoded-budget object after a service is
   installed rather than silently moving sources between managers.
6. Preserve service-construction behavior: `decodedByteCapacity` initializes a
   newly created budget only when no external shared budget was supplied.

## Tests

- Verify reducing service capacity evicts globally oldest decoded data across
  sources without changing source IDs.
- Verify an in-flight source read completes exactly once across a capacity
  reduction and queued work is not cancelled or restarted.
- Verify `Volume::setCacheBudget()` preserves its existing handle and warm data
  when increasing capacity.
- Verify reduction through `Volume::setCacheBudget()` uses the same source and
  enforces the new global limit.
- Retain and run existing concurrency reconfiguration tests.
- Build/run `test_chunk_cache`, `test_volume_local`, and VC3D; run
  `git diff --check`.

## Spec update

Specify that decoded RAM capacity and fetch concurrency are mutable global
service policy. Runtime policy changes preserve all source and queue state;
only decoded LRU eviction needed to satisfy a reduced capacity is allowed.

## Documentation updates

Update the ChunkCache and Volume API documentation to describe in-place global
capacity configuration and remove the obsolete source-handle reset statement.

## Changelog

Record the source- and queue-preserving RAM-capacity correction under
2026-08-15.

## Independent review

- A source must retain decoded byte accounting, LRU touches, and an eviction
  callback because it owns the entries, but it does not need a ceiling.
- The global budget already selects the oldest entry across all registered
  sources, so removing the local ceiling does not weaken total enforcement.
- Atomic capacity reads plus serialized budget enforcement permit concurrent
  completions during a reduction; each completion is accounted and then
  globally enforced without cancellation.
- Scheduler admission and task generations are untouched, so queued and
  running work cannot be restarted by this change.
- Persistent disk budget and write-format policy are separate concerns and are
  intentionally unchanged.
