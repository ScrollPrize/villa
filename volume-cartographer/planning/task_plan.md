# Task plan

## Current problem

`ChunkCacheService` already owns the source registry, decoded-byte budget,
active-view state, selection gate, probe scheduler, decode scheduler, and fetch
scheduler instances. However, each source `State` still selects a fetch
scheduler from its construction options. Reopening a service-retained source
refreshes fetchers but leaves the previous scheduler attached, so
`Volume::setIOThreads()` can clear data without changing actual concurrency.

The public shared-service `ChunkCache` constructor also reverses the desired
ownership: a source handle reaches into the service and registers itself.

## Implementation

### 1. Make source creation a service operation

1. Move source-open input types needed by both classes to namespace-level
   `ChunkCacheLevelInfo` and `ChunkCacheOptions`, retaining `ChunkCache` aliases
   for source compatibility.
2. Add `ChunkCacheService::openSource(identity, levels, fetchers, fill, dtype,
   options)`, returning `std::shared_ptr<ChunkCache>`.
3. Move source lookup, immutable metadata validation, source-ID allocation,
   source-state construction, budget registration, and fetcher refresh into
   service-owned registration helpers.
4. Give `ChunkCache` a private constructor accepting an already registered
   service and source state. Remove the public constructor that accepts a
   service and source identity.
5. Keep direct standalone `ChunkCache` constructors as compatibility
   conveniences. They create one service and register their sole source through
   the same shared helper; they must not create separate local schedulers.
6. Update `Volume` and service-sharing tests/callers to use `openSource()`.

### 2. Make fetch configuration service-global

1. Store one active `{maximum workers, adaptive}` configuration and active
   fetch scheduler on `ChunkCacheService::Impl`.
2. Keep scheduler instances service-owned so switching configurations does not
   destroy/join a running worker pool synchronously. Reuse an existing scheduler
   when returning to a previously used configuration.
3. Every `openSource()` applies its requested fetch configuration to the
   service. If it differs, it becomes the new global configuration; therefore
   the last source handle/configuration wins by documented contract.
4. Add a public service configuration method used by
   `Volume::setIOThreads()`. An explicit runtime change applies immediately to
   every registered source, including sources opened by other volumes sharing
   the service.
5. Remove fetch concurrency from immutable source metadata compatibility and
   stop treating `State::options_` as scheduler ownership. Source options keep
   only source-local cache/persistence policy.

### 3. Migrate all source work atomically

1. Select/create the replacement scheduler under service synchronization, then
   publish one migration through the shared scheduler selection gate.
2. For every registered source, under its state lock:
   - increment its scheduler epoch;
   - cancel the source group on the old scheduler;
   - install the service's active scheduler;
   - cancel stale probe/decode tasks carrying the old epoch;
   - retain decoded, missing, all-fill, demand, listener, and accounting state;
   - reset task IDs/error state only for unresolved demanded entries;
   - deterministically requeue those entries through the normal probe path.
3. Use a service configuration generation so overlapping configuration calls
   cannot install an older scheduler after a newer last-writer update.
4. Notify source waiters after migration. Running stale work may drain but its
   existing generation/epoch checks must prevent publication.

### 4. Make Volume always retain a service

1. `Volume::sharedChunkCache()` lazily creates and stores a service when none
   was supplied, then opens its source through that service.
2. `Volume::setIOThreads()` stores the requested mode and configures the
   retained service directly when present. It must not invalidate decoded
   source state or recreate the source handle solely to change concurrency.
3. If no service exists yet, the saved setting becomes the configuration when
   the lazily created service opens its first source.
4. `Volume::createChunkCache(options)` continues to create an isolated service
   for bounded prefill/redownload jobs, preserving their fixed concurrency and
   preventing them from changing the interactive service.

## Tests

- Convert shared-source construction tests to `service->openSource()` and prove
  direct standalone construction still creates a functional service domain.
- Prove source reuse returns the same numeric source ID and warm decoded data.
- Prove opening another source with a different worker setting changes the
  service globally and affects existing sources.
- Prove `Volume::setIOThreads()` after cache creation changes global service
  concurrency without evicting decoded chunks.
- Cover fixed-to-fixed, fixed-to-adaptive, and adaptive-to-fixed migration.
- Exercise a blocked old-scheduler fetch during migration; stale completion
  must not publish, retained demand must complete on the replacement scheduler,
  and no request may be duplicated or stranded.
- Prove isolated `Volume::createChunkCache()` configuration does not alter the
  volume's shared service.
- Run focused `test_chunk_cache`, `test_volume_local`, `test_volume_extras`, and
  the synthetic render smoke/Valgrind target because scheduler ownership is on
  the render hot path.

## Spec update

Update `planning/spec.md` to define `ChunkCacheService` as the sole owner of all
regular chunk schedulers and global fetch concurrency. Define `ChunkCache` as a
source-bound handle returned by `openSource()`, document last-writer-wins
service configuration, and distinguish separate services used by isolated
batch work.

## Documentation updates

Update `docs/remote_file_cache.md` and API comments with service/source-handle
ownership, global I/O-thread semantics, scheduler migration guarantees, and
the isolated-service rule for explicit batch caches. Correct stale text that
still describes fixed two-worker VC3D interactive reads.

## Changelog

Add one entry describing service-owned source creation and globally effective
fetch-concurrency changes without decoded-cache eviction.

## Independent plan review

- The distinction between service and source handle remains necessary because
  `IChunkedArray` is source-bound; merging them would add source IDs to every
  sampling call and immediately require another adapter.
- The plan removes infrastructure ownership from the handle without changing
  chunk identity, sampling, interpolation, or decoded data.
- Runtime migration must reuse the established scheduler epoch and stale-result
  checks rather than introduce a second cancellation mechanism.
- Applying options on each `openSource()` intentionally makes the most recent
  source configuration global. This is accepted behavior and must be explicit
  in docs and tests.
- Explicit batch caches require a distinct service, not a source-local
  scheduler, to retain isolation.
- Scheduler objects cannot be destroyed while holding service/state locks
  because their destructors may join workers; the service retains reusable
  scheduler instances until safe shutdown.
