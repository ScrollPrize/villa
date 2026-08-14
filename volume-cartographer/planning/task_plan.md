# Task plan

## Current problem

`ChunkCacheService::openSource()` currently accepts `ChunkCacheOptions`, which
mixes source-local persistence policy with service-global decoded-budget and
source-read concurrency policy. Every source acquisition calls
`configureServiceFetchScheduler()`, so opening or reopening a volume can change
global concurrency.

Changing concurrency currently replaces the service scheduler, cancels pending
tasks by scheduler epoch, and blindly requeues every demanded `InFlight` entry.
Already-running work cannot be cancelled, so a replacement scheduler starts a
second request for the same chunk and invalidates the first result. The focused
CI test explicitly expects two fetches and then underflows a one-shot test
latch. Making that latch multi-call-safe would preserve the wrong production
behavior.

`Volume::setIOThreads()` compounds the ownership problem: its name and location
look volume-specific, but it mutates the application-wide service shared by all
volumes.

## Implementation

### 1. Split service and source configuration

1. Introduce explicit service configuration containing:
   - aggregate decoded-byte capacity/budget;
   - physical source-read worker capacity;
   - current fixed/adaptive admission policy;
   - optional persisted adaptive-download state.
2. Introduce source-only acquisition options containing:
   - metadata-entry capacity and all-fill detection;
   - persistent-cache path and budget root;
   - persistent compression and quantization policy.
3. Remove decoded-budget and concurrency fields from source acquisition
   options. Do not retain deprecated aliases that allow a source to alter
   service policy.
4. Rename `ChunkCacheService::openSource()` to `acquireSource()` because it may
   either register a new source or return a handle to retained source state.
5. Keep source identity interning, metadata validation, fetcher refresh,
   numeric source-ID allocation, and source-handle creation in
   `acquireSource()`. None of those operations may alter scheduler policy.
6. Update standalone and isolated-cache creation paths to construct a separate
   service with their desired service policy, then acquire their source using
   source-only options.

### 2. Keep one source-read scheduler and reconfigure admission in place

1. Give each `ChunkCacheService` one source-read scheduler for its lifetime.
   Remove the scheduler-per-configuration map, active-scheduler replacement,
   fetch-configuration generations, and source scheduler migration.
2. Extend `ChunkRequestScheduler` with a synchronized in-place concurrency
   update:
   - fixed mode sets the admission limit to the requested value;
   - adaptive mode updates its bounds and resets only transient search/stability
     state while retaining the reusable long-term adaptive model;
   - increasing admission wakes workers immediately;
   - decreasing admission starts no additional tasks until active work falls
     below the new limit.
3. Keep queued tasks, task IDs, FIFO order, priorities, source demand, and
   scheduler groups untouched by a concurrency update.
4. Let running work complete and publish normally. A configuration update must
   not increment source epochs or fetch serials, cancel work, restart work, or
   invoke extra readiness/activity callbacks.
5. Configure a service with sufficient physical workers when it is constructed.
   Runtime admission must not silently exceed that capacity; invalid settings
   fail loudly.
6. Preserve adaptive transfer samples and persisted capacity data where valid.
   Switching modes resets only mode-specific transient probe state, not the
   queue or completed long-term model.

### 3. Make global ownership explicit at callers

1. Remove `Volume::setIOThreads()` and its stored per-volume `ioThreads_` state.
   Callers that intentionally change regular I/O concurrency must use the
   shared `ChunkCacheService` API.
2. Construct the production VC3D service once at application startup with the
   normal adaptive `[2,64]` source-read policy and restored adaptive state.
   Passing that service into windows, workspaces, and volumes must not modify
   its policy.
3. When a `Volume` has no supplied service, create its private service once:
   remote volumes use the existing adaptive default and local volumes use the
   existing fixed default. Source acquisition does not revisit that decision.
4. Explicit prefill, redownload, and batch operations continue to create an
   isolated service with their requested fixed concurrency. They may share a
   decoded-byte budget but never the regular scheduler.
5. Update all core, VC3D, Lasagna, benchmark, and test callers to pass service
   and source options at their proper ownership boundary.

## Tests

1. Replace `ChunkCacheService migrates unresolved work to replacement
   scheduler` with an in-place reconfiguration test:
   - start one blocked chunk and leave another queued;
   - increase admission and prove the queued chunk starts without restarting
     the blocked chunk;
   - release both and prove each key was fetched exactly once and each result
     published once.
2. Add a decrease test proving running work is not cancelled and no new task is
   admitted until active work falls below the new limit.
3. Cover fixed-to-fixed, fixed-to-adaptive, and adaptive-to-fixed updates while
   preserving pending order, view-relative priority, demand, and callbacks.
4. Prove acquiring and reacquiring sources cannot change service concurrency,
   while fetcher refresh still preserves source ID and decoded data.
5. Prove two sources share the explicitly configured service policy and an
   isolated batch service cannot alter it.
6. Remove tests of `Volume::setIOThreads()` and replace them with tests of
   service configuration before and after volume source acquisition.
7. Run:
   - `cmake --build build/ci-fast-core --target vc_test_core --parallel 4`
   - `build/ci-fast-core/bin/test_chunk_cache`
   - focused `test_volume_local`, `test_volume_extras`,
     `test_chunk_cache_persist`, and `test_zarr_chunk_fetcher`
   - `ctest --test-dir build/ci-fast-core --output-on-failure --parallel 4 -L '^vc-core$'`
   - VC3D compile/smoke and the synthetic render fixture.
8. Defer the known Valgrind role-attribution failure to its separate archived
   task, as requested; do not treat it as validation for this API correction.

## Spec update

Update `planning/spec.md` to state that:

- `ChunkCacheService::acquireSource()` is source-only and cannot modify service
  scheduling;
- source-read concurrency is configured explicitly once per service and may be
  changed only through the service API;
- runtime concurrency updates modify admission in place and never cancel,
  restart, duplicate, or invalidate running/queued work;
- `Volume` does not own or expose regular source-read concurrency;
- isolated operations use isolated services.

Remove the existing last-source-open-wins and scheduler-replacement language.

## Documentation updates

- Update `docs/remote_file_cache.md` with the service/source option split,
  source acquisition semantics, explicit global configuration, and in-place
  admission updates.
- Update `docs/api/Volume.md` to remove `setIOThreads()` and direct users to the
  service-level setting.
- Update public API comments in `ChunkCache.hpp`, `ChunkRequestScheduler.hpp`,
  and `Volume.hpp` so source acquisition and scheduling ownership cannot be
  confused again.

## Changelog

Add one entry recording that source acquisition no longer changes global I/O
policy and runtime concurrency changes preserve all running and queued work.

## Independent plan review

- **Task coverage:** The plan fixes both ownership errors: source acquisition no
  longer acts as a global setter, and volume-scoped configuration is removed.
- **CI root cause:** The failing latch test is replaced rather than weakened.
  The new invariant is one fetch per key across a concurrency change, so the
  production duplicate request that triggered the test failure is removed.
- **Specification consistency:** The plan preserves the established single
  application cache service, source-retained decoded data, numeric source IDs,
  explicit isolated services, queue priority, and non-cancellation of running
  work. It intentionally changes the recently documented last-source-wins and
  scheduler-replacement behavior because those contradict the clarified task.
- **No numeric/render change:** Chunk values, interpolation, transforms, and
  rendering are untouched. Only source registration and scheduler admission
  ownership change.
- **Concurrency safety:** In-place admission avoids the impossible operation of
  transferring a running function between executors. Running work remains on
  the same executor; pending work remains in the same queue.
- **Portability:** The implementation uses the existing mutex/CV worker model
  and adds no platform-specific synchronization.
- **Residual risk:** Mode changes must preserve adaptive-state locking and wake
  semantics. The fixed/adaptive transition tests and full core shard are
  required before completion.

## Adequacy

Yes. This design fixes the non-Valgrind CI failure at its source rather than
making the test tolerate duplicate work. It also removes the API path that
caused scheduler replacement during ordinary source acquisition. The solution
is adequate provided implementation uses one persistent scheduler with mutable
admission; replacing executors and attempting to migrate running work would
reintroduce the same side effects.
