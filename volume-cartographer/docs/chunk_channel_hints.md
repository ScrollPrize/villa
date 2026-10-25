# Generic channel read hints and decoded-cache partitions

These APIs belong to `vc::render`, not to the fiber tracer. They work with
`IChunkedArray`, `ChunkCache`, packed Zarr channel views and `Volume::readZYX`.
They do not change decoded values, interpolation or persistent cache identity.

## Hinting other channels

```cpp
using namespace vc::render;
ChunkRequestContext request;
request.channelHints = std::make_shared<ChunkChannelHints>(
    std::vector<std::shared_ptr<IChunkedArray>>{channelB, channelC});
auto a = channelA->getChunkBlocking(level, z, y, x, request);
// B and C are now scheduled into their ORDINARY caches for the same region.
// Later reads use the usual API, with no hint or special cache view:
auto b = channelB->getChunkBlocking(level, z, y, x);
```

The hint may contain just companion channels, or all channels including the
one being read. A caller can reuse the same immutable hint across its reads.
The request-aware `tryGetChunk`, `getChunkBlocking`, `prefetchChunks` and typed
static `Volume::readZYX(..., level, request)` overloads support it. Cache-only
lookups remain cache-only and never initiate companion loading.

- Hinted channels must have the same logical shape, levels and level transforms.
  Storage chunk sizes may differ: the covered region is mapped to all intersecting
  companion chunks. An incompatible origin or companion fails explicitly.
- Companion work is asynchronous ordinary prefetch/decode, not a barrier before
  returning A. A subsequent blocking read of B waits if it is still in flight.
  Entries remain governed by normal LRU eviction and byte budgets; no pinning or
  hidden retained decoded-payload cache is added.
- Packed channel views expose a shared `physicalCacheIdentity()`. Aliases of
  the same physical decoded payload are coalesced, preserving channel offsets.
  Packed CZYX already decodes its packed channels together; the hint adds no
  redundant decode. Separate arrays populate their own existing caches.
- Request context and priority are forwarded; the hint itself is removed before
  forwarding companion work, preventing recursive fan-out. For an ROI, hints
  expand once on its batched prefetch, not on every voxel/row acquisition.
- Missing/all-fill/error results retain their ordinary semantics. A companion
  error is visible when that channel is read; it does not invent an A error.
- Other `IChunkedArray` implementations inherit request-aware expansion, or must
  forward it if they override those overloads. Implementations that expose
  aliases should override `physicalCacheIdentity()` accordingly.

## Resource-scaled batch service

`ChunkCacheService::batchOptions(cores, decodedBytes)` returns an explicit batch
policy with `max(16, 2*cores)` local-read workers and `cores` decode workers.
It preserves the normal independent remote scheduler policy. Configure the
process service before opening sources. This does not change viewer defaults.

The whole-volume tracer configures this policy from the **actual granted** CPU
count. Its metadata preflight runs through the same launcher/build in a temporary
process, which exits before admission. The supervisor does not leave ungranted
worker pools alive. `loader.resource_scaled=false` is the explicit legacy control
for comparisons or hosts that already configured their process service.

## Optional worker-local decoded caches

`parentService->createDecodedPartition(bytes)` creates a separate decoded LRU and
byte budget while sharing the parent's local-read, remote-read, probe and decode
schedulers. It does not create another set of worker pools. Source/task IDs are
unique across the service family. Changes to scheduler concurrency and its
reporting are shared; decoded-byte budget changes remain partition-local.

`cache->inService(partition)` and `rebindChunkCache(array, partition)` reuse the
same source, fetchers and decoder, including packed channel offsets. Logical
source invalidation propagates across partitions, including cached missing/fill
entries. Partition destruction cancels its own sources without draining or
cancelling unrelated parent work. Rebinding to an unrelated service family is
rejected. Current partitions require RAM-only caches: sources with persistent
cache writers are rejected rather than creating competing disk writers.

The fiber chunk builder optionally uses this API with `worker_cache_bytes > 0`;
zero retains the shared cache. Independent partitions can duplicate decoded
overlapping chunks. Choose budgets using the aggregate worker count PLUS the
parent cache budget, not just the per-worker amount. Ordinary consumers (such as
filter-normal reads) can still populate that parent cache.

## Validation and measurements

`test_channel_read_hints` covers ordinary companion hits, different chunk sizes,
packed aliases, missing/fill/error semantics, invalidation, resource sizing and
partition teardown while unrelated work is pending. Existing cache, Zarr, ROI,
prediction and chunk-tracing suites cover numerical behavior.

`bench_chunk_cache_contention MAX_WORKERS OUTPUT.json` measures warmed shared
versus partitioned cache hits with disjoint and overlapping keys. Enable
`VC_SAMPLING_WAIT_PROFILE=1` for mutex/wait attribution. This excludes I/O,
decompression, tensor copying and tracing; it must not be treated as a full
pipeline speedup. Run it within the appropriate resource allocation.

The Paris4 512-core matched benchmark, exact commands, stage distributions and
raw artifacts are recorded in
`/ephemeral/hendrik/fiber3d_train2/evaluations/channel_loader_20261025_v1/REPORT.md`.
Shared warm-hit contention remains. Worker-local partitions are the measured
alternative in that experiment; the separate queued backend below is experimental.

## Optional queued batch loader

Configure `ChunkCacheService::Options::loaderBackend = LoaderBackend::Queued`
before opening sources. `queuedChunkWindow` limits simultaneously admitted
physical source/key pairs; zero selects `2 * (localReadWorkers + decodeWorkers)`.
The existing `Direct` backend remains the default. This is a static admission
window sized from worker capacity, not automatic observation of worker-queue
occupancy. Explicit window sweeps are supported.

```json
{
  "cache_bytes": 10737418240,
  "loader": {
    "resource_scaled": true,
    "backend": "queued",
    "queued_chunk_window": 0
  },
  "chunks": { "worker_cache_bytes": 0 }
}
```

This is a generic `vc::render` facility, not a tracer-specific cache. Native
`ChunkLoadQueue` accepts batches from concurrent callers. One coordinator merges
duplicate source/key requests, groups cache acquisition/poll/release per source,
and continuously replaces completed keys until the admission window is full.
It does not wait for an entire ROI to finish before admitting another. Existing
local-read, remote-fetch and decode workers do the actual work; the coordinator
does no blocking I/O or decoding. Completion/invalidation notifications wake it,
without timed polling. Foreground and numeric priority order pending admission;
key ordinal interleaves equal-priority request batches.

Resident terminal results bypass the coordinator: bulk reads inspect the batch
under one source lock, take owned payload references and submit only unresolved
keys. Results are scattered back into original order, including duplicates.
Blocking hits preserve LRU/demand bookkeeping; cached errors and fill results
also bypass the queue. Queued prefetch similarly omits resident keys. This avoids
a future/coordinator round trip for each already-decoded normal/filter sample.

`IChunkedArray::getChunksBlocking(keys, request)` preserves result order and
defaults to the existing scalar API. Queued `ChunkCache` and packed channel
views implement bulk acquisition. `Volume::readZYX` uses that API only when
`prefersBatchReads()` is true; direct sources keep their previous streaming copy.
Companion hints still populate ordinary caches and use the same queue. Packed
channel offsets, missing/fill/error handling and interpolation are unchanged.
Interactive/viewer scheduling and cache-only reads retain their existing paths.

Admitted keys use ordinary blocking-reader reservations until result handoff.
Invalidation wakes and retries pending reads. Source cancellation and queue
shutdown resolve futures, release reservations and unregister callbacks; cache
shutdown then drains the existing read/decode pools. Notifications do not retain
service handles. User chunk-ready listener exceptions are isolated so they cannot
kill a worker or prevent queue notification.

### Limits and observability

- The window bounds admitted keys, **not bytes**. Pending descriptors, returned
  ROI payloads and caller tensors are outside the decoded-cache budget. Bulk ROIs
  retain all returned payloads until copying finishes. Do not interpret 10 GiB
  decoded cache as a 10 GiB process-memory cap.
- Worker-local decoded partitions and the queued backend are deliberately
  separate alternatives; combining them fails explicitly. Direct partitions
  still support configurable per-worker budgets such as 1 GiB.
- Cancellation is source-wide or whole-queue; no per-request cancellation API.
- Custom source adapters must acquire reservations atomically and release all
  supplied reservations before potentially throwing. Callback failure may stop
  the whole queue; native cache adapters satisfy these ownership guarantees.
- The inbox itself has no independent descriptor limit. Callers must bound
  concurrent requests, as the native chunk worker pipeline already does.
- Metadata is still locked once per source batch, and completion workers still
  use ordinary cache locks. This is not a lock-free cache or decoder rewrite.
- `ChunkCacheService::loaderStats()` / Python `chunk_loader_stats()` expose
  requests, requested keys, coalescing, completed admissions, polls/wakes and
  in-flight/pending peaks. Completed admissions include cache hits/repeated keys,
  **not unique disk reads or decodes**. Reply bytes count logical references held
  for incomplete requests (shared payloads can be counted multiple times); they
  exclude results already handed to callers.

`test_chunk_load_queue` covers concurrent coalescing/admission, tiny budgets,
invalidation during reads, hints, packed bulk ROI/channel offsets, missing/fill/
error handling, throwing callbacks, cancellation and shutdown. The final
512-core Paris4 comparison (three repeats, same 10 GiB shared budget and grant)
is recorded at
`/ephemeral/hendrik/fiber3d_train2/evaluations/queued_loader_20261025_v2/REPORT.md`.
It is a cached-input benchmark, not a cold-disk or full-volume timing. No production
run is switched automatically; use the measured comparison before selecting it.

The subsequent fixed-work1/16/64 scaling and resident-hit/scheduler-wakeup trials
are recorded in `evaluations/queued_wake_scaling_20261025_v1/REPORT.md` under the
same evaluations root. Ordinary (non-maintenance-reserved) scheduler completion
now wakes one waiter for one released admission slot. Maintenance reservation,
shutdown and concurrency/adaptive changes retain broadcasts. This generic worker
optimization does not alter priority selection or numerical processing.

### Measured baseline (2026-10-25)

Retain the current resident-hit fast path and single-completion wakeup as the
implementation baseline; `Direct` remains the default and `Queued` is opt-in.
For the same 512 Paris4 cores, 10 GiB shared cache and six GPUs:

| Preparation workers | Direct wall time (s) | Queued wall time (s) |
|---:|---:|---:|
| 1 | 66.234 | 66.214 |
| 16 | 3.602 | 3.468 |
| 64 | 3.186 | 3.075 |

The 16/64 results are three-run means; 1 worker is a single verification run.
Read/decode pools scale with preparation workers, so these are not total thread
counts. Inputs were locally cached; decoded caches were cold per process. The
timings include preparation, tracing, filtering and output, not isolated loading.
All outputs matched the frozen baseline exactly. The complete report above
contains distributions, resource settings, commands and profiling counters.

This does **not** demonstrate bottleneck-free vc3d loading: scaling above 16
workers remains limited, residual cache locks remain, and neither cold storage
nor viewer throughput was measured. The queue is not automatically used by the
interactive viewer. Isolating ROI loading/decode/copy from seeding and filtering
is necessary before attributing the remaining plateau or claiming loader scaling.
