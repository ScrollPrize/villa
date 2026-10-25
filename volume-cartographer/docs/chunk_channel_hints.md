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

The public worker API remains **`Volume::readZYX` / bbox reads**. Workers submit
independent ROI calls concurrently; they do not need to enumerate storage chunks
or adopt a second read API. The protected implementation hook
`IChunkedArray::getChunksBlocking(keys, request)` preserves result order and
defaults to the existing scalar API. Its accessor lives only in
`core/src/render/ChunkReadBatch.hpp`. Both direct and queued `ChunkCache`, and
packed channel views, implement bulk acquisition. Direct bulk calls reserve and
prefetch the entire unresolved batch before waiting in result order, so one
caller can feed the read/decode pools. This does not create a second set of
pools. `Volume::readZYX` uses that API only when
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

Native source notifications include the changed key. The coordinator polls new
admissions, promoted requests and changed keys, not the entire active window on
every completion. Invalidation and legacy unkeyed source notifications rescan
the affected source. `loaderStats().queriedKeys` measures the actual key-query
work (distinct from the number of poll batches). All-fill classification of
owned decoded buffers runs outside the source metadata lock; publication still
validates generation and fetch serial under the lock. Neither change alters
decoded values or missing/error/fill handling.

Direct blocking readers wait on a counted per-key condition channel, independent
of the evictable cache entry. A normal completion wakes only that chunk's readers
plus global prefetch/drain waiters, instead of every reader of the source.
Invalidation, refresh, cancellation and other lifecycle broadcasts still wake all
registered keys. Registration and the wait predicate use the source mutex;
notification registry locking never acquires that source mutex. Multiple readers
of one key share the wait channel until the last exits, including error paths.

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

Follow-up end-to-end GPU tracing with decoder commit `b3fd49f8c` used the same
512 chunks, 176 nonempty, 1,028 paths, 10 GiB cache, refinement off and conflict
filtering on. All outputs matched exactly. Under one nonexclusive grant of
168 CPU cores and six GPUs (3 GiB/device), three repetitions gave:

| Preparation workers | Frozen reader mean (s) | Fast decoder mean (s) |
|---:|---:|---:|
| 16 | 3.652 | 3.407 |
| 64 | 3.265 | 3.185 |
| 168 | 3.445 | 3.494 |

The best new rate is 160.75 chunks/s, only 2.5% above its matched old control;
timing ranges overlap. Isolated decoder gains therefore do not demonstrate a
large end-to-end tracing speedup. At64 workers, GPU kernels sum to0.917s across
six devices and average process CPU use is22.21 cores. Stage times include
concurrent waits; these short runs include CUDA initialization and tails and do
not uniquely identify the remaining bottleneck or establish full-volume speed.
Source files were page-cache warm, decoded caches and output geometry fresh.
Full distributions, commands, grants and stage profiles are in
`/ephemeral/hendrik/fiber3d_train2/evaluations/trace_fast_decoder_20261025_v1/REPORT.md`.

Earlier scheduler comparison:

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

### Independent native reader benchmark

`core/test/bench_parallel_chunk_read.cpp` is a standalone `vc_core` executable,
not a fiber-tracing benchmark. Build target `bench_parallel_chunk_read`, then run:

```sh
<build>/bin/bench_parallel_chunk_read reader.json measurements.json
```

Example configuration (replace the array and keys with real existing chunks):

```json
{
  "array": "/path/to/uint8.zarr/0",
  "keys_zyx": [[12, 34, 56], [12, 34, 57]],
  "backend": "queued",
  "workers": 16,
  "pool_workers": 64,
  "batch": 32,
  "cache_bytes": 10737418240,
  "repeats": 3,
  "modes": ["raw", "decode", "copy", "cold", "hot", "roi", "roi-hot"]
}
```

- `workers` controls callers; `pool_workers` controls the existing read/decode
  pools through `batchOptions`. They are not a total process-thread count.
- `raw` reads and decodes through Zarr directly; `decode` uses retained encoded
  buffers; `copy` allocates/copies retained decoded buffers. These controls have
  no cache coordinator.
- `cold` invalidates the **decoded cache**, then reads batches through the
  internal cache scheduling helper (a diagnostic, **not** the worker-facing API).
  `hot` returns resident payload references, not copies of all bytes.
- `roi` / `roi-hot` call `Volume::readZYX` for channel zero, including the actual
  output allocation and C-order to column-major `Array3D` layout conversion.
  `roi_chunks` defaults to 1; set it to 2 for a 128³ ROI over eight 64³ storage chunks.
  `batch` groups caller jobs, not the ROI shape. The output is one channel; packed inputs still require
  decoding the whole physical multi-channel chunk.
- Input is currently uint8, 3-D ZYX or packed CZYX with the channels in one
  physical chunk. Unsupported dtypes fail explicitly. Use zero-fill arrays and
  existing, fully interior chunks: the current benchmark oracle assumes these
  properties (the production reader itself has broader fill/boundary support).
- Source pages and the exact decoded oracle are warmed before timing. This is
  **not cold-storage throughput**. Every result is compared exactly with the
  oracle outside timing. Worker startup/join and validation are excluded.
- JSON records wall/user/system time, voluntary context switches, per-job
  mean/p50/p95 latency, caller-side cache-lock waits, decode counts and queue
  work. `logical_bytes` distinguishes single-channel ROI output from full
  physical chunks. Caller lock counters do not cover every internal worker lock.
- Diagnostic modes `copy-reuse` and `roi-hot-reuse` preallocate and pre-touch
  outputs on the same worker/static job partition before timing. The latter
  still performs the ordinary ROI fill/layout conversion. `fixed_jobs` selects
  static assignment for other controls too. Recorded worker CPU positions expose
  migrations; these are not CPU-pinned NUMA benchmarks.
- Linux diagnostics additionally record process backing-device read/write bytes
  and read characters; rusage supplies page faults and block-IO counters on Unix.
  Unsupported backing-IO counters are null, not assumed zero. Minor faults mean
  memory mapping/allocation work, not disk reads. Payload GiB/s is not measured
  memory-controller bandwidth.
- The decoded-cache budget is not a total-memory cap. Encoded/decoded reference
  buffers and returned output are additionally retained by the benchmark.

The reusable approved launcher is `scripts/run_fiber_tests`; see
[`FIBER_TEST_RUNNER.md`](../../scripts/FIBER_TEST_RUNNER.md) for the local locked
input suites. Frozen baseline builds must not be rebuilt with candidate sources.

### Public ROI API validation (2026-10-25)

The follow-up comparison uses **only existing `Volume::readZYX` calls** in the
timed worker loop. It covers 2048 separate 64³ ROIs and 256 disjoint 128³ ROIs (each
spanning eight 64³ storage chunks). Both produce 0.5 GiB of presence output while
decoding 3.5 GiB of packed seven-channel data. Multi-ROI tests keep the same 224-pool
setting and vary caller count; no Python processing, inference or fiber tracing
is involved. See the complete generated report at:

`/ephemeral/hendrik/fiber3d_train2/evaluations/reader_roi_multi_after_20261025_v1/REPORT.md`

The keyed direct-reader wakeups remove a measured high-concurrency regression:
224-caller multi-ROI mean falls from 0.349 s to 0.167 s. Mean voluntary context switches
fall from about 202,000 to 19,000 per pass. This is an ROI-path result, **not** the
earlier internal chunk-list submission comparison. Exact voxel checks and 166
native regressions passed. Direct remains default; queued remains opt-in.

These are three-repeat, warm-filesystem, cold-decoded-cache measurements on the
locked Paris4 fixture, GCC 13 RelWithDebInfo. They do not imply linear scaling to
224 callers or full-volume/cold-disk throughput: 256 ROI jobs give only 1.14 jobs per
caller at that setting. Decode/allocation/copy controls and all distributions
are in the report. Existing production processes were not rebuilt or restarted.

### Remaining scaling attribution (2026-10-25)

Independent matched-input controls (three repeats, same build/fixture above)
are in `reader_scaling_diagnosis_20261025_v1/REPORT.md` under the evaluations root.
Matched caller/pool settings1/16/64 give public ROI means6.925/0.452/0.179s:
15.3x and38.8x scaling. The earlier fixed224-pool sweep was not core-count
scaling: even its1caller ran substantial background work. Pool counts are not
total core counts here either; process CPU equivalents are recorded.

All timed passes recorded zero backing-device read bytes and major faults.
In-memory decode, with no cache coordinator, takes0.179s at64workers, approximately
the whole ROI pipeline's elapsed time. Allocation/page faults are a demonstrated
cost: copying3.5GiB at64workers takes0.109s with fresh buffers and about916k minor
faults, versus0.045s and zero faults with pre-touched reused buffers. The kernel
CPU cost falls from5.337s to0.002s. Reused-copy throughput itself plateaus from
16to64workers (86.5to77.7GiB/s payload); this does **not** prove hardware DRAM
saturation because placement is unpinned and hardware counters are unavailable.
Decode allocation/redundant initialization/copies are the next evidence-backed
optimization targets. No production algorithm or buffer policy changed in this
diagnostic; exact byte/voxel checks passed for all cases.

### Decode buffer recycling

The ordinary native Zarr decode stage can reuse released dense allocations.
`DecodedChunkCacheBudget` owns exact-size idle buffers. Published chunk bytes
remain immutable shared leases; their custom deleter returns the allocation only
after the final reader/persistent writer releases it. Weak budget ownership
prevents cycles and permits readers to outlive their source/service safely.

Idle vector **capacity** plus resident decoded bytes share the configured decoded
ceiling. Idle storage is discarded before useful cache entries; shrinking the
budget also trims it, and late reader releases respect the new limit. As before,
this is a retained-cache budget, not a cap on external leases or in-flight decode.
Budget stats expose `recycledBytes` and cumulative `reusedBuffers` separately.

If a decode needs a new buffer while the cache is full, it may use the existing
global LRU to retire a victim **before** allocation, rather than at publication.
The existing preferred-victim and parked-reader protections remain active. It
does not wait for a busy enforcement mutex: allocation is the fallback. Victim
selection can still acquire source locks; this is not a wait-free operation.
This earlier eviction is a deliberate cache-policy timing change, so bounded-cache
benchmarks check decode counts/refetches as well as elapsed time.

`ZarrArray::decode_chunk_payload` no longer copies compressed input before passing
it to a codec. Shared `decode_chunk_payload_into` logic also serves the existing
`read_chunk_into`, using codec-into callbacks and in-place byte swapping. V2
filters and codecs without an into callback retain allocating fallback behavior.
The strict payload helper checks full size, while the file-read API preserves its
legacy oversized-prefix and short-payload-false fallback behavior. Blosc/Gzip
short/incomplete decodes now throw rather than exposing zero or stale tail bytes;
custom into callbacks must likewise fully initialize output or throw.

This does not change the public ROI API, channel-hint behavior, scheduling backend,
precision or disk formats. Reuse currently feeds ordinary split-transfer decoding;
sharded-object and persistent-read decoding retain allocating behavior. Ordinary
all-fill/stale/error publication returns unused storage too; Delta3D rejected and
all-fill results still release it normally. Decode-stage telemetry includes buffer
acquisition/early eviction, not only the codec kernel.

The native regression suite covers lease lifetime, concurrent shrink/release,
all-fill, ordinary streaming eviction, exact codecs, endian/filter fallback and
strict corrupt-size handling. Run `scripts/run_fiber_tests` with local suite
`reader-decode-reuse`; generate comparisons with `reader-decode-reuse-report`.
All measurements/builds are isolated from running production binaries.

Measured final results (`reader_decode_reuse_20261025_v2/REPORT.md` under the
evaluations root), same input and three repeats:

| ROI workload | Callers | Before mean | After mean | Speedup |
|---|---:|---:|---:|---:|
| 10 GiB cache, first + two re-decodes | 1 | 6.925 s | 6.471 s | 1.07x |
| 10 GiB cache, first + two re-decodes | 16 | 0.452 s | 0.365 s | 1.24x |
| 10 GiB cache, first + two re-decodes | 64 | 0.179 s | 0.170 s | 1.05x |
| 512 MiB cache, streaming 3.5 GiB | 16 | 0.380 s | 0.336 s | 1.13x |
| 512 MiB cache, streaming 3.5 GiB | 64 | 0.178 s | 0.156 s | 1.14x |

The first allocating pass changes little. The64caller repeat measurements are
noisy (0.126–0.175s), so the report retains them all rather than selecting the
fastest. Isolated64worker in-memory decode is0.179s with allocation versus0.085s
into pre-touched buffers; that2.10x is **not** end-to-end ROI speedup. Streaming
at64callers has2051decodes/pass before and2052after on average for2048requested
chunks: extra eviction/refetch work remains small in this fixture.16callers
decode exactly2048chunks in both versions. All bytes/voxels match;199native
regressions and12launcher tests pass. No production process was restarted.

### Generic ROI layout and destination reuse

Storage chunks are X-contiguous, while the existing `Array3D` output is
Z-contiguous. A naive X-inner copy writes with a 16 KiB stride for a 128³ uint8
ROI. `copyStrided3D` transposes small tiles in stack-local scratch storage,
then writes contiguous Z runs. This shared `Volume::readZYX` path applies to
every `IChunkedArray` decoder, not just Zarr or fiber tracing. It preserves the
public output layout, typed values, packed-channel offsets and edge clipping.

Successful interior reads overwrite every output cell, so they no longer clear
the entire output first. Missing/all-fill chunks explicitly fill their regions;
out-of-volume padding remains zero. Callers can retain an already-sized output
between reads to avoid allocation. A failed read throws and its partial output
must not be used. This is not zero-copy: assembling arbitrary ROIs from shared
immutable storage chunks into a differently laid-out contiguous array still
requires a copy.

For decoder authors, `ChunkFetchResult::decodeBuffer` is the optional exclusive
destination contract: consume it when present, otherwise allocate; fully write
the decoded output or return an error. Contents are unspecified before decode.
This works independently of codec: a synthetic non-Zarr decoder test verifies
the same allocation is reused only after its final reader releases it. Codecs
without destination support keep their allocating fallback. Pool diagnostics
report buffer requests, successful reuse, capacity rejections, trimmed buffers/
bytes, and busy opportunistic eviction attempts.

Idle-pool shrinking now frees only the excess capacity, not the entire pool.
It detaches buffers under the pool mutex and frees them outside it, without
allocating a temporary discard list. Failed pool insertion removes empty buckets;
late reader releases and concurrent shrinking remain safe.

The matched `reader_layout_20261025_v1/REPORT.md` measurements (same fixture,
three repeats) isolate this layout work from prior decode improvements:

| Callers | Resident ROI before / after | Decode + ROI before / after |
|---:|---:|---:|
| 1 | 2.078 / 0.469 s | 6.471 / 5.919 s |
| 16 | 0.139 / 0.039 s | 0.365 / 0.357 s |
| 64 | 0.049 / 0.020 s | 0.170 / 0.140 s |

These are read/assembly times, not tracer times. Bounded-cache streaming sees
little improvement from layout alone. Pool counters show capacity rejection and
busy opportunistic eviction, not bulk trimming, dominate remaining missed reuse.
We retain allocation fallback instead of forcing readers/decoders to serialize
or silently growing the configured cache budget. Final partial-trim checks are
recorded separately in `reader_layout_20261025_v2/REPORT.md`.
Those final streaming16/64means are0.329/0.148s versus a fresh frozen-baseline
rerun0.339/0.165s. Copy-only timings0.327/0.155s are close; partial trim is not
a demonstrated major throughput win. Final204native and12launcher tests pass,
with exact benchmark voxel comparisons and zero physical reads during timing.

### All-channel throughput and actual layout chain

The reader-only fixture now supports `roi_channels` (default1). For multiple
channels it requires unsharded full-channel packed physical chunks, creates
ordinary shared-cache channel views, and calls public `Volume::readZYX` for
each channel of each spatial ROI. Every output channel is checked against an
independent C-order decode oracle. No new production read API was added.

Matched7-channel measurements (`reader_multichannel_20261025_v1/REPORT.md` under
the evaluations root),64callers,three repeats,3.5GiB decoded and output:

| Operation | Mean time | Useful output GiB/s |
|---|---:|---:|
| memcpy into reused storage | 0.0432 s | 81.08 |
| Decode into reused storage | 0.0853 s | 41.05 |
| Decode with fresh storage | 0.1661 s | 21.08 |
| Resident ROI, reused output | 0.0876 s | 39.94 |
| Resident ROI, fresh output | 0.1335 s | 26.22 |
| Full read/decode/ROI,10GiB cache | 0.2621 s | 13.36 |
| Full read/decode/ROI,512MiB cache | 0.5252 s | 6.66 |

Full10GiB-cache one-channel output is3.52GiB/s (0.1419s for0.5GiB). Seven channels
provide3.79x the useful throughput, not7x: decode is shared but copying/allocation
of the six companion outputs is additional work. Both variants perform exactly
2048physical decodes per pass with10GiB retained cache. At512MiB and64callers,
all-channel decode counts6669/6216/6575 show3.17x amplification: companion reads
can revisit a physical chunk after eviction.16caller512MiB runs remain2048decodes
and9.04GiB/s. Each128³ ROI requires14MiB packed data, so64simultaneous ROIs already
span896MiB, exceeding512MiB before other in-flight work. Do not infer production
full-tracer timing from this intentionally small-cache control.

Input and final GPU output **are X-fast spatially**, but the CPU intermediate is
not. Actual indexing:

- Zarr CZYX C-order: `x + X*(y + Y*(z + Z*c))` (planar channels).
- `Array3D` ZYX: `z + Z*(y + Y*x)` (legacy column-major, Z-fast).
- `FiberChunkTensor::packGpu`: `6*((z*Y+y)*X+x)+c` (six interleaved byte channels),
  plus a separate X-fast float-width buffer with sampling/scaling.

Thus an X-fast -> Z-fast -> X-fast round-trip really exists. The blocked copy
accelerates its first half; it does not remove the intermediate. A future
layout-aware ROI/tensor integration could avoid it, preserving legacy callers
and sampling numerics. This task only measured/inspected, not changed layouts.
GPU packing and width interpolation are excluded from these timings, as is
production presence-first seed scanning that can delay/skip companions. All
timed source reads hit the filesystem page cache. CPU placement is unpinned;
throughput is payload, not measured DRAM bandwidth. Use approved runner suites
`reader-multichannel` and `reader-multichannel-report`.
