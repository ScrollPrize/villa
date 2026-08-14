# Remote File Cache

`vc/core/util/RemoteFileCache.hpp` provides persistent exact-byte caching for
one arbitrary remote file or object. It is not a recursive directory cache and
does not replace the existing remote Zarr object cache.

## API

Call `vc::core::util::cacheRemoteFile(source, options)`. The caller supplies a
cache root and a relative destination below that root. The returned result
contains the local payload path, normalized endpoint, a cache-hit flag, and an
optional persistent-budget read pin.

Built-in fetching supports HTTP, HTTPS, S3, and region-qualified S3 locations.
`RemoteFileCacheOptions::fetcher` can provide another fetch-to-temporary-file
transport. Explicit `HttpAuth` is passed to the built-in transport; otherwise
the existing AWS credential loader is used where required.

`CacheFirst` reuses a valid entry. `Refresh` fetches and atomically replaces
it. `invalidateRemoteFileCacheEntry()` removes one known destination. There is
no automatic TTL, ETag check, or prefix traversal.

## Entry Contract

The payload is stored at the caller-selected destination. Its adjacent
`<payload>.vc-remote-file.json` sidecar records:

- sidecar format version;
- canonical query/fragment-free source location;
- exact payload size;
- `managed` or `unmanaged` accounting class.

A cache hit requires both files and matching identity, size, and accounting.
Downloads use unique temporary files and publish payload plus sidecar by
rename. A failed refresh restores the previous valid pair. Concurrent requests
for the same destination and source are coalesced within one process.

The readable source is stored in plaintext, but query strings and fragments
are removed before persistence because they may contain signed credentials.
The full caller-supplied locator is used only for the request. Error and
Lasagna hit/download diagnostics redact query strings.

## Disk Accounting

`Managed` payloads reserve space and hold read pins through
`PersistentZarrCacheBudget`. Startup scanning recognizes managed remote-file
payloads so accounting survives process restart. Budget eviction removes the
payload; a later lookup treats the remaining or missing sidecar as a miss.

`Unmanaged` payloads are small control files outside the evictable Zarr-byte
budget. Cached Lasagna manifests use this class. Callers must choose the class
deliberately and keep large data payloads managed.

## Lasagna Layout

`remoteFileCachePath()` maps a validated remote source to a readable path:

```text
<remote-cache-root>/remote_sources/https/example.org/run/file.json
<remote-cache-root>/remote_sources/s3/bucket/run/file.json
```

Scheme, authority, and object-path components are mirrored directly. Empty,
traversing, or platform-invalid components are rejected. Direct remote
Lasagna manifests use this path for the manifest and its sidecar; relative
Zarr groups share the readable parent cache directory. Existing local
manifests with an explicit adjacent `lasagna-remote.json` remain supported and
its `artifact_url` remains authoritative.

There is no lookup or migration for the earlier development-only cache
layout. Reopening a remote source populates the readable layout from a cold
cache.

## VC3D Download Diagnostics

VC3D's existing application cache status bar reports fixed-GiB RAM and disk
usage. An idle remote volume appears as:

```text
RAM 3.2/10.0 disk 82.4/500.0 GiB  net idle  Z sens: 1.0
```

During active remote downloads it includes in-flight count, recent throughput,
and unresolved requests by pyramid level:

```text
RAM 3.2/10.0 disk 82.4/500.0 GiB  net 16x 42.7MiB/s q1 8/0/3  Z sens: 1.0
```

`q1` means level 1 is the first pyramid level with queued or running chunk
requests. Slash-separated values cover consecutive levels through the last
nonzero level; interior zeros are retained. Queue information is omitted when
no remote fetch is active, and local volumes omit the network field entirely.

Counts come from `ChunkCache` state, so repeated requests for one unresolved
chunk count once. Cache diagnostics and Z-scroll sensitivity are composed into
one permanent status label so growing queue text cannot overlap a neighboring
status widget.

The MiB/s field is not derived from completed-chunk wall spans. Scoped remote
Zarr reads report encoded HTTP response-body increments from the common curl
callback. The shared source scheduler aggregates those bytes across concurrent
requests over the last five active transfer seconds; intervals with no measured
transfer active do not dilute the result. The adaptive controller consumes this
same measurement. A fetcher that cannot report progress uses a compatibility
fallback of mean individual request bandwidth multiplied by the admission
represented by its completion samples.

For render-order debugging, start VC3D with
`--debug-download-queue`. Every shared slice viewer, including plane, segment,
strip, generated annotation, and Spiral views, then overlays pixels belonging
to chunks which are actively being fetched from a remote source. The overlay
uses stable colors by pyramid level at 50-percent opacity and includes both the
base and volume-overlay sources.

The overlay tracks actual source fetch start/stop events. It does not show
requests waiting in a queue, persistent-cache reads, decoding, or decoded-cache
hits. Each accepted render builds compact `uint16` pixel-to-chunk maps for its
requested and fallback levels. This full-frame pass and its retained maps add
CPU and memory overhead, so the flag is intended only for diagnostics and is
disabled by default.

## VC3D Regular Chunk Cache

VC3D creates one application-lifetime `vc::render::ChunkCacheService` and
injects it through `CWindow`/`CState` into every `Volume`. A `ChunkCache` is a
source-bound `IChunkedArray` handle over a service-retained source state. Main
views, Spiral views, overlays, and `SurfaceCache` fillers therefore reuse the
same decoded source chunks and in-flight requests.

Shared sources are acquired only through `ChunkCacheService::acquireSource()`.
Standalone `ChunkCache` construction remains a convenience for tests and batch
callers, but internally creates and retains a separate one-source service. The
handle never creates probe, source-read, or decode schedulers of its own.

Source registration is a cold path. `Volume::chunkCacheSourceIdentity()` uses a
canonical local path, or a normalized remote URL plus selected base scale, and
the service interns it to a monotonic `VolumeSourceId`. Hot `ChunkKey` values
contain only that integer plus level and chunk coordinates; paths, URIs,
credentials, and disk-cache directories are absent from render-time hashes.
Registering an existing identity validates immutable source metadata and fails
if it is incompatible. A compatible registration adopts the newly opened
per-level fetchers, which allows refreshed temporary credentials to recover an
existing source without changing its numeric ID or dropping decoded chunks.
Probe, source-read, persistent-decode, and fetched-decode tasks carry an
immutable fetcher handle plus a separate fetcher generation. Work from an older
fetcher may drain, but cannot advance stages or publish after refresh. Source
identity, cache paths, and logs remain independent of credential material.

The service retains one source-local entry map, LRU, request state, listeners,
and diagnostics record per interned source. All source states use one aggregate
decoded-byte budget and the service's shared probe, source-read, and decode
schedulers. Decoded payloads remain ordinary heap-owned byte vectors; they are
not memory mapped.

Source-read concurrency is global to a service. The service constructor fixes
the physical worker capacity and initial fixed/adaptive admission policy.
`acquireSource()` accepts only source-local metadata and persistent-cache
options, so acquiring or reacquiring a source cannot change concurrency.
`configureFetchConcurrency()` updates admission on the existing scheduler in
place: queued and running work remains attached, no request is restarted, and
completed work publishes normally. Explicit prefill, redownload, and batch
caches obtain isolation by owning a separate service, not by attaching a
private scheduler to a source handle.

Changing the current volume drops viewer references but keeps the `Volume`'s
lightweight source handle and service state, so switching A -> B -> A neither
reopens the source nor refetches resident chunks. Cache retention is
capacity-bound, not guaranteed. Explicit volume invalidation clears only that
source and rejects stale fetch completion.
No switch-time cancellation of already-running fetch/decode work is
implemented. Interactive ownership comes only from explicit versioned per-view
demand snapshots; pending work without current interactive or background
ownership is canceled, while already-running work may drain.

Interactive VC3D renders publish versioned per-view demand before normal
sampling starts. A sparse, stratified viewport probe associates missing chunks
with retained screen-space occurrences. The randomized 8-pixel probe interval
controls coverage only. Occurrences are deduplicated per chunk and level by the
chunk's projected representative footprint, derived from chunk shape, declared
level transform, and the render's explicit pixels-per-base-voxel scale. This
collapses planar duplicates while retaining spatially separate appearances of
one chunk on folded surfaces.

The viewer owns the latest pointer focus and captures it in each accepted render
job. The render worker computes nearest per-chunk distances from its local
occurrence list, then atomically publishes the complete demand snapshot and
re-sorts pending work. Mouse and Agent Bridge canvas movement only atomically
marks the active view; it never walks demand or scheduler entries. Pending GUI
work is ordered by relative coarse level, active view, captured pointer distance
(or viewport center before pointer activity), then FIFO. Chunks missed by the
probe still enter the GUI lane without a location and sort last within their
view and level.

Whole-view closure clears that view across the shared service. Different-source
overlay disable/replacement instead clears only the overlay handle's bound
source and current render version, preserving base and other-view ownership.
Queued overlay callbacks verify attachment/opacity generation before requesting
another render. Same-source base and overlay demand remains merged and a
base-only render replaces it normally.

Plane and generated surface coordinates are declared in level-0/base-volume
voxel units, so camera scale is framebuffer pixels per base voxel in every
slice viewer. Each base or overlay source analytically selects one source Zarr
level for the complete render from that scale and its own level transforms.
Fallback bounds use the same declared scale; they never estimate scale from
generated coordinates or cache contents. `QuadSurface::scale()` only maps its
point grid to base-voxel surface coordinates and is not a second LOD.

GUI and non-GUI callers use separate pending lanes in three shared scheduler
stages:

1. A 32-worker local probe stage classifies persistent encoded data, persistent
   empty markers, and cache misses using filesystem metadata only.
2. A source-read stage performs remote downloads, or direct source reads when
   no persistent cache is configured. Its physical capacity and initial
   admission policy are service construction options; only
   `ChunkCacheService::configureFetchConcurrency()` changes admission later.
3. An eight-worker CPU stage reads and decodes persistent hits or decodes
   successful source reads.

The local probe never reads or decodes payloads. Consequently it can classify
the visible working set quickly, send known misses to the network, and send
known hits to CPU decoding without either class blocking discovery of the
other. HTTP chunk absence is established by the chunk `GET`; the pipeline does
not add a separate remote `HEAD` request.

Normal VC3D remote reads share an adaptive 2-64 source admission limit. VC3D
stores the controller's settled limit, long-term bandwidth EMA, and saturated
per-worker capacity model in `VC3D.ini` on clean shutdown. The next run starts
at that limit instead of two. Only reusable capacity data is restored: epoch
samples, probe phase, direction history, and the stability window are reset, so
the controller performs frequent initial up/down probes around the prior limit
before returning to its normal stability-dependent cadence.

Each saturated controller epoch requires at least five active transfer seconds
and one successful completion per admitted worker. Request completion supplies
p90 latency and saturation/failure information; it is not the primary payload
bandwidth clock. Failed, missing, or underfilled-tail requests reset the current
capacity epoch so later work cannot inherit idle time.

All three schedulers are work-conserving and admit one background item after at
most seven consecutive GUI items while both lanes are nonempty. Existing
queued chunks are reprioritized in place when a newer view snapshot references
them, and each handoff recomputes current view-relative priority. View-demand
publication updates all three queues atomically. Running work is allowed to
finish, and this separation does not yet enforce a pyramid-level barrier across
different stages.

The viewport probe reuses the render's geometry path. Direct surface rendering
generates the full coordinate/normal matrices once and uses them for both the
probe and pixel sampling. Fully SurfaceCache-backed rendering probes the shared
`SurfaceGeometryTileCache`; the following tile fills consume the same geometry
tiles. This keeps the probe from introducing a competing surface-coordinate
cache.

`SurfaceCache` level keys name the selected source Zarr level. Its surface-grid
sampling stride is derived cache detail, not an independently selected surface
LOD.

Persistent Zarr cache directory selection and file naming are unchanged and
remain separate from in-memory source identity. Surface image and geometry
caches also remain separate derived caches, but their raw input chunks come
from the regular service. The former Spiral plane regular-cache setting was
removed because Spiral no longer owns a competing decoded pool.
