# Adaptive and channel-selective decoded caching

Both features are opt-in. Fixed-byte, whole-chunk caching remains the default. All behavior lives in C++; Python calls the same implementation.

## Memory budget

```python
from vc_reader.reader.memory_cache import MemoryCache
from vc_reader import ZarrReader

cache = MemoryCache("auto", fraction=0.5, smoothing_seconds=10)
volume = ZarrReader.open(path, memory_cache=cache)
```

C++ uses `vcr::MemoryCache::automatic(0.5, 10)` as `zarr::OpenOptions::memoryCache`. Share this cache between volumes to give them one combined budget.

During active reads/writes, at most once per second, the target is `fraction * (available_memory + this_cache_resident_bytes)`. Including its own resident bytes avoids shrinking the target just because the cache fills. The initial budget uses that target immediately. Later changes use a time-based EMA with a default 10-second time constant: the update fraction is `1-exp(-elapsed/10)`, independent of the number of reads. Set `smoothing_seconds=0` to disable smoothing.

The smoothed budget is capped by `resident_bytes + fraction * available_memory`: a stale high target cannot authorize growth into unavailable memory. This cap can reduce the budget faster than the EMA when headroom disappears. There are no extra hidden pressure thresholds. Independent active caches react to one another's allocations; two equally busy 50% caches tend toward one third each of the otherwise available pool. This is a budget, not an allocation reservation or an exact equilibrium guarantee.

Linux uses `MemAvailable`, capped by cgroup memory headroom including reclaimable inactive file pages. macOS uses a conservative free+inactive-page estimate. There is no swap budget. The fraction defaults to 0.5 only when auto mode is requested. An idle cache refreshes on its next access or explicit `refresh_budget()`; it has no background thread. Referenced buffers and in-flight decoding may outlive LRU eviction, so the budget is not a hard process-RSS limit.

## Channel policy and reads

```python
volume.request_channel_cache([4, 5, 6])
```

C++ uses `array.request_channel_cache({4,5,6})`. Axis zero must be the channel axis, with the existing C-order storage convention. The policy is shared by array identity across live readers in the process. Registrations accumulate; a second reader registering `[1,4]` makes the relevant set `[1,4,5,6]`. Cache entries use channel-aware keys and the existing decoded cache budget, not an additional compressed-byte cache. Requests for channels outside the registered set are still valid and cache the requested channel.

Existing `read_many` combines multiple channel regions into one physical-chunk operation. Point batches may also contain multiple channel coordinates. Nearest, linear, cache-only, full-chunk and region reads retain their value/fill semantics. A full-chunk request explicitly requests every channel. Channel policy is separate from the decoded cache instance: readers wishing to share cached bytes must also share their `MemoryCache`.

Raw data reads only requested channel ranges. Blosc reads its header and block directory, coalesces the required compressed blocks, and decodes each required block once. Relevant channels wholly covered by those blocks are retained without fetching extra blocks. Shuffle, bitshuffle and short final blocks are handled by the existing Blosc decoder. Other codecs decode the full chunk and retain requested and registered relevant channels. Channel planes may span multiple physical chunks.

Local files and ranged HTTP/S3/SFTP sources, including indexed Zarr shards, use the same codec-level selection. Remote range reads validate ranges and use the source ETag when available. Partial remote reads populate decoded RAM only: they never write an incomplete payload into the whole-chunk disk cache. Existing disk-cache chunks support selective reads; whole-source reads still populate the ordinary disk cache. No partial-file disk-cache format is introduced.

`Array::stats()` / Python `stats()` expose `channel_read_bytes` and `channel_decoded_bytes` for the selective path. The latter includes unavoidable complete codec blocks. `remote_bytes` includes actual HTTP payload bytes, including probes. These counters measure application I/O, not physical device traffic or page-cache misses.

## Lock profiling

### Shared chunk-load coordination

Whole-chunk and channel-selective decoded reads use the same per-key in-flight result, across reader instances and local, remote, or computed sources. The key includes the decoded-cache quality namespace. A short-lived registration entry shares the producer's result or exception with concurrent waiters; failures do not poison subsequent retries. Partial results can supply the requested channel planes; another request loads only planes still missing. Resident cache hits bypass registration entirely.

Registration maps use independent hash shards (hardware concurrency by default), like the decoded LRU, but do not share its locks. No registration lock covers source I/O, decoding, cache admission, or waiting on another producer. Different chunks that hash to the same shard only serialize their brief registration/completion steps, not their source work. Cache-only requests return a miss rather than wait on an active source load. Explicit caller-thread-only computed sources retain caller-thread execution. Uncached raw range streaming retains bounded range reads rather than materializing an entire chunk just to share it.

See [coordination benchmarks](../benchmarks/singleflight.md) for reproducible whole/selected read cases and sustained contention tests, including oversubscribed users.

For sampled producer/waiter/queued-task counts on real workloads, see [coordination profiling](coordination-profiling.md).

C++ callers can opt in with `MemoryCache::enable_profiling()` and read `profile()` snapshots. Counters distinguish acquisition wait and time holding shard mutexes for get, put, erase and clear, aggregated across shards. Memory-availability probe time and refresh call counts are separate. The legacy `refresh` lock-timing member is zero in the sharded implementation: pressure trimming across shards is not currently instrumented. Profiling is off by default; enabled timing uses the real steady clock, independently of the injectable budget clock. Counters are updated under the owning mutex without another counter lock. Clock reads and counter updates perturb the measured workload, so compare a profiling-disabled control. Wait times sum across threads: they are not wall-clock percentages. Two readers sharing a cache report the same counters; do not sum those snapshots twice.

## Validation

Build and run `test_native_auto_cache`, `test_native_channel_decode`, and `test_native_channel_cache` with the existing CMake native-test configuration. Python tests include `tests/test_auto_memory_cache.py`. Tests use automatically cleaned fixtures, compare values to whole-chunk reads, and assert actual range-read byte counts and retained cache sizes.

## Sharded decoded-memory LRU

The decoded cache hashes the complete cache key (including volume, chunk and channel) across independent LRU lists and mutexes. The default shard count is the hardware concurrency, with a minimum of one; native callers can pass an explicit shard count to the `MemoryCache` constructor for controlled comparisons. A one-shard cache retains global LRU ordering. Multiple shards use local LRU ordering instead; this may change cache hit rates, never decoded values.

Hits, misses and replacements that preserve the byte count require no shared atomic increment: their counters and bookkeeping stay under the owning shard lock. Byte growth/release uses a single atomic resident-byte counter; the budget is another atomic value. Auto mode still has one budget and one EMA for the entire cache, not one per shard. Unused capacity is available to every shard, including a chunk larger than an equal share of the budget.

Equal-size replacements update the existing entry in place and release the old payload outside the shard lock. Aggregate statistics visit the shards and are not a globally atomic snapshot during concurrent operations. Clear also visits shards, so callers requiring an empty cache must stop concurrent insertions first.

## Background maintenance and caller assistance

Fixed-size and auto-sized caches start four maintenance workers by default; zero-byte fixed caches need none. Each writer can try one sampled batch inline when it crosses the eviction watermark and no other reclaimer is active. If it encounters contention or doesn't reclaim enough, the background workers take over. Selection is based on pressure and contention, not the number of callers. Native callers can construct `MemoryCache(budget_or_options, shards, false)` for synchronous eviction, or enable a custom maintenance policy on that cache before concurrent use. Injected auto-budget callbacks must support access from the maintenance thread. Construct native caches after forking; do not use or destroy inherited caches in the child. Python reader wrappers rebuild their cache on first use in a child process. Destruction stops and joins the maintenance threads.

Shard-local LRU lists record a shared millisecond epoch on access. A maintenance worker advances the epoch periodically; hits do not increment a global access counter. Shards publish changed tail epochs into a contiguous atomic array. One atomic byte claim covers a bounded prefix of entries. Payload releases from several shards are combined up to 64 entries / 16 MiB per batch and performed outside shard locks. Byte reservations stay charged until release completes; external reader references can still keep the actual payload alive.

The default scans eight tails. A positive scan count selects the oldest available tail among that many consecutive shards starting at a per-consumer pseudorandom position. Entries must be at least seven eighths as old as the selected tail when revalidated under lock. The entry limit is `min(64, 1 + 4 * cutoff_age_in_milliseconds)`; current-epoch ties allow only one entry. A sampled cursor stores only its one candidate. Explicitly calling `cache.enable_maintenance(workers, 0)` on a synchronous cache selects the full-directory cohort policy instead. Neither policy guarantees exact global LRU order.

Headroom is dynamic, not a fixed partition. It starts at 1% of the shared budget and responds to offered growth bandwidth, observed time to recover the low watermark, and caller-assistance events. Offered growth includes requests that initially could not reserve memory, rather than only counting successful admissions. Rate smoothing has a 100 ms time constant; headroom grows with a 50 ms time constant and decays over 2 seconds. Bounds are a 0.1%-of-budget / 64-byte floor and a 5% cap, clamped for tiny budgets. The cap is not the normal reservation. Maintenance starts when free bytes fall below 75% of the current headroom and aims to restore the full headroom. Effective headroom is clamped immediately when an auto budget shrinks. Headroom replenishment can continue after producers stop; it is optional preparation for future admissions, not deferred enforcement of the hard budget.

Admission reserves bytes atomically against the shared budget. If a fitting chunk cannot reserve enough bytes, its caller releases its insertion-shard lock, helps reclaim a batch through the same global-recency routine, then retries with a fresh lookup. If another worker already claimed the available eviction work, it yields and retries. A large incoming chunk can request more free bytes than the normal headroom. Fitting chunks are not skipped because maintenance is behind; only chunks larger than the current entire budget remain uncacheable. This fallback adds work only on a reservation failure, but can substantially affect tail latency under sustained overload. No fairness or bounded admission latency is promised under unlimited competing writers.

`maintenance_stats()` reports admission bytes, total eviction bytes/batches, effective headroom, pending reclamation, oversized admission skips, caller-assisted puts/bytes/batches, candidate-array scans, prefix byte claims and evicted entry counts. `wait_for_maintenance()` is an explicit diagnostic drain barrier with a timeout, not part of ordinary reads or writes. Auto-budget shrink is asynchronous in this mode, so existing occupancy can temporarily exceed a newly reduced budget; new growth is still constrained by admission reservations. Statistics are concurrent snapshots, not a transactionally consistent view.

`test_native_cache_maintenance` covers 1/2/4/8 maintenance workers, concurrent admissions, no capacity-based skips, global cold-before-hot eviction, reader-held payload lifetime, large incoming chunks, budget shrink, clear and destruction. Use the contention benchmark's maintenance and pacing arguments to compare foreground latency, accepted work, hit retention, assistance and headroom—not just raw request rate.

