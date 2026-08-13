# Task log

## Findings

- Source fetch scheduling currently allocates a fixed worker pool selected by
  `maxConcurrentReads`; probe and decode work already use separate pools.
- The status bandwidth currently sums completed bytes over a fixed three-second
  wall-clock window. That estimate is not tied to chunk count or current
  parallelism.
- Encoded chunk bytes and source-fetch start/completion timestamps are available
  at the source-stage boundary without changing fetcher or decoder behavior.
- A scheduler admission limit can adapt concurrency without rebuilding worker
  pools or changing pending-task order.

## Accepted limitation

- The controller performs no exploratory increase. If two downloads cannot
  expose available bandwidth, the estimate may remain conservative.

## Implementation notes

- `ChunkRequestScheduler` now supports a dynamic admission limit independently
  of its allocated worker count. Priority queues and selection ordering are
  unchanged.
- The shared adaptive source scheduler has 64 workers, starts with two admitted
  transfers, and retains at most 256 successful encoded-transfer samples.
- Both adaptive and fixed source schedulers compute bandwidth over their latest
  `admission_limit * 4` successful chunks. Fixed schedulers report that estimate
  but never change admission.
- Normal remote `Volume::sharedChunkCache()` enables adaptation. Explicit
  `Volume::createChunkCache(options)` jobs use private fixed source schedulers
  while retaining the shared decoded-byte budget.
- The existing status formatter receives the scheduler estimate through
  `ChunkCache::Stats`; its displayed `Nx` remains actual in-flight fetches.

## Validation

- Built `test_chunk_cache`, `test_chunk_cache_persist`,
  `test_download_queue_stats`, `test_volume_chunk_errors`, and `VC3D`.
- New deterministic tests cover 2 MiB chunks at 2, 4, and 100 MiB/s, the
  two-worker minimum, 64-worker maximum, actual admission, and fixed mode.
- Three consecutive focused CTest runs passed before upper-clamp coverage was
  added; the rebuilt complete cache suite then passed twice.
- During repeated broad runs, the pre-existing
  `ChunkCache reprioritizes pending decode work by view-relative level` exact
  order assertion failed twice and passed on immediate reruns. The adaptive
  tests execute earlier and passed; no production change was made for that
  unrelated timing-sensitive test.
