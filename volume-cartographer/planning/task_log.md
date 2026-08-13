# Task log

## Planning review

- Current `ChunkCacheService` has a four-worker probe scheduler and download
  schedulers keyed by `maxConcurrentReads`; there is no decode scheduler.
- A persistent probe currently reads and decodes a cache hit before processing
  another probe. A miss is submitted to the download scheduler only after that
  combined operation finishes.
- Production Zarr fetch currently performs HTTP/local source read and decoding
  in one `IChunkFetcher::fetch()` call. The download worker also rechecks the
  persistent cache, coupling local read, remote transfer, and decode again.
- The HTTP Zarr path discovers chunk absence through the data `GET`; no remote
  stat queue is needed for this task.
- The three schedulers already can share the atomic selection gate. Entry task
  tracking, reprioritization, and invalidation must be expanded from two to
  three stages.

## Deviations

- None.

## Implementation

- Added split encoded-transfer and decode hooks to `IChunkFetcher`, with
  compatibility defaults for existing synthetic/custom fetchers.
- Production Zarr source reads now return encoded chunk payloads. Decoding and
  persistence-format selection happen on the independent decode stage.
- Replaced the combined four-worker persistent read/decode pool with a
  32-worker filesystem classification queue and an eight-worker decode queue.
  Download concurrency remains controlled by `maxConcurrentReads`.
- Persistent probes retain compressed-entry preference and raw fallback, but
  defer file reads, cache decompression, and Zarr decoding to decode workers.
- Added per-entry decode task tracking and included the decode scheduler in
  atomic reprioritization, source invalidation, and exclusive-view compaction.
- Remote activity diagnostics continue to bracket source reads only; decode and
  persistent-cache work do not appear as downloads.

## Validation

- `test_chunk_cache`: 50 test cases passed, including blocked cached-decode vs
  remote-admission, encoded download/decode separation, decode priority, and
  decode invalidation fixtures.
- `test_zarr_chunk_fetcher`: 13 test cases passed.
- `test_chunk_cache_persist`: 17 test cases passed, including corrupt
  compressed-cache fallback.
- `test_chunked_plane_sampler`: 16 test cases passed.
- `VC3D` built successfully.
- Synthetic fixture and no-site coordinator tests passed.
- The fresh eight-case Valgrind rendering gate passed. Modeled score ratios to
  reference were serial `full_res=1.029`, `fallback_3=1.032`,
  `mixed_correlated=1.021`, `mixed_shuffled=1.023`; parallel
  `full_res=0.960`, `fallback_3=1.040`, `mixed_correlated=1.038`, and
  `mixed_shuffled=1.029`. All remained below the `1.10x` limit.
