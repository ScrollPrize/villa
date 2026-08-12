# Task log

## 2026-08-12

- Read the requested workflow, current chunk-cache statistics and scheduling
  paths, slice status rendering, settings persistence, and viewer-manager global
  update path.
- Plan review: the plan covers every task requirement and preserves the spec's
  diagnostic-only boundary. Queue counts are sourced once from cache entries.
- Added mutex-protected per-level unresolved request accounting to `ChunkCache`.
  Duplicate requests and reprioritization retain one count; resolution,
  invalidation, and stale-view cancellation remove it.
- Initially added a per-slice diagnostic setting, then removed that approach
  after confirming VC3D already has a shared cache status bar.
- Added a shared queue formatter to the existing cache-status path. Remote idle
  volumes show `net idle`; active remote downloads append `qK A/B/C` after the
  compact throughput field. Local volumes omit network information.
- Normalized RAM and disk cache usage to fixed GiB units.
- Documented the status fields in `docs/remote_file_cache.md`.
- Validation:
  - `cmake -S volume-cartographer -B volume-cartographer/build`
  - `cmake --build volume-cartographer/build --target test_chunk_cache test_download_queue_stats VC3D -j4`
  - `ctest --test-dir volume-cartographer/build --output-on-failure -R '^(test_chunk_cache|download_queue_stats)$'`
- Result: VC3D built successfully and both focused tests passed.
- Existing Qt deprecation and MOC completeness warnings remained during the
  full build; this task introduced no new build errors.
- No task simplifications, deferred requirements, or implementation deviations.
