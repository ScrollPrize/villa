# Task: reconfigure global RAM cache capacity in place

Make decoded RAM capacity exclusively owned by `ChunkCacheService` and its
shared decoded-budget manager.

- Runtime cache-size changes must preserve sources, source IDs, fetchers,
  queued work, running work, and decoded entries that remain within budget.
- Reducing capacity may evict decoded LRU entries, but must not cancel or
  restart probe, source-read, or decode work.
- Remove the redundant per-source decoded-byte ceiling.
- `Volume::setCacheBudget()` must configure the attached service in place and
  must not invalidate or reacquire the source.
- Existing service-wide concurrency changes must retain their current
  queue/source-preserving behavior.
