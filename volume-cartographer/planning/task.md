# Task: make chunk source handles service-owned

Refactor regular chunk caching so every source is opened by a
`ChunkCacheService`, every `ChunkCache` is only a lightweight source-bound
`IChunkedArray` handle, and fetch concurrency is configured once for the whole
service.

- `ChunkCacheService::openSource(...)` creates or reacquires source handles.
- A cache without an supplied service creates and retains a new service; it
  does not construct source-local schedulers.
- Probe, fetch, and decode schedulers and fetch concurrency belong exclusively
  to the service.
- Changing I/O threads updates every source in that service. The most recent
  configuration wins globally.
- Scheduler replacement preserves decoded chunks, source IDs, demand,
  listeners, and cache accounting while safely moving unresolved work.
- Explicit bounded/batch caches remain isolated by using a separate service.
