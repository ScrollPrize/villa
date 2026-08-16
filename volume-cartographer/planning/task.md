# Task

Separate persistent-cache population from decoded-data prefetch without
creating a second download system.

- Normal prefetch and blocking sampling must always use the process-global
  decoded chunk cache with no special per-caller cache services.
- Open Data prefill and Settings "Redownload cache" must write exact source
  payloads to the persistent disk cache without decoding or retaining decoded
  chunks.
- Persistence-only work must use the process service's existing source-read
  scheduler, run at maintenance priority behind interactive and ordinary
  background work, and deduplicate source reads with simultaneous decoded
  requests for the same chunk.
- Standalone processes and explicitly low-level batch/test construction may
  continue to own a separate cache service.
