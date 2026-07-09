# Current Task

Update 2D fiber-strip prefetch for parallel, transparent chunk fetching.

- Add a VC3D coordinate-dependency function that returns chunk information
  without sampling image values or performing blocking downloads.
- Return all required chunks for a coordinate envelope, including enough
  information for Python to derive remote chunk URLs and persistent cache paths.
- Keep local path checking, deduplication, downloads, retry handling, and
  missing-chunk marker writes in Python.
- Run sample coordinate generation and chunk dependency mapping in parallel with
  the download loop, using several dependency workers and up to 16 download
  workers.
- Retry transient connection/download failures for up to 10 minutes.
- Do not repeatedly download chunks that are known missing. Missing chunks use
  the VC3D persistent cache convention:
  `<cache>/level_<level>/<iz>/<iy>/<ix>.empty`.
- Remove temporary debug-profiling output from prefetch.
- Prefetch progress should track:
  - samples processed for chunk information;
  - chunks queued for download and chunks already downloaded;
  - cache hits;
  - known-missing chunks;
  - errors;
  - ETA based on download rate and observed hit/missing ratio while dependency
    generation is still running.
