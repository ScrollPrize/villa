# Task Plan: Parallel Transparent Prefetch

## Scope

Replace the current 2D fiber-strip prefetch implementation with a transparent
two-stage pipeline:

1. Generate deterministic CP-local prefetch envelopes and map them to required
   base-volume chunks.
2. Deduplicate, classify, and download those chunks in Python while respecting
   the VC3D persistent cache layout, including `.empty` markers for known
   missing chunks.

The training, runner, augment-vis, and normal image-loading paths must keep
their existing shared source-strip and coordinate-generation behavior.

## Requirements From The Todo

- Add a VC3D dependency-only binding that does not call blocking image sampling.
- The binding returns all chunk dependencies for explicit coordinate envelopes.
- Python performs cache-path checking and global chunk deduplication.
- Coordinate/dependency generation runs in parallel with chunk fetching.
- Several dependency workers may run concurrently to keep CPU-side geometry and
  chunk mapping busy.
- Downloading runs Python-side with bounded concurrency, capped at 16 workers.
- Transient connection failures are retried until a 10-minute deadline.
- Missing zarr chunks are not repeatedly fetched.
- Missing chunks are recorded using VC3D-compatible persistent-cache markers:
  `<cache>/level_<level>/<iz>/<iy>/<ix>.empty`.
- Temporary prefetch debug/profiling rows are removed.
- Progress reports samples processed, queued chunks, downloads, hits, missing
  chunks, errors, throughput, and ETA.

## Design

### VC3D Binding

- Extend the VC3D Python binding with a dependency/introspection function for
  coordinate sampling, separate from `sample_coords`.
- Reuse the existing `ChunkedPlaneSampler::collectCoordsDependencies(...)`
  logic so chunk coverage matches the production coordinate sampler.
- Return chunk keys as structured dictionaries or tuples:
  `level, iz, iy, ix`.
- Also expose enough cache metadata from `Volume` for Python to derive paths:
  `remote_url`, `remote_cache_path`, chunk shape/grid if needed, and the
  persistent extension used for present chunks.
- Keep this function non-fetching. It must not call
  `ChunkCache::prefetchChunks(...)`, `getChunkBlocking(...)`, or
  `sampleCoordsFineToCoarse(...)`.
- Preserve the existing blocking `sample_coords` image-loading path for actual
  batches.

### Python Chunk Request Model

- Add a small prefetch request type in `sampling.py` or a helper module:
  `store_identity, level, iz, iy, ix, remote_url, cache_path, empty_path`.
- For VC3D requests, use the VC3D persistent cache layout:
  - data path under `<remote_cache_path>/level_<level>/<iz>/<iy>/<ix><ext>`;
  - missing marker under
    `<remote_cache_path>/level_<level>/<iz>/<iy>/<ix>.empty`.
- The prefetcher treats existing data files as hits.
- The prefetcher treats existing `.empty` files as known-missing hits.
- If a download returns a definitive not-found result, write the `.empty`
  marker and count it as known missing.
- If a download has a transient failure, retry until the configured retry
  deadline.

### Remote Downloading

- Implement download logic in Python for the base-volume chunks produced by the
  dependency stage.
- Derive chunk URLs from the opened remote zarr URL and chunk key/path mapping
  used by VC3D.
- Write to a temporary file in the cache directory, then atomically rename.
- Do not overwrite existing data or `.empty` markers discovered after a request
  was queued.
- Keep downloads bounded by `prefetch_workers`, capped at 16.
- Use a retry loop with backoff until `volume_cache_retry_seconds` if configured,
  otherwise use the todo default of 600 seconds for prefetch downloads.
- Distinguish:
  - cache hit;
  - known missing marker;
  - newly downloaded;
  - newly marked missing;
  - transient failure still retrying;
  - final error.

### Parallel Pipeline

- Dependency producers:
  - iterate deterministic sample indices in the same order as training;
  - build the shared CP-local source-strip once per sample;
  - derive all configured strip-z offset envelopes from that source;
  - call the VC3D dependency-only function for each envelope;
  - push chunk requests into a bounded queue.
- Download consumers:
  - deduplicate chunk requests globally by `(store_identity, level, iz, iy, ix)`;
  - perform path checks before queueing network work;
  - download missing real chunks with the bounded worker pool;
  - write `.empty` for definitive missing chunks.
- The pipeline should allow dependency generation to continue while downloads
  are in flight.
- Use deterministic sample indexing only; parallel scheduling must not change
  which samples/chunks are requested.

### Progress Output

- Remove the temporary per-source/per-offset debug timing table.
- Print a concise progress line, refreshed in place when possible:
  - samples_done / samples_total;
  - patches_done / patches_total;
  - unique_chunks_seen;
  - queued_for_download;
  - cache_hits;
  - known_missing;
  - downloaded;
  - download_errors;
  - MiB downloaded;
  - current MiB/s;
  - ETA.
- ETA should account for incomplete dependency generation by estimating final
  chunk count from the observed chunks-per-sample and hit/missing/download
  ratios so far.
- The final summary should print one stable multi-field line suitable for logs.

## Spec Update

Update `planning/specs.md`:

- Replace the current prefetch statement that says VC3D prefetch uses the same
  blocking coordinate sampling/cache path with the new dependency-only prefetch
  semantics.
- State that normal image loading still uses blocking `sample_coords`, while
  prefetch uses dependency-only chunk discovery plus Python-side cache/dl logic.
- State that missing chunks are represented by VC3D-compatible `.empty` marker
  files in the persistent cache.
- State that prefetch is parallelized as dependency producers plus bounded
  download consumers, capped at 16 downloads.
- State that the prefetch path remains base-volume-only and augmentation-envelope
  based.

## Docs Updates

Update `docs/code_structure.md`:

- Document the new VC3D dependency-only binding and Python prefetch pipeline.
- Document the cache file conventions for data chunks and `.empty` missing
  markers.
- Document the prefetch progress counters and retry behavior.
- Remove references that describe prefetch as sampling image values and
  discarding them.

Update `planning/task_log.md` after implementation with:

- Important implementation decisions.
- Any deviations from this plan.
- Validation commands and results.

Update `planning/changelog.md` with a short entry after implementation.

## Testing

- Add or update unit tests with fake/local samplers for:
  - dependency requests are deduplicated globally;
  - existing data cache paths are counted as hits and not downloaded;
  - existing `.empty` paths are counted as known missing and not downloaded;
  - definitive 404/not-found responses write `.empty` markers;
  - transient download failures retry and eventually succeed/fail by deadline;
  - progress summary counters are internally consistent.
- Add a focused VC3D-adapter test if feasible without network:
  - monkeypatch/fake `vc.volume.Volume` to verify dependency-only calls do not
    call `sample_coords`;
  - verify base-coordinate to selected-level scaling is preserved in dependency
    collection.
- Run:
  - `python -m py_compile` for touched Python files;
  - focused `fiber_trace_2d` tests with plugin autoload disabled;
  - training and runner CLI help smoke checks.
- Do not require network access for automated tests.

## Risks And Checks

- The remote chunk URL mapping must exactly match VC3D's zarr fetcher. If the
  binding cannot expose enough information, add the minimal VC3D accessor rather
  than duplicating hidden path logic in Python.
- Python-side downloads must write cache bytes in the same encoded/raw format
  expected by VC3D for that volume. If VC3D stores transformed persistent bytes,
  expose the expected remote/cache path and extension through VC3D rather than
  guessing.
- `.empty` markers must be written only for definitive missing chunks, not for
  transient network errors.
- Parallel dependency workers must not mutate shared loader state in a way that
  changes deterministic sample selection.
