# Task Plan: Python-Owned Parallel Prefetch Downloads

## Scope

The prefetch path should have one clean split of responsibility:

- VC3D calculates required base-volume chunk dependencies for the configured
  strip coordinate envelopes.
- Python owns everything after that: cache-path checks, deduplication,
  `.empty` marker handling, downloading, retrying, temporary files, atomic
  renames, and progress reporting.

Do not add or keep VC3D code whose purpose is downloading chunks, reading chunk
payloads, or writing persistent-cache files for this prefetch feature.

## Requirements

- Revert accidental VC3D changes unrelated to dependency calculation:
  - remove any new VC3D blocking chunk-fetch/prefetch API;
  - remove any Python binding that exposes VC3D chunk fetching;
  - remove any VC3D persistent-cache write changes made only for Python
    prefetch downloading, except the requested `.empty` marker semantic change.
- Change VC3D's existing `.empty` marker writer to create zero-byte marker
  files. VC3D reads `.empty` markers by existence only, so this should preserve
  read behavior while matching the Python marker convention.
- Keep only the VC3D dependency/chunk-calculation function if it is already
  needed for prefetch.
- `.empty` files are zero-byte empty marker files:
  - no temporary file is needed for `.empty`;
  - create the final marker directly for definitive missing chunks only;
  - never create `.empty` for transient network/decode/cache errors.
- Python must perform the actual data download/cache write:
  - derive each chunk's remote URL and final persistent cache path;
  - download/decode/write the chunk payload in Python as required by the cache
    format VC3D will later read;
  - write payloads to a unique temp file in the same cache directory;
  - atomically rename temp file to the final data path after a complete write;
  - clean up temp files on failed attempts where possible.
- Use up to 16 parallel Python download workers.
- Do not use VC3D `sample_coords`, VC3D image sampling, or VC3D chunk-fetch
  calls as the production download path.
- Remove production prefetch fallbacks that call VC3D `prefetch_chunk`,
  VC3D `read_chunk`, `sample_coords`, or generic `store[...]` indexing for
  remote VC3D chunk fetching.
- Ensure VC3D chunk dependency output provides enough information for Python to
  consume directly:
  - remote zarr chunk key/URL;
  - final persistent cache data path;
  - expected persistent cache extension/format;
  - `.empty` marker path.
- Python must not reconstruct VC3D remote/cache paths for VC3D chunk requests.
- Keep dependency discovery independent of the specific random augmentation
  draw by using the configured maximum augmentation envelope.
- Preserve deterministic sample order and deterministic set of prefetched
  samples/chunks.

## Design

### VC3D Dependency Boundary

- Audit `volume-cartographer` diffs and revert changes that fetch or write
  chunks.
- Keep or extend the dependency-only API so each dependency item includes chunk
  coordinates plus VC3D-owned path/format metadata: remote chunk source/path,
  final persistent cache data path, expected cache format/extension, and
  `.empty` marker path. It must not perform chunk download/fetch or image
  sampling.
- The Python side should treat VC3D dependency output as immutable chunk IDs:
  `(level, iz, iy, ix)` plus VC3D-provided remote/cache path metadata.
- Update VC3D `writePersistentEmpty(...)` so it opens/truncates the marker file
  and writes no bytes.
- Do not add a VC3D `prefetch_chunk` or equivalent download API for this
  feature.

### Python Chunk Requests

- Represent each chunk request with:
  - store identity;
  - zarr level;
  - chunk index `(iz, iy, ix)`;
  - VC3D-provided remote chunk URL or remote store key;
  - VC3D-provided final persistent cache data path including extension;
  - VC3D-provided expected payload/cache format;
  - VC3D-provided `.empty` marker path.
- Deduplicate requests globally by the stable chunk identity.
- Before queueing a download, check:
  - existing data path: count as cache hit;
  - existing `.empty`: count as known missing;
  - otherwise queue one download for that chunk.
- If the request lacks a remote source or exact final cache path/format, fail
  prefetch with a clear error. Do not reconstruct VC3D paths in Python and do
  not fall back to VC3D chunk reads.

### Python Download And Cache Write

- Implement a Python download/write helper with this sequence:
  1. Re-check final data path and `.empty` marker after the chunk reaches a
     worker.
  2. Fetch the remote chunk payload or decoded chunk bytes in the format needed
     by the VC3D persistent cache.
  3. Write to a unique temp path in the final cache directory.
  4. Flush/close the temp file.
  5. Atomically rename with `os.replace(temp_path, final_path)`.
  6. Remove the temp file if the attempt fails before rename.
- Unique temp names should include at least final filename, process id, thread
  id or random token, and an attempt token.
- Use direct final-file creation for zero-byte `.empty` only when the remote
  result is a definitive not-found/missing chunk.
- Retry transient failures until the configured retry deadline, defaulting to
  600 seconds for prefetch.
- For remote VC3D prefetch, the implementation must not call:
  - `request.store.prefetch_chunk`;
  - `request.store.read_chunk`;
  - `request.store[...]`;
  - VC3D `sample_coords`.
- Preserve tests/local fake stores with explicit fake download hooks instead of
  relying on generic store indexing.

### Parallelism

- Dependency workers:
  - generate deterministic sample coordinate envelopes;
  - call dependency-only VC3D chunk calculation;
  - push deduplicated chunk requests toward the download stage.
- Download workers:
  - run Python download/cache writes;
  - cap concurrency at 16;
  - avoid duplicate concurrent writes for the same chunk.
- Parallel scheduling must not affect which samples or chunks are requested.

### Progress Output

- Keep two progress lines or two clearly separated progress sections:
  - sample/dependency progress with its own ETA;
  - download progress with its own ETA.
- Report:
  - samples done/total;
  - patches done/total;
  - unique chunks seen;
  - cache hits;
  - known missing markers;
  - chunks needing download;
  - chunks downloaded;
  - active/download errors;
  - MiB downloaded and MiB/s;
  - separate sample ETA and download ETA.
- Download ETA must account for incomplete dependency generation. While sample
  dependency workers are still running, estimate the final download-needed count
  from observed chunks-per-sample and observed cache-hit / known-missing /
  download-needed ratios, then combine that estimate with the observed download
  completion rate.
- Once dependency generation is complete, download ETA uses the exact remaining
  download-needed count.
- Do not label unresolved queued chunks as downloaded.
- Do not print temporary per-offset profiling rows in normal prefetch mode.

## Spec Update

Update `planning/specs.md` to state:

- Prefetch uses VC3D only for dependency/chunk calculation.
- Python owns download/cache behavior.
- Data chunks are written through unique temp files followed by atomic rename.
- `.empty` markers are direct zero-byte empty marker files and are only created
  for definitive missing chunks.
- VC3D also writes zero-byte `.empty` markers and reads them by existence.
- Up to 16 Python download workers are used.
- VC3D image sampling and chunk fetching are not part of production prefetch
  downloading.
- Remote/cache path metadata must be explicit enough that Python never guesses
  or reconstructs VC3D persistent cache paths/formats.
- Prefetch download ETA is extrapolated from observed download rate and observed
  hit/missing/download-needed ratio while dependency generation is incomplete.

## Docs Updates

Update `docs/code_structure.md` to document:

- The dependency-only VC3D boundary.
- The Python prefetch pipeline and worker roles.
- Cache hit, known-missing, download, retry, and atomic-write behavior.
- Zero-byte `.empty` marker behavior on both Python and VC3D sides.
- The two progress views and what their ETAs mean.

Update planning docs after implementation:

- `planning/status.md`: current checklist and validation status.
- `planning/task_log.md`: implementation decisions, deviations, commands, and
  results.
- `planning/changelog.md`: one concise entry for the prefetch fix.

## Testing

- Add/update unit tests with fake chunk stores/downloaders for:
  - data cache hit skips download;
  - `.empty` marker skips download;
  - definitive missing response creates a zero-byte `.empty` marker directly;
  - transient failure retries and eventually succeeds/fails by deadline;
  - data writes use unique temp files and atomic rename;
  - duplicate requests trigger at most one download;
  - download worker limit is capped at 16;
  - progress counters distinguish chunks needed, downloaded, hits, and missing;
  - ETA estimation uses hit/missing/download-needed ratios before dependency
    generation is complete and exact remaining download count after it is
    complete.
- Add a regression check that the prefetch path does not call VC3D image
  sampling or VC3D chunk fetching.
- Add/adjust a VC3D unit or smoke check, if available, that
  `writePersistentEmpty(...)` creates an existing zero-byte marker accepted by
  `readPersistentEmpty(...)`.
- Run:
  - `python -m py_compile` for touched Python files;
  - focused `fiber_trace_2d` pytest suite with plugin autoload disabled;
  - CLI smoke for prefetch help or a mocked/local prefetch path.
- Do not require network access for automated tests.

## Risks And Checks

- The Python writer must produce exactly the cache file format VC3D expects for
  the current zarr codec using VC3D-provided path/format metadata. If VC3D does
  not provide enough metadata, stop and report the missing metadata instead of
  guessing or reconstructing paths in Python.
- Atomic rename only protects final visibility if the temp file is in the same
  filesystem/directory as the final cache path.
- Zero-byte `.empty` marker creation must be limited to definitive not-found
  cases.
- Reverting VC3D fetch code must not remove dependency-only chunk calculation.
