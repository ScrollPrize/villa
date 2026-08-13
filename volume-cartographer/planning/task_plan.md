# Task plan

## Scope and invariants

- Preserve decoded bytes, rendered values, fill detection, error behavior,
  persistent file formats, source identity, and cache budget accounting.
- Keep the existing shared scheduler priority policy for pending work. Every
  stage handoff recomputes priority from current per-view demand.
- Use one application-service-wide 32-worker local probe scheduler. A probe
  performs filesystem classification only and never reads or decodes a chunk.
- Keep remote download concurrency controlled by `maxConcurrentReads`.
- Add one application-service-wide CPU decode scheduler. Decoding is never
  performed by local-probe or production Zarr download workers.
- Preserve compatibility for synthetic/custom `IChunkFetcher`
  implementations through default split-fetch methods; production Zarr
  fetchers provide a genuinely encoded download/decode split.

## Pipeline implementation

1. Add an encoded-fetch/decode boundary to `IChunkFetcher`. The compatibility
   default treats the existing `fetch()` result as already decoded, while the
   Zarr implementation downloads encoded chunk payloads and decodes them only
   when invoked by the decode stage.
2. Replace persistent read-and-decode probing with a lightweight classifier
   that records existing compressed/raw data paths or an empty marker. Preserve
   compressed-cache preference and raw fallback behavior.
3. Add separate probe, fetch, and decode task IDs to each unresolved cache
   entry and separate scheduler references in source state.
4. Route persistent data hits to local read/decode jobs, empty hits directly to
   resolution, and misses directly to remote fetch jobs.
5. Route successful remote fetch payloads to decode jobs; resolve remote
   missing/error results directly. Keep remote activity callbacks and transfer
   rate accounting around network work only.
6. Publish decoded results through the existing single store path and preserve
   writeback behavior for downloaded chunks.
7. Extend reprioritization, atomic publication, invalidation, and exclusive
   view cancellation to all three queues.

## Testing

- Extend `test_chunk_cache` with deterministic stage-separation coverage:
  - a blocked cached decode does not prevent a classified cache miss from
    reaching the download queue;
  - downloaded encoded bytes are decoded on the decode stage;
  - persistent hits never touch the remote fetch path;
  - pending decode work follows current view-relative priority;
  - invalidation cancels pending work in every stage.
- Run:

  ```bash
  cmake --build volume-cartographer/build --target test_chunk_cache test_zarr_chunk_fetcher test_chunked_plane_sampler VC3D -j4
  ctest --test-dir volume-cartographer/build --output-on-failure -R '^(test_chunk_cache|test_zarr_chunk_fetcher|test_chunked_plane_sampler)$'
  git diff --check
  ```

- Run the synthetic rendering benchmark after functional validation. It is
  virtualized and may run without waiting for an otherwise idle host.

## Spec update

- Replace the combined probe/fetch-decode queue description with the explicit
  three-stage pipeline and worker ownership.
- State that local cache classification precedes decode/download admission and
  that all stage handoffs recompute current view-relative priority.

## Docs updates

- Update `docs/remote_file_cache.md` with the local probe, remote transfer, and
  decode responsibilities and concurrency controls.

## Changelog update

- Add a dated entry after implementation and validation.

## Independent plan review

- The plan preserves the specification's asynchronous rendering, shared source
  state, atomic priority publication, non-cancellation of running work, and
  unchanged numerical output.
- The main implementation risk is payload ownership between download and
  decode. Tests must cover encoded persistence and generation checks at every
  handoff so stale payloads cannot publish or consume unbounded retained state.
- No requirement is deferred.
