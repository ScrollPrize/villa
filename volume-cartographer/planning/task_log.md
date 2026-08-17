# Direct Zarr mirror disk-cache task log

## 2026-08-16

- Committed the preceding shared cache-service and persistence work as
  `03fac62b9` before starting this layout change.
- Confirmed legacy caches support per-chunk mixed representations. Legacy
  `.zst` wins over `.bin`, corrupt `.zst` falls back to `.bin`, and `.empty` is
  considered only when no data representation exists.
- Confirmed `ZarrChunkFetcher::sourceChunkKey()` already exposes the physical
  source-relative object key needed by a direct mirror.
- Confirmed remote Zarr metadata is fetched during pyramid opening, before
  normal chunk-cache work, so metadata mirroring must be integrated with the
  remote opener rather than inferred from decoded chunks.
- Independent review found that the first plan incorrectly treated an extracted
  sharded inner payload as a complete source object. The revised plan separates
  logical decode keys from physical storage-object keys and requires full-shard
  download, persistence, read deduplication, and decode fanout.
- The revised plan also makes physical download notifications distinct from
  logical overlay/ready notifications, moves mirror bookkeeping outside the
  Zarr root, requires explicit metadata collection, and makes prefill,
  redownload, and budget accounting storage-object-aware.
- Implemented immutable layout selection: a complete legacy cache footprint
  retains legacy behavior, native/empty cache roots use direct mirrors, and
  ambiguous nonempty roots fail rather than mixing representations.
- Added exact metadata publication and source-relative object persistence with
  validated store keys. Structural metadata is protected from eviction while
  still respecting the disk free-space floor.
- Separated logical decoded chunks from physical storage objects throughout
  probe, persistent-read, source-download, write, and decode stages. Concurrent
  inner-chunk requests now share one complete outer-shard transfer and exact
  write, then decode independently from the shared payload.
- Updated `.empty` handling so whole missing shards receive one outer marker,
  while missing inner entries in present shards do not create false sidecars.
- Made Open Data prefill and redownload enumerate physical storage objects in
  mirror mode and retain the existing logical legacy scanners otherwise.
- Removed VC3D remote-cache recompression controls and production writes while
  retaining mixed legacy `.bin`, `.zst`, `.c3d`, `.source`, and `.empty`
  decoding/writing compatibility.
- Added regression coverage for mirror selection, exact-byte reopen, complete
  shard coalescing and notifications, missing-shard semantics, unsafe metadata
  paths, protected metadata accounting, and legacy selection.
- Final concurrency review corrected physical-transfer activity retirement so
  consumers joining during the post-download mirror write cannot retain stale
  active notifications, and retired fetcher generations cannot decrement a
  newer transfer's in-flight count.
- Native budget discovery now parses v2/v3 array metadata and validates exact
  chunk-key syntax. Unrelated files inside an array directory remain protected
  and untracked rather than becoming eviction candidates.
- Mirror layout selection now validates physical storage-object support before
  admitting any request, so incompatible generic fetchers fail immediately
  instead of leaving logical requests unresolved.
- Validation:
  - `cmake --build volume-cartographer/build -j 8`
  - `ctest --test-dir volume-cartographer/build -j 8 --output-on-failure`
    (`150/150` passed)
  - `git diff --check`
- The first aggregate rebuild exposed a corrupt generated `test_atlas.cpp.o`.
  Forcing that target to rebuild cleared the build-tree artifact; the following
  complete build and test run passed.
- No implementation deviations from the reviewed full-shard mirror plan.
