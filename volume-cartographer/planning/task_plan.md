# Direct Zarr mirror disk-cache plan

## Scope and invariants

- Preserve rendering values, decoded bytes, interactive priority, adaptive
  concurrency, and persistent-budget guarantees.
- New-format remote caches support one persistence representation only: an
  incomplete byte-for-byte Zarr mirror plus adjacent `.empty` markers.
- Existing legacy caches remain readable and writable in their legacy layout;
  they are never migrated or mixed with mirror paths automatically.
- Treat a logical decode chunk and a physical Zarr storage object as separate
  identities. An unsharded object covers one logical chunk; a shard can cover
  many logical inner chunks.
- Never persist a shard byte range or extracted inner payload as though it were
  a native Zarr storage object.
- Keep general C3D/VCZ1 codec support. Remove only decoded remote-cache
  recompression production and controls.

## 1. Cache-layout selection

1. Add immutable `PersistentCacheLayout::{Legacy,ZarrMirror}` policy to source
   options/state and compare it during shared-source compatibility checks.
2. Detect legacy only from the complete legacy hierarchy
   `level_<n>/<z>/<y>/<x>.<bin|zst|c3d|source|empty>` or an existing legacy
   prefill marker. A `.empty` file or source group named `level_N` alone is not
   a legacy footprint.
3. Existing valid Zarr metadata selects mirror. Missing/empty cache paths
   default to mirror. If a nonempty path matches neither format, fail loudly.
4. Persist layout bookkeeping outside the mirrored Zarr directory. Legacy wins
   only when its complete footprint is present, preserving old cache identity
   without placing private markers into a native mirror.

## 2. Explicit storage-object model

1. Extend `IChunkFetcher` with a storage-object descriptor for each logical
   `ChunkKey`:
   - physical source-relative object key;
   - fixed-width outer storage coordinates and logical level;
   - inner coordinates within the object;
   - storage-object grid/coverage needed by prefill and reverse lookup.
2. For unsharded arrays, outer and logical coordinates are identical. For
   sharded arrays, derive outer coordinates by dividing by the declared
   inner-chunks-per-shard and retain the inner offset separately.
3. Add shared `utils::ZarrArray` helpers rather than duplicate shard parsing:
   - map logical indices to the physical storage-object key;
   - fetch a complete encoded storage object;
   - extract/decode one logical inner chunk from complete object bytes;
   - parse a source-relative storage key back to outer coordinates.
4. Retain physical array paths across logical base-scale rebasing explicitly.
   All mirror, prefill, redownload, and empty-marker paths use the physical key.

## 3. Storage-object queue state and deduplication

1. Keep decoded RAM entries keyed by logical `ChunkKey` as today. Add
   source-local transient storage-object state keyed by
   `(logical level, outer z, outer y, outer x)`; exact source paths remain cold
   metadata, not hot render keys.
2. A storage-object state owns at most one pending/running probe, disk read,
   source download, and atomic write. It retains a set of logical decode
   consumers plus optional maintenance ownership.
3. Multiple logical requests mapping to one shard join the same state. Joining
   must never issue duplicate stat/read/download/write work.
4. Pending storage work receives the best current priority among its logical
   consumers. Adding, replacing, or closing view demand recomputes that best
   priority and reprioritizes the current stage. Maintenance is lowest priority
   and cannot lower interactive/background ownership.
5. If all logical consumers become stale and no maintenance owner remains,
   cancel pending storage stages. Running work drains as elsewhere in the cache.
6. On disk hit or completed download, hold one immutable shared storage payload
   and fan out independently prioritized decode tasks for every still-current
   logical consumer. Release it after writers/decoders drop their references.
7. A later request after transient state retirement may reread the persisted
   object, but concurrent requests for the same object remain fully coalesced.

## 4. Stage transitions and notifications

1. Logical request -> storage descriptor -> shared persistent probe.
2. Probe outcomes:
   - exact object exists: queue one disk read;
   - adjacent `.empty` exists: resolve all covered logical consumers missing;
   - neither exists: join/queue one physical source-object download.
3. A source download fetches the complete object, including an entire shard.
   Network byte accounting and adaptive admission measure the physical transfer.
4. After successful transport, atomically write the exact bytes to the original
   Zarr key. Decode from the same immutable bytes after write completion rather
   than rereading disk. If persistence is refused/fails, rendering may still
   decode the downloaded bytes and reports the persistence failure to
   maintenance callers.
5. Download activity/status counts physical storage-object transfers. The
   debug overlay remains logical: on transfer start notify every requested
   logical consumer mapped to that object; a consumer joining an in-flight
   transfer receives a start event immediately. On network completion or
   consumer removal, emit the matching stop event exactly once.
6. Logical chunk-ready callbacks fire independently after each inner decode is
   published. Completing a shard download does not claim every covered inner
   chunk is decoded or resident.
7. Stage handoff validates source/cache/fetcher generations for both the object
   and each consumer. A stale consumer cannot publish, while other consumers of
   the same object continue normally.

## 5. Mirror paths, metadata, and missing data

1. Mirror data goes to `<mirror-root>/<physical-source-object-key>` with no
   extension or payload transformation. Successful writes remove a stale
   adjacent `.empty` marker.
2. A missing unsharded object writes `<chunk-key>.empty`. A missing whole shard
   writes `<shard-key>.empty`. Missing inner chunks inside a present shard are
   represented only by the shard index and do not receive sidecars.
3. Explicitly collect required metadata while initially opening the remote
   pyramid, then atomically publish it after the final existing cache identity
   and layout are known. Fetch root and discovered-array `.zgroup`, `.zattrs`,
   `.zarray`, or `zarr.json` as applicable; do not rely on incidental probes.
4. Validate every store-relative key before filesystem joining. Reject absolute
   and traversing paths.
5. Structural metadata is protected from eviction. It uses free-space-aware
   write reservation but is excluded from the evictable chunk-capacity total.
6. Extend local multiscale discovery to honor root multiscale dataset paths so
   a valid mirror is reopenable even when array paths are not numeric.

## 6. Legacy compatibility

1. Retain legacy logical-chunk probing and precedence:
   `.source`, valid recompressed `.zst`, primary `.bin`/`.c3d`/source `.zst`,
   then `.empty`.
2. Retain legacy writing for legacy-selected paths. Generic decoded chunks may
   be written as `.bin`; source-native representations and maintenance
   `.source` keep their current handling.
3. Remove production creation of recompressed decoded `.zst` entries. Keep old
   `.zst` decoding and corrupt-`.zst` fallback to `.bin`.
4. Keep deprecated cache compression option fields source-compatible if needed,
   but make them ignored for writes and document that no new recompression is
   performed. Do not remove general compression codec APIs.

## 7. Prefill, redownload, and bookkeeping

1. Prefill enumerates the physical storage-object grid, not the logical inner
   grid. Each shard is downloaded at most once.
2. Redownload scans represented exact source objects and refreshes each physical
   object once. Fetcher reverse mapping validates mirror object paths and maps
   them to storage descriptors without guessing v2/v3 separators or rebasing.
3. Legacy redownload retains its existing logical-path scanner and behavior.
4. Move mirror prefill completion markers to application bookkeeping outside
   the mirror. Continue recognizing old in-cache legacy markers.
5. Maintenance ownership shares probe/read/download/write states with normal
   demand, remains lowest priority, performs no decoded-RAM insertion, and gets
   completion only after the selected representation is durably published.

## 8. VC3D recompression removal

1. Remove the remote-cache compression checkbox, quantization selector,
   "Compress existing cache" action, compression-worker control, settings
   keys, and startup application of recompression defaults.
2. Remove cache-writer calls to `cacheCompress`; legacy raw writes remain raw.
3. Keep the redownload action with layout-specific discovery described above.
4. Update disk-cache wording to describe exact remote Zarr payloads rather than
   compressed decoded bytes.

## 9. Persistent budget

1. Register/derive concrete mirror array roots and chunk-key encodings from
   parsed metadata. Count/evict only validated data-object paths and `.empty`
   markers; unknown files remain untracked and non-evictable.
2. Keep legacy accounting unchanged.
3. Reserve exact object writes against their stale data/empty counterpart.
   Metadata uses protected free-space reservation and cannot become an eviction
   victim.
4. Reading one shard pins that physical file once while all inner decoders hold
   the shared payload.

## 10. Tests

- Layout detection: new/empty and native metadata -> mirror; complete legacy
  footprints -> legacy; incidental `level_N`/`.empty` does not misclassify;
  unknown nonempty and conflicting layouts fail or select documented legacy.
- Metadata is byte-identical at exact paths, traversal is rejected, metadata is
  protected from eviction, and mirrors with numeric/nonnumeric datasets reopen.
- Unsharded ordinary fetch writes exact encoded bytes at the source key; reopen
  decodes without a source request.
- Sharded tests prove multiple inner logical requests issue one complete-object
  fetch/write/read, decode the correct independent inner chunks, and emit
  balanced logical activity/ready notifications.
- Missing whole shard creates one outer `.empty`; missing inner entries in a
  present shard create no sidecar and return fill/missing correctly.
- A joining higher-priority logical request promotes the shared object stage;
  stale consumers cannot cancel or corrupt remaining consumers.
- Base-scale tests cover physical paths in reads, prefill, redownload, markers,
  and budget entries.
- Prefill/redownload operate once per physical object and maintenance-only work
  does not insert decoded RAM.
- Legacy mixed formats retain current precedence, corrupt fallback, and writes.
- No production remote-cache path creates recompressed decoded `.zst` entries.
- Budget scanning counts only validated mirror data/markers and never metadata
  or unknown files.
- Build affected core/VC3D/Python targets, run focused cache/Zarr/volume/UI
  tests, then run the configured volume-cartographer CTest suite.

## Spec update

- Add the logical-decode versus physical-storage-object distinction and exact
  queue mapping/fanout contract.
- Specify immutable layout detection and legacy precedence.
- Specify new caches as incomplete native Zarr mirrors, including full-shard
  downloads and whole-object `.empty` semantics.
- Specify physical transfer versus logical overlay/ready notifications.
- Remove remote-cache recompression production/settings while retaining legacy
  compressed-entry decoding.
- Document metadata protection and storage-object budget accounting.

## Docs updates

- Update `docs/remote_file_cache.md` with both layouts, detection, exact paths,
  sharded storage-object behavior, notification semantics, `.empty`, prefill,
  redownload, and legacy support.
- Update `docs/api/Volume.md` where persistent remote cache behavior appears.
- Remove user-facing remote-cache recompression documentation.

## Changelog

- Add one entry for exact native Zarr mirror caches by default, complete-shard
  coalescing, automatic legacy compatibility, and removal of cache
  recompression production.
