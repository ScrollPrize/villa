# Integrating immutable stores with upstream numbered versions

This is the implementation contract and progress record for integrating the immutable/sharded-store branch with current upstream. It supersedes conflicting versioning and conversion details in the earlier v2 documents. Paragraphs are not manually line-wrapped.

## Scope and identities

- Use the next upstream SQLite schema version (21 if upstream remains at 20), not an independent “vcobj v2” schema family. An ordinary store remains an ordinary SQLite file. Optional packed-container framing has its own physical format version, independent of the logical SQLite schema.
- Retain upstream functionality, including checked transactions, numbered object and relationship versions, lineage/generation UUIDs, receipts, hosted clients/server, discovery operations, and history queries where history is retained. Do not restore the obsolete hub/replica implementation that upstream removed.
- Retain our history-off policy, sealing, direct-key store_references and overlays, independent sharding, binary UUIDs/hashes, direct geometry descriptors, optional spatial indexing, packed VFS, and fast shared-stream batch import. Retain the default of 32 import workers.
- The object UUID is the complete logical key, including across shards and referenced stores. Relationships store endpoint UUIDs directly; no proxy objects, aliases, per-object routing directories, endpoint directories, or duplicated endpoint rows. Fixed key-routing metadata determines the responsible database.
- UUIDs use one 16-byte binary storage format. Chunk hashes remain binary content identifiers, not version counters.

## One commit clock and one record version

- Rename our numeric store generation to `commitSequence`/`commit_sequence`. This is the sole store-wide logical change counter and is independent of history retention.
- Retain upstream's generation UUID as a lineage identifier, not a number. Store identity, lineage identity, object identity, object version, and schema version have distinct meanings.
- New objects and relationships start at version 1. A changed record advances from its current version once per transaction, even if several setters modify it. No-op updates do not advance it. Deletion retains the last version in a tombstone so recreating the same UUID cannot reuse an old state identifier.
- Advance the store clock once for a committed logical change. Rollbacks, rolled-back savepoints, and no-op transactions do not advance it. Changes to store_references, sealing and logical provisioning count; physical repacking and cache maintenance alone do not.
- Preserve the dependency ordering rule: a derived store's commit sequence is greater than every sealed referenced dependency. Allocate `max(current sequence, referenced dependency sequences) + 1` when required. Sequence gaps are permitted; change feeds must handle them. Unrelated stores are not ordered by their sequence values alone.
- Current record versions must be stored/read without joining retained history. History-on archives use these same versions and commit sequences, not another incrementing counter. History-off does not create archival version rows merely to provide current-state freshness checks.
- Remove the old per-record numeric `generation` and `data_generation` fields and their duplicate bookkeeping. Update approvals, geometry repair guards, serializers, Python bindings and VC3D consumers to use the appropriate record version or store-state token. Do not merely rename old numeric values: legacy mutation counts are not necessarily numbered versions.
- History removal never resets current versions or the commit sequence. Enabling history archives the current state under its existing identities. The redesigned lazy/delta history scheme remains deferred.

## Geometry is ordinary object state

The object's metadata/current descriptor identifies its geometry chunks. Changing the descriptor or chunk references changes the object and its ordinary version; history records that object-state change. There is no separate geometry-change counter. Current geometry stays directly on the object, not in extra current-geometry or current-payload tables. History-on may retain existing archival representations until the later history redesign.

Geometry caches may conservatively invalidate on any object-version change. Existing content hashes/descriptors can identify identical payloads without introducing another counter. Geometry relationships must still be checked against the relevant geometry/anchors: changing an unrelated metadata field must not imply that the relationship's geometry is wrong. In-transaction read-your-writes and view invalidation must remain correct even when several edits share one eventual committed version.

Approval metadata belongs to the version it approves. Migrate legacy approval-generation stamps to the numbered-version model, preserving approval provenance and invalidation behavior; changes within the same transaction must not accidentally approve later unreviewed edits.

## Store references, sharding and history-off

- Store references (`store_reference`) target sealed stores and pin `(store UUID, lineage UUID, commitSequence)`. A mismatch fails explicitly. An immutable base can be shadowed by an ordinary same-UUID object/relationship in the writable overlay; deletion uses the existing state field, not a separate tombstone table.
- Retain `state` with Active/Retired/Deleted for objects and Active/Deleted for relationships. Retain location search APIs and fixed database/table routing metadata. All data tables follow the shard ownership plan; provisioning metadata remains small and store-local.
- Retain the pre-integration lifecycle simplification: retiring an object tombstones its current incident relationships atomically; restoring does not resurrect them. History-on retains earlier observations through historical selections. Do not reintroduce per-edge lifecycle-binding tables or claim surviving current links need a separate lifecycle epoch check.
- Independent shard writers construct one unpublished logical import state with known versions and a shared lineage/commit token. They directly write known UUID references without cross-shard joins or validation reads. Finalization must not reopen all shards to rediscover information the importer already owns.
- History-off still supports current reads, checked writes, versions and change detection. Historical retrieval/change replay unavailable without retained records must fail or explicitly invalidate the caller's cursor; never fabricate old states or a complete feed. No history-only tables, triggers or per-object history rows may be required in slim imports.
- Keep the fast importer isolated from interactive edit bookkeeping, with bounded queues, independent SQLite writers and RAM-first packed output. No regressions to generated control points, JSON payload dumping, per-record UUID readbacks, per-edge endpoint metadata, or duplicate geometry mappings.

## Conversion policy

- Support only upstream formats as migration inputs. Our unpublished experimental schema-18/v2 layout is discarded and has no compatibility promise or migration path. Do not retain its converter.
- Convert upstream schema 20 to the new schema through a separate offline converter. Preserve upstream's existing supported schema-16–19 conversions and restrictions, chaining their result into the new conversion where appropriate. Reject unsupported layouts rather than guessing from a version number.
- Preserve store and lineage UUIDs, numbered versions, historical references, timestamps, payloads, current approvals and supported artifacts. Convert legacy serialized records to the new fields without inventing chronology.
- The normal runtime opens only the supported current schema. It must not contain branches that read old record layouts or automatically modify old stores during open.
- Conversion creates a new file, validates completeness and consistency, and leaves the source untouched. Explicit in-place replacement, if requested, uses a validated sibling staging file and durable atomic replacement. Partial failures must leave the source recoverable.

## Implementation sequence

1. Record this contract, fetch upstream, and preserve the pre-integration branch state in Git.
2. Integrate upstream additions without reintroducing superseded hub/replica code or undoing the direct-key/slim-store changes.
3. Implement schema 21/current version and commit tracking, history-off behavior, geometry descriptors, approvals and store_reference state tokens.
4. Reconcile the offline upstream converters; remove the unpublished experimental-format conversion path.
5. Update native/Python/hosted/VC3D consumers, documentation and tests.
6. Run core, adapter, Python, conversion and import regression tests; benchmark matched prepared and AFV inputs with the existing fixed runner. Each import benchmark is limited to 30 seconds; estimate full-volume time instead of launching long imports.

## Verification checklist

- [x] Fresh upstream fetched and integration recorded.
- [x] Single current schema, ordinary SQLite and optional packed stores open correctly.
- [x] Versions/commit sequence: create, repeated edits, approvals, no-ops, rollback/savepoints, delete/recreate, history toggles and lineage changes.
- [x] History-on round trips retain upstream numbered references and historical contents; history-off creates no history-only storage.
- [x] Geometry/metadata edits share object versioning; cache invalidation and relationship checks remain correct.
- [x] Sealed dependency pins include lineage and commit sequence; overlays, tombstones, routing and sequence gaps work.
- [x] No duplicate per-record generation/data-generation fields, endpoint directories or alias rows remain.
- [x] Upstream 20 and supported earlier converter fixtures preserve current and historical data; unpublished experimental layouts are not accepted.
- [x] Hosted APIs, Python bindings and VC3D consumers use the reconciled identities.
- [x] Core/adapters/Python tests pass; prepared and AFV import correctness and throughput are measured.
- [x] Format/API documentation describes the implemented schema, with old v2 terminology removed or explicitly historical.

## Progress

- Contract committed before integration as `e43271649`. Pre-integration branch head: `ef3442aeb`; fetched upstream head: `2144fdd3f` (schema 20).
- Schema 21, ordinary record versions, direct geometry descriptors, store_reference lineage/commit pins and explicit upstream conversion are implemented. Final validation: 49 native tests pass (9.25 s; the external-credential SSH test is skipped), and all 126 Python tests pass (8.62 s), including hosted-server tests and importer readback. No deferred delta-history redesign was implemented.
- History-off request retry receipts are operational state, not retained object history: create their existing standalone receipt table lazily only when an explicit request UUID is submitted. Slim initial imports never need this table. History-policy changes preserve already-issued receipts.
- Conversion preserves logical identities, so hosted metadata-cache keys include the schema-22 wire namespace. Existing raw geometry/artifact blocks remain reusable. Geometry-only and metadata-plus-geometry edits preserve explicitly recorded historical relationship pins; an unpublished relationship cannot claim a historical state that never existed.

### Repeating validation

The local argument-free entrypoint is `bash local/scripts/run_vcobj_tests`; its ignored local configs select the native, Python and bounded benchmark phases. The shareable native entrypoint remains `bash vc_object_store/scripts/run_cpp_tests.sh`. For a Python-enabled adapter build, run `ctest --test-dir <build> --output-on-failure` and `VC_OBJECT_STORE_BUILD_PYTHON_PATH=<build>/python VC_OBJECT_SERVER=<build>/bin/vc_object_server python -m pytest vc_object_store/tests/python -q`. Configure through the build instructions in the repository README; no dependencies are installed by these tests.

Validation here was on Linux amd64, not macOS/arm64. VC3D storage-field consumers were updated and audited, but the complete Qt application was not rebuilt in this verification. Preserving a historical link after an edit is intentionally unavailable with history off: callers must explicitly remove/revalidate it or enable history/use a history-enabled overlay; the UI reports that restriction rather than silently retaining stale evidence.

## Matched import check

The integration was checked with 50,000 source fibers (15,486,987 XYZ points), one volume descriptor and 300,000 constraints. Both paths use the same native importer, 32 readers/writers for parallel cases, 512 MiB SQLite caches, a 4 GiB RAM-first packed budget and one final exporter to tmpfs. Build: GCC 13.3, RelWithDebInfo, SQLite 3.45.1, Linux amd64. Preparation is excluded from prepared-input ingestion; AFV ingestion includes parallel source decoding. Persistence is measured separately. Readback is outside these timings and covered by the native/Python regression tests. Each benchmark invocation has a 30-second hard limit.

| Path | Workers | Before ingestion (s, one trial) | After ingestion median (s) | After mean / min–max (s) | After persistence median (s) |
|---|---:|---:|---:|---:|---:|
| Prepared native records | 1 | 4.122 | 4.252 | 4.252 / 4.252–4.252 | 0.394 |
| Prepared native records | 32 | 0.219 | 0.229 | 0.255 / 0.215–0.321 | 0.315 |
| AFV source streaming | 32 | 0.924 | 0.918 | 0.910 / 0.892–0.919 | 0.330 |

There are three after-trials for each 32-worker path and one for the sequential control. Prepared ingestion scales 18.6× from one worker to 32 at the median. These short checks show no material throughput regression; they are not measurements of full-volume runtime or sustained physical-disk write speed. The source has unequal fiber/point/constraint sampling fractions, so multiplying total time by one sampling factor would give an invalid full-volume estimate. Default parallelism remains 32.
