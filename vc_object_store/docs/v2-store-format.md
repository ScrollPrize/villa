# V2 store format

The logical store format is version 2, the SQLite application ID is `0x56434f42`, and the SQL schema version is 18. An ordinary v2 store is still a plain SQLite file. Optional [collections](collections.md) compose independent databases; optional [packed files](packed-stores.md) put them in one immutable VFS container. SQLite 3.37 or later is required for reliable table versus virtual-table-shadow catalog discovery. Schema 17 is rejected by normal opens; use the separate validated converter below. Do not change an old database's `user_version` manually. The redesigned lazy/delta history in the [planning document](v2-provisioning-plan.md) is explicitly deferred.

## Persistent policy and identity

| Table | Contents |
| --- | --- |
| `store_identity` | Existing logical store UUID |
| `store_manifest` | Format version, committed generation, history-enabled, spatial-enabled and sealed flags |
| `store_required_features` | Required capabilities; unknown entries cause opening to fail |
| `database_identity` | Independent physical database UUID and SQL schema version |
| `table_directory` | Binary UUID v4 and SQL name of each application table instance |
| `table_key_columns` | Ordered primary-key column names for each table UUID |
| `store_write_state` | Transactional dirty marker, cleared at commit |
| `key_allocators` | Object/edge namespace UUID, reserved bounds and next ID; NULL means exhausted |

The store, physical database and table identities are different. Reopening, sealing or copying the same database preserves them; a new independent database gets new identities. Indexes, views, SQLite-owned tables and virtual-table shadow tables do not get independent table UUIDs. An empty primary-key list means no declared primary key, not that an implicit rowid is a portable persistent key. The directory is populated at creation and refreshed on schema-changing transactions and before sealing. Table registration uses bound catalog parameters rather than interpolated identifiers.

The store generation begins at 1 and increments once per committed logical write, including sealing and provisioning changes. A no-op transaction, rollback or rolled-back savepoint does not increment it. The marker is transactional; SQLite's cumulative change count is not used. Counter overflow is rejected. Objects, relationships, newly installed geometry, approval stamps, journal entries (including deletions), and retained object/relationship revisions use the committing store generation. An unchanged record retains its last-modified generation. Several geometry edits within one transaction share the generation; immutable revision UUIDs and insertion order preserve their distinct ancestry. Journal sequence numbers remain event/replay cursors, not global generations. Converted pre-v2 history retains its original information; a zero history-publication generation denotes a version recorded before this global-generation scheme.

Imports and synchronization publish content at the destination's generation, never copy another database's counter into current records. Author, time, approval provenance, geometry revision UUIDs and payload hashes remain preserved. Canonical synchronization content omits local `generation`/`approved_generation` and, for managed geometry, `data_generation`; equality, conflict detection and acknowledgement hashes compare this content while hub journal versions retain compare-and-set ordering. External-format data generations are still content and are not omitted. Newly transferred historical geometry receives its local installation generation; already-present immutable geometry keeps its original local generation. Reimporting identical metadata or retransferring identical geometry is idempotent despite different local counters. This does not replicate the source's historical publication timeline. A migrated in-flight acknowledgement made with the older hash format can conservatively surface a conflict; it cannot authorize an overwrite of a newer hub version.

## History-off current storage

`OpenOptions::history = HistoryMode::Off` selects a newly created store's persistent history policy. Opening uses the stored policy, regardless of the default creation options. This does not disable SQLite's rollback/WAL transaction safety.

History-off retains ordinary typed objects, relationships, lifecycle bindings, tags and content-addressed binary geometry chunks. It omits the application journal, historical object and relationship revisions, historical geometry revisions and deltas, and history-only triggers. Current geometry uses:

| Table | Contents |
| --- | --- |
| `current_geometry` | One current descriptor and opaque geometry token per object |
| `current_geometry_chunks` | Current descriptor/array/chunk-to-content-hash mapping |
| `current_shared_payloads` | Live chunk references from shared sequence metadata |
| `current_edge_metadata` | Current metadata dependencies required to validate relationships |
| `current_changed_objects`, `current_unused_chunks` | Transaction work queues for validation and reclamation |

Read-only `geometry_revisions`, `geometry_heads` and `geometry_revision_chunks` views expose current descriptors to the existing verified geometry reader. These views do not retain superseded geometry. Edits replace the current descriptor, preserve relationship validation, and reclaim unreferenced current payloads at commit. Historical reads and history-dependent checked writes fail explicitly. History-on keeps the existing history layout; the redesigned delayed/delta history is deferred. History enablement therefore currently writes the existing baseline representation rather than the proposed future lazy baseline.

`changeHistoryMode(path, HistoryMode::On)` enables retained history; `changeHistoryMode(path, HistoryMode::Off)` is the explicit remove-history-and-disable operation. The command-line equivalents are `vc_object_convert --enable-history STORE` and `vc_object_convert --remove-history-and-disable STORE`. These are offline maintenance operations: exclusive access, sibling staging, verified materialization, durable atomic replacement. They reject sealed or hosted/replica stores and history removal that would discard geometry still pinned by an active relationship. They never reset the generation counter. Sealed sources require a new export instead of an in-place policy change.

Materializing a different geometry representation mints a new current geometry token and updates its current relationship/spatial references. Callers must close and reopen their handles and reload these tokens after maintenance; an old immutable token is never reused for changed representation content. Compatible extension tables, views and triggers are preserved. Extensions that depend on removed history cause conversion to fail without replacing the source.

Typed geometry snapshots, current native export/copy, current project/document/umbilicus helpers and catalog fetching also work without history. A current `GeometryView` exposes `objectGeneration()` and an empty `objectRevision()` in history-off mode. Transformed writes validate both the captured geometry token and object generation, plus the current volume transform. Explicit old-version selections and restoring an older catalog publication require retained history.

For checked session writes without historical UUIDs, capture `store.provisioning(txn).generation` in the same read as the inputs and use `ObjectStoreSession::actAtGeneration`. It rejects any intervening store commit before invoking the callback. Ordinary current-record editing helpers do this automatically. Fetchers use `FetchWrite::atGeneration` with the generation observed before downloading; history-on fetchers retain their revision-selection call. History-off session changes emit `StoreSessionEvent::fullInvalidation = true` and the new `storeGeneration`, not fabricated journal entries. Consumers using `touches()` automatically invalidate their caches; consumers inspecting `changes` directly must handle `fullInvalidation` too.

## Relationship observations and geometry payloads

`edges.relation_key` is an immutable optional observation key (`TEXT NOT NULL DEFAULT ''`). The uniqueness index is `(kind,a_id,b_id,a_dependencies,b_dependencies,relation_key)`. Empty keys preserve the existing one-logical-relation rule; distinct keys allow distinct measurements between the same endpoint/dependency pair. Native C++ uses `EdgeRecord::relationKey`; JSON/Python use `relation_key`. Constraint import uses a stable source constraint identity, not invented geometry dependencies or unique edge kinds.

Geometry chunks contain lossless binary payloads addressed by their SHA-256 hash. An explicitly declared collection may instead hold a NULL payload with a `collection_external_chunks` entry identifying an immutable dependency and verified byte length. Such data is read through the same geometry/metadata reader and hash checks; missing mappings/resolvers fail, never become empty bytes. Plain stores cannot insert external placeholders. See [collections](collections.md) for copy-on-write geometry.

## V1 conversion

Run `vc_object_convert STORE` with every other user of the database closed. The tool accepts schema 17 only, obtains exclusive SQLite access, makes a consistent sibling backup, provisions v2, and compares every original table/column/row including binary payload bytes. It runs integrity and foreign-key checks and reopens the result through the v2 API before durable atomic replacement. Failure leaves the original logical contents intact; staging files are cleaned. Existing history, UUIDs, geometry bytes and source provenance are preserved.

Pre-v2 per-record counters remain preserved freshness tokens until a record is next changed; they must not be interpreted as a historical global timeline. Newly added historical publication-generation columns use zero for pre-v2 rows. New writes use the unified store generation. Conversion does not fabricate old chronology.

## Optional spatial storage

`OpenOptions::spatialIndex = false` omits `object_bounds`, spatial-region tables and their R-tree/shadow structures when creating a store. Opening honors the persisted flag. Geometry access, metadata, graph queries and non-spatial object selection remain available. Typed ingestion summaries such as fiber length/bbox and maintained surface area are not silently removed.

Bounding-box selection, spatial-region queries, geometry-region reads and surface proximity queries throw `object store: spatial index is absent from this store`. They do so even when another filter would otherwise produce an empty result. On-demand spatial indexing is not implemented yet.

## Sealing

`seal(writeTxn)` is atomic with the caller's write. After commit, new writable opens and writes through already-open writable handles fail. Ordinary readonly opens continue to work. A readonly handle is not equivalent to a sealed store; sealing is persistent policy. This is API-enforced immutability, not protection against arbitrary raw SQL or filesystem modification. Collections only accept sealed dependencies and check their UUID/generation before use. Packed containers additionally reject physical writes through their VFS.

```cpp
OpenOptions options;
options.author = "importer";
options.history = HistoryMode::Off;
options.spatialIndex = false;
auto store = ObjectStore::create("fibers.vcobj", options);
{
    auto txn = store.write();
    // Use normal typed object, geometry and relationship insertion APIs.
    txn.commit();
}
{
    auto txn = store.write();
    store.seal(txn);
    txn.commit();
}
auto readonly = ObjectStore::open("fibers.vcobj", OpenOptions{.readOnly = true});
auto txn = readonly.read();
auto policy = readonly.provisioning(txn);
auto directory = readonly.databaseIdentity(txn);
```

## Scope boundaries

The C++ `ObjectCollection` interface is the composed-store interface. Ordinary `ObjectStore` opens reject the `collection-v1` feature instead of silently returning only a local shard; `physicalPartitionOnly` is explicit low-level access. SQL virtual-table composition, a collection-wide spatial accelerator, GUI collection editing and the redesigned historical checkpoint/delta layout are not introduced here. Missing spatial support remains an explicit error, not an empty result. Native AFV/AFC import and its separately configured performance workflow are documented in [AFV import](afv-v2-import.md).

## Python policies

The existing Python `ObjectStore.create` accepts keyword-only `history_enabled=False` and `spatial_index=False`. These are creation policies, not open-time overrides. `store.provisioning()` reports persistent policy and generation; `store.database_identity()` reports physical/table identities. `store.seal()` seals in its own transaction, or `txn.seal()` can seal atomically with writes in `with store.write() as txn`. The module-level `enable_history(path)`, `remove_history_and_disable(path)` and `convert_v1_to_v2(path)` call the same offline native maintenance operations; close all handles first. History removal is explicit and destructive, and callers must reload geometry tokens afterward. History and spatial queries retain the native explicit capability errors. Collection composition remains a C++ API.

## Regression workflow

Run `bash vc_object_store/scripts/run_cpp_tests.sh` from the repository. The argument-free runner builds the standalone C++ tests, writes persistent build/log artifacts under `local/vcobj-tests` by default, and creates automatically cleaned temporary fixtures under that root. `VC_OBJECT_STORE_TEST_ROOT`, `VC_OBJECT_STORE_BUILD_JOBS` and `VC_OBJECT_STORE_TEST_JOBS` configure location and concurrency. The SFTP fixture requires permission to run its local server. No dependency installation is performed.
