# Store format and immutable policies (schema 21)

The current SQLite schema is 21 and the application ID is `0x56434f42`. There is no independent “v2” logical format family. An ordinary store is a plain SQLite file. Optional [collections](collections.md) compose independent databases, and optional [packed files](packed-stores.md) put them in an immutable VFS container with its own physical format version. SQLite 3.37 or later is required for reliable table versus virtual-table-shadow catalog discovery. The normal runtime accepts only the current schema. Never change `user_version` manually.

The [shared API contract](store-api-contract.md) defines resolution and consumer responsibilities. The lazy/delta history redesign is not part of schema 21.

## Identities and versions

All SQL UUID fields use the same 16-byte SQLite BLOB representation; public APIs and JSON use canonical UUID strings. Object UUIDs are the complete logical keys across local records, shards and referenced stores. SHA-256 payload keys and SQL references are 32-byte BLOBs. Public hashes and JSON descriptor references remain hexadecimal. Neither object keys nor payload hashes are counters.

| Table | Contents |
| --- | --- |
| `store_identity` | Logical store UUID |
| `store_manifest` | Schema version, lineage UUID, commit sequence, retained-history/spatial/sealed policies |
| `store_required_features` | Required capabilities; unknown entries reject opening |
| `database_identity` | Independent physical database UUID and schema version |
| `table_directory` | Binary UUID v4 and SQL name of each application table instance |
| `table_key_columns` | Ordered primary-key column names for each table UUID |
| `extension_tables` (optional) | Typed-extension identity, format version, input-state references and extension-owned provisioning; see [extensions](extensions.md) |
| `store_write_state` | Transactional dirty marker, cleared at commit |
| `key_allocators` | Namespace UUID, reserved bounds and next ID; NULL means exhausted |

Logical store, lineage, physical database and table identities have different meanings. Reopening/sealing preserves them. Independent database copies receive independent physical/table identities. Indexes, views, SQLite-owned tables and virtual-table shadow tables do not receive table UUIDs. A missing declared primary key does not make an implicit rowid a portable persistent key. The directory is populated at creation and refreshed after schema changes and before sealing.

The store's `commitSequence` advances once per committed logical change, including sealing and provisioning. It does not advance on no-op transactions, rollback or rolled-back savepoints. It also remains available without history. The lineage `generation` is a UUID, not a number; unrelated stores are not ordered by their sequences.

Objects and relationships have their own numbered `version`, starting at 1 and advancing once per transaction that changes them, regardless of the number of setters. Current records carry their versions directly, without joining archival history. They also record their publishing lineage and commit sequence. Deletion preserves the last version in a tombstone; recreating the same UUID advances beyond it. Approval stamps use the approved record version. There are no old numeric record `generation` or `data_generation` fields.

Graph snapshots expose these ordinary object versions as C++ `GraphReadView::versions` and Python `graph.versions`, parallel to their object UUID list. The obsolete numeric `generations` graph property is not retained as an alias.

A historical reference is `(store UUID, lineage UUID, object UUID, version)`. Existing upstream archives use these same versions and commit sequences. Geometry descriptors are ordinary object state: changing the descriptor changes the object's version. Chunk hashes identify payload content, not another geometry-change counter.

History's `object_revisions.geometry_uuid` and geometry's `geometry_revisions.parent_uuid` store exact geometry references, not local-only foreign keys. A reference is the same geometry UUID in both history modes. The descriptor may be current object state or a retained archive, locally or in a sealed dependency. The shared resolver validates and reads that record in the transaction's pinned scope. Metadata history retains the original descriptor/reference unchanged; a geometry edit can directly reference its inherited parent without creating a duplicate base revision. This adds no tables, counters or payload-registration records.

## History-off current storage

`OpenOptions::history = HistoryMode::Off` selects a new store's persistent policy. Opens honor the stored policy, not the creation-option default. This does not disable SQLite rollback/WAL transaction safety.

History-off retains typed current objects, relationships, lifecycle state, tags and content-addressed geometry chunks. It omits application journals, archived object/relationship/geometry versions, deltas and history-only triggers. Current version checks, typed transactions, receipts and session change detection still work; historical retrieval cannot invent unavailable states. Explicit request UUIDs lazily create operational `transaction_receipts` for durable retry/idempotency; these are not object-history rows and survive history-policy changes. Initial bulk imports supply no request UUID and omit that table entirely.

| Current storage | Contents |
| --- | --- |
| `objects.geometry_manifest` | Geometry descriptor with ordered chunk hashes and shared metadata references |
| `objects.storage_ref`, `data_checksum` | Current geometry UUID and content checksum |
| `geometry_chunks` | Binary content hash and non-null payload |

There are no separate current-geometry, current-chunk-map, shared-payload-reference or geometry-head tables. Both history policies use the object's current descriptor. History-enabled stores additionally retain the existing archives until the later history redesign.

Each new geometry state receives an ordinary UUID, including repeated writes in one transaction and independent overlay branches. Metadata-only edits retain that UUID; the view's `version()` reports the selected object version. History-off retains only the current descriptor, while history-on also retains previous geometry states. Both use the same descriptor/payload resolver, and stale history-off writer views are rejected once their UUID is replaced. Read snapshots retain SQLite isolation. The existing generic `storage_ref` column holds the canonical UUID text; geometry/archive reference columns use binary UUIDs. An index on current `storage_ref` supports direct UUID lookup without scanning objects.

Edit validation uses transaction-local C++ snapshots and savepoint rollback, not temporary SQL tables. Immutable batch construction does not create interactive-edit snapshots. Superseded payloads are not scanned/reclaimed on every write; explicit reclamation is a separate maintenance concern.

### Lifecycle and relationships

`objects.state` is Active=0, Retired=1 or Deleted=2. `edges.state` permits Active=0 or Deleted=2. Public `retired`/historical deletion indicators are projections, not independent current-state columns. A same-key deletion row blocks fallback to referenced stores; only an absent UUID falls through. An explicit recreate can replace the deletion marker, but does not implicitly revive geometry or relationships.

Retiring an object removes its current incident relationships. Restoring it does not silently recreate them. Current relationships to retired/deleted endpoints are rejected. History-enabled stores retain earlier states; history-off cannot retain a historical-only relationship.

Edges store endpoint UUIDs directly in `a_uuid`/`b_uuid`, identically for local, sibling-shard and referenced-store objects. There are no proxy objects, endpoint directories, external-key aliases, negative handles or per-object routing rows. Native APIs resolve keys through fixed routing metadata when edit validation requires endpoints; batch construction writes supplied keys without cross-shard lookups.

Each edge UUID identifies one observation. Different UUIDs may describe the same endpoint pair and kind; duplicate live UUIDs are rejected. There is no extra relation key or endpoint-pair uniqueness constraint. Relationship tags remain ordinary tags, not additional endpoint objects.

No persistent current edge-geometry/lifecycle bindings duplicate endpoint state. Geometry/dependency changes require an explicit relationship decision before commit. Transaction-local decisions cannot approve a later unreviewed endpoint change. Archived relationships retain exact endpoint versions when history exists. An archived object revision is referenced directly by its binary UUID; otherwise an exact current-only version is encoded as `(store UUID, lineage UUID, object UUID, version)`—three binary 16-byte UUIDs and a little-endian 64-bit version (56 bytes). Both forms use the shared exact-version reader, whether local or referenced; there is no special external-endpoint encoding or relaxed dependency check. A missing version is never silently substituted by a current one.

### History maintenance and sessions

`changeHistoryMode(path, HistoryMode::On)` enables retained history. `changeHistoryMode(path, HistoryMode::Off)` explicitly removes it and disables further retention. The CLI forms are `vc_object_convert --enable-history STORE` and `vc_object_convert --remove-history-and-disable STORE`. Close every other handle first. These offline operations stage and validate a sibling replacement, atomically replacing the original only after success; sealed sources require a new export. Current record versions and commit sequence never reset. Enabling history archives existing current state under its existing identities.

Geometry UUIDs are preserved during history-policy maintenance; reopen handles and reload views afterward. Compatible extensions survive; incompatible dependencies fail conversion without replacing the source.

Current native export/copy, typed snapshots, projects/documents/umbilici and catalog fetching work with either policy. Checked edits always use ordinary numbered versions and typed transaction requests; no separate store-generation-only write API is needed. `StoreStatus.historyEnabled` describes feed availability. A history-off session detects lineage/commit changes and emits `StoreSessionEvent.revalidate=true` with no fabricated journal entries. Clients must re-read cached state. Hosted snapshots similarly validate a stable commit sequence instead of replaying absent history.

Copying a project forks its logical and physical store identities, preserves retained object history and artifacts, and clears operational request receipts: a request issued to the source must not replay as if it had been issued to the new store. The copied publication advances its own commit sequence and starts mutable; the source remains unchanged.

## Optional spatial storage

`OpenOptions::spatialIndex = false` omits object-bounds, spatial-region and R-tree/shadow tables for a new store. Geometry, graph queries and ordinary selection remain available. Existing typed ingestion summaries, such as fiber length/bbox and surface area, are not silently removed.

Spatial selection, region reads and nearest-surface queries fail explicitly when the index is absent, even if another filter would produce no records. On-demand spatial construction is not implemented yet.

## Sealed dependencies, sharding and packed containers

`seal(writeTxn)` persists immutability atomically. After commit, new writable opens and already-open handles cannot write. Merely opening read-only does not seal a store. This is API-level protection, not protection against arbitrary raw SQL or filesystem modification.

Store references (`store_reference`) pin a sealed dependency's store UUID, lineage UUID and commit sequence. A mismatch fails. A derived store's sequence is greater than all pinned dependencies; allocation uses the maximum relevant sequence plus one, so sequence gaps are valid. Physical repacking preserves these logical identities.

A shard import is one unpublished state with a shared lineage and commit token. Independent writers directly create object/edge keys and descriptors. The final coordinator writes known routing metadata, without reopening shards to rediscover or validate records. All data tables follow ownership; metadata remains small and local. No cross-database transaction is claimed.

The optional `uuid-hash-routing-v1` feature maps UUIDs using FNV-1a64 over their 16 canonical bytes, modulo shard count. This requires only per-shard metadata. Ordinary `ObjectStore` uses the same logical-store resolver as the [collection API](collections.md), including direct routing, overlays and dependency precedence. `physicalPartitionOnly` is explicit low-level maintenance access, not the consumer read path.

Geometry hashes resolve through pinned dependencies without local placeholder rows. Missing payloads/resolvers fail, never returning empty substitute data. Unchanged borrowed chunks stay in their original database. Packed containers have separate physical framing and do not change the logical SQLite schema.

## Conversion

Use `vc_object_convert old.vcobj new.vcobj` for supported upstream schemas 16–20. Earlier upstream conversion restrictions still apply; schema 20 is then rebuilt into schema 21. Normal runtime opens never auto-upgrade old tables. The unpublished experimental v2/schema18 layout is not a supported migration input.

Conversion preserves supported store/lineage identities, numbered versions, history, payloads, timestamps, approvals and artifacts. It validates the new output before success and leaves the source untouched. Explicit in-place replacement uses a sibling staging file and durable atomic replacement. Unknown/incompatible layouts or extensions fail rather than silently losing information. See the converter's report for input-specific restrictions.

## Python policies and testing

`ObjectStore.create(..., history_enabled=False, spatial_index=False)` selects creation policies. `store.provisioning()` reports `schema_version`, lineage `generation`, `commit_sequence` and flags. `store.database_identity()` reports physical/table identities. Use `store.seal()` or `txn.seal()`. Module-level `enable_history(path)` and `remove_history_and_disable(path)` invoke native offline maintenance. There is no experimental `convert_v1_to_v2` binding.

Run `bash vc_object_store/scripts/run_cpp_tests.sh`. The argument-free runner writes build/log artifacts under `local/vcobj-tests` by default and owns automatically cleaned fixtures there. `VC_OBJECT_STORE_TEST_ROOT`, `VC_OBJECT_STORE_BUILD_JOBS` and `VC_OBJECT_STORE_TEST_JOBS` configure location/concurrency. Tests install no dependencies.
