# Draft vcobj v2 provisioning plan

Status: implementation specification, based on `origin/main` at `3fbb3bbad`. See [v2-store-format.md](v2-store-format.md), [collections.md](collections.md) and [packed-stores.md](packed-stores.md) for the implemented schema/APIs. The redesigned history representation and lazy history sections below remain explicitly deferred; they are not claims about current history-on storage. This plan covers the intended infrastructure, not the abandoned experimental importer.

The goal is a versioned logical store manifest that separates object identity and routing from physical storage. The same model should support ordinary SQLite files, independent database shards, linked derived stores, and an optional packed VFS container.

## Implementation scope

Implement v2 provisioning, sealed readonly stores, linking, optional sharding, optional packed VFS storage, and completely history-off stores now. Defer the redesigned historical checkpoint/delta representation and lazy history-on layout described later in this plan. History-on retains its existing representation for this implementation.

History mode is persistent store-wide policy, not an importer flag or a per-handle option. All handles, transactions, importers and edit APIs obey it. History-off stores omit history-only tables, triggers, journals, superseded states and historical payloads; they retain only the descriptors and chunks required for current data. Do not implement history-off by writing history and deleting it afterward. Requests for unavailable historical operations fail explicitly. Current-state geometry, graph operations, transaction safety and generation-based change detection remain supported. SQLite rollback/WAL journals for transaction safety are distinct from application history.

The deferred history redesign must not prevent implementing history-off. Enabling history uses the existing history-on machinery until the follow-up redesign; disabling retained history requires the explicit remove-history-and-disable operation, never a silent open option. Linked sealed stores retain their own immutable policy; the mutable overlay's policy governs its local writes.

V1 compatibility belongs in a separate conversion tool, not parallel legacy provisioning paths in normal v2 operations. Convert into a new sibling file, preserve existing contents and retained history by default, validate completeness and consistency, close and durably finalize the result, then atomically replace the original under exclusive access. A failed conversion leaves the original intact. Plain v2 stores remain ordinary SQLite databases; only packed containers require framing and custom VFS loading.

## Goals and boundaries

- Slim batch import/export: current objects, relationships and geometry without history, with optional spatial indexes and derived accounting omitted.
- Immutable published shards and linked stores. Parallel import uses independent SQLite databases, not concurrent writers to tables in one database.
- Derived graph data and edits in a new store, with deeper dependency chains and optional history.
- One ordinary writable SQLite database per mutable overlay initially. No custom distributed transactions or in-place editing of published shards.
- Both vcobj API routing and optional SQL composition over the same metadata.
- Full-volume fiber importing as the implementation performance workload.

## Current format and versioning

Current main has SQLite application ID `0x56434f42`, schema version **17**, and a store UUID. Opening requires the exact schema version. There are no table UUIDs or shard/container manifests. See [Database.cpp](../src/Database.cpp), [Schema.hpp](../src/internal/Schema.hpp), and [ObjectStoreSession.hpp](../include/vc_object_store/objects/ObjectStoreSession.hpp).

Keep three versions distinct:

| Version | Meaning |
| --- | --- |
| Store format, proposed v2 | Linking, identity, routing, history modes and capabilities |
| SQL schema version | Actual table layouts; advance beyond 17 rather than resetting it |
| Container format version | Optional VFS header and database extent mapping |

An ordinary unsharded SQLite file can be a v2 store. The container is optional. Unknown required features fail explicitly. Missing optional accelerators never mean that their underlying data is empty.

Existing schema-17 stores use the separate validated conversion tool described above; adding metadata alone does not make current readers support new storage semantics.

## Provisioning records

These are proposed logical records; exact SQL names and layouts remain open. Use small structured tables and binary UUIDs. Routing must not require parsing per-object JSON.

| Record | Required information |
| --- | --- |
| Store manifest | Store UUID, current generation, format version, required features, history mode, sealed/mutable state |
| Database directory | Database UUID, SQL schema/profile, external-file or embedded-container location |
| Table directory | Table UUID v4, database UUID, SQL table name, logical role, key definition and schema version |
| Key-space reservations | Key-space UUID, reserved integer ranges and owner; distinct from populated ranges |
| Routing rules | Logical table, routing key, range boundaries, partition rule and destination table UUIDs |
| Linked stores | Target store UUID, expected generation, locator hints and dependency/overlay role; target must be sealed |
| Overlay declarations | Inherited records/properties overridden or deleted, and explicit precedence |
| Accelerator directory | Index kind, coverage, backing table/database, source store UUID and generation, and availability |
| Container directory | Database UUID to logical size and physical extents, or contiguous offset/length |

The container also needs a small bootstrap header identifying its version and the manifest database's location. Finding the manifest must not require first reading the manifest.

## Identities and routing

Distinguish logical ownership from storage location:

- Store UUID: logical ownership.
- Database UUID: an independent SQLite database.
- Table UUID: a particular table instance.
- Object/edge UUID: persistent entity identity.
- Store UUID plus generation: the logical state referenced by a dependency.

Repacking a file must not change its logical identities. A path is a locator, not an identity; resolved dependencies must match their declared identity and state. A collection is represented by a store manifest and its links, rather than requiring a separate identity for every dependency-list file.

## Shared generation scheme

Introduce one persistent store-wide generation, shared by all logical changes in a committed transaction. It is independent of history retention and journal storage. Current main has per-record generations, journal sequences and geometry revisions, but no single store-wide generation covering all changes.

### Transaction rules

- A new independent store starts at generation 1.
- A content-changing transaction commits at `max(current generation, generations of all linked stores) + 1`.
- Creating a store with dependencies starts at `max(generations of linked stores) + 1`.
- Assign the generation within the write transaction and persist it atomically with the changes. Rollbacks do not advance it; no-op transactions do not advance it either.
- Every object, relationship, geometry version and deletion changed by that transaction is stamped with the same generation. Unchanged records keep their last-modified generation.
- Logical changes include dependency declarations and routing/ownership metadata, not just object payloads. Physical repacking or cache maintenance alone does not change the logical generation.
- Do not reset or reuse generations when removing history or journals. Reject counter overflow rather than wrapping.

Generations belong to the store that commits the change. For an inherited object, distinguish its owning identity from the overlay store and generation that published an override. Independent branches need distinct store UUIDs; generations may coincide across unrelated stores.

### Uses

| Use | Generation semantics |
| --- | --- |
| Whole-store change detection | Compare store UUID and current generation |
| Dependency validation | Require the target UUID and generation recorded in the link |
| Object and relationship freshness | Record the generation of the last modifying transaction |
| Geometry change tracking | Stamp geometry versions with the committing generation |
| Incremental synchronization and events | Select retained changes after generation G across all record kinds |
| Change-sets and journals | Every entry in one transaction shares its generation; entry IDs may distinguish entries |
| Accelerator validity | Record the source store UUID and generation covered by the index |
| Future historical snapshots | Resolve the state after a specified committed generation |

Use this scheme instead of separate journal-sequence and geometry-revision progression cursors. Logical geometry version identity must remain stable for references and reconstruction, but need not require an allocated revision UUID and row for every initial object: store/entity/component/generation can identify it. Existing revision-UUID APIs need explicit adaptation or compatibility mapping. Chunk hashes still identify shared content. Schema versions and runtime attachment identities remain separate concepts.

Incremental replay requires retained change records, including deletions. A history/journal-free store still supports change detection but need not support replaying all changes since an arbitrary old generation.

### Immutable dependency ordering

Links are allowed only to sealed immutable stores, not merely handles opened read-only on otherwise mutable databases. Every link records the target's UUID and expected generation. Resolve and verify both; a mismatch is an error, not a request to silently follow a newer state or reconstruct an older one.

The outgoing-link store must have a strictly higher generation than every linked store. Adding a link uses the transaction rule above, jumping above a new dependency's generation when necessary. Linked targets cannot subsequently advance. This gives increasing generations from base to derived states through the whole dependency chain without coordinating independent writers.

For example, a sealed base at 10 can be linked by a derived store starting at 11. Linking that derived store to another sealed dependency at 30 advances its next committed state to at least 31. A further independent child can link it only after it is sealed, and starts above its final generation.

A higher generation orders dependent states; it does not imply that an unrelated store contains another store's changes. Always retain the store UUID in state identifiers. Reject dependency cycles and inconsistent repeated references even when generation ordering is also checked.

### Sharded batch publication

Independent import workers build unpublished shards under one reserved logical store generation. Finalization publishes the complete store manifest at that generation only after all shards are ready. There is no reader-visible partial store and no need for a transaction spanning the shard databases. Subsequent edits use the writable overlay's generation counter.

## Integer IDs and reservations

Reserve ranges before parallel import and give workers predetermined IDs. Related tables such as `objects` and `fibers` share the same object key space. Current `objects.id` is shared across kinds: do not allocate overlapping fiber/surface IDs unless references also include a namespace. Initially retain one object key space and a separate edge key space.

Independent stores cannot guarantee globally non-overlapping integer ranges merely by inspecting their existing links. References therefore need namespace qualification; reservations provide fast routing within that namespace. Cross-store edges cannot rely on the current local integer foreign keys alone.

Support these explicit routing rules:

- One or more ranges.
- Modulo partitioning.
- Bit-based partitioning.
- A precisely specified, versioned hash rule where needed.

Each partition group has one authoritative rule, not independently conflicting rules at store, database and table levels. Record reserved ranges separately from occupied ranges so future allocation cannot reuse unpopulated reserved IDs.

Edge lookup by edge ID and adjacency lookup by object ID are different operations. Provision both explicitly. Without an adjacency accelerator, adjacency searches query the relevant edge partitions; an object's own shard does not necessarily contain all its incident edges.

## Linked stores and overlays

Support chains such as an immutable base, a classification overlay, and a later normal-alignment overlay. Seal each layer before another store links to it. Each link records the parent's UUID and generation for change detection, with the child's generation strictly greater as specified above. Links do not require historical snapshot retrieval or a separate revision selector.

Linked-store entries do not duplicate the parent's ranges. Load the dependency manifests and assemble the routing map. Reject cycles, conflicting ownership and ambiguous overlays. Multiple links to the same dependency must resolve consistently.

For edits in the local overlay:

- Resolve unchanged data from the parent.
- Store changed metadata locally.
- Reference new geometry chunks while retaining unchanged parent chunk references.
- Represent deletions with explicit tombstones.
- Allocate new objects from newly reserved IDs.

An override retains the inherited object's identity; it is not a new object. The exact choice between property patches and replacement metadata records remains open. Either choice must have explicit precedence and deletion semantics.

Initially all mutable work lives in one ordinary SQLite database. Linked stores and published shards are read-only. No cross-shard write transaction is needed.

## History and slim stores

Keep these properties independent:

| Property | Choices |
| --- | --- |
| Mutability | Writable or sealed |
| History | Off or on |
| Storage | Ordinary, sharded or packed |
| Accelerators | Explicitly available or absent |

A slim immutable import has history off and can omit spatial indexes, journals and derived accounting. Current geometry descriptors and chunk references remain required to read the data; they are not optional history. Do not invent control points for traced fibers that only have line points.

Spatial queries without a spatial accelerator initially throw an explicit error. Later, an accelerator can be built in a separate file linked to the immutable source's UUID and generation.

History records changes in the writable overlay against its immutable baseline, without copying the parent's entire history or geometry. References distinguish the object's owning store from the store holding a newer version.

History mode rules:

- A newly created store may start history-off.
- History can subsequently be enabled; the baseline must remain reconstructible.
- History cannot simply be switched off while retained history exists.
- An explicit remove-history-and-disable operation materializes the retained current state and removes obsolete history.
- For a sealed store this produces a new export, not an in-place modification.
- Removing history must not silently invalidate external references to old states.

Direct edits still use ordinary SQLite transactions even with history off. Enabling history records a store-level retention boundary, not a full copy of every object. Preserve pre-edit information lazily as described below. Earlier history-off states are not retroactively recoverable.

### One authoritative current state

Ordinary typed object/detail records and edges are the current state. Current geometry is attached directly to the object through its array descriptors and ordered chunk references. Do not duplicate current records into serialized history caches or manufacture an initial revision merely to read an object.

An unchanged object needs no historical representation rows, even in a store with history enabled. History-only tables may be created lazily when first needed; their absence has explicit schema/capability semantics, not an inferred error-recovery path. All readers use one common current-state interface.

| Current structure | Proposed slimmer treatment |
| --- | --- |
| Initial `object_revisions` metadata copy | Omit; retain old state only when an edit requires it |
| `object_heads` | Omit; typed current records and their generations identify the head |
| `object_head_records` full serialized current record | Omit; read the authoritative typed records |
| Initial `geometry_revisions` wrapper | Omit the history wrapper; retain the necessary current geometry descriptor |
| `geometry_heads` | Omit the separate revision indirection; associate current geometry directly with its object |
| `geometry_revision_chunks` | Retain necessary ordered chunk mappings, addressed by current geometry or historical representation |
| Initial `relationship_revisions` copies | Omit; current edges are authoritative until superseded |
| Duplicated current relationship membership in `object_relationship_heads` | Use indexed current edges; retain historical edge/membership changes only as needed |
| Separate current geometry materialization/cache copies | Current geometry is already materialized; any additional cache is optional and not authoritative |

This replaces the relevant schema and API paths; it is not a second slim path beside the old implementation. Do not simply drop tables while callers still depend on them. Historical endpoint bindings, relationship applicability, deletions, stable point identities and sparse editing semantics must remain representable.

For bulk creation, a fixed `initial_generation` in store metadata can supply the implicit generation of untouched records. It never follows the advancing store generation. Changed or subsequently inserted records require explicit generations. Whether to omit the initial per-record integer is a packing choice to measure; avoiding whole redundant rows is the primary simplification.

### Writing history only when needed

With history off, update current data without creating historical versions. With history on, capture the information necessary to recover the superseded state in the same transaction that publishes the new state. Only changed components need history: a metadata-only edit must not create a geometry version.

For example, an object imported at generation 10 initially has only current records. Its first edit at 25 creates the required 10-to-25 history information and replaces current records with generation 25. A reversible delta plus the materialized state at 25 can preserve the old state without a full copy at 10. Never create initial revision records for unrelated untouched objects.

Reuse immutable geometry chunks. Preserve changed array descriptors, chunk references or exact sparse byte changes; do not copy all geometry bytes for a new version. Structural edits must retain enough information to reconstruct both layouts, including removed arrays/elements. Deletion retains the required old state and an explicit absence/tombstone boundary. New objects likewise have an explicit creation boundary, so earlier snapshots do not invent their presence.

An immutable linked base can supply the old state without a local copy. Local history records the qualified base identity and exact generation when needed.

### Representation records and payload storage

Use a proposed `state_representations` table for history, generalized across objects, edges and their components. Full checkpoints and deltas belong in this table; no separate bases table or eagerly populated logical-version table is required. Exact SQL layout remains to be finalized around the read paths below.

| Field group | Purpose |
| --- | --- |
| Representation ID | Identifies the physical representation, independently of logical version identity |
| Entity kind/key and component | Selects object metadata, geometry, edge state, etc.; namespace-qualified when external |
| Source and target generations | Exact endpoints of a delta; a full checkpoint has only its represented generation |
| Encoding and codec version | Full checkpoint, reversible delta, or explicitly directed delta |
| Source/target verification information | Detects applying a delta to the wrong contents |
| Payload reference and encoded/decoded sizes | Locates stored bytes and estimates reconstruction cost |

Large payloads reside in the shared data chunk store. Full geometry checkpoints reference array descriptors and ordinary geometry chunks; delta payloads contain encoded changes. Ordered mappings retain chunk order and lengths for payloads spanning multiple chunks. Small metadata may be inline if measurements justify it. Do not add another copy of chunk bytes to a history table.

The current state serves as an implicit full checkpoint with an exact component generation. A historical reference must never mean an unqualified, moving "current" state. Each edit must atomically preserve the previous endpoint or extend/reorganize the reconstruction chain so all retained targets remain recoverable. Explicit checkpoint entries are created only when needed; existing immutable chunks can be referenced without copying their bytes.

### Efficient historical retrieval

Organize each entity/component history into ordered checkpoint intervals, not an arbitrary graph requiring recursive per-version SQL lookups. Prefer reversible deltas between consecutive changed versions, with full checkpoints selected to bound replay work. Logical chronology and generation IDs do not depend on the chosen physical reconstruction direction.

Required read paths:

1. **Current data:** route once, read current typed records/descriptors and batch fetch the requested chunks. Do not traverse history.
2. **Target generation G:** indexed seek for the latest component change at or before G, considering the current generation and initial/retention boundary. A generation at which the component did not change selects its preceding state. Respect creation, deletion and unavailable-history boundaries.
3. **Checkpoint selection:** find at most the closest full checkpoint on either side of the selected version. Include the materialized current endpoint or qualified immutable base when applicable.
4. **Path selection:** compare estimated payload bytes, decoded work and replay count. Use one eligible side, not both complete payload ranges.
5. **Replay:** fetch intervening delta descriptors in an ordered range query, batch payload reads by database/chunk location, and replay forward or backward. Verify exact endpoint contents and return the selected logical version.

Use an index ordered by entity/component/generation for version and delta range lookups, plus a partial checkpoint index or equivalent small directory for the two neighboring full-state seeks. Keep enough interval size/count information to choose a path without loading both sides' payloads; cumulative totals are an option. The first retained boundary must also be directly discoverable when no full initial checkpoint was stored.

Two-sided retrieval requires deltas that can actually be applied from either checkpoint. Reversible encodings retain overwritten/deleted information; a forward-only patch is not automatically invertible. If a directed encoding is chosen for efficiency, record its direction and guarantee at least one complete eligible path. Never choose a checkpoint solely because its generation is closer.

For partial geometry reads, index affected arrays/chunks within delta payloads so unrelated changes need not be decoded. Layout changes may require reading descriptor changes and remapping chunk coordinates; do not assume chunk indices remain stable through structural edits. Batched reads across many entities must also avoid one SQL round trip per revision or per chunk.

Historical adjacency needs endpoint-indexed edge lifetime/change information, created only as history requires it. The current-edge index alone cannot find deleted historical edges. Do not reconstruct every object's history to answer an adjacency query, or recreate two copies of the initial graph to enable it.

### Reorganizing physical history

Keep the current state materialized. Checkpoint placement and forward/backward representations can be changed later without changing logical generations, identities or reconstructed bytes. A full checkpoint shares unchanged chunks; it need not duplicate the complete geometry payload.

Publish replacement representations atomically before reclaiming old ones, respecting active readers and retained references. Prevent reconstruction cycles and cap replay bytes/steps with checkpoints. A sealed store cannot be rewritten; reorganizing it requires export into a new physical file. Repacking unchanged logical history is distinct from deleting retained history.

### Historical generations

When history is enabled, change-sets and their record versions carry the shared transaction generation. A future snapshot at generation G resolves each record to its latest retained version at or before G, including relationship changes, deletions and the dependency declarations valid at that generation. An inherited baseline remains available through its immutable links.

The generation counter alone does not preserve past contents. Historical retrieval is limited to retained history and is not required for the initial change-detection feature. Do not silently substitute a current state for an unavailable historical one. History removal preserves the current generation lineage and advances the counter for the removal operation; it never resets it.

## VFS allocation and buffered import

During construction, each independent SQLite database gets a logical address space. MiB-sized extents map that space into a shared container file. Keep the allocation directory compact enough to hold in memory.

RAM residency, dirty masks, block-write counters and LRU state are runtime bookkeeping, not permanent per-object metadata. Background writers persist dirty extents; later changes make those extents dirty again. Block-write counters are only for I/O race protection and are unrelated to logical store generations.

Flushing and eviction are separate:

- Flush cold dirty regions, coalescing adjacent writes where useful.
- Retain flushed blocks as clean cached data.
- Evict cold clean blocks when the memory budget requires it.
- Reload evicted contents for reads or partial updates; full overwrites need not reload old contents.
- Track block-write counters so an in-flight write cannot mark newer changes clean.

A full VFS chunk is not necessarily finished: SQLite may rewrite it later. Flush partial chunks as well when memory pressure or finalization requires it.

Anonymous RAM buffering plus explicit writes gives control over buffering. Writable disk-backed mmap remains subject to kernel writeback; do not assume that mmap alone provides the same behavior. Memory mapping can be an optional backend, not part of the persistent format contract.

SQLite's VFS supplies read, write, truncate, size and sync interfaces. Preserve sync semantics; disposable bulk-import mode must explicitly relax durability rather than silently ignore sync requests. See the [SQLite VFS interface](https://www.sqlite.org/c3ref/io_methods.html).

### Finalization and compact export

An explicit error-reporting `finalize()` drains writes, persists the complete directory and publishes the finished store. Incomplete builds are not valid published stores. Closing waits for pending work, but destructor-only finalization is insufficient for reporting failed writes.

Optional compact export copies finished databases into contiguous ranges in the final container and replaces their extent maps with offset/length entries. This removes container allocation slack. SQLite's internal free pages require separate compaction; packing does not remove them automatically.

The container VFS does not implement cross-database transactions. Published containers are immutable; future edits go into a new overlay database.

## Read interfaces

Both interfaces share the same manifest and resolver:

1. vcobj API routing over independent SQLite connections.
2. Optional SQL composition through views or a custom virtual table.

`UNION ALL` alone does not provide overlay precedence or custom routing. SQLite virtual-table callbacks can expose the shared resolver to SQL. Avoid requiring every shard to be attached to one connection, since `ATTACH` has a connection-level database-count limit. See [virtual tables](https://www.sqlite.org/vtab.html) and [ATTACH](https://www.sqlite.org/lang_attach.html).

Cross-database relationships require application-level integrity handling; ordinary local SQLite foreign keys do not enforce those relationships. The manifest must make the boundary explicit rather than silently treating remote references as local IDs.

## Implementation and validation sequence

1. Define v2 metadata, shared generations, invariants and capability-aware opening.
2. Add history-off current-state storage and sealed export while preserving the existing history-on layout. Defer lazy history and redesigned indexed checkpoint/delta retrieval to the follow-up.
3. Add generation-checked immutable links, cross-store references and one writable overlay.
4. Add independent-file shards using the same routing metadata.
5. Add packed VFS storage and buffered import.
6. Optionally expose the composed store through SQL.

At each storage stage, use the full-volume fiber import and report preparation, ingestion, durable finalization, output size and peak RAM separately. Compare object/constraint counts, geometry round trips and adjacency results against the unsharded representation. Keep dataset paths and machine-specific settings in local configuration, not in the reusable implementation.

Cover missing dependencies, incompatible versions, range conflicts, overlay precedence, forbidden writes, history transitions, and publication failure. Generation tests must cover atomic stamping across record kinds, rollback/no-op behavior, history-off changes, removal without counter reset, dependency generation jumps, rejection of mutable or mismatched targets, and unpublished shards remaining invisible until complete publication. History tests must verify that untouched imports and history enablement create no per-object initial history, first edits preserve exact old states, and metadata-only edits do not duplicate geometry. Test forward/backward replay, checkpoint replacement, partial geometry reads, structural changes, historical adjacency, creation/deletion, and missing/pruned history boundaries.

Measure current and intermediate-version reads as well as writes: query counts, payload bytes fetched/decoded, peak memory, and latency across both short and long histories. Compare sparse-region versus full reads and cold versus warm caches on real fiber data. Demonstrate indexed seeks and batched range reads; do not accept storage savings that require whole-history or whole-store scans. VFS tests must include rewrites during flushing, eviction/reload, partial writes, truncation and reopening compacted containers. Preserve Ubuntu/macOS and amd64/arm64 portability.

## Decisions still open

- Exact SQL schemas and required feature identifiers; schema-17 conversion follows the separate validated replacement policy above.
- Namespace-qualified key encoding and allocation across independent branches.
- Overlay patch versus replacement-record representation.
- Integrity verification beyond the store UUID and generation change token.
- Initial mandatory lookup indexes versus optional accelerators.
- History checkpoint thresholds, reversible codecs, payload packing and the measured benefit of implicit initial per-record generations.
- VFS extent size, dirty tracking granularity, memory budget and flush policy.
- Safe handling of external historical pins during history removal.

Resolve these before freezing the format. Physical performance choices should remain replaceable without changing object identities or linked-store semantics.
