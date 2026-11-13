# Immutable stores and numbered versions

The current runtime uses SQLite schema 22. Immutable stores, sharding and extensions use the same numbered-version model as ordinary stores. The [store format](store-format.md), [shared API contract](store-api-contract.md), [versioned-reference contract](versioned-reference-contract.md) and [conversion guide](schema-conversion.md) define the corresponding storage and API requirements.

## Scope and identities

- Use SQLite schema 22, not an independent “vcobj v2” schema family. An ordinary store remains an ordinary SQLite file. Optional packed-container framing has its own physical format version, independent of the logical SQLite schema.
- Retain upstream functionality, including checked transactions, numbered object and relationship versions, lineage/generation UUIDs, receipts, hosted clients/server, discovery operations, and history queries where history is retained. Do not restore the obsolete hub/replica implementation that upstream removed.
- History-off policy, sealing, direct-key store_references and overlays, independent sharding, binary UUIDs/hashes, direct geometry descriptors, optional spatial indexing, packed VFS and shared-stream batch import coexist with numbered versions. The default is 32 import workers.
- The object UUID is the complete logical key, including across shards and referenced stores. Relationships store endpoint UUIDs directly; no proxy objects, aliases, per-object routing directories, endpoint directories, or duplicated endpoint rows. Fixed key-routing metadata determines the responsible database.
- UUIDs use one 16-byte binary storage format. Chunk hashes remain binary content identifiers, not version counters.

## One commit clock and one record version

- The numeric store change counter is `commitSequence`/`commit_sequence`. This is the sole store-wide logical change counter and is independent of history retention.
- The generation UUID is a lineage identifier, not a number. Store identity, lineage identity, object identity, object version, and schema version have distinct meanings.
- New objects and relationships start at version 1. A changed record advances from its current version once per transaction, even if several setters modify it. No-op updates do not advance it. Deletion retains the last version in a tombstone so recreating the same UUID cannot reuse an old state identifier.
- Advance the store clock once for a committed logical change. Rollbacks, rolled-back savepoints, and no-op transactions do not advance it. Changes to store_references, sealing and logical provisioning count; physical repacking and cache maintenance alone do not.
- Preserve the dependency ordering rule: a derived store's commit sequence is greater than every sealed referenced dependency. Allocate `max(current sequence, referenced dependency sequences) + 1` when required. Sequence gaps are permitted; change feeds must handle them. Unrelated stores are not ordered by their sequence values alone.
- Current record versions must be stored/read without joining retained history. History-on archives use these same versions and commit sequences, not another incrementing counter. History-off does not create archival version rows merely to provide current-state freshness checks.
- Remove the old per-record numeric `generation` and `data_generation` fields and their duplicate bookkeeping. Update approvals, geometry repair guards, serializers, Python bindings and VC3D consumers to use the appropriate record version or store-state token. Do not merely rename old numeric values: legacy mutation counts are not necessarily numbered versions.
- History removal never resets current versions or the commit sequence. Enabling history archives the current state under its existing identities. The redesigned lazy/delta history scheme remains deferred.

## Geometry is ordinary object state

Each geometry state has an ordinary UUID in both history policies; metadata-only edits retain it, and history-policy maintenance preserves it. The object's metadata/current descriptor identifies its geometry chunks. Changing the descriptor or chunk references changes the object and its ordinary version; history records that object-state change. There is no separate geometry-change counter. Current geometry stays directly on the object, not in extra current-geometry or current-payload tables. History-on may retain existing archival representations until the later history redesign.

Geometry caches may conservatively invalidate on any object-version change. Existing content hashes/descriptors can identify identical payloads without introducing another counter. Geometry relationships must still be checked against the relevant geometry/anchors: changing an unrelated metadata field must not imply that the relationship's geometry is wrong. In-transaction read-your-writes and view invalidation must remain correct even when several edits share one eventual committed version.

Approval metadata belongs to the version it approves. Migrate legacy approval-generation stamps to the numbered-version model, preserving approval provenance and invalidation behavior; changes within the same transaction must not accidentally approve later unreviewed edits.

## Store references, sharding and history-off

- Store references (`store_reference`) target sealed stores and pin `(store UUID, lineage UUID, commitSequence)`. A mismatch fails explicitly. An immutable base can be shadowed by an ordinary same-UUID object/relationship in the writable overlay; deletion uses the existing state field, not a separate tombstone table.
- Retain `state` with Active/Retired/Deleted for objects and Active/Deleted for relationships. Retain location search APIs and fixed database/table routing metadata. All data tables follow the shard ownership plan; provisioning metadata remains small and store-local.
- Retiring an object tombstones its current incident relationships atomically; restoring does not resurrect them. History-on retains earlier observations through historical selections. Do not reintroduce per-edge lifecycle-binding tables or claim surviving current links need a separate lifecycle epoch check.
- Independent shard writers construct one unpublished logical import state with known versions and a shared lineage/commit token. They directly write known UUID references without cross-shard joins or validation reads. Finalization must not reopen all shards to rediscover information the importer already owns.
- History-off still supports current reads, checked writes, versions and change detection. Historical retrieval/change replay unavailable without retained records must fail or explicitly invalidate the caller's cursor; never fabricate old states or a complete feed. No history-only tables, triggers or per-object history rows may be required in slim imports.
- Keep the fast importer isolated from interactive edit bookkeeping, with bounded queues, independent SQLite writers and RAM-first packed output. No regressions to generated control points, JSON payload dumping, per-record UUID readbacks, per-edge endpoint metadata, or duplicate geometry mappings.

History-off request retry receipts are operational state, not retained object history. Their standalone receipt table is created lazily when an explicit request UUID is submitted; slim initial imports do not require it. History-policy changes preserve already-issued receipts.

## Conversion policy

- Support only upstream formats as migration inputs. Our unpublished experimental schema-18/v2 layout is discarded and has no compatibility promise or migration path. Do not retain its converter.
- Convert supported upstream schemas 16–21 to schema 22 through the separate offline converter. Preserve the earlier upstream conversion restrictions and the current schema-21 conversion's identity, history, dependency and artifact guarantees. Reject unsupported layouts rather than guessing from a version number.
- Preserve store and lineage UUIDs, numbered versions, historical references, timestamps, payloads, current approvals and supported artifacts. Convert legacy serialized records to the new fields without inventing chronology.
- The normal runtime opens only the supported current schema. It must not contain branches that read old record layouts or automatically modify old stores during open.
- Conversion creates a new file, validates completeness and consistency, and leaves the source untouched. Explicit in-place replacement, if requested, uses a validated sibling staging file and durable atomic replacement. Partial failures must leave the source recoverable.
