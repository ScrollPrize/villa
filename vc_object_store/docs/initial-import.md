# Initial batch construction

`objects/InitialImport.hpp` provides source-independent C++ construction of a sealed, history-off, spatial-index-free collection. Source readers supply native `ObjectRecord`, optional `GeometryData`, and `InitialCollectionEdge` records. The API owns SQLite connections, transactions, object-ID reservations, shard provisioning, optional index finalization, sealing, publication and failure cleanup. Edges use binary UUID primary keys without numeric row IDs or edge-ID reservations. It does not require AFV or any intermediate database.

Optional typed extension data uses the same construction path. Supply table definitions in `InitialImportOptions::extensions`, then attach `InitialExtensionRow` values to each `InitialObjectRecord::extensions` or `InitialCollectionEdge::extensions`. Definitions are installed in each shard before importing records; rows follow their owning object's or edge's UUID routing and transaction. These initial result tables are immutable (`versioned=false`). See [typed extensions](extensions.md) for input provenance, columns, indexes and editable extensions outside initial construction.

```cpp
#include <vc_object_store/objects/InitialImport.hpp>
using namespace vc::objects;

InitialImportOptions options;
options.store.author = "batch-import";
// Default: one ordinary SQLite file. Optional independent object-row ranges:
options.shards.resize(2);
options.shards[0].objectIds = {1, 1000000};
options.shards[1].objectIds = {1000001, 2000000};
// Optional slim edge layout: only uuid/a_uuid/b_uuid/kind/commit_sequence.
options.store.edgeColumns = std::vector<std::string>{};
options.store.edgeAdjacencyIndex = false; // Full graph scans do not need it.

auto result = importInitialCollection("fibers.vcobj", options,
    [&](size_t shardIndex, InitialShardImport& shard) {
        // One callback per independent worker; each reader/buffer is worker-local.
        // Read or construct batches from your source, then:
        std::vector<InitialObjectRecord> objects;
        // ...fill objects whose UUID hashes to shardIndex via uuidShardIndex,
        // with explicit numeric IDs in this shard's range or automatic IDs...
        auto identities = shard.appendObjects(objects);
        // Repeat appendObjects until this shard's objects are complete.
    },
    [&](size_t shardIndex, InitialShardImport& shard) {
        // Only this worker's object producer has finished; other workers may still be writing.
        shard.importEdges([&](std::vector<InitialCollectionEdge>& edges) {
            // ...fill one nonempty batch using endpoint ObjectKeys (object UUID plus a routing hint),
            // including objects owned by any sibling shard...
            // Return false with an empty batch at EOF.
            return !edges.empty();
        });
    });
```

One unpacked shard produces an ordinary SQLite file. Multiple unpacked shards produce a directory with `collection.vcobj` and independent shard databases. Optional `options.packed` produces one framed packed-VFS file instead. The logical store UUID is newly generated per import, shared by its shards and available as `shard.storeUuid()`; source namespaces belong in provenance and entity UUID derivation, not publication identity.

Finishing a batch is direct construction, not discovery of an existing collection. Each new shard has exactly one writer connection and no reader pool. Its worker returns the committed identity, lineage/commit sequence and reserved ranges; the coordinator writes the routing manifest from those descriptors using one additional writer. No new shard or completed collection is reopened, read back, or passed through `publishShards` during construction. The manifest writer shares the format encoder with other collection operations, not their discovery/validation workflow. External pre-existing dependencies still need readers; they are not newly built shards. Each thread-confined importer uses one reader per dependency database, regardless of the ordinary `OpenOptions::readerPoolSize`; larger pools would be idle and multiply file-descriptor use across input and output shards. Durability/atomic file exposure follows once all writers close. Regression tests count actual SQLite connection creations to enforce this boundary.

Each shard's numeric object-row ranges must be positive and disjoint from other shards. Edges have no numeric ranges. Object UUIDs and edge UUIDs must be globally unique within the new logical collection. Native SQLite uniqueness checks remain active within each shard. With default `uuidHashRouting=true`, the producer assigns both object and edge records using `uuidShardIndex(uuid, shardCount)`; the manifest stores fixed per-shard rules, not per-object entries. Reference endpoints need not belong to the edge's shard. Explicit `uuidHashRouting=false` supports arbitrary legacy partitioning with fanout UUID reads. References to objects in another publication use the same object UUID and a sealed dependency in `options.dependencies`. `ObjectKey::storeUuid` is only a routing/provenance hint; it does not distinguish two records with the same object UUID. Neither producer form inspects another worker's live transaction.

For relations between newly created sibling shards, use `importInitialCollection(destination, options, objectProducer, edgeProducer)`. Each worker completes its own object producer before its edge producer runs, independently of the other workers. There is no endpoint directory, merge, global object barrier or replacement registry. Edges may reference UUIDs whose objects have not yet been inserted. The object phase cannot import edges, and the edge phase cannot append objects. A failure prevents publication. Sibling shards never become mutable SQLite store_references, and no connection crosses worker threads. Cross-shard UUID uniqueness and reference consistency are the producer's responsibility.

Objects are insertion-only: no updates, replacement, invented control points, or ordinary editing callbacks. Supplied geometry uses the same native encoders and chunk format as ordinary geometry storage. Every edge stores its two original object UUIDs directly as 16-byte BLOBs, regardless of where the objects reside. Trusted batch construction does not look up endpoint existence, retirement, geometry dependencies or cross-table UUID collisions. Producers must supply consistent data; ordinary editing APIs retain their validation. SQL uniqueness and applicable local foreign keys, record-format checks and rollback remain active. Retirement and lifecycle epoch are fields on the object row; immutable builders omit sync identity. Object creation uses new-publication provenance and pending approval, not restoration of old approval records. One edge producer is permitted per shard. It may only supply data, not mutate or commit the destination. Exceptions poison the construction even if the producer catches them; no partial collection is published. Existing destinations are never overwritten.

`InitialImportOptions::beforePublish` is an optional source-completion callback. It runs once on the calling thread after all successful shard producers have joined, while the destination is still private and before manifest construction or final persistence/publication. Use it to join source readers and compare their aggregate counters with declared totals. Throwing removes the private staging, leaving the destination available for a corrected retry. It adds no database reopening, row scan, or readback validation; producers that do not need a completion check leave it unset.

`InitialEdgeImportOptions` controls optional deferred nonunique indexes and SQL statement batch size. There are no endpoint caches. The default `sqlBatchRows=1` still processes a stream of native batches; it does not mean an editing transaction per row. Larger SQL VALUES statements are optional, not assumed faster. Unique indexes and foreign keys remain active. Rebuilding deferred indexes is part of edge construction, not hidden after timing ends.

### Deferred extension transformation

When a streaming producer needs a final reduction before its measurement rows are ready, list the immutable extension table names in `options.edges.stagedExtensions`. Definitions still belong in `options.extensions`, and edge batches still carry ordinary `InitialExtensionRow` values. The importer writes selected rows to private SQLite TEMP tables instead of their final extension tables; other extensions in the same batch are inserted normally. Staging uses the existing shard writer connection, with no additional database, connection pool or published extension definition. It is optional and does not change core schema 22 or a table's declared format version.

After `shard.importEdges(...)` has drained the shard's input, call the following on the same shard worker, before returning from its edge producer:

```cpp
const int64_t transferred = shard.transferStagedExtensionRows(
    "ext_example_measurements",
    [&](ExtensionRow& row) {
        // Apply the completed reduction to this ordinary row in column order.
        // Preserve its owning UUID, schema and other unchanged values.
    },
    4096); // Maximum rows materialized for one transformation batch.
```

The callback runs once per staged row; transformed values pass through the normal extension row encoder into the final table. `InitialExtensionRow::stagingPayload` optionally supplies opaque construction-only bytes. For a nonempty payload, the callback receives one extra final BLOB value after the declared columns; transfer strips that value before insertion. The bytes occupy only the TEMP table and queue accounting, never extension provisioning or final rows. Supplying a payload for a non-staged table is an error. Callbacks must preserve the declared column count, with or without that final staging value.

The return value is the row count. Transfer deletes the temporary rows and drops the private staging table when complete. Every selected stage must be transferred, even when empty: sealing rejects any remaining stage table. A missing stage, invalid callback/batch size, malformed transformed row, SQL error or callback exception poisons construction and prevents publication. Transfer itself has savepoint rollback; the top-level builder also cleans all unpublished output, including when a producer catches its failure. This is a construction-only facility, not an editable result or a reader-side correction mechanism.

Transfer batches bound the temporary C++ row buffers, not the accumulated TEMP table. Construction currently uses SQLite `temp_store=MEMORY`; all pending staged rows consume memory outside packed-VFS `residentBytes`. During transfer, temporary SQLite pages and the growing final database can coexist. Allow headroom for both, per-writer caches and source/queue buffers; a packed resident budget is not a process-wide memory cap. Transfer is part of ingestion, before final persistence. Per-shard callback durations overlap and may include waiting for the producer's final reduction, so report them separately rather than adding them to ingestion wall time.

Private immutable builders use 64-KiB SQLite pages; ordinary editable stores and existing files keep their page size. This is ordinary SQLite framing and needs no custom reader. Schema 22 uses UUID-only edge tables. New relation rows share their logical commit state; provenance is written only into columns selected by the layout. Empty metadata/dependency values are normalized to SQL NULL. Explicit point/channel/metadata dependencies retain the general serializer and record-format validation without checking the target object. Omitting optional columns is an explicit creation choice: supplying a non-default value for an absent field fails rather than silently losing it. See [store format](store-format.md#uuid-only-edges-and-optional-columns).

Private construction connections are owned exclusively by their producer thread and omit SQLite connection mutexes; editable stores retain their ordinary mutex protection. Edge batches are normalized in place, reuse binary UUIDs, and borrow stable payload strings only while their SQL statements are alive. No endpoint or cross-table namespace scan runs at finalization. A failed finalization poisons the private import, including when a producer catches the exception.

For RAM-first ingestion, set `options.packed` with `disposableBulk=true`, `memoryFirst=true`, `backgroundFlush=false`, and a sufficient `residentBytes` budget. Payload pages stay resident until finalization; exceeding the budget fails instead of silently spilling. Finalization writes and durably publishes the compact file. SQLite caches and source batches require additional memory. Alternatively enable background flushing explicitly. See [packed stores](packed-stores.md) for the modes and failure behavior.

Results separate per-shard creation, native object/edge construction, source callback time and seal/commit, plus wall ingestion, manifest construction, final persistence and total time. Per-shard times overlap and must not be summed as wall time. Benchmark readback is caller-side validation and should be reported separately. For slow full-volume trials, use a short external time cap and label extrapolations: incomplete ingestion does not measure final index construction, manifest publication or persistence.

The lower-level `ObjectCollection::importInitialObjects` and `importInitialEdges` operate within a caller-owned transaction and provide savepoint rollback. `ObjectCollection::checkpoint()` flushes the local writer's committed WAL; call it only after ending write transactions, never from an edge producer. Use the top-level builder when the API should own the entire construction lifecycle. The AFV command is one [source adapter and validation harness](afv-import.md), not the implementation of vcobj's construction workflow.

Run `bash vc_object_store/scripts/run_cpp_tests.sh` for native regression coverage. `test_initial_import` covers bounded staged transfer, typed value preservation, coexistence with immediate extension rows, no extra SQLite connections, forgotten transfers and failed-transform cleanup for ordinary and packed collections. Persistent builds/logs remain under `VC_OBJECT_STORE_TEST_ROOT` (default `local/vcobj-tests`). If that filesystem lacks atomic no-replace directory renames, set `VC_OBJECT_STORE_FIXTURE_ROOT` to an existing local filesystem directory supporting them. Only disposable fixtures go there; the script removes its private fixture directory on exit, including failures. Unsupported atomic directory publication fails explicitly rather than using a racy check-and-rename fallback; packed and ordinary single-file outputs use their own atomic file publication.

## Parallel source streaming

`objects/imports/ParallelImportStream.hpp` is the shared transport used by the AFV adapter and prepared-input benchmark. Source tasks are coarse chunks, independent of destination shards. A fixed reader pool claims tasks from an atomic work queue, constructs native records, and routes their UUIDs into bounded destination queues. Each SQLite writer consumes only its own queue. There is no per-object directory or serial payload partition/copy. `readers` and `shards` can be chosen independently.

Parallel streaming defaults to 32 readers and 32 destination shards. The collection builder must be given matching shard specifications. The lower-level `InitialImportOptions` still defaults to one ordinary SQLite file; choosing parallel streaming does not change ordinary non-sharded store creation.

An `ObjectEmitter` accepts an `ImportObject` containing either an owned `InitialObjectRecord` or a borrowed pointer to one. Borrowed records and their geometry must remain immutable and alive until the stream and importer finish. Writers collect pointers and call `InitialShardImport::appendObjectRefs`; the lower-level `ObjectCollection::importInitialObjectRefs` has the same transactional semantics as owned batches. This avoids copying prepared XYZ arrays without creating a second storage path. Edge emitters accept ordinary `InitialCollectionEdge` records.

Extension values travel in those same records and count toward owned payload byte budgets. Borrowed prepared records include borrowed extension values under the same lifetime rule. No secondary measurement stream, per-row source-state copy or endpoint-directory scan is required.

Call `stream.start(objectTaskCount, edgeTaskCount, objectTask, edgeTask)` before invoking the collection builder, then drain `popObjects` and `popEdges` in its respective callbacks. A callback receives its reader index, task index and emitter, so it can reuse a thread-owned source cursor. Readers that finish their object tasks can prepare edges while other readers complete their final geometry chunks; there is no reader-join barrier between phases. Destination writers still append their objects before their edges. Every producer/writer exception must be forwarded to `stream.fail(std::current_exception())` before rethrowing; call `stream.finish()` in `beforePublish` to join readers and propagate failures before the destination becomes visible, then perform any source-total checks there. Destruction cancels and joins an abandoned stream. No partial store is published.

`batchRows` bounds record count per emitted destination batch, `readerBytes` bounds pending payload per reader, and `queueBytes` bounds queued payload per stream across destination shards. One oversized record/batch per destination is allowed to avoid deadlock. Borrowed prepared payloads are already resident outside this budget. When the reader budget is reached, the largest pending destination batch is flushed, rather than flushing every small batch. Deallocation occurs outside queue locks. Reports expose task counts, per-reader work and queue-wait time; worker times overlap and are not wall import time.

Sources that already own a parallel worker pool can use `startEdgeProducer(workerCount, producer)` instead of `start`. The producer receives one thread-exclusive emitter per source worker, joins all source work before returning, and emits directly to the same bounded shard queues. No additional preparation pool or globally ordered source queue is introduced. Source identifiers must be supplied deterministically by the caller, independently of arrival order. Failures use the same cancellation and unpublished-store cleanup as ordinary streaming imports.

## Geometry construction

Coordinate/source metadata is parsed and serialized once. The initial object INSERT includes the complete geometry descriptor, token and checksum, without a subsequent object UPDATE.

Initial geometry uses the ordinary manifest encoder, sequence identity/annotation format, current-descriptor writer and chunk writer. Preparation consumes per-point source annotations into shared metadata and computes managed object fields before the single initial object insertion; surfaces also derive their grid dimensions and area before inserting the detail row. It does not call `putGeometry`, mutate metadata afterward, stamp an existing record, scan incident edges or repair an existing geometry. Preparation and publication share the caller's rollback boundary. Newly inserted chunks need no redundant byte readback: SQLite has stored the supplied payload. When a hash already exists, its bytes are still compared, including pinned external payloads, so corrupt or colliding hashes are rejected.

Relationship anchors are supplied by the producer without owner-geometry reads during batch construction. Neither local nor sibling/external endpoints create a descriptor copy, subtype row, geometry copy or registration entry. Current relations refer to the current endpoints, with later ordinary edits checked transactionally. Historical relationships retain endpoint version references only when history is enabled.
