# Initial batch construction

`objects/InitialImport.hpp` provides source-independent C++ construction of a sealed, history-off, spatial-index-free v2 collection. Source readers supply native `ObjectRecord`, optional `GeometryData`, and `InitialCollectionEdge` records. The API owns SQLite connections, transactions, ID reservations, endpoint reference creation, shard provisioning, index finalization, sealing, publication and failure cleanup. It does not require AFV or any intermediate database.

```cpp
#include <vc_object_store/objects/InitialImport.hpp>
using namespace vc::objects;

InitialImportOptions options;
options.store.author = "batch-import";
// Default: one ordinary SQLite file. Optional independent shard ranges:
options.shards = {{{1, 1000000}, {1, 10000000}},
                  {{1000001, 2000000}, {10000001, 20000000}}};

auto result = importInitialCollection("fibers.vcobj", options,
    [&](size_t shardIndex, InitialShardImport& shard) {
        // One callback per independent worker; each reader/buffer is worker-local.
        // Read or construct batches from your source, then:
        std::vector<InitialObjectRecord> objects;
        // ...fill objects, with explicit IDs in this shard's range or automatic IDs...
        auto identities = shard.appendObjects(objects);
        // Repeat appendObjects until this shard's objects are complete.
    },
    [&](size_t shardIndex, InitialShardImport& shard) {
        // Every shard's objects now exist in the build-private directory.
        shard.importEdges([&](std::vector<InitialCollectionEdge>& edges) {
            // ...fill one nonempty batch using qualified endpoint ObjectKeys,
            // including objects owned by any sibling shard...
            // Return false with an empty batch at EOF.
            return !edges.empty();
        });
    });
```

One unpacked shard produces an ordinary SQLite file. Multiple unpacked shards produce a directory with `collection.vcobj` and independent shard databases. Optional `options.packed` produces one framed packed-VFS file instead. The logical store UUID is newly generated per import, shared by its shards and available as `shard.storeUuid()`; source namespaces belong in provenance and entity UUID derivation, not publication identity.

Each shard's numeric object and edge ranges must be positive and disjoint from other shards. Object UUIDs and edge UUIDs must be globally unique within the new logical collection. Native SQLite uniqueness checks remain active within each shard. References to objects in another publication use their qualified `{storeUuid, objectUuid}` keys and a sealed dependency in `options.dependencies`. The single producer form supports each shard's own objects and sealed dependencies; it does not inspect another worker's live transaction.

For relations between newly created sibling shards, use `importInitialCollection(destination, options, objectProducer, edgeProducer)`. Each worker completes its object producer before any edge producer runs. Accepted objects contribute a compact, immutable endpoint directory containing their qualified key, numeric ID, lifecycle status, and geometry dependency information (array shapes, stable point identities and surface validity). XYZ payload buffers are not retained. Each worker accumulates its own entries without per-object shared locking; the coordinator merges them once, rejects duplicate qualified object identities, and makes the directory available to all edge producers. The object phase cannot import edges, and the edge phase cannot append objects. A failure wakes waiting peers and prevents publication. Sibling shards never become mutable SQLite links, and no connection crosses worker threads. The one-phase form has no global endpoint registry; in that form cross-shard UUID uniqueness remains the producer's responsibility. Edge UUID uniqueness across shards remains the producer's responsibility in both forms.

Objects are insertion-only: no updates, replacement, invented control points, or ordinary editing callbacks. Supplied geometry uses the same native encoders and chunk format as ordinary geometry storage. Metadata, tags, endpoint dependency validation, lifecycle state, unique constraints and ordinary local foreign keys retain their native semantics. Cross-database endpoints use compact qualified reference keys without copied endpoint records; their owner and dependencies are checked by native validation, not SQLite cross-database foreign keys. Current edges to missing or retired endpoints are rejected. Object creation uses new-publication provenance and pending approval, not restoration of old approval records. All owned objects precede edge import; one edge producer is permitted per shard. It may only supply data, not mutate or commit the destination. Exceptions poison the construction even if the producer catches them; no partial collection is published. Existing destinations are never overwritten.

`InitialEdgeImportOptions` controls bounded endpoint caches, optional deferred nonunique indexes, and SQL statement batch size. The default `sqlBatchRows=1` still processes a stream of native batches; it does not mean an editing transaction per row. Larger SQL VALUES statements are optional, not assumed faster. Unique indexes and foreign keys remain active. Rebuilding deferred indexes is part of edge construction, not hidden after timing ends.

For RAM-first ingestion, set `options.packed` with `disposableBulk=true`, `memoryFirst=true`, `backgroundFlush=false`, and a sufficient `residentBytes` budget. Payload pages stay resident until finalization; exceeding the budget fails instead of silently spilling. Finalization writes and durably publishes the compact file. SQLite caches, source batches and native lookup caches require additional memory. Alternatively enable background flushing explicitly. See [packed stores](packed-stores.md) for the modes and failure behavior.

Results separate per-shard creation, native object/edge construction, source callback time and seal/commit, plus wall ingestion, manifest construction, final persistence and total time. Per-shard times overlap and must not be summed as wall time. Benchmark readback is caller-side validation and should be reported separately. For slow full-volume trials, use a short external time cap and label extrapolations: incomplete ingestion does not measure final index construction, manifest publication or persistence.

The lower-level `ObjectCollection::importInitialObjects` and `importInitialEdges` operate within a caller-owned transaction and provide savepoint rollback. `ObjectCollection::checkpoint()` flushes the local writer's committed WAL; call it only after ending write transactions, never from an edge producer. Use the top-level builder when the API should own the entire construction lifecycle. The AFV command is one [source adapter and validation harness](afv-v2-import.md), not the implementation of vcobj's construction workflow.

Run `bash vc_object_store/scripts/run_cpp_tests.sh` for native regression coverage. Persistent builds/logs remain under `VC_OBJECT_STORE_TEST_ROOT` (default `local/vcobj-tests`). If that filesystem lacks atomic no-replace directory renames, set `VC_OBJECT_STORE_FIXTURE_ROOT` to an existing local filesystem directory supporting them. Only disposable fixtures go there; the script removes its private fixture directory on exit, including failures. Unsupported atomic directory publication fails explicitly rather than using a racy check-and-rename fallback; packed and ordinary single-file outputs use their own atomic file publication.

## Geometry construction

Initial geometry uses the ordinary manifest encoder, sequence identity/annotation format, current-descriptor writer and chunk writer. Preparation consumes per-point source annotations into shared metadata and computes managed object fields before the single initial object insertion; surfaces also derive their grid dimensions and area before inserting the detail row. It does not call `putGeometry`, mutate metadata afterward, stamp an existing record, scan incident edges or repair an existing geometry. Preparation and publication share the caller's rollback boundary. Newly inserted chunks need no redundant byte readback: SQLite has stored the supplied payload. When a hash already exists, its bytes are still compared, including pinned external payloads, so corrupt or colliding hashes are rejected.

Indexed relationship anchors to dependency geometry are validated against the owner without storing a local descriptor, metadata copy, subtype row or geometry chunks. Newly created sibling endpoints use the compact prepared directory from the object phase, including stable point IDs and surface validity where needed. Whole-geometry relationships with no point references need only endpoint existence/lifecycle validation. Current relations do not store duplicate geometry-version, lifecycle, or metadata-binding rows: they refer to the current endpoints, with later changes checked transactionally. Historical relationships retain endpoint version references only when history is enabled.
