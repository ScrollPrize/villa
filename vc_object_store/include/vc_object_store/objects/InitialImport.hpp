#pragma once
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include <functional>

namespace vc::objects {

struct InitialShardSpec {
    std::array<int64_t,2> objectIds{1,INT64_MAX};
    std::array<int64_t,2> edgeIds{1,INT64_MAX};
};
struct InitialImportOptions {
    OpenOptions store=[] {OpenOptions o;o.history=HistoryMode::Off;o.spatialIndex=false;return o;}();
    std::vector<InitialShardSpec> shards{InitialShardSpec{}};
    // Producers MUST send each object/edge UUID to uuidShardIndex(uuid,N).
    // False is explicit compatibility for arbitrary partitions: UUID reads
    // then fan out instead of claiming a routing contract they cannot honor.
    bool uuidHashRouting = true;
    std::vector<std::filesystem::path> dependencies;
    std::optional<PackedStoreOptions> packed;
    InitialEdgeImportOptions edges;
};
struct InitialShardResult {
    int64_t objects=0,edges=0;
    double creationSeconds=0,objectSeconds=0,edgeSeconds=0,producerSeconds=0,sealCommitSeconds=0;
};
struct InitialImportResult {
    // Public root locator: the file, or directory/collection.vcobj for plain shards.
    std::filesystem::path path;
    std::string storeUuid;
    std::vector<InitialShardResult> shards;
    double ingestionSeconds=0,manifestSeconds=0,persistenceSeconds=0,totalSeconds=0;
};

// Available only during the producer callback, on its assigned worker thread.
// No mutable store/transaction handle is exposed. Objects precede relations
// within each worker. Endpoint UUIDs are supplied directly by the controlled
// producer, including references to sibling shards and sealed dependencies.
class InitialShardImport {
public:
    ~InitialShardImport();
    InitialShardImport(const InitialShardImport&)=delete;
    InitialShardImport& operator=(const InitialShardImport&)=delete;
    const std::string& storeUuid() const;
    std::vector<InitialObjectIdentity> appendObjects(std::span<const InitialObjectRecord>);
    int64_t importEdges(const std::function<bool(std::vector<InitialCollectionEdge>&)>&);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit InitialShardImport(std::unique_ptr<Impl>);
    friend InitialImportResult importInitialCollection(const std::filesystem::path&,
        const InitialImportOptions&,const std::function<void(size_t,InitialShardImport&)>&);
    friend InitialImportResult importInitialCollection(const std::filesystem::path&,
        const InitialImportOptions&,const std::function<void(size_t,InitialShardImport&)>&,
        const std::function<void(size_t,InitialShardImport&)>&);
};

// Constructs, seals and atomically publishes one new collection. The API owns
// independent worker connections, transactions, generations, index finalization,
// linking, VFS lifetime and failure cleanup. Producers only supply native data.
// One uncompressed shard produces an ordinary SQLite file; multiple uncompressed
// shards produce a directory. Supplying packed produces one framed VFS file.
// History-off/spatial-off initial construction only; editing uses other APIs.
// Dependencies must already be sealed; sibling shards are not mutable links.
// A fresh logical UUID is allocated for every import; creationStoreUuid must be
// unset. Producers must supply globally unique object and edge UUIDs across all
// shards. Native uniqueness is enforced per shard. UUID-hash routing sends any
// repeated key to the same shard, without constructing a per-object directory.
InitialImportResult importInitialCollection(const std::filesystem::path& destination,
    const InitialImportOptions&,
    const std::function<void(size_t,InitialShardImport&)>& producer);

// Two callbacks per worker: its object producer finishes before its edge
// producer starts, without a cross-worker barrier or endpoint directory.
// Edges address original UUIDs without opening another worker's connection.
// Object producers may only append objects; edge producers may only import
// edges. Failure in either phase cancels peers and prevents publication.
InitialImportResult importInitialCollection(const std::filesystem::path& destination,
    const InitialImportOptions&,
    const std::function<void(size_t,InitialShardImport&)>& objectProducer,
    const std::function<void(size_t,InitialShardImport&)>& edgeProducer);

} // namespace vc::objects
