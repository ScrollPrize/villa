#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"

namespace vc::objects {
namespace detail { struct CollectionReadContext; struct InitialCollectionAccess; }
// The UUID is the complete logical identity. storeUuid is a routing hint and
// provenance for numeric lookup, never a second part of object identity.
struct ObjectKey {
    std::string storeUuid, objectUuid;
    auto operator<=>(const ObjectKey& other) const { return objectUuid<=>other.objectUuid; }
    bool operator==(const ObjectKey& other) const { return objectUuid==other.objectUuid; }
};
struct CollectionObject { ObjectKey key; ObjectRecord record; };
struct CollectionEdge { ObjectKey key, a, b; EdgeRecord record; };
struct InitialCollectionEdge { ObjectKey a, b; EdgeRecord record; };
// Source-independent initial records. Coordinates are already in the declared
// native frame; no synthetic control points or source-format interpretation.
struct InitialObjectRecord {
    ObjectRecord record;
    std::optional<int64_t> id;
    std::optional<GeometryData> geometry;
};
struct InitialObjectIdentity { ObjectKey key; int64_t id; };
struct InitialEdgeImportOptions {
    bool deferIndexes = true;
    std::size_t sqlBatchRows = 1;
};
struct CollectionMember {
    std::string databaseUuid, storeUuid;
    int64_t generation = 0;
    std::filesystem::path locator;
    bool partition = false;
    std::array<int64_t,2> objectIds{}, edgeIds{};
};
// Rules operate on positive signed SQLite IDs. One group has one rule kind;
// ranges may have multiple disjoint entries per destination. HashV1 is the
// documented SplitMix64 finalizer, not std::hash or a platform-dependent hash.
enum class RoutingKind { Range, Modulo, Bits, HashV1, UuidHashV1 };
// Stable FNV-1a64 over the canonical 16 UUID bytes, modulo shardCount.
// Use this exact function in producers declaring UUID-hash routing.
std::size_t uuidShardIndex(const std::string& uuid, std::size_t shardCount);
struct KeyRoute {
    std::string keySpace; // objects/edges: numeric IDs; objects.uuid/edges.uuid: UUID keys
    RoutingKind kind = RoutingKind::Range;
    int64_t lower = 1, upper = INT64_MAX;
    uint64_t divisor = 1, remainder = 0;
    unsigned shift = 0;
    std::string databaseUuid;
    bool matches(int64_t id) const;
    bool matchesUuid(const std::string& uuid) const;
};
class ObjectCollection;
class CollectionTxn {
public:
    CollectionTxn(CollectionTxn&&) noexcept;
    ~CollectionTxn();
    void commit();
    bool writable() const;
private:
    friend class ObjectCollection;
    struct Impl;
    explicit CollectionTxn(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};

// Composed vcobj API over independent SQLite connections, without ATTACH's
// database limit. The local ObjectStore is one atomic writable overlay; every
// dependency/partition is sealed and verified by UUID+generation on open.
class ObjectCollection {
public:
    static ObjectCollection create(const std::filesystem::path&, const std::vector<std::filesystem::path>& bases, const OpenOptions&);
    static ObjectCollection open(const std::filesystem::path&, const OpenOptions&);
    // Publish independently imported, sealed physical partitions of one logical
    // store. Each must have the same store UUID/generation and disjoint reserved
    // object/edge ID ranges. The manifest is sealed in that same generation.
    static ObjectCollection publishShards(const std::filesystem::path&, const std::vector<std::filesystem::path>&, const OpenOptions&, const std::vector<KeyRoute>& routes = {});
    ObjectCollection(ObjectCollection&&) noexcept;
    ~ObjectCollection();
    CollectionTxn read();
    CollectionTxn write();
    std::string uuid(const CollectionTxn&) const;
    StoreProvisioning provisioning(const CollectionTxn&) const;
    // Flush the local writer's committed WAL; linked readonly stores are unchanged.
    void checkpoint();
    std::vector<CollectionMember> members() const;
    std::optional<CollectionObject> object(const CollectionTxn&, const ObjectKey&) const;
    // Numeric lookup routes through partition reservations. Declared UUID-hash
    // routes query only the selected shard per layer (no fallback fanout).
    // Legacy arbitrary partitions without UUID routes require explicit fanout.
    // It never interprets a hole in a
    // reserved range as permission to allocate that ID in another partition.
    std::optional<CollectionObject> objectById(const CollectionTxn&, const std::string& owner, int64_t id) const;
    void scanObjects(const CollectionTxn&, const std::function<bool(const CollectionObject&)>&, const std::optional<std::string>& kind = {}) const;
    std::optional<GeometryRevision> currentGeometry(const CollectionTxn&, const ObjectKey&) const;
    std::vector<uint8_t> readGeometryArray(const CollectionTxn&, const ObjectKey&, const std::string& array) const;
    std::string putGeometry(CollectionTxn&, const ObjectKey&, const GeometryData&);
    // Same sparse-edit rules as ObjectStore. Inherited incident edges must be
    // explicitly removed/recreated locally before changing their geometry.
    std::string editGeometry(CollectionTxn&, const ObjectKey&, const std::vector<GeometryArrayEdit>&,
                             const std::string& expectedRevision, const std::vector<GeometryEdgeEdit>& = {});
    std::vector<CollectionEdge> edgesOf(const CollectionTxn&, const ObjectKey&) const;
    std::vector<CollectionEdge> edgesByKind(const CollectionTxn&, const std::string&) const;
    std::optional<CollectionEdge> edge(const CollectionTxn&, const ObjectKey&) const;
    std::optional<CollectionEdge> edgeById(const CollectionTxn&, const std::string& owner, int64_t id) const;
    ObjectKey upsertObject(CollectionTxn&, const ObjectRecord&);
    void replaceObject(CollectionTxn&, const ObjectKey&, const ObjectRecord&);
    void setMeta(CollectionTxn&, const ObjectKey&, const std::string&, const std::string& json);
    void deleteObject(CollectionTxn&, const ObjectKey&);
    ObjectKey addEdge(CollectionTxn&, const ObjectKey& a, const ObjectKey& b, const EdgeRecord&);
    ObjectKey insertEdgeWithId(CollectionTxn&, const ObjectKey& a, const ObjectKey& b, const EdgeRecord&, int64_t id);
    // Initial-only append: no updates/replacements, no existing local edges,
    // history off. A failed batch rolls back every object and geometry row.
    std::vector<InitialObjectIdentity> importInitialObjects(CollectionTxn&,
        std::span<const InitialObjectRecord>);
    // Construct the first edge graph of a history-off store. The producer fills
    // each batch and returns true, or returns false with an empty batch at EOF.
    // It may read external inputs, but must not mutate this collection. All
    // batches form one atomic operation; failure restores records/allocators/
    // indexes. Existing owned objects and sealed dependencies are supported.
    // Bulk creation stamps use one publication timestamp per batch. Ordinary
    // editing APIs retain their per-operation behavior.
    int64_t importInitialEdges(CollectionTxn&,
        const std::function<bool(std::vector<InitialCollectionEdge>&)>& producer,
        const InitialEdgeImportOptions& = {});
    void removeEdge(CollectionTxn&, const ObjectKey&);
    void seal(CollectionTxn&);
private:
    friend class CollectionTxn;
    friend struct detail::CollectionReadContext;
    friend struct detail::InitialCollectionAccess;
    struct Impl;
    explicit ObjectCollection(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};
}
