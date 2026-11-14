#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"

namespace vc::objects {
namespace detail { struct CollectionReadContext; struct InitialCollectionAccess; struct CollectionScanAccess; }
// The UUID is the complete logical identity. storeUuid is a routing hint and
// provenance for numeric lookup, never a second part of object identity.
struct ObjectKey {
    std::string storeUuid, objectUuid;
    auto operator<=>(const ObjectKey& other) const { return objectUuid<=>other.objectUuid; }
    bool operator==(const ObjectKey& other) const { return objectUuid==other.objectUuid; }
};
struct CollectionObject { ObjectKey key; ObjectRecord record; };
struct CollectionEdge { ObjectKey key, a, b; EdgeRecord record; };
struct InitialCollectionEdge { ObjectKey a, b; EdgeRecord record; std::vector<InitialExtensionRow> extensions; };
// Source-independent initial records. Coordinates are already in the declared
// native frame; no synthetic control points or source-format interpretation.
struct InitialObjectRecord {
    ObjectRecord record;
    std::optional<int64_t> id;
    std::optional<GeometryData> geometry;
    std::vector<InitialExtensionRow> extensions;
};
struct InitialObjectIdentity { ObjectKey key; int64_t id; };
struct InitialEdgeImportOptions {
    bool deferIndexes = true;
    std::size_t sqlBatchRows = 1;
    // TEMP-only initial measurements, reconciled before final transfer.
    std::vector<std::string> stagedExtensions;
};
struct CollectionMember {
    std::string databaseUuid, storeUuid, generation;
    int64_t commitSequence = 0;
    std::filesystem::path locator;
    bool partition = false;
    std::array<int64_t,2> objectIds{};
};
// Numeric object rules operate on positive signed SQLite IDs; edges use UUIDs. One group has one rule kind;
// ranges may have multiple disjoint entries per destination. HashV1 is the
// documented SplitMix64 finalizer, not std::hash or a platform-dependent hash.
enum class RoutingKind { Range, Modulo, Bits, HashV1, UuidHashV1 };
// Stable FNV-1a64 over the canonical 16 UUID bytes, modulo shardCount.
// Use this exact function in producers declaring UUID-hash routing.
std::size_t uuidShardIndex(const std::string& uuid, std::size_t shardCount);
struct KeyRoute {
    std::string keySpace; // objects: numeric IDs; objects.uuid/edges.uuid: UUID keys
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
// dependency is sealed or history-enabled and verified by store/lineage UUID
// and exact commit sequence. Physical partitions remain sealed-only.
class ObjectCollection {
public:
    static ObjectCollection create(const std::filesystem::path&, const std::vector<std::filesystem::path>& bases, const OpenOptions&);
    static ObjectCollection open(const std::filesystem::path&, const OpenOptions&);
    // Publish independently imported, sealed physical partitions of one logical
    // store. Each must have the same store/lineage UUID and commit sequence and
    // disjoint reserved object ID ranges; edges are UUID-only. The manifest has the same state pin.
    static ObjectCollection publishShards(const std::filesystem::path&, const std::vector<std::filesystem::path>&, const OpenOptions&, const std::vector<KeyRoute>& routes = {});
    ObjectCollection(ObjectCollection&&) noexcept;
    ~ObjectCollection();
    CollectionTxn read();
    CollectionTxn write();
    std::string uuid(const CollectionTxn&) const;
    StoreProvisioning provisioning(const CollectionTxn&) const;
    // True only when every database in the consumed scope is sealed/history-off.
    bool supportsTrustedGeometryReads(const CollectionTxn&) const;
    struct GeometryArrayColumns {
        StoreStatus source;
        std::vector<std::array<unsigned char,16>> objectUuids;
        PackedGeometryArray geometry;
        std::map<std::string,double> timings;
    };
    // Owned current geometry, sorted and deduplicated by requested UUID.
    // Uses the caller's snapshot with one worker. Parallel reads require a
    // sealed or history-enabled root and exact dependency pins, and use
    // independent guarded native snapshots over one shared routing topology.
    // No geometry/history/mapping rules are replaced by this batch API.
    GeometryArrayColumns readGeometryArrayColumns(const CollectionTxn&,
        std::span<const std::array<unsigned char,16>> objectUuids,
        const std::string& arrayName,const std::optional<std::string>& objectKind=std::nullopt,
        const std::optional<std::string>& volume=std::nullopt,
        const std::optional<StoreStatus>& expectedPosition=std::nullopt,std::size_t workers=32,bool profile=false,
        bool trustedImmutable=false) const;
    void createExtensionTable(CollectionTxn&, const ExtensionTable&);
    void replaceExtensionRows(CollectionTxn&, const std::string&, const ObjectKey&,
                              std::span<const ExtensionRow>);
    std::vector<ExtensionTableInfo> extensionTables(const CollectionTxn&) const;
    VolumeTransforms volumeTransforms(const CollectionTxn&) const;
    void appendExtensionRows(CollectionTxn&, const std::string& table, std::span<const ExtensionRow>);
    // Immutable rows overlay by primary key; versioned fields follow the winning
    // owner, including clears and deletions. A routed UUID prefix visits only the
    // relevant partition in each layer. Unfiltered scans visit all partitions.
    void scanExtensionRows(const CollectionTxn&, const std::string&,
        const std::function<bool(const ExtensionRow&)>&, std::span<const ExtensionValue> keyPrefix = {},
        const ExtensionKeyFilter& keyFilter = {}) const;
    // Exact version references retain their meaning across overlays. No
    // substitution of a newer local owner or dependency version is allowed.
    std::optional<ObjectRevision> objectAtVersion(const CollectionTxn&, const ObjectVersion&) const;
    std::optional<RelationshipRevision> relationshipAtVersion(const CollectionTxn&, const RelationshipVersion&) const;
    void scanExtensionRowsAtVersion(const CollectionTxn&, const std::string&, const ObjectVersion&,
        const std::function<bool(const ExtensionRow&)>&) const;
    void scanExtensionRowsAtVersion(const CollectionTxn&, const std::string&, const RelationshipVersion&,
        const std::function<bool(const ExtensionRow&)>&) const;
    std::vector<uint8_t> readGeometryArrayAtVersion(const CollectionTxn&, const ObjectVersion&, const std::string& array) const;
    // Flush the local writer's committed WAL; referenced readonly stores are unchanged.
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
    std::optional<uint64_t> artifactLength(const CollectionTxn&,const std::string& hash) const;
    std::vector<uint8_t> readArtifact(const CollectionTxn&,const std::string& hash,uint64_t offset=0,
                                    const std::optional<uint64_t>& count=std::nullopt) const;
    std::string putGeometry(CollectionTxn&, const ObjectKey&, const GeometryData&);
    // Same sparse-edit rules as ObjectStore. Inherited incident edges must be
    // explicitly removed/recreated locally before changing their geometry.
    std::string editGeometry(CollectionTxn&, const ObjectKey&, const std::vector<GeometryArrayEdit>&,
                             const std::string& expectedRevision, const std::vector<GeometryEdgeEdit>& = {});
    std::vector<CollectionEdge> edgesOf(const CollectionTxn&, const ObjectKey&) const;
    std::vector<CollectionEdge> edgesByKind(const CollectionTxn&, const std::string&) const;
    std::optional<CollectionEdge> edge(const CollectionTxn&, const ObjectKey&) const;
    ObjectKey upsertObject(CollectionTxn&, const ObjectRecord&);
    // Metadata writes create a same-key record override, not a local dataset.
    // The geometry descriptor is copied; payloads/archives and extension
    // snapshots remain borrowed from pinned bases;
    // file references are rebased if their owning directory changes.
    void replaceObject(CollectionTxn&, const ObjectKey&, const ObjectRecord&);
    void setMeta(CollectionTxn&, const ObjectKey&, const std::string&, const std::string& json);
    void deleteObject(CollectionTxn&, const ObjectKey&);
    ObjectKey addEdge(CollectionTxn&, const ObjectKey& a, const ObjectKey& b, const EdgeRecord&);
    // Initial-only append: no updates/replacements, no existing local edges,
    // history off. A failed batch rolls back every object and geometry row.
    std::vector<InitialObjectIdentity> importInitialObjects(CollectionTxn&,
        std::span<const InitialObjectRecord>);
    // Borrowed records: no geometry copy; pointed-to records live through this call.
    std::vector<InitialObjectIdentity> importInitialObjectRefs(CollectionTxn&,
        std::span<const InitialObjectRecord* const>);
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
    friend void copyObjectCollection(const std::filesystem::path&, const std::filesystem::path&, const OpenOptions&);
    friend class CollectionTxn;
    friend struct detail::CollectionReadContext;
    friend struct detail::CollectionScanAccess;
    friend struct detail::InitialCollectionAccess;
    struct Impl;
    explicit ObjectCollection(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};
// Resolve a recorded complete store position to an independently openable root
// in this collection's provisioned dependency closure. Never resolves a shard
// partition as a public source; mismatches and ambiguous roots are errors.
std::filesystem::path resolveStorePosition(const ObjectCollection&, const CollectionTxn&,
                                         const ExtensionSource&);

// Independent mutable copy of a resolved collection snapshot. The root is a
// normal SQLite database; dependencies/partitions live in <destination>.stores
// and artifacts in its ordinary sidecar. Relative locators make this bundle
// movable together. Referenced immutable stores retain their logical identity.
// A partitioned root stays sealed behind a new writable overlay; an ordinary
// root is forked directly, preserving its retained versions under the new ID.
// The destination and both sidecars must be absent. Failure cleans owned files.
void copyObjectCollection(const std::filesystem::path& source, const std::filesystem::path& destination,
                          const OpenOptions&);
}
