#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"

namespace vc::objects {
namespace detail { struct CollectionReadContext; }
// Namespace-qualified logical identities. Local SQL row IDs are deliberately
// not treated as globally unique across independently created stores.
struct ObjectKey {
    std::string storeUuid, objectUuid;
    auto operator<=>(const ObjectKey&) const = default;
};
struct CollectionObject { ObjectKey key; ObjectRecord record; };
struct CollectionEdge { ObjectKey key, a, b; EdgeRecord record; };
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
enum class RoutingKind { Range, Modulo, Bits, HashV1 };
struct KeyRoute {
    std::string keySpace; // objects or edges
    RoutingKind kind = RoutingKind::Range;
    int64_t lower = 1, upper = INT64_MAX;
    uint64_t divisor = 1, remainder = 0;
    unsigned shift = 0;
    std::string databaseUuid;
    bool matches(int64_t id) const;
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
    std::vector<CollectionMember> members() const;
    std::optional<CollectionObject> object(const CollectionTxn&, const ObjectKey&) const;
    // Numeric lookup routes through partition reservations; UUID lookup is
    // indexed within each relevant partition. It never interprets a hole in a
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
    void removeEdge(CollectionTxn&, const ObjectKey&);
    void seal(CollectionTxn&);
private:
    friend class CollectionTxn;
    friend struct detail::CollectionReadContext;
    struct Impl;
    explicit ObjectCollection(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};
}
