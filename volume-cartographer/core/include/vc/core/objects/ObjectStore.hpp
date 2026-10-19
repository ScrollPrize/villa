#pragma once
#include "vc/core/storage/RecordDb.hpp"
#include "vc/core/objects/Geometry.hpp"
namespace vc::objects {
using namespace vc::storage;
// Independently openable versioned object storage. Contains no graph tables.
class ObjectStore : public RecordDb {
public:
    static ObjectStore create(const std::filesystem::path&, const OpenOptions&);
    static ObjectStore open(const std::filesystem::path&, const OpenOptions&);
    // Stable store identity, independent of its filesystem location.
    std::string uuid(const Txn&) const;
    // Immutable geometry. A null expectedRevision means "expect no geometry";
    // pass the current revision UUID to replace it. Stale writers fail, even
    // when their proposed bytes match. Identical current content is a no-op.
    // The write is savepoint-protected and invalidates object approval.
    std::string putGeometry(WriteTxn& txn, const std::string& objectUuid,
                            const GeometryData& data,
                            const std::optional<std::string>& expectedRevision = std::nullopt);
    std::optional<GeometryRevision> currentGeometry(const Txn& txn, const std::string& objectUuid) const;
    std::optional<GeometryRevision> geometryRevision(const Txn& txn, const std::string& revisionUuid) const;
    std::vector<GeometryRevision> geometryHistory(const Txn& txn, const std::string& objectUuid) const;
    // Byte ranges allow bounded loading without materializing a whole array.
    // Every touched chunk is verified against its SHA-256 content address.
    std::vector<uint8_t> readGeometryArray(const Txn& txn, const std::string& revisionUuid,
        const std::string& arrayName, int64_t byteOffset = 0,
        const std::optional<int64_t>& byteCount = std::nullopt) const;
private:
    explicit ObjectStore(std::unique_ptr<Impl>);
};
} // namespace vc::objects
