#pragma once
#include <filesystem>
#include <memory>
#include <vector>
#include "vc_object_store/objects/ObjectCollection.hpp"
struct sqlite3;
namespace vc::objects { struct OpenOptions; }
namespace vc::objects::detail {
// Connections are pinned by the caller's transaction, in overlay precedence
// order. Lease/setup happens on the caller thread; each connection may then be
// scanned by ONE worker. No collection callbacks or concurrent caller access.
struct CollectionScanPartition {
    sqlite3* db;
    std::vector<KeyRoute> uuidRoutes;
};
struct CollectionScanAccess {
    static std::vector<CollectionScanPartition> partitions(const ObjectCollection&, const CollectionTxn&);
};
// Offline maintenance already holds the root connection exclusively. Reuse
// its snapshot, open only policy-eligible pinned dependencies, and install the common verified
// reference resolver until this guard is destroyed.
std::shared_ptr<void> attachCollectionReferenceResolver(sqlite3*, const std::filesystem::path&);
std::shared_ptr<void> attachCollectionReferenceResolver(sqlite3*, const std::filesystem::path&,const OpenOptions&);
}
