#pragma once
#include "geometry/GeometryReader.hpp"

namespace vc::objects::detail {
using SharedJson = nlohmann::json;
// Raw content-addressed payload access, local or explicitly linked. Callers
// account budgets and verify the appropriate content hash exactly once.
int64_t geometryPayloadSize(sqlite3*, const std::string& hash);
bool hasExternalGeometryPayload(sqlite3*, const std::string& hash);
std::vector<uint8_t> readGeometryPayload(sqlite3*, const std::string& hash);
std::vector<uint8_t> readExternalGeometryPayload(sqlite3*, const std::string& hash);
std::vector<std::string> reachableGeometryMetadata(sqlite3*, const SharedJson& manifest);
std::vector<uint8_t> readSharedPayload(sqlite3*, const std::string& name, const std::string& hash);
std::string storeSharedPayload(sqlite3*, const std::string&, std::span<const uint8_t>);
// Install an initial full descriptor whose chunks are already registered
// locally (payloads may be pinned external references). Uses the ordinary
// publication, generation, geometry indexing and dependency machinery.
std::string publishReferencedGeometry(ObjectStore&, WriteTxn&, const std::string&, const SharedJson&);
// Structural operations materialize a regular chunk snapshot; subsequent
// coordinate edits use the same delta path as newly imported geometry.
std::string publishStructuralSnapshot(ObjectStore&, WriteTxn&, const std::string&, const GeometryData&, SharedJson,
    const std::string&, const std::vector<GeometryEdgeEdit>&,
    const std::optional<std::pair<std::string,std::vector<int64_t>>>& changedIds = std::nullopt);
// The edges publishStructuralSnapshot(data, changedIds) requires decisions for.
std::vector<EdgeRecord> structuralSnapshotEdges(const ObjectStore&, const Txn&, const std::string&, const GeometryData&,
    const std::optional<std::pair<std::string,std::vector<int64_t>>>& changedIds = std::nullopt);
}
