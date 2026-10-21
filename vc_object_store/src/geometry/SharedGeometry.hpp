#pragma once
#include "geometry/GeometryReader.hpp"

namespace vc::objects::detail {
using SharedJson = nlohmann::json;
std::vector<uint8_t> readSharedPayload(sqlite3*, const std::string& name, const std::string& hash);
bool identityManaged(sqlite3*, const std::string&);
std::string storeSharedPayload(sqlite3*, const std::string&, std::span<const uint8_t>);
// Structural operations materialize a regular chunk snapshot; subsequent
// coordinate edits use the same delta path as newly imported geometry.
std::string publishStructuralSnapshot(ObjectStore&, WriteTxn&, const std::string&, const GeometryData&, SharedJson,
    const std::string&, const std::vector<GeometryEdgeEdit>&,
    const std::optional<std::pair<std::string,std::vector<int64_t>>>& changedIds = std::nullopt);
}
