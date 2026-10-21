#pragma once
#include "geometry/GeometryReader.hpp"

namespace vc::objects::detail {
using SharedJson = nlohmann::json;
std::vector<uint8_t> readSharedPayload(sqlite3*, const std::string& name, const std::string& hash);
bool sharedArray(const SharedJson&);
bool identityManaged(sqlite3*, const std::string&);
std::vector<uint8_t> readSharedArray(sqlite3*, const SharedJson&, const std::string&, int64_t, int64_t);
std::vector<uint8_t> readSharedRectangle(sqlite3*, const SharedJson&, const std::string&, int64_t, int64_t, int64_t, int64_t);
std::string storeSharedPayload(sqlite3*, const std::string&, std::span<const uint8_t>);
void referenceSharedArrays(sqlite3*, const std::string&, const SharedJson&);
void editSharedArray(sqlite3*, SharedJson&, const GeometryArrayEdit&);
void indexSharedGeometry(sqlite3*, int64_t, const std::string&, SharedJson&);
std::string publishSharedGeometry(ObjectStore&, WriteTxn&, const std::string&, SharedJson,
    const std::string&, const std::vector<GeometryEdgeEdit>&,
    const std::optional<std::pair<std::string,std::vector<int64_t>>>& changedIds = std::nullopt);
}
