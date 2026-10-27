#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include <functional>
struct sqlite3;

namespace vc::objects::detail {
using Bounds = std::array<double, 6>;
void validateBounds(const Bounds&);
std::optional<Bounds> recordBounds(const ObjectRecord&);
void indexGeometryBounds(sqlite3*, int64_t objectId, const std::string& revision, const GeometryData&);
void updateEditedGeometryBounds(sqlite3*, int64_t objectId, const std::string& revision,
                                const std::vector<GeometryArrayEdit>&);
std::string spatialOverlapSql(int firstParameter = 1);
// Surfaces whose geometry is not local yet (catalog placeholders) whose record
// bounds overlap `bounds` (source XYZ), filtered like a geometry query: optional
// JSON id list and exact coordinate declaration. UUID order.
std::vector<std::string> remoteSurfacesIn(sqlite3*, const Bounds& bounds, const std::optional<std::string>& includedIds,
                                          const std::optional<std::string>& coordinates);
void setSpatialBounds(sqlite3*, int64_t objectId, const std::optional<Bounds>&,
                      const std::optional<std::string>& revision = std::nullopt);
void refreshGeometryBounds(sqlite3*, const std::string& objectUuid);
void createSpatialSchema(sqlite3*);
}
