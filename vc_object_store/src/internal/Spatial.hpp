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
void migrateSpatialRegions(sqlite3*);
std::string spatialOverlapSql(int firstParameter = 1);
void setSpatialBounds(sqlite3*, int64_t objectId, const std::optional<Bounds>&,
                      const std::optional<std::string>& revision = std::nullopt);
void refreshGeometryBounds(sqlite3*, const std::string& objectUuid);
void createSpatialSchema(sqlite3*);
void backfillSpatialBounds(sqlite3*);
class GeometryReader;
void visitIndexedRegions(sqlite3*, const SpatialRegion&, const Bounds&, GeometryReader&,
                         const std::function<void(const SpatialRegion&)>&);
std::vector<SpatialRegion> refineIndexedRegions(sqlite3*, const std::vector<SpatialRegion>&, const ObjectSelection&);
}
