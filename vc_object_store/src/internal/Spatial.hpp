#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include <functional>
struct sqlite3;

namespace vc::objects::detail {
using Bounds = std::array<double, 6>;
void validateBounds(const Bounds&);
std::optional<Bounds> recordBounds(const ObjectRecord&);
void indexGeometryBounds(sqlite3*, int64_t objectId, const std::string& revision, const GeometryData&);
// The grid cells a sparse edit of a surface revision touches (cells with an
// edited vertex or validity) and their area in that revision; nullopt when the
// edit is no surface point/validity edit, or touches too much of the grid to
// update the area cell by cell.
struct EditedSurfaceCells {
    std::vector<std::pair<int64_t, int64_t>> cells;  // (row, column), row-major
    double area = 0;
};
std::optional<EditedSurfaceCells> editedSurfaceCells(sqlite3*, const std::string& revision,
                                                     const std::vector<GeometryArrayEdit>&);
// After an edit: bounds of the dirty regions, and the area (by the cells'
// change when `before` holds them, else recomputed).
void updateEditedGeometryBounds(sqlite3*, int64_t objectId, const std::string& revision,
                                const std::vector<GeometryArrayEdit>&,
                                const std::optional<EditedSurfaceCells>& before = std::nullopt);
std::string spatialOverlapSql(int firstParameter = 1);
// The query-space box of a slab (validated orthonormal axes), with a margin of
// a few ulps so inclusive overlap tests never miss geometry on its faces.
Bounds slabBounds(const RegionSlice&);
// Surfaces whose geometry is not local yet (catalog placeholders) whose record
// bounds overlap `bounds` (source XYZ), filtered like a geometry query: optional
// JSON id list and exact coordinate declaration. UUID order.
std::vector<std::string> remoteSurfacesIn(sqlite3*, const Bounds& bounds, const std::optional<std::string>& includedIds,
                                          const std::optional<std::string>& coordinates);
void setSpatialBounds(sqlite3*, int64_t objectId, const std::optional<Bounds>&,
                      const std::optional<std::string>& revision = std::nullopt);
void refreshGeometryBounds(sqlite3*, const std::string& objectUuid);
// A surface revision's area in stored units squared (geometry::surfaceGridArea
// over its valid vertices, read in row bands) and grid size as the object's
// surface detail; nothing for other kinds.
void refreshSurfaceArea(sqlite3*, int64_t objectId, const std::string& revision);
void createSpatialSchema(sqlite3*);
}
