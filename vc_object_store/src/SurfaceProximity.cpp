#include "geometry/GeometryReader.hpp"
#include "vc_object_store/geometry/ArrayBytes.hpp"
#include "geometry/ArrayLayout.hpp"
#include "internal/CoordinateFilter.hpp"
#include "internal/Database.hpp"
#include "internal/Spatial.hpp"
#include "internal/VolumeFrames.hpp"
#include "spatial/SurfaceTiles.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <map>

namespace vc::objects {
using namespace detail;
namespace {
bool before(const SurfaceProximityHit& a, const SurfaceProximityHit& b)
{
    return a.distance != b.distance ? a.distance < b.distance : a.objectUuid < b.objectUuid;
}
}

std::vector<SurfaceProximityHit> ObjectStore::nearestSurfaces(const ReadTxn& txn, const SurfaceProximityQuery& requested) const
{
    auto query = requested;
    if(query.volume&&!query.transforms) query.transforms=std::make_shared<VolumeTransforms>(volumeTransforms(txn));
    auto* db = static_cast<sqlite3*>(connection(txn));
    for (double x : query.point) if (!std::isfinite(x)) throw std::invalid_argument("surface proximity point must be finite");
    if (!std::isfinite(query.tolerance) || query.tolerance < 0)
        throw std::invalid_argument("surface proximity tolerance must be finite and nonnegative");
    std::vector<SurfaceProximityHit> result;
    if (query.volume) {
        if (query.coordinates) throw std::invalid_argument("volume and coordinates filters are exclusive");
        if (!std::isfinite(query.targetScale) || query.targetScale <= 0)
            throw std::invalid_argument("surface proximity target scale must be finite and positive");
        for (const auto& frame : volumeFrames(db, *query.volume, std::string("surface"), false, query.transforms.get())) {
            auto stored = query;
            stored.volume.reset();
            stored.coordinates = frame.coordinates;
            stored.toTarget=VolumeAffine::scale(query.targetScale)*frame.mapping;
            for (auto hit : nearestSurfaces(txn, stored)) result.push_back(std::move(hit));
        }
        std::sort(result.begin(), result.end(), before);
        return result;
    }
    std::optional<std::string> includedIds;
    if (query.uuids) {
        std::string ids = "[";
        for (const auto& uuid : *query.uuids)
            if (const auto id = objectIdByUuid(db, parseUuid(uuid))) ids += (ids.size() > 1 ? "," : "") + std::to_string(*id);
        if (ids.size() == 1) return result;
        includedIds = ids + "]";
    }
    SurfaceNearest nearest(query.point, query.tolerance, query.toTarget);
    const Bounds& box = nearest.box();
    const auto sourceBox=query.toTarget?transformBounds(box,query.toTarget->inverse()):box;
    if (impl_->options.remoteGeometryReached)
        if (const auto remote = remoteSurfacesIn(db, sourceBox, includedIds, query.coordinates); !remote.empty())
            impl_->options.remoteGeometryReached(remote);
    std::string sql = "SELECT o.uuid,h.revision_uuid,b.row_start,b.col_start,b.row_count,b.col_count "
        "FROM spatial_regions_rtree r CROSS JOIN spatial_regions b ON b.id=r.id CROSS JOIN objects o ON o.id=b.object_id "
        "JOIN geometry_heads h ON h.object_uuid=o.uuid LEFT JOIN object_lifecycle l ON l.object_id=o.id "
        "WHERE o.kind='surface' AND b.array_name='points' AND COALESCE(l.retired,0)=0 AND " + spatialOverlapSql();
    if (includedIds) sql += " AND o.id IN (SELECT value FROM json_each(?8))";
    if (query.coordinates) sql += coordinateFilterSql("o", "?7");
    sql += " ORDER BY o.uuid,b.row_start,b.col_start";
    Statement tiles(db, sql);
    for (int i = 0; i < 6; ++i) tiles.bind(i + 1, sourceBox[i]);
    if (query.coordinates) tiles.bind(7, coordinateFilterValue(*query.coordinates));
    if (includedIds) tiles.bind(8, *includedIds);
    std::optional<RetainedGeometryReader> reader;
    std::string object, objectRevision;
    auto finish = [&] {
        if (auto hit = nearest.take()) result.push_back({object, objectRevision, hit->grid, hit->point, hit->distance});
    };
    while (tiles.row()) {
        const auto uuid = formatUuid(tiles.blob16(0));
        if (uuid != object) {
            finish();
            object = uuid;
            objectRevision = formatUuid(tiles.blob16(1));
            reader.emplace(db, objectRevision);
        }
        const auto row0 = tiles.i64(2), col0 = tiles.i64(3), rows = tiles.i64(4), cols = tiles.i64(5);
        if (rows < 2 || cols < 2) continue;
        nearest.add(surfaceTile(**reader, row0, col0, rows, cols), row0, col0, rows, cols);
    }
    finish();
    std::sort(result.begin(), result.end(), before);
    return result;
}
}
