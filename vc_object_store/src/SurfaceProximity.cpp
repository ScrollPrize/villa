#include "geometry/GeometryReader.hpp"
#include "vc_object_store/geometry/ArrayBytes.hpp"
#include "geometry/ArrayLayout.hpp"
#include "internal/CoordinateFilter.hpp"
#include "internal/Database.hpp"
#include "internal/Spatial.hpp"
#include "internal/VolumeFrames.hpp"
#include "spatial/TriangleDistance.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <map>

namespace vc::objects {
using namespace detail;
namespace {
struct Vec3 {
    std::array<double, 3> v;
    double operator[](size_t i) const { return v[i]; }
    double dot(const Vec3& o) const { return v[0] * o.v[0] + v[1] * o.v[1] + v[2] * o.v[2]; }
};
Vec3 operator-(const Vec3& a, const Vec3& b) { return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}}; }
Vec3 operator+(const Vec3& a, const Vec3& b) { return {{a[0] + b[0], a[1] + b[1], a[2] + b[2]}}; }
Vec3 operator*(double s, const Vec3& a) { return {{s * a[0], s * a[1], s * a[2]}}; }
Vec3 operator*(const Vec3& a, double s) { return s * a; }
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
    Bounds box;
    for (int axis = 0; axis < 3; ++axis) {
        box[axis] = query.point[axis] - query.tolerance;
        box[axis + 3] = query.point[axis] + query.tolerance;
    }
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
    const Vec3 p{query.point};
    const double toleranceSq = query.tolerance * query.tolerance;
    std::optional<RetainedGeometryReader> reader;
    bool hasValid = false;
    std::optional<SurfaceProximityHit> best;
    double bestSq = INFINITY;
    auto finish = [&] { if (best) result.push_back(std::move(*best)); best.reset(); bestSq = INFINITY; };
    std::string object;
    while (tiles.row()) {
        const auto uuid = formatUuid(tiles.blob16(0));
        const auto revision = formatUuid(tiles.blob16(1));
        if (uuid != object) {
            finish();
            object = uuid;
            reader.emplace(db, revision);
            hasValid = (*reader)->manifest().at("arrays").contains(surfaceValidChannel);
        }
        const auto row0 = tiles.i64(2), col0 = tiles.i64(3), rows = tiles.i64(4), cols = tiles.i64(5);
        if (rows < 2 || cols < 2) continue;
        const auto bytes = (*reader)->rectangle("points", row0, col0, rows, cols);
        const auto valid = hasValid ? (*reader)->rectangle(surfaceValidChannel, row0, col0, rows, cols) : std::vector<uint8_t>{};
        std::vector<Vec3> world(size_t(rows * cols));
        std::vector<uint8_t> usable(world.size());
        for (size_t i = 0; i < world.size(); ++i) {
            std::array<float, 3> q; copyScalars<float>(std::span(bytes).subspan(i * 12, 12), q);
            world[i] = {{q[0], q[1], q[2]}};
            if(query.toTarget)world[i].v=query.toTarget->apply(world[i].v);
            usable[i] = vc::geometry::validSurfacePoint(q) && (!hasValid || valid[i]);
        }
        for (int64_t r = 0; r + 1 < rows; ++r)
            for (int64_t c = 0; c + 1 < cols; ++c) {
                const auto ia = r * cols + c, ib = ia + 1, id = ia + cols, ie = id + 1;
                if (!usable[ia] || !usable[ib] || !usable[id] || !usable[ie]) continue;
                const auto &a = world[ia], &b = world[ib], &d = world[id], &e = world[ie];
                bool outside = false;
                for (int axis = 0; axis < 3; ++axis)
                    outside |= (a[axis] < box[axis] && b[axis] < box[axis] && d[axis] < box[axis] && e[axis] < box[axis]) ||
                               (a[axis] > box[axis + 3] && b[axis] > box[axis + 3] && d[axis] > box[axis + 3] && e[axis] > box[axis + 3]);
                if (outside) continue;
                for (int t = 0; t < 2; ++t) {
                    const auto hit = t ? closestPointOnTriangle(p, b, e, d) : closestPointOnTriangle(p, a, b, d);
                    if (!(hit.distSq <= toleranceSq) || hit.distSq >= bestSq) continue;
                    const double u = std::clamp(t ? hit.bary[0] + hit.bary[1] : hit.bary[1], 0.0, 1.0);
                    const double v = std::clamp(t ? hit.bary[1] + hit.bary[2] : hit.bary[2], 0.0, 1.0);
                    bestSq = hit.distSq;
                    best = SurfaceProximityHit{uuid, revision, {double(col0 + c) + u, double(row0 + r) + v},
                                               hit.closest.v, std::sqrt(hit.distSq)};
                }
            }
    }
    finish();
    std::sort(result.begin(), result.end(), before);
    return result;
}
}
