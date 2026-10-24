#include "spatial/SurfaceIndex.hpp"
#include "spatial/TriangleDistance.hpp"
#include <limits>
#include <map>
namespace vc::object_links {
namespace {
bool valid(const cv::Vec3f& p)
{
    return p[0] != -1 && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}
}
void SurfaceIndex::rebuild(const std::vector<SurfacePtr>& surfaces)
{
    surfaces_ = surfaces;
    tiles_.clear();
    std::vector<objects::detail::BoundsIndex::Box> boxes;
    for (size_t i = 0; i < surfaces.size(); ++i) {
        const auto& points = surfaces[i]->rawPoints();
        for (int row = 0; row + 1 < points.rows; row += 8) {
            for (int col = 0; col + 1 < points.cols; col += 8) {
                objects::detail::BoundsIndex::Box box{INFINITY, INFINITY, INFINITY,
                                                     -INFINITY, -INFINITY, -INFINITY};
                for (int r = row; r < std::min(row + 9, points.rows); ++r) {
                    for (int c = col; c < std::min(col + 9, points.cols); ++c) {
                        const auto point = points(r, c);
                        if (!valid(point)) continue;
                        for (int axis = 0; axis < 3; ++axis) {
                            box[axis] = std::min(box[axis], double(point[axis]));
                            box[axis + 3] = std::max(box[axis + 3], double(point[axis]));
                        }
                    }
                }
                if (!std::isfinite(box[0])) continue;
                boxes.push_back(box);
                tiles_.push_back({i, row, col});
            }
        }
    }
    bounds_.build(std::move(boxes));
}

std::vector<SurfaceIndex::LookupResult> SurfaceIndex::locateAll(const PointQuery& query) const
{
    if (!(query.tolerance > 0) || !std::isfinite(query.tolerance)) return {};
    objects::detail::BoundsIndex::Box box;
    cv::Vec3f low, high;
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(query.worldPoint[axis])) return {};
        // Round outwards to allow float triangle arithmetic at large world coordinates.
        const float slack = 8 * std::numeric_limits<float>::epsilon() *
                            (std::abs(query.worldPoint[axis]) + query.tolerance + 1.f);
        box[axis] = double(query.worldPoint[axis]) - query.tolerance - slack;
        box[axis + 3] = double(query.worldPoint[axis]) + query.tolerance + slack;
        low[axis] = query.worldPoint[axis] - query.tolerance - slack;
        high[axis] = query.worldPoint[axis] + query.tolerance + slack;
    }
    std::vector<size_t> candidates;
    bounds_.query(box, [&](size_t i) { candidates.push_back(i); });
    std::sort(candidates.begin(), candidates.end());
    struct Best {
        LookupResult hit;
        float distanceSquared = INFINITY;
    };
    std::map<size_t, Best> best;
    for (size_t index : candidates) {
        const auto& tile = tiles_[index];
        const auto& surface = surfaces_[tile.surface];
        if (query.surfaces.includeRaw && !query.surfaces.includeRaw->contains(surface.get())) continue;
        const auto& points = surface->rawPoints();
        for (int r = tile.row; r < std::min(tile.row + 8, points.rows - 1); ++r) {
            for (int c = tile.col; c < std::min(tile.col + 8, points.cols - 1); ++c) {
                const auto a = points(r, c), b = points(r, c + 1);
                const auto d = points(r + 1, c), e = points(r + 1, c + 1);
                bool outside = false;
                for (int axis = 0; axis < 3; ++axis) {
                    outside |= (a[axis] < low[axis] && b[axis] < low[axis] &&
                                d[axis] < low[axis] && e[axis] < low[axis]) ||
                               (a[axis] > high[axis] && b[axis] > high[axis] &&
                                d[axis] > high[axis] && e[axis] > high[axis]);
                }
                if (outside || !valid(a) || !valid(b) || !valid(d) || !valid(e)) continue;
                for (int t = 0; t < 2; ++t) {
                    const auto hit = t ? objects::detail::closestPointOnTriangle(query.worldPoint, b, e, d)
                                       : objects::detail::closestPointOnTriangle(query.worldPoint, a, b, d);
                    if (!std::isfinite(hit.distSq) || hit.distSq > query.tolerance * query.tolerance) continue;
                    auto& current = best[tile.surface];
                    if (hit.distSq >= current.distanceSquared) continue;
                    const float u = std::clamp(t ? hit.bary[0] + hit.bary[1] : hit.bary[1], 0.f, 1.f);
                    const float v = std::clamp(t ? hit.bary[1] + hit.bary[2] : hit.bary[2], 0.f, 1.f);
                    current = {LookupResult{surface, {c + u, r + v}, std::sqrt(hit.distSq)}, hit.distSq};
                }
            }
        }
    }
    std::vector<LookupResult> result;
    for (const auto& [id, hit] : best) result.push_back(hit.hit);
    return result;
}

std::optional<SurfaceIndex::LookupResult> SurfaceIndex::locate(const PointQuery& query) const
{
    auto hits = locateAll(query);
    if (hits.empty()) return {};
    return *std::min_element(hits.begin(), hits.end(), [](const auto& a, const auto& b) {
        return a.distance < b.distance;
    });
}
} // namespace vc::object_links
