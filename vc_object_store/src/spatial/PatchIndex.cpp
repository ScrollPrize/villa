#include "spatial/PatchIndex.hpp"
#include <limits>
#include <map>
namespace vc::object_links {
namespace {
struct TriangleHit {
    cv::Vec3f closest, bary;
    float distSq;
};
bool valid(const cv::Vec3f& p)
{
    return p[0] != -1 && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}
TriangleHit closestPointOnTriangle(const cv::Vec3f& p,
                                   const cv::Vec3f& a,
                                   const cv::Vec3f& b,
                                   const cv::Vec3f& c)
{
    TriangleHit hit;

    const cv::Vec3f ab = b - a;
    const cv::Vec3f ac = c - a;
    const cv::Vec3f ap = p - a;

    float d1 = ab.dot(ap);
    float d2 = ac.dot(ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        hit.closest = a;
        hit.bary = {1.0f, 0.0f, 0.0f};
        const cv::Vec3f d = p - hit.closest;
        hit.distSq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
        return hit;
    }

    const cv::Vec3f bp = p - b;
    float d3 = ab.dot(bp);
    float d4 = ac.dot(bp);
    if (d3 >= 0.0f && d4 <= d3) {
        hit.closest = b;
        hit.bary = {0.0f, 1.0f, 0.0f};
        const cv::Vec3f d = p - hit.closest;
        hit.distSq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
        return hit;
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        hit.closest = a + v * ab;
        hit.bary = {1.0f - v, v, 0.0f};
        const cv::Vec3f d = p - hit.closest;
        hit.distSq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
        return hit;
    }

    const cv::Vec3f cp = p - c;
    float d5 = ab.dot(cp);
    float d6 = ac.dot(cp);
    if (d6 >= 0.0f && d5 <= d6) {
        hit.closest = c;
        hit.bary = {0.0f, 0.0f, 1.0f};
        const cv::Vec3f d = p - hit.closest;
        hit.distSq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
        return hit;
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        hit.closest = a + w * ac;
        hit.bary = {1.0f - w, 0.0f, w};
        const cv::Vec3f d = p - hit.closest;
        hit.distSq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
        return hit;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        hit.closest = b + w * (c - b);
        hit.bary = {0.0f, 1.0f - w, w};
        const cv::Vec3f d = p - hit.closest;
        hit.distSq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
        return hit;
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    float u = 1.0f - v - w;
    hit.closest = a + ab * v + ac * w;
    hit.bary = {u, v, w};
    const cv::Vec3f d = p - hit.closest;
    hit.distSq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
    return hit;
}

}
void PatchIndex::rebuild(const std::vector<SurfacePtr>& surfaces)
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

std::vector<PatchIndex::LookupResult> PatchIndex::locateAll(const PointQuery& query) const
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
                    const auto hit = t ? closestPointOnTriangle(query.worldPoint, b, e, d)
                                       : closestPointOnTriangle(query.worldPoint, a, b, d);
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

std::optional<PatchIndex::LookupResult> PatchIndex::locate(const PointQuery& query) const
{
    auto hits = locateAll(query);
    if (hits.empty()) return {};
    return *std::min_element(hits.begin(), hits.end(), [](const auto& a, const auto& b) {
        return a.distance < b.distance;
    });
}
} // namespace vc::object_links
