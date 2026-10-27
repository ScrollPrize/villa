#include "spatial/SurfaceTiles.hpp"
#include "vc_object_store/geometry/SegmentClip.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc::objects::detail {
std::vector<SurfaceTileRect> surfaceTiles(int64_t rows, int64_t columns)
{
    std::vector<SurfaceTileRect> tiles;
    if (!rows || !columns) return tiles;
    constexpr int64_t cells = surfaceTileCells;
    const int64_t tileColumns = (std::max<int64_t>(1, columns - 1) + cells - 1) / cells;
    for (int64_t row = 0; row < std::max<int64_t>(1, rows - 1); row += cells)
        for (int64_t t = 0; t < tileColumns; ++t)
            tiles.push_back({row, t * cells, std::min(cells + 1, rows - row), std::min(cells + 1, columns - t * cells)});
    return tiles;
}

namespace {
using Point = SurfaceCut::Point;
Point project(const RegionSlice& view, const Point& point) {
    Point p{};
    const std::array<Point, 3> axes{view.u, view.v, view.normal};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) p[i] += (point[j] - view.origin[j]) * axes[i][j];
    return p;
}
// project(view, point)[2] with the same arithmetic.
double normalDistance(const RegionSlice& view, const Point& point) {
    double d = 0;
    for (int j = 0; j < 3; ++j) d += (point[j] - view.origin[j]) * view.normal[j];
    return d;
}
size_t hash(const std::pair<int64_t, int64_t>& e) {
    const uint64_t h = uint64_t(e.first) * 0x9e3779b97f4a7c15ull ^ uint64_t(e.second) * 0xc2b2ae3d27d4eb4full;
    return h ^ (h >> 29);
}
}

SurfaceCut::SurfaceCut(const RegionSlice& view, const std::optional<VolumeAffine>& toTarget, int64_t columns)
    : view_(view), toTarget_(toTarget), columns_(columns),
      planeOffset_((view.localBounds[2] + view.localBounds[5]) / 2) {}

bool SurfaceCut::reaches(const std::array<double, 6>& box) const
{
    bool below = false, above = false;
    double magnitude = 1;
    std::array<double, 8> distances;
    for (int corner = 0; corner < 8; ++corner) {
        Point p{box[corner & 1 ? 3 : 0], box[corner & 2 ? 4 : 1], box[corner & 4 ? 5 : 2]};
        if (toTarget_) p = toTarget_->apply(p);
        for (const double x : p) magnitude = std::max(magnitude, std::abs(x));
        distances[corner] = normalDistance(view_, p) - planeOffset_;
    }
    const double margin = 1e-9 * magnitude;
    for (const double d : distances) {
        above |= d >= -margin;
        below |= d < margin;
    }
    return above && below;
}

// The slot of `edge` (open addressing: no allocation per edge), and whether it was just added.
std::pair<size_t, bool> SurfaceCut::insert(const GridEdge& edge)
{
    if ((slots_.size() + 1) * 2 > table_.size()) grow();
    for (size_t i = hash(edge) & (table_.size() - 1);; i = (i + 1) & (table_.size() - 1)) {
        if (table_[i] == none) {
            table_[i] = slots_.size();
            slots_.push_back({edge});
            return {table_[i], true};
        }
        if (slots_[table_[i]].edge == edge) return {table_[i], false};
    }
}

void SurfaceCut::grow()
{
    table_.assign(std::max<size_t>(1024, table_.size() * 2), none);
    for (size_t s = 0; s < slots_.size(); ++s)
        for (size_t i = hash(slots_[s].edge) & (table_.size() - 1);; i = (i + 1) & (table_.size() - 1))
            if (table_[i] == none) { table_[i] = s; break; }
}

uint64_t SurfaceCut::add(const SurfaceTile& tile, int64_t row0, int64_t col0, int64_t rows, int64_t cols)
{
    uint64_t tests = 0;
    world_.resize(tile.points.size());
    distance_.resize(world_.size());
    const auto& usable = tile.valid;
    // Only usable vertices take part in a cell; a region whose usable
    // vertices all lie on one side of the plane has no crossing cell.
    bool below = false, above = false;
    for (size_t i = 0; i < world_.size(); ++i) {
        if (!usable[i]) continue;
        const auto& q = tile.points[i];
        world_[i] = {q[0], q[1], q[2]};
        if (toTarget_) world_[i] = toTarget_->apply(world_[i]);
        distance_[i] = normalDistance(view_, world_[i]) - planeOffset_;
        (distance_[i] >= 0 ? above : below) = true;
    }
    if (!above || !below) return tests;
    auto global = [&](int64_t r, int64_t c) { return (row0 + r) * columns_ + col0 + c; };
    auto crossing = [&](int64_t a, int64_t b, int64_t ga, int64_t gb) {
        if (ga > gb) { std::swap(a, b); std::swap(ga, gb); }
        const auto [slot, inserted] = insert({ga, gb});
        if (inserted) {
            const double t = distance_[a] / (distance_[a] - distance_[b]);
            auto& crossing = slots_[slot].crossing;
            for (int k = 0; k < 3; ++k) crossing.point[k] = world_[a][k] + t * (world_[b][k] - world_[a][k]);
            const std::array<double, 2> ca{double(ga % columns_), double(ga / columns_)};
            const std::array<double, 2> cb{double(gb % columns_), double(gb / columns_)};
            for (int k = 0; k < 2; ++k) crossing.grid[k] = ca[k] + t * (cb[k] - ca[k]);
        }
        return slot;
    };
    for (int64_t r = 0; r + 1 < rows; ++r)
        for (int64_t c = 0; c + 1 < cols; ++c) {
            const std::array<int64_t, 4> v{r * cols + c, r * cols + c + 1, (r + 1) * cols + c + 1, (r + 1) * cols + c};
            if (!usable[v[0]] || !usable[v[1]] || !usable[v[2]] || !usable[v[3]]) continue;
            const bool side0 = distance_[v[0]] >= 0;
            if (side0 == (distance_[v[1]] >= 0) && side0 == (distance_[v[2]] >= 0) && side0 == (distance_[v[3]] >= 0)) continue;
            const std::array<int64_t, 4> g{global(r, c), global(r, c + 1), global(r + 1, c + 1), global(r + 1, c)};
            for (const auto& tri : {std::array<int, 3>{0, 1, 3}, std::array<int, 3>{1, 2, 3}}) {
                ++tests;
                std::array<size_t, 3> cut;
                int cuts = 0;
                for (int e = 0; e < 3; ++e) {
                    const auto a = tri[e], b = tri[(e + 1) % 3];
                    if ((distance_[v[a]] >= 0) != (distance_[v[b]] >= 0)) cut[cuts++] = crossing(v[a], v[b], g[a], g[b]);
                }
                if (cuts == 2) segments_.emplace_back(cut[0], cut[1]);
            }
        }
    return tests;
}

// Chain segments through their shared grid edges, in emission order. A grid
// edge borders at most two triangles, each cut by one segment.
std::vector<std::deque<size_t>> SurfaceCut::chains() const
{
    auto slots = slots_;
    for (size_t i = 0; i < segments_.size(); ++i)
        for (const auto slot : {segments_[i].first, segments_[i].second}) {
            auto& through = slots[slot].segments;
            if (through[1] != none) throw std::logic_error("surface region: grid edge on three triangles");
            through[through[0] == none ? 0 : 1] = i;
        }
    std::vector<std::deque<size_t>> result;
    std::vector<bool> used(segments_.size());
    for (size_t start = 0; start < segments_.size(); ++start) {
        if (used[start]) continue;
        used[start] = true;
        std::deque<size_t> chain{segments_[start].first, segments_[start].second};
        auto extend = [&](bool front) {
            for (;;) {
                const auto end = front ? chain.front() : chain.back();
                const auto& next = slots[end].segments;
                const auto it = std::find_if(next.begin(), next.end(), [&](size_t s) { return s != none && !used[s]; });
                if (it == next.end()) return;
                used[*it] = true;
                const auto other = segments_[*it].first == end ? segments_[*it].second : segments_[*it].first;
                if (front) chain.push_front(other); else chain.push_back(other);
            }
        };
        extend(false); extend(true);
        result.push_back(std::move(chain));
    }
    return result;
}

bool SurfaceCut::clip(const Point& a, const Point& b, double& first, double& last) const
{
    const std::array<double, 6> planeBox{view_.localBounds[0], view_.localBounds[1], -1,
                                         view_.localBounds[3], view_.localBounds[4], 1};
    auto pa = project(view_, a), pb = project(view_, b);
    pa[2] = pb[2] = 0;
    return vc::geometry::clipSegment(pa, pb, planeBox, first, last);
}

SurfaceNearest::SurfaceNearest(const std::array<double, 3>& point, double tolerance, const std::optional<VolumeAffine>& toTarget)
    : point_{point}, toleranceSq_(tolerance * tolerance), toTarget_(toTarget)
{
    for (int axis = 0; axis < 3; ++axis) {
        box_[axis] = point[axis] - tolerance;
        box_[axis + 3] = point[axis] + tolerance;
    }
}

void SurfaceNearest::add(const SurfaceTile& tile, int64_t row0, int64_t col0, int64_t rows, int64_t cols)
{
    const auto& usable = tile.valid;
    world_.resize(tile.points.size());
    for (size_t i = 0; i < world_.size(); ++i) {
        const auto& q = tile.points[i];
        world_[i] = {{q[0], q[1], q[2]}};
        if (toTarget_) world_[i].v = toTarget_->apply(world_[i].v);
    }
    const auto& box = box_;
    for (int64_t r = 0; r + 1 < rows; ++r)
        for (int64_t c = 0; c + 1 < cols; ++c) {
            const auto ia = r * cols + c, ib = ia + 1, id = ia + cols, ie = id + 1;
            if (!usable[ia] || !usable[ib] || !usable[id] || !usable[ie]) continue;
            const auto &a = world_[ia], &b = world_[ib], &d = world_[id], &e = world_[ie];
            bool outside = false;
            for (int axis = 0; axis < 3; ++axis)
                outside |= (a[axis] < box[axis] && b[axis] < box[axis] && d[axis] < box[axis] && e[axis] < box[axis]) ||
                           (a[axis] > box[axis + 3] && b[axis] > box[axis + 3] && d[axis] > box[axis + 3] && e[axis] > box[axis + 3]);
            if (outside) continue;
            for (int t = 0; t < 2; ++t) {
                const auto hit = t ? closestPointOnTriangle(point_, b, e, d) : closestPointOnTriangle(point_, a, b, d);
                if (!(hit.distSq <= toleranceSq_) || hit.distSq >= bestSq_) continue;
                const double u = std::clamp(t ? hit.bary[0] + hit.bary[1] : hit.bary[1], 0.0, 1.0);
                const double v = std::clamp(t ? hit.bary[1] + hit.bary[2] : hit.bary[2], 0.0, 1.0);
                bestSq_ = hit.distSq;
                best_ = Hit{{double(col0 + c) + u, double(row0 + r) + v}, hit.closest.v, std::sqrt(hit.distSq)};
            }
        }
}

std::optional<SurfaceNearest::Hit> SurfaceNearest::take()
{
    auto hit = std::move(best_);
    best_.reset();
    bestSq_ = INFINITY;
    return hit;
}
}
