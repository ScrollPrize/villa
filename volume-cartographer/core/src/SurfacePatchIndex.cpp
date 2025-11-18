#include "vc/core/util/SurfacePatchIndex.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>
#include <unordered_map>

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "vc/core/util/Surface.hpp"

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace {

inline bool equalVec3f(const cv::Vec3f& a, const cv::Vec3f& b)
{
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

inline bool shouldSampleIndex(int idx, int count, int stride)
{
    if (count <= 0) {
        return false;
    }
    if (idx < 0 || idx >= count) {
        return false;
    }
    if (stride <= 1) {
        return true;
    }
    const int lastIndex = count - 1;
    return (idx % stride == 0) || (idx == lastIndex);
}

cv::Vec3f makePtrFromAbsCoord(const QuadSurface* surface, float absX, float absY)
{
    const cv::Vec3f center = surface->center();
    const cv::Vec2f scale = surface->scale();
    return cv::Vec3f(
        absX - center[0] * scale[0],
        absY - center[1] * scale[1],
        0.0f
    );
}

struct TriangleHit {
    cv::Vec3f closest{0, 0, 0};
    cv::Vec3f bary{0, 0, 0}; // weights for vertices (sum to 1, >= 0)
    float distSq = std::numeric_limits<float>::max();
};

inline float clamp01(float v) {
    return std::max(0.0f, std::min(1.0f, v));
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
        hit.distSq = cv::norm(p - hit.closest, cv::NORM_L2SQR);
        return hit;
    }

    const cv::Vec3f bp = p - b;
    float d3 = ab.dot(bp);
    float d4 = ac.dot(bp);
    if (d3 >= 0.0f && d4 <= d3) {
        hit.closest = b;
        hit.bary = {0.0f, 1.0f, 0.0f};
        hit.distSq = cv::norm(p - hit.closest, cv::NORM_L2SQR);
        return hit;
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        hit.closest = a + v * ab;
        hit.bary = {1.0f - v, v, 0.0f};
        hit.distSq = cv::norm(p - hit.closest, cv::NORM_L2SQR);
        return hit;
    }

    const cv::Vec3f cp = p - c;
    float d5 = ab.dot(cp);
    float d6 = ac.dot(cp);
    if (d6 >= 0.0f && d5 <= d6) {
        hit.closest = c;
        hit.bary = {0.0f, 0.0f, 1.0f};
        hit.distSq = cv::norm(p - hit.closest, cv::NORM_L2SQR);
        return hit;
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        hit.closest = a + w * ac;
        hit.bary = {1.0f - w, 0.0f, w};
        hit.distSq = cv::norm(p - hit.closest, cv::NORM_L2SQR);
        return hit;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        hit.closest = b + w * (c - b);
        hit.bary = {0.0f, 1.0f - w, w};
        hit.distSq = cv::norm(p - hit.closest, cv::NORM_L2SQR);
        return hit;
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    float u = 1.0f - v - w;
    hit.closest = a + ab * v + ac * w;
    hit.bary = {u, v, w};
    hit.distSq = cv::norm(p - hit.closest, cv::NORM_L2SQR);
    return hit;
}

struct CellKey {
    QuadSurface* surface = nullptr;
    int row = 0;
    int col = 0;

    bool operator==(const CellKey& other) const noexcept
    {
        return surface == other.surface && row == other.row && col == other.col;
    }
};

struct CellKeyHash {
    std::size_t operator()(const CellKey& key) const noexcept
    {
        std::size_t h = std::hash<QuadSurface*>{}(key.surface);
        h ^= static_cast<std::size_t>(key.row) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= static_cast<std::size_t>(key.col) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

} // namespace

struct SurfacePatchIndex::Impl {
    struct PatchRecord {
        QuadSurface* surface = nullptr;
        int i = 0;
        int j = 0;
        std::array<cv::Vec3f, 4> corners{};

        bool operator==(const PatchRecord& other) const noexcept {
            return surface == other.surface &&
                   i == other.i &&
                   j == other.j &&
                   std::equal(corners.begin(), corners.end(),
                              other.corners.begin(), equalVec3f);
        }
    };

    struct TriangleRecord {
        QuadSurface* surface = nullptr;
        int i = 0;
        int j = 0;
        int triangleIndex = 0;
        std::array<cv::Vec3f, 3> world{};
        std::array<cv::Vec3f, 3> surfaceParams{};

        bool operator==(const TriangleRecord& other) const noexcept {
            return surface == other.surface &&
                   i == other.i &&
                   j == other.j &&
                   triangleIndex == other.triangleIndex &&
                   std::equal(world.begin(), world.end(),
                              other.world.begin(), equalVec3f) &&
                   std::equal(surfaceParams.begin(), surfaceParams.end(),
                              other.surfaceParams.begin(), equalVec3f);
        }
    };

    using Point3 = bg::model::point<float, 3, bg::cs::cartesian>;
    using Box3 = bg::model::box<Point3>;
    using Entry = std::pair<Box3, PatchRecord>;
    using PatchTree = bgi::rtree<Entry, bgi::quadratic<32>>;
    using TriangleEntry = std::pair<Box3, TriangleRecord>;
    using TriangleTree = bgi::rtree<TriangleEntry, bgi::quadratic<32>>;

    std::unique_ptr<PatchTree> tree;
    std::unique_ptr<TriangleTree> triangleTree;
    struct CellEntry {
        bool hasPatch = false;
        Entry patch;
        std::vector<TriangleEntry> triangles;
    };

    size_t patchCount = 0;
    size_t triangleCount = 0;
    float bboxPadding = 0.0f;
    int samplingStride = 1;

    std::unordered_map<CellKey, CellEntry, CellKeyHash> cellEntries;

    struct PatchHit {
        bool valid = false;
        float u = 0.0f;
        float v = 0.0f;
        float distSq = std::numeric_limits<float>::max();
    };

    static std::vector<std::pair<CellKey, CellEntry>>
    collectEntriesForSurface(QuadSurface* surface,
                             float bboxPadding,
                             int samplingStride,
                             int rowStart,
                             int rowEnd,
                             int colStart,
                             int colEnd);
    static bool buildCellEntry(QuadSurface* surface,
                               const cv::Mat_<cv::Vec3f>& points,
                               int col,
                               int row,
                               float bboxPadding,
                               CellEntry& outEntry);

    void removeCellEntry(const CellKey& key);
    void insertCells(const std::vector<std::pair<CellKey, CellEntry>>& cells);
    void removeCells(QuadSurface* surface,
                     int rowStart,
                     int rowEnd,
                     int colStart,
                     int colEnd);

    bool replaceSurfaceEntries(QuadSurface* surface,
                               std::vector<std::pair<CellKey, CellEntry>>&& newCells);

    bool removeSurfaceEntries(QuadSurface* surface);

    static PatchHit evaluatePatch(const PatchRecord& rec, const cv::Vec3f& point) {
        PatchHit best;

        const auto& p00 = rec.corners[0];
        const auto& p10 = rec.corners[1];
        const auto& p11 = rec.corners[2];
        const auto& p01 = rec.corners[3];

        // Triangle 0: (p00, p10, p01)
        {
            TriangleHit tri = closestPointOnTriangle(point, p00, p10, p01);
            if (tri.distSq < best.distSq) {
                best.valid = true;
                best.distSq = tri.distSq;
                best.u = clamp01(tri.bary[1]);
                best.v = clamp01(tri.bary[2]);
            }
        }

        // Triangle 1: (p10, p11, p01)
        {
            TriangleHit tri = closestPointOnTriangle(point, p10, p11, p01);
            if (tri.distSq < best.distSq) {
                best.valid = true;
                best.distSq = tri.distSq;
                float u = clamp01(tri.bary[0] + tri.bary[1]);
                float v = clamp01(tri.bary[1] + tri.bary[2]);
                best.u = u;
                best.v = v;
            }
        }

        return best;
    }
};

SurfacePatchIndex::SurfacePatchIndex()
    : impl_(std::make_unique<Impl>())
{}

SurfacePatchIndex::~SurfacePatchIndex() = default;
SurfacePatchIndex::SurfacePatchIndex(SurfacePatchIndex&&) noexcept = default;
SurfacePatchIndex& SurfacePatchIndex::operator=(SurfacePatchIndex&&) noexcept = default;

std::vector<std::pair<CellKey, SurfacePatchIndex::Impl::CellEntry>>
SurfacePatchIndex::Impl::collectEntriesForSurface(QuadSurface* surface,
                                                  float bboxPadding,
                                                  int samplingStride,
                                                  int rowStart,
                                                  int rowEnd,
                                                  int colStart,
                                                  int colEnd)
{
    std::vector<std::pair<CellKey, CellEntry>> result;
    if (!surface) {
        return result;
    }
    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return result;
    }

    const int rows = points->rows;
    const int cols = points->cols;
    const int cellRowCount = rows - 1;
    const int cellColCount = cols - 1;
    if (cellRowCount <= 0 || cellColCount <= 0) {
        return result;
    }

    rowStart = std::max(0, rowStart);
    rowEnd = std::min(cellRowCount, rowEnd);
    colStart = std::max(0, colStart);
    colEnd = std::min(cellColCount, colEnd);
    if (rowStart >= rowEnd || colStart >= colEnd) {
        return result;
    }

    samplingStride = std::max(1, samplingStride);

    for (int j = rowStart; j < rowEnd; ++j) {
        if (!shouldSampleIndex(j, cellRowCount, samplingStride)) {
            continue;
        }
        for (int i = colStart; i < colEnd; ++i) {
            if (!shouldSampleIndex(i, cellColCount, samplingStride)) {
                continue;
            }

            CellEntry entry;
            if (!buildCellEntry(surface, *points, i, j, bboxPadding, entry)) {
                continue;
            }

            result.emplace_back(CellKey{surface, j, i}, std::move(entry));
        }
    }

    return result;
}

void SurfacePatchIndex::rebuild(const std::vector<QuadSurface*>& surfaces, float bboxPadding)
{
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    impl_->bboxPadding = bboxPadding;
    impl_->cellEntries.clear();
    impl_->patchCount = 0;
    impl_->triangleCount = 0;

    const size_t surfaceCount = surfaces.size();
    if (surfaceCount == 0) {
        impl_->tree.reset();
        impl_->triangleTree.reset();
        impl_->samplingStride = std::max(1, impl_->samplingStride);
        return;
    }

    impl_->samplingStride = std::max(1, impl_->samplingStride);

    std::vector<std::vector<std::pair<CellKey, Impl::CellEntry>>> cellsPerSurface(surfaceCount);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) if(surfaceCount > 1)
#endif
    for (int idx = 0; idx < static_cast<int>(surfaceCount); ++idx) {
        QuadSurface* surface = surfaces[idx];
        cellsPerSurface[idx] = Impl::collectEntriesForSurface(surface,
                                                              bboxPadding,
                                                              impl_->samplingStride,
                                                              0,
                                                              std::numeric_limits<int>::max(),
                                                              0,
                                                              std::numeric_limits<int>::max());
    }

    std::vector<Impl::Entry> entries;
    std::vector<Impl::TriangleEntry> triangleEntries;
    size_t estimatedPatches = 0;
    size_t estimatedTriangles = 0;
    for (const auto& cells : cellsPerSurface) {
        estimatedPatches += cells.size();
        for (const auto& cell : cells) {
            estimatedTriangles += cell.second.triangles.size();
        }
    }
    entries.reserve(estimatedPatches);
    triangleEntries.reserve(estimatedTriangles);

    for (auto& cells : cellsPerSurface) {
        for (auto& cell : cells) {
            if (cell.second.hasPatch) {
                entries.push_back(cell.second.patch);
            }
            for (const auto& tri : cell.second.triangles) {
                triangleEntries.push_back(tri);
            }
            impl_->cellEntries.emplace(cell.first, std::move(cell.second));
        }
    }

    impl_->patchCount = entries.size();
    impl_->triangleCount = triangleEntries.size();
    if (entries.empty()) {
        impl_->tree.reset();
    } else {
        impl_->tree = std::make_unique<Impl::PatchTree>(entries.begin(), entries.end());
    }

    if (triangleEntries.empty()) {
        impl_->triangleTree.reset();
    } else {
        impl_->triangleTree = std::make_unique<Impl::TriangleTree>(triangleEntries.begin(),
                                                                   triangleEntries.end());
    }
}

void SurfacePatchIndex::clear()
{
    if (impl_) {
        impl_->tree.reset();
        impl_->triangleTree.reset();
        impl_->patchCount = 0;
        impl_->bboxPadding = 0.0f;
        impl_->cellEntries.clear();
        impl_->samplingStride = 1;
        impl_->triangleCount = 0;
    }
}

bool SurfacePatchIndex::empty() const
{
    return !impl_ || !impl_->tree || impl_->patchCount == 0;
}

std::optional<SurfacePatchIndex::LookupResult>
SurfacePatchIndex::locate(const cv::Vec3f& worldPoint, float tolerance, QuadSurface* targetSurface) const
{
    if (!impl_ || !impl_->tree || tolerance <= 0.0f) {
        return std::nullopt;
    }

    const float tol = std::max(tolerance, 0.0f);
    Impl::Point3 min_pt(worldPoint[0] - tol, worldPoint[1] - tol, worldPoint[2] - tol);
    Impl::Point3 max_pt(worldPoint[0] + tol, worldPoint[1] + tol, worldPoint[2] + tol);
    Impl::Box3 query(min_pt, max_pt);

    std::vector<Impl::Entry> candidates;
    impl_->tree->query(bgi::intersects(query), std::back_inserter(candidates));

    const float toleranceSq = tol * tol;
    SurfacePatchIndex::LookupResult best;
    float bestDistSq = toleranceSq;
    bool found = false;

    for (const auto& entry : candidates) {
        const Impl::PatchRecord& rec = entry.second;
        if (targetSurface && rec.surface != targetSurface) {
            continue;
        }

        Impl::PatchHit hit = Impl::evaluatePatch(rec, worldPoint);
        if (!hit.valid || hit.distSq > bestDistSq) {
            continue;
        }

        const cv::Vec3f center = rec.surface->center();
        const cv::Vec2f scale = rec.surface->scale();
        const float absX = static_cast<float>(rec.i) + hit.u;
        const float absY = static_cast<float>(rec.j) + hit.v;
        cv::Vec3f ptr = {
            absX - center[0] * scale[0],
            absY - center[1] * scale[1],
            0.0f
        };

        best.surface = rec.surface;
        best.ptr = ptr;
        best.distance = std::sqrt(hit.distSq);
        bestDistSq = hit.distSq;
        found = true;
    }

    if (!found) {
        return std::nullopt;
    }

    return best;
}

void SurfacePatchIndex::queryBox(const Rect3D& bounds,
                                 QuadSurface* targetSurface,
                                 std::vector<PatchCandidate>& outCandidates) const
{
    outCandidates.clear();
    if (!impl_ || !impl_->tree) {
        return;
    }

    Impl::Point3 min_pt(bounds.low[0], bounds.low[1], bounds.low[2]);
    Impl::Point3 max_pt(bounds.high[0], bounds.high[1], bounds.high[2]);
    Impl::Box3 query(min_pt, max_pt);

    std::vector<Impl::Entry> candidates;
    impl_->tree->query(bgi::intersects(query), std::back_inserter(candidates));

    outCandidates.reserve(outCandidates.size() + candidates.size());
    for (const auto& entry : candidates) {
        const Impl::PatchRecord& rec = entry.second;
        if (targetSurface && rec.surface != targetSurface) {
            continue;
        }

        PatchCandidate candidate;
        candidate.surface = rec.surface;
        candidate.i = rec.i;
        candidate.j = rec.j;
        candidate.corners = rec.corners;
        outCandidates.push_back(candidate);
    }
}

void SurfacePatchIndex::queryTriangles(const Rect3D& bounds,
                                       QuadSurface* targetSurface,
                                       std::vector<TriangleCandidate>& outCandidates) const
{
    outCandidates.clear();
    if (!impl_ || !impl_->triangleTree) {
        return;
    }

    Impl::Point3 min_pt(bounds.low[0], bounds.low[1], bounds.low[2]);
    Impl::Point3 max_pt(bounds.high[0], bounds.high[1], bounds.high[2]);
    Impl::Box3 query(min_pt, max_pt);

    std::vector<Impl::TriangleEntry> candidates;
    impl_->triangleTree->query(bgi::intersects(query), std::back_inserter(candidates));

    outCandidates.reserve(outCandidates.size() + candidates.size());
    for (const auto& entry : candidates) {
        const Impl::TriangleRecord& rec = entry.second;
        if (targetSurface && rec.surface != targetSurface) {
            continue;
        }

        TriangleCandidate candidate;
        candidate.surface = rec.surface;
        candidate.i = rec.i;
        candidate.j = rec.j;
        candidate.triangleIndex = rec.triangleIndex;
        candidate.world = rec.world;
        candidate.surfaceParams = rec.surfaceParams;
        outCandidates.push_back(candidate);
    }
}

bool SurfacePatchIndex::Impl::removeSurfaceEntries(QuadSurface* surface)
{
    if (!surface) {
        return false;
    }

    std::vector<CellKey> keys;
    keys.reserve(cellEntries.size());
    for (const auto& entry : cellEntries) {
        if (entry.first.surface == surface) {
            keys.push_back(entry.first);
        }
    }

    if (keys.empty()) {
        return false;
    }

    for (const auto& key : keys) {
        removeCellEntry(key);
    }

    if (tree && patchCount == 0) {
        tree.reset();
    }
    if (triangleTree && triangleCount == 0) {
        triangleTree.reset();
    }

    return true;
}

bool SurfacePatchIndex::Impl::replaceSurfaceEntries(
    QuadSurface* surface,
    std::vector<std::pair<CellKey, CellEntry>>&& newCells)
{
    if (!surface) {
        return false;
    }

    removeSurfaceEntries(surface);
    insertCells(newCells);
    return !newCells.empty();
}

namespace {
struct IntersectionEndpoint {
    cv::Vec3f world;
    cv::Vec3f param;
};

bool pointsApproximatelyEqual(const cv::Vec3f& a, const cv::Vec3f& b, float epsilon)
{
    return cv::norm(a - b) <= epsilon;
}
} // namespace

std::optional<SurfacePatchIndex::TriangleSegment>
SurfacePatchIndex::clipTriangleToPlane(const TriangleCandidate& tri,
                                       const PlaneSurface& plane,
                                       float epsilon)
{
    std::array<float, 3> distances{};
    int positive = 0;
    int negative = 0;
    int onPlane = 0;

    for (size_t idx = 0; idx < tri.world.size(); ++idx) {
        float d = plane.scalarp(tri.world[idx]);
        distances[idx] = d;
        if (d > epsilon) {
            ++positive;
        } else if (d < -epsilon) {
            ++negative;
        } else {
            ++onPlane;
        }
    }

    if (positive == 0 && negative == 0 && onPlane == 0) {
        return std::nullopt;
    }

    if ((positive == 0 && negative == 0) && onPlane == 3) {
        // Triangle lies entirely on plane; fall through to treat edges as intersection.
    } else if (positive == 0 && negative == 0 && onPlane == 0) {
        return std::nullopt;
    } else if (positive == 0 && negative == 0 && onPlane == 1) {
        return std::nullopt;
    } else if (positive == 0 && negative == 0 && onPlane == 2) {
        // Edge on the plane; vertices already counted below.
    } else if (positive == 0 || negative == 0) {
        // Triangle is fully on one side of plane.
        return std::nullopt;
    }

    std::vector<IntersectionEndpoint> endpoints;
    const float mergeDistance = epsilon * 4.0f;

    auto addEndpoint = [&](const cv::Vec3f& world, const cv::Vec3f& param) {
        for (const auto& existing : endpoints) {
            if (pointsApproximatelyEqual(existing.world, world, mergeDistance)) {
                return;
            }
        }
        endpoints.push_back({world, param});
    };

    auto addVertexIfOnPlane = [&](int idx) {
        if (std::abs(distances[idx]) <= epsilon) {
            addEndpoint(tri.world[idx], tri.surfaceParams[idx]);
        }
    };

    auto addEdgeIntersection = [&](int a, int b) {
        float da = distances[a];
        float db = distances[b];

        if ((da > epsilon && db > epsilon) || (da < -epsilon && db < -epsilon)) {
            return;
        }

        if (std::abs(da) <= epsilon && std::abs(db) <= epsilon) {
            addEndpoint(tri.world[a], tri.surfaceParams[a]);
            addEndpoint(tri.world[b], tri.surfaceParams[b]);
            return;
        }

        if ((da > epsilon && db < -epsilon) || (da < -epsilon && db > epsilon)) {
            const float denom = da - db;
            if (std::abs(denom) <= std::numeric_limits<float>::epsilon()) {
                return;
            }
            const float t = da / denom;
            cv::Vec3f world = tri.world[a] + t * (tri.world[b] - tri.world[a]);
            cv::Vec3f param = tri.surfaceParams[a] + t * (tri.surfaceParams[b] - tri.surfaceParams[a]);
            addEndpoint(world, param);
        } else if (std::abs(da) <= epsilon) {
            addEndpoint(tri.world[a], tri.surfaceParams[a]);
        } else if (std::abs(db) <= epsilon) {
            addEndpoint(tri.world[b], tri.surfaceParams[b]);
        }
    };

    addVertexIfOnPlane(0);
    addVertexIfOnPlane(1);
    addVertexIfOnPlane(2);
    addEdgeIntersection(0, 1);
    addEdgeIntersection(1, 2);
    addEdgeIntersection(2, 0);

    if (endpoints.size() < 2) {
        return std::nullopt;
    }

    if (endpoints.size() > 2) {
        float bestDist = -1.0f;
        std::pair<size_t, size_t> bestPair = {0, 1};
        for (size_t a = 0; a < endpoints.size(); ++a) {
            for (size_t b = a + 1; b < endpoints.size(); ++b) {
                float dist = cv::norm(endpoints[a].world - endpoints[b].world);
                if (dist > bestDist) {
                    bestDist = dist;
                    bestPair = {a, b};
                }
            }
        }
        IntersectionEndpoint first = endpoints[bestPair.first];
        IntersectionEndpoint second = endpoints[bestPair.second];
        endpoints.clear();
        endpoints.push_back(first);
        endpoints.push_back(second);
    }

    TriangleSegment segment;
    segment.surface = tri.surface;
    segment.world = {endpoints[0].world, endpoints[1].world};
    segment.surfaceParams = {endpoints[0].param, endpoints[1].param};
    return segment;
}

bool SurfacePatchIndex::updateSurface(QuadSurface* surface)
{
    if (!impl_ || !surface) {
        return false;
    }
    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->rows < 2 || points->cols < 2) {
        return false;
    }

    auto cells = Impl::collectEntriesForSurface(surface,
                                                impl_->bboxPadding,
                                                impl_->samplingStride,
                                                0,
                                                points->rows - 1,
                                                0,
                                                points->cols - 1);
    return impl_->replaceSurfaceEntries(surface, std::move(cells));
}

bool SurfacePatchIndex::updateSurfaceRegion(QuadSurface* surface,
                                            int rowStart,
                                            int rowEnd,
                                            int colStart,
                                            int colEnd)
{
    if (!impl_ || !surface) {
        return false;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->rows < 2 || points->cols < 2) {
        return false;
    }

    const int cellRowCount = points->rows - 1;
    const int cellColCount = points->cols - 1;
    rowStart = std::max(0, rowStart);
    rowEnd = std::min(cellRowCount, rowEnd);
    colStart = std::max(0, colStart);
    colEnd = std::min(cellColCount, colEnd);
    if (rowStart >= rowEnd || colStart >= colEnd) {
        return false;
    }

    impl_->removeCells(surface, rowStart, rowEnd, colStart, colEnd);
    auto cells = Impl::collectEntriesForSurface(surface,
                                                impl_->bboxPadding,
                                                impl_->samplingStride,
                                                rowStart,
                                                rowEnd,
                                                colStart,
                                                colEnd);
    impl_->insertCells(cells);
    return !cells.empty();
}

bool SurfacePatchIndex::removeSurface(QuadSurface* surface)
{
    if (!impl_ || !surface) {
        return false;
    }
    return impl_->removeSurfaceEntries(surface);
}

bool SurfacePatchIndex::setSamplingStride(int stride)
{
    stride = std::max(1, stride);
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    if (impl_->samplingStride == stride) {
        return false;
    }
    impl_->samplingStride = stride;
    impl_->tree.reset();
    impl_->triangleTree.reset();
    impl_->cellEntries.clear();
    impl_->patchCount = 0;
    impl_->triangleCount = 0;
    return true;
}

int SurfacePatchIndex::samplingStride() const
{
    if (!impl_) {
        return 1;
    }
    return std::max(1, impl_->samplingStride);
}
bool SurfacePatchIndex::Impl::buildCellEntry(QuadSurface* surface,
                                             const cv::Mat_<cv::Vec3f>& points,
                                             int col,
                                             int row,
                                             float bboxPadding,
                                             CellEntry& outEntry)
{
    const cv::Vec3f& p00 = points(row, col);
    const cv::Vec3f& p10 = points(row, col + 1);
    const cv::Vec3f& p01 = points(row + 1, col);
    const cv::Vec3f& p11 = points(row + 1, col + 1);

    if (p00[0] == -1.0f || p10[0] == -1.0f || p01[0] == -1.0f || p11[0] == -1.0f) {
        return false;
    }

    PatchRecord rec;
    rec.surface = surface;
    rec.i = col;
    rec.j = row;
    rec.corners = {p00, p10, p11, p01};

    cv::Vec3f low = rec.corners[0];
    cv::Vec3f high = rec.corners[0];
    for (const cv::Vec3f& c : rec.corners) {
        low[0] = std::min(low[0], c[0]);
        low[1] = std::min(low[1], c[1]);
        low[2] = std::min(low[2], c[2]);
        high[0] = std::max(high[0], c[0]);
        high[1] = std::max(high[1], c[1]);
        high[2] = std::max(high[2], c[2]);
    }

    if (bboxPadding > 0.0f) {
        low -= cv::Vec3f(bboxPadding, bboxPadding, bboxPadding);
        high += cv::Vec3f(bboxPadding, bboxPadding, bboxPadding);
    }

    Point3 min_pt(low[0], low[1], low[2]);
    Point3 max_pt(high[0], high[1], high[2]);

    outEntry.patch = std::make_pair(Box3(min_pt, max_pt), rec);
    outEntry.hasPatch = true;
    outEntry.triangles.clear();
    outEntry.triangles.reserve(2);

    auto addTriangle = [&](const std::array<cv::Vec3f, 3>& worldPts,
                           const std::array<cv::Vec2f, 3>& gridPts,
                           int triIndex) {
        TriangleRecord triRec;
        triRec.surface = surface;
        triRec.i = col;
        triRec.j = row;
        triRec.triangleIndex = triIndex;
        triRec.world = worldPts;
        for (size_t v = 0; v < triRec.surfaceParams.size(); ++v) {
            const float absX = gridPts[v][0];
            const float absY = gridPts[v][1];
            triRec.surfaceParams[v] = makePtrFromAbsCoord(surface, absX, absY);
        }

        cv::Vec3f triLow = worldPts[0];
        cv::Vec3f triHigh = worldPts[0];
        for (const cv::Vec3f& pt : worldPts) {
            triLow[0] = std::min(triLow[0], pt[0]);
            triLow[1] = std::min(triLow[1], pt[1]);
            triLow[2] = std::min(triLow[2], pt[2]);
            triHigh[0] = std::max(triHigh[0], pt[0]);
            triHigh[1] = std::max(triHigh[1], pt[1]);
            triHigh[2] = std::max(triHigh[2], pt[2]);
        }

        Point3 triMin(triLow[0], triLow[1], triLow[2]);
        Point3 triMax(triHigh[0], triHigh[1], triHigh[2]);
        outEntry.triangles.emplace_back(Box3(triMin, triMax), triRec);
    };

    addTriangle(
        {p00, p10, p01},
        {cv::Vec2f(static_cast<float>(col), static_cast<float>(row)),
         cv::Vec2f(static_cast<float>(col + 1), static_cast<float>(row)),
         cv::Vec2f(static_cast<float>(col), static_cast<float>(row + 1))},
        0
    );

    addTriangle(
        {p10, p11, p01},
        {cv::Vec2f(static_cast<float>(col + 1), static_cast<float>(row)),
         cv::Vec2f(static_cast<float>(col + 1), static_cast<float>(row + 1)),
         cv::Vec2f(static_cast<float>(col), static_cast<float>(row + 1))},
        1
    );

    return true;
}

void SurfacePatchIndex::Impl::removeCellEntry(const CellKey& key)
{
    auto it = cellEntries.find(key);
    if (it == cellEntries.end()) {
        return;
    }

    if (tree && it->second.hasPatch) {
        tree->remove(it->second.patch);
        if (patchCount > 0) {
            --patchCount;
        }
    }

    if (triangleTree) {
        for (const auto& tri : it->second.triangles) {
            triangleTree->remove(tri);
            if (triangleCount > 0) {
                --triangleCount;
            }
        }
    }

    cellEntries.erase(it);
}

void SurfacePatchIndex::Impl::insertCells(const std::vector<std::pair<CellKey, CellEntry>>& cells)
{
    for (const auto& cell : cells) {
        if (cell.second.hasPatch) {
            if (!tree) {
                tree = std::make_unique<PatchTree>();
            }
            tree->insert(cell.second.patch);
            ++patchCount;
        }

        if (!cell.second.triangles.empty()) {
            if (!triangleTree) {
                triangleTree = std::make_unique<TriangleTree>();
            }
            for (const auto& tri : cell.second.triangles) {
                triangleTree->insert(tri);
                ++triangleCount;
            }
        }

        cellEntries[cell.first] = cell.second;
    }
}

void SurfacePatchIndex::Impl::removeCells(QuadSurface* surface,
                                          int rowStart,
                                          int rowEnd,
                                          int colStart,
                                          int colEnd)
{
    if (!surface) {
        return;
    }
    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->rows < 2 || points->cols < 2) {
        return;
    }

    const int cellRowCount = points->rows - 1;
    const int cellColCount = points->cols - 1;

    rowStart = std::max(0, rowStart);
    rowEnd = std::min(cellRowCount, rowEnd);
    colStart = std::max(0, colStart);
    colEnd = std::min(cellColCount, colEnd);

    if (rowStart >= rowEnd || colStart >= colEnd) {
        return;
    }

    std::vector<CellKey> keys;
    keys.reserve(static_cast<size_t>(rowEnd - rowStart) * static_cast<size_t>(colEnd - colStart));

    for (int row = rowStart; row < rowEnd; ++row) {
        if (!shouldSampleIndex(row, cellRowCount, samplingStride)) {
            continue;
        }
        for (int col = colStart; col < colEnd; ++col) {
            if (!shouldSampleIndex(col, cellColCount, samplingStride)) {
                continue;
            }
            CellKey key{surface, row, col};
            if (cellEntries.find(key) != cellEntries.end()) {
                keys.push_back(key);
            }
        }
    }

    for (const auto& key : keys) {
        removeCellEntry(key);
    }

    if (tree && patchCount == 0) {
        tree.reset();
    }
    if (triangleTree && triangleCount == 0) {
        triangleTree.reset();
    }
}
