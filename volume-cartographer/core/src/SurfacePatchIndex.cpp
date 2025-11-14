#include "vc/core/util/SurfacePatchIndex.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>
#include <unordered_map>

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/iterator/function_output_iterator.hpp>

#include "vc/core/util/Surface.hpp"


namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace {

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
    std::uint64_t packed = 0;

    CellKey() = default;
    CellKey(QuadSurface* surf, int rowIndex, int colIndex)
        : surface(surf),
          packed(pack(rowIndex, colIndex))
    {}

    static std::uint64_t pack(int rowIndex, int colIndex) noexcept
    {
        auto r = static_cast<std::uint64_t>(static_cast<std::uint32_t>(rowIndex));
        auto c = static_cast<std::uint64_t>(static_cast<std::uint32_t>(colIndex));
        return (r << 32) | c;
    }

    int rowIndex() const noexcept
    {
        return static_cast<int>(packed >> 32);
    }

    int colIndex() const noexcept
    {
        return static_cast<int>(packed & 0xffffffffULL);
    }

    std::uint64_t packedIndex() const noexcept
    {
        return packed;
    }

    bool operator==(const CellKey& other) const noexcept
    {
        return surface == other.surface && packed == other.packed;
    }
};

} // namespace

struct SurfacePatchIndex::Impl {
    struct PatchRecord {
        QuadSurface* surface = nullptr;
        int i = 0;
        int j = 0;

        bool operator==(const PatchRecord& other) const noexcept {
            return surface == other.surface &&
                   i == other.i &&
                   j == other.j;
        }
    };

    struct TriangleRecord {
        QuadSurface* surface = nullptr;
        int i = 0;
        int j = 0;
        int triangleIndex = 0;

        bool operator==(const TriangleRecord& other) const noexcept {
            return surface == other.surface &&
                   i == other.i &&
                   j == other.j &&
                   triangleIndex == other.triangleIndex;
        }
    };

    using Point3 = bg::model::point<float, 3, bg::cs::cartesian>;
    using Box3 = bg::model::box<Point3>;
    using Entry = std::pair<Box3, PatchRecord>;
    using PatchTree = bgi::rtree<Entry, bgi::quadratic<32>>;

    std::unique_ptr<PatchTree> tree;
    struct CellEntry {
        bool hasPatch = false;
        Entry patch;
    };

    struct SurfaceCellMask {
        int rows = 0;
        int cols = 0;
        int activeCount = 0;
        std::vector<uint8_t> states;
        std::unordered_map<std::size_t, Entry> cachedEntries;

        void clear()
        {
            rows = 0;
            cols = 0;
            activeCount = 0;
            states.clear();
            cachedEntries.clear();
        }

        void ensureSize(int rowCount, int colCount)
        {
            rowCount = std::max(rowCount, 0);
            colCount = std::max(colCount, 0);
            const std::size_t required = static_cast<std::size_t>(rowCount) * colCount;
            if (rowCount <= 0 || colCount <= 0) {
                clear();
                return;
            }
            if (rows == rowCount && cols == colCount && states.size() == required) {
                return;
            }
            rows = rowCount;
            cols = colCount;
            activeCount = 0;
            states.assign(required, 0);
            cachedEntries.clear();
        }

        bool validIndex(int row, int col) const
        {
            return row >= 0 && row < rows && col >= 0 && col < cols;
        }

        std::size_t index(int row, int col) const
        {
            return static_cast<std::size_t>(row) * cols + col;
        }

        bool isActive(int row, int col) const
        {
            if (!validIndex(row, col)) {
                return false;
            }
            return states[index(row, col)] != 0;
        }

        void setActive(int row, int col, bool active)
        {
            if (!validIndex(row, col)) {
                return;
            }
            const std::size_t idx = index(row, col);
            const uint8_t next = active ? 1u : 0u;
            const uint8_t prev = states[idx];
            if (prev == next) {
                return;
            }
            states[idx] = next;
            activeCount += active ? 1 : -1;
        }

        Entry* entryAt(int row, int col)
        {
            if (!validIndex(row, col)) {
                return nullptr;
            }
            auto it = cachedEntries.find(index(row, col));
            if (it == cachedEntries.end()) {
                return nullptr;
            }
            return &it->second;
        }

        const Entry* entryAt(int row, int col) const
        {
            if (!validIndex(row, col)) {
                return nullptr;
            }
            auto it = cachedEntries.find(index(row, col));
            if (it == cachedEntries.end()) {
                return nullptr;
            }
            return &it->second;
        }

        void storeEntry(int row, int col, const Entry& entry)
        {
            if (!validIndex(row, col)) {
                return;
            }
            cachedEntries[index(row, col)] = entry;
        }

        void eraseEntry(int row, int col)
        {
            if (!validIndex(row, col)) {
                return;
            }
            cachedEntries.erase(index(row, col));
        }

        bool empty() const
        {
            return activeCount == 0;
        }
    };

    size_t patchCount = 0;
    float bboxPadding = 0.0f;
    int samplingStride = 1;

    std::unordered_map<QuadSurface*, SurfaceCellMask> surfaceCellRecords;

    SurfaceCellMask& ensureMask(QuadSurface* surface);

    std::optional<Entry> makePatchEntry(const CellKey& key) const;

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
    static bool loadPatchCorners(const PatchRecord& rec,
                                 std::array<cv::Vec3f, 4>& outCorners);
    static bool loadTriangleGeometry(const TriangleRecord& rec,
                                     std::array<cv::Vec3f, 3>& world,
                                     std::array<cv::Vec3f, 3>& surfaceParams);
    void removeCellEntry(SurfaceCellMask& mask,
                         QuadSurface* surface,
                         int row,
                         int col);
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

        std::array<cv::Vec3f, 4> corners;
        if (!loadPatchCorners(rec, corners)) {
            return best;
        }

        const auto& p00 = corners[0];
        const auto& p10 = corners[1];
        const auto& p11 = corners[2];
        const auto& p01 = corners[3];

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

            result.emplace_back(CellKey(surface, j, i), std::move(entry));
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
    impl_->surfaceCellRecords.clear();
    impl_->patchCount = 0;
    impl_->surfaceCellRecords.clear();

    const size_t surfaceCount = surfaces.size();
    if (surfaceCount == 0) {
        impl_->tree.reset();
        impl_->samplingStride = std::max(1, impl_->samplingStride);
        return;
    }

    impl_->samplingStride = std::max(1, impl_->samplingStride);

    std::vector<Impl::Entry> entries;
    for (QuadSurface* surface : surfaces) {
        auto cells = Impl::collectEntriesForSurface(surface,
                                                    bboxPadding,
                                                    impl_->samplingStride,
                                                    0,
                                                    std::numeric_limits<int>::max(),
                                                    0,
                                                    std::numeric_limits<int>::max());

        entries.reserve(entries.size() + cells.size());

        for (auto& cell : cells) {
            if (cell.second.hasPatch) {
                entries.push_back(cell.second.patch);
            }
            auto& mask = impl_->ensureMask(cell.first.surface);
            mask.setActive(cell.first.rowIndex(), cell.first.colIndex(), cell.second.hasPatch);
        }
    }

    impl_->patchCount = entries.size();
    if (entries.empty()) {
        impl_->tree.reset();
    } else {
        impl_->tree = std::make_unique<Impl::PatchTree>(entries.begin(), entries.end());
    }
}

void SurfacePatchIndex::clear()
{
    if (impl_) {
        impl_->tree.reset();
        impl_->patchCount = 0;
        impl_->bboxPadding = 0.0f;
        impl_->surfaceCellRecords.clear();
        impl_->samplingStride = 1;
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

    const float toleranceSq = tol * tol;
    SurfacePatchIndex::LookupResult best;
    float bestDistSq = toleranceSq;
    bool found = false;
    struct SurfaceInfo {
        cv::Vec3f center;
        cv::Vec2f scale;
    };
    std::unordered_map<QuadSurface*, SurfaceInfo> surfaceInfoCache;
    surfaceInfoCache.reserve(4);
    auto ensureSurfaceInfo = [&](QuadSurface* surface) -> const SurfaceInfo& {
        auto it = surfaceInfoCache.find(surface);
        if (it != surfaceInfoCache.end()) {
            return it->second;
        }
        SurfaceInfo info{surface->center(), surface->scale()};
        auto [insertIt, _] = surfaceInfoCache.emplace(surface, info);
        return insertIt->second;
    };

    auto processEntry = [&](const Impl::Entry& entry) {
        const Impl::PatchRecord& rec = entry.second;
        if (targetSurface && rec.surface != targetSurface) {
            return;
        }

        Impl::PatchHit hit = Impl::evaluatePatch(rec, worldPoint);
        if (!hit.valid || hit.distSq > bestDistSq) {
            return;
        }

        const SurfaceInfo& info = ensureSurfaceInfo(rec.surface);
        const float absX = static_cast<float>(rec.i) + hit.u;
        const float absY = static_cast<float>(rec.j) + hit.v;
        cv::Vec3f ptr = {
            absX - info.center[0] * info.scale[0],
            absY - info.center[1] * info.scale[1],
            0.0f
        };

        best.surface = rec.surface;
        best.ptr = ptr;
        bestDistSq = hit.distSq;
        found = true;
    };

    impl_->tree->query(
        bgi::intersects(query),
        boost::make_function_output_iterator(processEntry));

    if (!found) {
        return std::nullopt;
    }

    best.distance = std::sqrt(bestDistSq);
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

    auto emitCandidate = [&](const Impl::Entry& entry) {
        const Impl::PatchRecord& rec = entry.second;
        if (targetSurface && rec.surface != targetSurface) {
            return;
        }

        PatchCandidate candidate;
        candidate.surface = rec.surface;
        candidate.i = rec.i;
        candidate.j = rec.j;
        if (!Impl::loadPatchCorners(rec, candidate.corners)) {
            return;
        }
        outCandidates.push_back(std::move(candidate));
    };

    impl_->tree->query(bgi::intersects(query),
                       boost::make_function_output_iterator(emitCandidate));
}

void SurfacePatchIndex::queryTriangles(const Rect3D& bounds,
                                       QuadSurface* targetSurface,
                                       std::vector<TriangleCandidate>& outCandidates) const
{
    outCandidates.clear();
    forEachTriangle(bounds, targetSurface, [&](const TriangleCandidate& candidate) {
        outCandidates.push_back(candidate);
    });
}

void SurfacePatchIndex::forEachTriangle(const Rect3D& bounds,
                                        QuadSurface* targetSurface,
                                        const std::function<void(const TriangleCandidate&)>& visitor) const
{
    forEachTriangleImpl(bounds, targetSurface, nullptr, visitor);
}

void SurfacePatchIndex::forEachTriangle(const Rect3D& bounds,
                                        const std::unordered_set<QuadSurface*>& targetSurfaces,
                                        const std::function<void(const TriangleCandidate&)>& visitor) const
{
    if (targetSurfaces.empty()) {
        return;
    }
    forEachTriangleImpl(bounds, nullptr, &targetSurfaces, visitor);
}

void SurfacePatchIndex::forEachTriangleImpl(
    const Rect3D& bounds,
    QuadSurface* targetSurface,
    const std::unordered_set<QuadSurface*>* filterSurfaces,
    const std::function<void(const TriangleCandidate&)>& visitor) const
{
    if (!visitor || !impl_ || !impl_->tree) {
        return;
    }

    Impl::Point3 min_pt(bounds.low[0], bounds.low[1], bounds.low[2]);
    Impl::Point3 max_pt(bounds.high[0], bounds.high[1], bounds.high[2]);
    Impl::Box3 query(min_pt, max_pt);

    auto emitFromPatch = [&](const Impl::Entry& entry) {
        const Impl::PatchRecord& rec = entry.second;
        if (targetSurface && rec.surface != targetSurface) {
            return;
        }
        if (filterSurfaces && filterSurfaces->find(rec.surface) == filterSurfaces->end()) {
            return;
        }

        Impl::TriangleRecord triRec;
        triRec.surface = rec.surface;
        triRec.i = rec.i;
        triRec.j = rec.j;

        for (int triIdx = 0; triIdx < 2; ++triIdx) {
            TriangleCandidate candidate;
            candidate.surface = rec.surface;
            candidate.i = rec.i;
            candidate.j = rec.j;
            candidate.triangleIndex = triIdx;
            triRec.triangleIndex = triIdx;
            if (!Impl::loadTriangleGeometry(triRec, candidate.world, candidate.surfaceParams)) {
                continue;
            }

            Rect3D triBounds;
            triBounds.low = candidate.world[0];
            triBounds.high = candidate.world[0];
            triBounds = expand_rect(triBounds, candidate.world[1]);
            triBounds = expand_rect(triBounds, candidate.world[2]);
            if (!intersect(bounds, triBounds)) {
                continue;
            }

            visitor(candidate);
        }
    };

    impl_->tree->query(bgi::intersects(query),
                       boost::make_function_output_iterator(emitFromPatch));
}

bool SurfacePatchIndex::Impl::removeSurfaceEntries(QuadSurface* surface)
{
    if (!surface) {
        return false;
    }

    auto it = surfaceCellRecords.find(surface);
    if (it == surfaceCellRecords.end() || it->second.empty()) {
        return false;
    }

    SurfaceCellMask& mask = it->second;
    for (int row = 0; row < mask.rows; ++row) {
        for (int col = 0; col < mask.cols; ++col) {
            if (mask.isActive(row, col)) {
                removeCellEntry(mask, surface, row, col);
            }
        }
    }
    surfaceCellRecords.erase(it);

    if (tree && patchCount == 0) {
        tree.reset();
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

    std::array<IntersectionEndpoint, 6> endpoints{};
    size_t endpointCount = 0;
    const float mergeDistance = epsilon * 4.0f;

    auto addEndpoint = [&](const cv::Vec3f& world, const cv::Vec3f& param) {
        for (size_t idx = 0; idx < endpointCount; ++idx) {
            if (pointsApproximatelyEqual(endpoints[idx].world, world, mergeDistance)) {
                return;
            }
        }
        if (endpointCount < endpoints.size()) {
            endpoints[endpointCount++] = {world, param};
        }
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

    if (endpointCount < 2) {
        return std::nullopt;
    }

    if (endpointCount > 2) {
        float bestDist = -1.0f;
        std::pair<size_t, size_t> bestPair = {0, 1};
        for (size_t a = 0; a < endpointCount; ++a) {
            for (size_t b = a + 1; b < endpointCount; ++b) {
                float dist = cv::norm(endpoints[a].world - endpoints[b].world);
                if (dist > bestDist) {
                    bestDist = dist;
                    bestPair = {a, b};
                }
            }
        }
        IntersectionEndpoint first = endpoints[bestPair.first];
        IntersectionEndpoint second = endpoints[bestPair.second];
        endpoints[0] = first;
        endpoints[1] = second;
        endpointCount = 2;
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
    int samplingStride = impl_->samplingStride;
    const int rowSpan = rowEnd - rowStart;
    const int colSpan = colEnd - colStart;
    if (samplingStride > 1 && (rowSpan < samplingStride || colSpan < samplingStride)) {
        // Small dirty regions can otherwise end up deleting sampled cells without
        // re-inserting replacements because the stride skips every local index.
        samplingStride = 1;
    }
    auto cells = Impl::collectEntriesForSurface(surface,
                                                impl_->bboxPadding,
                                                samplingStride,
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
    impl_->surfaceCellRecords.clear();
    impl_->patchCount = 0;
    return true;
}

int SurfacePatchIndex::samplingStride() const
{
    if (!impl_) {
        return 1;
    }
    return std::max(1, impl_->samplingStride);
}

std::optional<SurfacePatchIndex::Impl::Entry>
SurfacePatchIndex::Impl::makePatchEntry(const CellKey& key) const
{
    if (!key.surface) {
        return std::nullopt;
    }

    PatchRecord rec;
    rec.surface = key.surface;
    rec.i = key.colIndex();
    rec.j = key.rowIndex();

    std::array<cv::Vec3f, 4> corners;
    if (!loadPatchCorners(rec, corners)) {
        return std::nullopt;
    }

    cv::Vec3f low = corners[0];
    cv::Vec3f high = corners[0];
    for (const cv::Vec3f& c : corners) {
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
    return Entry(Box3(min_pt, max_pt), rec);
}

SurfacePatchIndex::Impl::SurfaceCellMask&
SurfacePatchIndex::Impl::ensureMask(QuadSurface* surface)
{
    auto& mask = surfaceCellRecords[surface];
    const cv::Mat_<cv::Vec3f>* points = surface ? surface->rawPointsPtr() : nullptr;
    const int rowCount = points ? std::max(0, points->rows - 1) : 0;
    const int colCount = points ? std::max(0, points->cols - 1) : 0;
    mask.ensureSize(rowCount, colCount);
    return mask;
}

bool SurfacePatchIndex::Impl::loadPatchCorners(const PatchRecord& rec,
                                               std::array<cv::Vec3f, 4>& outCorners)
{
    if (!rec.surface) {
        return false;
    }
    const cv::Mat_<cv::Vec3f>* points = rec.surface->rawPointsPtr();
    if (!points) {
        return false;
    }
    const int rows = points->rows;
    const int cols = points->cols;
    if (rows < 2 || cols < 2) {
        return false;
    }

    const int row = rec.j;
    const int col = rec.i;
    if (row < 0 || col < 0 || row + 1 >= rows || col + 1 >= cols) {
        return false;
    }

    const cv::Vec3f& p00 = (*points)(row, col);
    const cv::Vec3f& p10 = (*points)(row, col + 1);
    const cv::Vec3f& p01 = (*points)(row + 1, col);
    const cv::Vec3f& p11 = (*points)(row + 1, col + 1);

    if (p00[0] == -1.0f || p10[0] == -1.0f || p01[0] == -1.0f || p11[0] == -1.0f) {
        return false;
    }

    outCorners = {p00, p10, p11, p01};
    return true;
}

bool SurfacePatchIndex::Impl::loadTriangleGeometry(const TriangleRecord& rec,
                                                   std::array<cv::Vec3f, 3>& world,
                                                   std::array<cv::Vec3f, 3>& surfaceParams)
{
    if (!rec.surface) {
        return false;
    }

    PatchRecord patch{rec.surface, rec.i, rec.j};
    std::array<cv::Vec3f, 4> corners;
    if (!loadPatchCorners(patch, corners)) {
        return false;
    }

    const float baseX = static_cast<float>(rec.i);
    const float baseY = static_cast<float>(rec.j);

    auto makeParam = [&](float dx, float dy) {
        return makePtrFromAbsCoord(rec.surface, baseX + dx, baseY + dy);
    };

    if (rec.triangleIndex == 0) {
        world = {corners[0], corners[1], corners[3]};
        surfaceParams = {
            makeParam(0.0f, 0.0f),
            makeParam(1.0f, 0.0f),
            makeParam(0.0f, 1.0f)
        };
    } else if (rec.triangleIndex == 1) {
        world = {corners[1], corners[2], corners[3]};
        surfaceParams = {
            makeParam(1.0f, 0.0f),
            makeParam(1.0f, 1.0f),
            makeParam(0.0f, 1.0f)
        };
    } else {
        return false;
    }

    return true;
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

    std::array<cv::Vec3f, 4> corners = {p00, p10, p11, p01};
    cv::Vec3f low = corners[0];
    cv::Vec3f high = corners[0];
    for (const cv::Vec3f& c : corners) {
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

    return true;
}

void SurfacePatchIndex::Impl::removeCellEntry(SurfaceCellMask& mask,
                                              QuadSurface* surface,
                                              int row,
                                              int col)
{
    if (!surface || !mask.isActive(row, col)) {
        mask.eraseEntry(row, col);
        return;
    }

    bool removed = false;
    if (tree) {
        const Entry* cachedEntry = mask.entryAt(row, col);
        if (cachedEntry) {
            removed = tree->remove(*cachedEntry);
        } else {
            if (auto entry = makePatchEntry(CellKey(surface, row, col))) {
                removed = tree->remove(*entry);
            }
        }
        if (removed && patchCount > 0) {
            --patchCount;
        }
    }

    mask.setActive(row, col, false);
    mask.eraseEntry(row, col);
}

void SurfacePatchIndex::Impl::insertCells(const std::vector<std::pair<CellKey, CellEntry>>& cells)
{
    for (const auto& cell : cells) {
        QuadSurface* surface = cell.first.surface;
        if (!surface) {
            continue;
        }
        auto& mask = ensureMask(surface);
        const int row = cell.first.rowIndex();
        const int col = cell.first.colIndex();

        if (cell.second.hasPatch) {
            if (!tree) {
                tree = std::make_unique<PatchTree>();
            }
            tree->insert(cell.second.patch);
            ++patchCount;
            mask.setActive(row, col, true);
            mask.storeEntry(row, col, cell.second.patch);
        } else {
            mask.setActive(row, col, false);
            mask.eraseEntry(row, col);
        }
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
    auto surfaceIt = surfaceCellRecords.find(surface);
    if (surfaceIt == surfaceCellRecords.end() || surfaceIt->second.empty()) {
        return;
    }

    SurfaceCellMask& mask = surfaceIt->second;
    const int cellRowCount = mask.rows;
    const int cellColCount = mask.cols;
    if (cellRowCount <= 0 || cellColCount <= 0) {
        return;
    }

    rowStart = std::max(0, rowStart);
    rowEnd = std::min(cellRowCount, rowEnd);
    colStart = std::max(0, colStart);
    colEnd = std::min(cellColCount, colEnd);

    if (rowStart >= rowEnd || colStart >= colEnd) {
        return;
    }

    for (int row = rowStart; row < rowEnd; ++row) {
        for (int col = colStart; col < colEnd; ++col) {
            if (mask.isActive(row, col)) {
                removeCellEntry(mask, surface, row, col);
            }
        }
    }

    if (tree && patchCount == 0) {
        tree.reset();
    }
    if (mask.empty()) {
        mask.clear();
        surfaceCellRecords.erase(surfaceIt);
    }
}
