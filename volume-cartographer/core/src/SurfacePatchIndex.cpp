#include "vc/core/util/SurfacePatchIndex.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>


#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/iterator/function_output_iterator.hpp>
#include <vc_reader/tifxyz.hpp>

#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/Logging.hpp"


namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace {

using SurfacePtr = SurfacePatchIndex::SurfacePtr;

struct TriangleHit {
    cv::Vec3f closest{0, 0, 0};
    cv::Vec3f bary{0, 0, 0}; // weights for vertices (sum to 1, >= 0)
    float distSq = std::numeric_limits<float>::max();
};

inline float clamp01(float v) {
    return std::max(0.0f, std::min(1.0f, v));
}

static constexpr char kCacheMagic[8] = {'V', 'C', 'S', 'P', 'I', 'D', 'X', '2'};
static constexpr std::uint32_t kCacheVersion = 1;

std::string normalized_surface_path(const QuadSurface* surface)
{
    if (!surface) {
        return {};
    }
    std::filesystem::path p = surface->path;
    if (p.empty()) {
        return {};
    }
    std::error_code ec;
    p = std::filesystem::absolute(p, ec);
    if (ec) {
        p = surface->path;
    }
    return p.lexically_normal().generic_string();
}

std::uint64_t fnv1a_update(std::uint64_t h, const void* data, std::size_t size)
{
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ULL;
    }
    return h;
}

std::uint64_t fnv1a_update_string(std::uint64_t h, const std::string& value)
{
    const std::uint64_t len = value.size();
    h = fnv1a_update(h, &len, sizeof(len));
    return fnv1a_update(h, value.data(), value.size());
}

void hash_path_identity(std::uint64_t& h, const std::filesystem::path& path)
{
    const std::string pathString = path.lexically_normal().generic_string();
    h = fnv1a_update_string(h, pathString);

    std::error_code ec;
    const bool exists = std::filesystem::exists(path, ec);
    h = fnv1a_update(h, &exists, sizeof(exists));
    if (!exists || ec) {
        return;
    }

    const bool regular = std::filesystem::is_regular_file(path, ec);
    h = fnv1a_update(h, &regular, sizeof(regular));
    if (regular && !ec) {
        const auto size = std::filesystem::file_size(path, ec);
        if (!ec) {
            h = fnv1a_update(h, &size, sizeof(size));
        }
    }

    const auto mtime = std::filesystem::last_write_time(path, ec);
    if (!ec) {
        const auto ticks = mtime.time_since_epoch().count();
        h = fnv1a_update(h, &ticks, sizeof(ticks));
    }
}

template <typename T>
bool write_pod(std::ostream& os, const T& value)
{
    os.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return static_cast<bool>(os);
}

template <typename T>
bool read_pod(std::istream& is, T& value)
{
    is.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(is);
}

bool write_string(std::ostream& os, const std::string& value)
{
    const std::uint64_t size = value.size();
    return write_pod(os, size) &&
           (size == 0 || static_cast<bool>(os.write(value.data(), static_cast<std::streamsize>(size))));
}

bool read_string(std::istream& is, std::string& value)
{
    std::uint64_t size = 0;
    if (!read_pod(is, size) || size > (1ULL << 32)) {
        return false;
    }
    value.assign(static_cast<std::size_t>(size), '\0');
    return size == 0 ||
           static_cast<bool>(is.read(value.data(), static_cast<std::streamsize>(size)));
}

inline bool isFinitePoint(const cv::Vec3f& p) noexcept
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}


inline bool isValidBounds(const cv::Vec3f& low, const cv::Vec3f& high) noexcept
{
    return isFinitePoint(low) && isFinitePoint(high)
        && low[0] <= high[0] && low[1] <= high[1] && low[2] <= high[2];
}

struct RawSurfaceFilter {
    QuadSurface* only = nullptr;
    bool hasInclude = false;
    bool rejectAllIncludes = false;
    std::unordered_set<QuadSurface*> include;
    std::unordered_set<QuadSurface*> exclude;

    bool accepts(QuadSurface* surface) const
    {
        if (!surface || (only && surface != only)) {
            return false;
        }
        if (hasInclude) {
            const bool included = include.find(surface) != include.end();
            if (rejectAllIncludes || !included) {
                return false;
            }
        }
        return exclude.find(surface) == exclude.end();
    }
};

RawSurfaceFilter makeRawSurfaceFilter(const SurfacePatchIndex::SurfaceFilter& filter)
{
    RawSurfaceFilter raw;
    raw.only = filter.only ? filter.only.get() : nullptr;

    if (filter.includeRaw) {
        raw.hasInclude = true;
        raw.include = *filter.includeRaw;
        raw.rejectAllIncludes = raw.include.empty();
    } else if (filter.include) {
        raw.hasInclude = true;
        raw.include.reserve(filter.include->size());
        for (const auto& surface : *filter.include) {
            if (surface) {
                raw.include.insert(surface.get());
            }
        }
        raw.rejectAllIncludes = raw.include.empty();
    }

    if (filter.excludeRaw) {
        raw.exclude = *filter.excludeRaw;
    }
    if (filter.exclude) {
        raw.exclude.reserve(filter.exclude->size());
        for (const auto& surface : *filter.exclude) {
            if (surface) {
                raw.exclude.insert(surface.get());
            }
        }
    }

    return raw;
}

// A tifxyz directory read through vc_reader in bands of rows, each vertex with its validity
// (the coordinate rule and mask.tif), so a surface is indexed from its files without loading
// it whole. The two most recent bands are kept; a cell spans at most one band boundary.
class TifxyzBands {
public:
    explicit TifxyzBands(const std::filesystem::path& dir)
        : reader_(dir.string(), /*useMask=*/true)
    {
        const auto shape = reader_.shape();
        rows_ = int(shape[0]);
        cols_ = int(shape[1]);
    }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    // The vertex at (row, col) when it is valid; nothing otherwise.
    std::optional<cv::Vec3f> at(int row, int col) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const Band& band = this->band(row);
        const std::size_t index = std::size_t(row - band.first) * std::size_t(cols_) + std::size_t(col);
        if (!band.coordinates.valid[index]) {
            return std::nullopt;
        }
        const float* xyz = band.coordinates.xyz.data() + index * 3;
        return cv::Vec3f(xyz[0], xyz[1], xyz[2]);
    }

private:
    static constexpr int kBandRows = 256;
    struct Band {
        int first = -1;
        vcr::tifxyz::Coordinates coordinates;
    };
    const Band& band(int row) const
    {
        const int first = row - row % kBandRows;
        for (const auto& band : bands_) {
            if (band.first == first) {
                return band;
            }
        }
        Band& band = bands_[std::size_t(next_)];
        next_ = (next_ + 1) % int(bands_.size());
        band.first = first;
        band.coordinates = reader_.window({first, 0, std::min(kBandRows, rows_ - first), cols_});
        return band;
    }

    mutable std::mutex mutex_;
    mutable vcr::tifxyz::Reader reader_;
    mutable std::array<Band, 2> bands_;
    mutable int next_ = 0;
    int rows_ = 0;
    int cols_ = 0;
};

struct LoadedSurface {
    SurfacePtr surface;
    bool wasLoaded = false;
    bool loadedForIndex = false;
};

struct SurfaceForIndex {
    SurfacePtr surface;
    std::shared_ptr<TifxyzBands> mapped;
    int rows = 0;
    int cols = 0;
};

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

struct CellKey {
    SurfacePtr surface;
    std::uint64_t packed = 0;

    CellKey() = default;
    CellKey(const SurfacePtr& surf, int rowIndex, int colIndex)
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

    bool operator==(const CellKey& other) const noexcept
    {
        return surface.get() == other.surface.get() && packed == other.packed;
    }
};

} // namespace

struct SurfacePatchIndex::Impl {
    struct PatchRecord {
        // Raw pointer: the SurfaceRecord in surfaceRecords owns the
        // shared_ptr and outlives every PatchRecord that references it.
        QuadSurface* surface = nullptr;
        const TifxyzBands* mapped = nullptr;
        int i = 0;
        int j = 0;
        // NOTE: stride is not stored — it is always equal to
        // SurfacePatchIndex::Impl::samplingStride (global per-index).

        bool operator==(const PatchRecord& other) const noexcept {
            return surface == other.surface &&
                   mapped == other.mapped &&
                   i == other.i &&
                   j == other.j;
        }
    };

    using Point3 = bg::model::point<float, 3, bg::cs::cartesian>;
    using Box3 = bg::model::box<Point3>;
    using Entry = std::pair<Box3, PatchRecord>;
    using PatchTree = bgi::rtree<Entry, bgi::quadratic<32>>;

    // Volume-wide quantization domain for cachedBoxes compression.
    // All bboxes inserted into the rtree are snapped (quantize-then-dequantize)
    // so on-disk bytes match rtree::remove's operator== lookup exactly.
    // 101000-unit span / 65535 steps ≈ 1.54 unit precision, plenty for cell-
    // granularity spatial queries (cells are typically 2+ units wide).
    static constexpr float kQLow = -1000.0f;
    static constexpr float kQHigh = 100000.0f;
    static constexpr float kQRange = kQHigh - kQLow;

    struct QBox {
        uint16_t lo[3];
        uint16_t hi[3];
    };

    // Boxes are snapped outward (lower corners down, upper corners up), so a
    // snapped box contains its geometry. The 0.01-step margin absorbs the
    // float rounding of a snapped coordinate, so snapping a snapped box again
    // returns it unchanged (removal matches boxes exactly).
    static uint16_t qEnc(float v, bool upper) noexcept {
        if (!std::isfinite(v)) return 0;
        double t = (double(v) - kQLow) * (65535.0 / kQRange);
        t = upper ? std::ceil(t - 0.01) : std::floor(t + 0.01);
        if (t < 0.0) return 0;
        if (t > 65535.0) return 65535;
        return static_cast<uint16_t>(t);
    }
    static float qDec(uint16_t c) noexcept {
        return kQLow + float(c) * (kQRange / 65535.0f);
    }
    static QBox boxQuantize(const Box3& b) noexcept {
        QBox q;
        q.lo[0] = qEnc(b.min_corner().get<0>(), false);
        q.lo[1] = qEnc(b.min_corner().get<1>(), false);
        q.lo[2] = qEnc(b.min_corner().get<2>(), false);
        q.hi[0] = qEnc(b.max_corner().get<0>(), true);
        q.hi[1] = qEnc(b.max_corner().get<1>(), true);
        q.hi[2] = qEnc(b.max_corner().get<2>(), true);
        return q;
    }
    static Box3 boxDequantize(const QBox& q) noexcept {
        return Box3(Point3(qDec(q.lo[0]), qDec(q.lo[1]), qDec(q.lo[2])),
                    Point3(qDec(q.hi[0]), qDec(q.hi[1]), qDec(q.hi[2])));
    }
    static Box3 boxSnap(const Box3& b) noexcept {
        return boxDequantize(boxQuantize(b));
    }

    std::unique_ptr<PatchTree> tree;
    struct CellEntry {
        bool hasPatch = false;
        Entry patch{Box3(Point3(0,0,0), Point3(0,0,0)), PatchRecord{}};
    };

    struct SurfaceCellMask {
        int rows = 0;
        int cols = 0;
        int tileStride = 1;
        int tileRows = 0;
        int tileCols = 0;
        int activeCount = 0;
        std::vector<uint8_t> states;
        // Dense vector of each tile anchor's quantized R-tree Box3. Entries
        // are keyed by compact tile slot, while public/internal callers still
        // pass native grid cell anchors (row,col). PatchRecord keeps native
        // anchors so query/update behavior does not change.
        // Validity is determined by states[] bits, not by a sentinel QBox.
        std::vector<QBox> cachedBoxes;
        std::unordered_set<std::size_t> pendingCells;  // Tile slots needing R-tree update

        void clear()
        {
            rows = 0;
            cols = 0;
            tileStride = 1;
            tileRows = 0;
            tileCols = 0;
            activeCount = 0;
            states.clear();
            cachedBoxes.clear();
            cachedBoxes.shrink_to_fit();
            pendingCells.clear();
        }

        void ensureSize(int rowCount, int colCount, int stride)
        {
            rowCount = std::max(rowCount, 0);
            colCount = std::max(colCount, 0);
            stride = std::max(1, stride);
            if (rowCount <= 0 || colCount <= 0) {
                clear();
                return;
            }
            const int requiredTileRows = (rowCount + stride - 1) / stride;
            const int requiredTileCols = (colCount + stride - 1) / stride;
            const std::size_t required =
                static_cast<std::size_t>(requiredTileRows) * requiredTileCols;
            const std::size_t requiredBytes = (required + 7u) / 8u;
            if (rows == rowCount && cols == colCount && tileStride == stride
                && tileRows == requiredTileRows && tileCols == requiredTileCols
                && states.size() == requiredBytes
                && cachedBoxes.size() == required) {
                return;
            }
            rows = rowCount;
            cols = colCount;
            tileStride = stride;
            tileRows = requiredTileRows;
            tileCols = requiredTileCols;
            activeCount = 0;
            states.assign(requiredBytes, 0);
            cachedBoxes.assign(required, QBox{});
            pendingCells.clear();
        }

        bool validIndex(int row, int col) const
        {
            return row >= 0 && row < rows && col >= 0 && col < cols;
        }

        std::optional<std::size_t> tileIndex(int row, int col) const
        {
            if (!validIndex(row, col) || tileStride <= 0 || tileRows <= 0 || tileCols <= 0) {
                return std::nullopt;
            }
            if ((row % tileStride) != 0 || (col % tileStride) != 0) {
                return std::nullopt;
            }
            const int tileRow = row / tileStride;
            const int tileCol = col / tileStride;
            if (tileRow < 0 || tileRow >= tileRows || tileCol < 0 || tileCol >= tileCols) {
                return std::nullopt;
            }
            return static_cast<std::size_t>(tileRow) * tileCols + tileCol;
        }

        std::pair<int, int> nativeAnchor(std::size_t tileIdx) const
        {
            const int tileRow = static_cast<int>(tileIdx / tileCols);
            const int tileCol = static_cast<int>(tileIdx % tileCols);
            return {tileRow * tileStride, tileCol * tileStride};
        }

        bool isActive(int row, int col) const
        {
            const auto idx = tileIndex(row, col);
            if (!idx) {
                return false;
            }
            return (states[*idx >> 3] >> (*idx & 7u)) & 1u;
        }

        void setActive(int row, int col, bool active)
        {
            const auto idx = tileIndex(row, col);
            if (!idx) {
                return;
            }
            const std::size_t byte = *idx >> 3;
            const uint8_t bit = static_cast<uint8_t>(1u << (*idx & 7u));
            const bool prev = (states[byte] & bit) != 0;
            if (prev == active) {
                return;
            }
            if (active) {
                states[byte] |= bit;
                ++activeCount;
            } else {
                states[byte] &= static_cast<uint8_t>(~bit);
                --activeCount;
            }
        }

        std::optional<Box3> bboxAt(int row, int col) const
        {
            const auto idx = tileIndex(row, col);
            if (!idx || !isActive(row, col)) {
                return std::nullopt;
            }
            return boxDequantize(cachedBoxes[*idx]);
        }

        void storeBox(int row, int col, const Box3& box)
        {
            const auto idx = tileIndex(row, col);
            if (!idx) {
                return;
            }
            cachedBoxes[*idx] = boxQuantize(box);
        }

        void eraseBox(int row, int col)
        {
            // No-op: validity tracked via states[]. Leaving stale bytes in
            // cachedBoxes is fine because bboxAt() checks isActive() first.
            (void)row; (void)col;
        }

        bool empty() const
        {
            return activeCount == 0;
        }

        // Pending update tracking methods
        void queueUpdate(int row, int col)
        {
            const auto idx = tileIndex(row, col);
            if (!idx) {
                return;
            }
            pendingCells.insert(*idx);
        }

        void clearPending(int row, int col)
        {
            const auto idx = tileIndex(row, col);
            if (!idx) {
                return;
            }
            pendingCells.erase(*idx);
        }

        bool isPending(int row, int col) const
        {
            const auto idx = tileIndex(row, col);
            if (!idx) {
                return false;
            }
            return pendingCells.count(*idx) > 0;
        }

        bool hasPending() const
        {
            return !pendingCells.empty();
        }

        void clearAllPending()
        {
            pendingCells.clear();
        }
    };

    // Surface record holding the shared_ptr and associated mask
    struct SurfaceRecord {
        SurfacePtr surface;  // Keeps the surface alive
        std::shared_ptr<TifxyzBands> mapped;
        SurfaceCellMask mask;
    };

    static constexpr int kMinTileStride = 8;
    size_t patchCount = 0;
    float bboxPadding = 0.0f;
    // samplingStride: triangulation stride — controls how finely the
    // visitor emits triangles within a tile (user-facing).
    // tileStride: rtree-storage stride — one entry per tileStride×tileStride
    // source-mesh region. Keeping tileStride decoupled from (and at least
    // kMinTileStride) keeps rtree memory bounded even when the user wants
    // stride-1 triangulation. Always a multiple of samplingStride.
    int samplingStride = 1;
    int tileStride = kMinTileStride;
    bool readOnly = false;

    static int computeTileStride(int triStride) noexcept {
        const int s = std::max(1, triStride);
        if (s >= kMinTileStride) return s;
        // Round kMinTileStride up to the nearest multiple of s so that
        // sub-iteration inside a tile divides evenly.
        return ((kMinTileStride + s - 1) / s) * s;
    }

    // Maps raw pointer -> record (for fast lookup while keeping surface alive via shared_ptr in record)
    std::unordered_map<QuadSurface*, SurfaceRecord> surfaceRecords;
    std::unordered_map<QuadSurface*, uint64_t> surfaceGenerations;  // For undo/redo detection

    SurfaceCellMask& ensureMask(const SurfacePtr& surface);
    SurfaceRecord* getRecord(QuadSurface* raw);

    std::optional<Entry> makePatchEntry(const CellKey& key) const;

    struct PatchHit {
        bool valid = false;
        float u = 0.0f;
        float v = 0.0f;
        float distSq = std::numeric_limits<float>::max();
    };

    static std::vector<std::pair<CellKey, CellEntry>>
    collectEntriesForSurface(const SurfacePtr& surface,
                             float bboxPadding,
                             int samplingStride,
                             int rowStart,
                             int rowEnd,
                             int colStart,
                             int colEnd);
    static std::vector<Entry> collectEntriesForRebuild(QuadSurface* surface,
                                                       const cv::Mat_<cv::Vec3f>& points,
                                                       SurfaceCellMask& mask,
                                                       float bboxPadding,
                                                       int samplingStride);
    static std::vector<Entry> collectMappedEntriesForRebuild(QuadSurface* surface,
                                                             const TifxyzBands& points,
                                                             SurfaceCellMask& mask,
                                                             float bboxPadding,
                                                             int samplingStride);
    static bool buildEntryForCell(QuadSurface* surface,
                                  const cv::Mat_<cv::Vec3f>& points,
                                  int col,
                                  int row,
                                  int stride,
                                  float bboxPadding,
                                  Entry& outEntry);
    static bool buildMappedEntryForCell(QuadSurface* surface,
                                        const TifxyzBands& points,
                                        int col,
                                        int row,
                                        int stride,
                                        float bboxPadding,
                                        Entry& outEntry);
    static bool buildCellEntry(const SurfacePtr& surface,
                               const cv::Mat_<cv::Vec3f>& points,
                               int col,
                               int row,
                               int stride,
                               float bboxPadding,
                               CellEntry& outEntry);
    static bool loadPatchCorners(const PatchRecord& rec,
                                 int stride,
                                 std::array<cv::Vec3f, 4>& outCorners);
    static Entry buildEntryFromCorners(const PatchRecord& rec,
                                       const std::array<cv::Vec3f, 4>& corners,
                                       float bboxPadding);
    static Entry buildEntryFromBbox(const PatchRecord& rec,
                                    cv::Vec3f low,
                                    cv::Vec3f high,
                                    float bboxPadding);
    void removeCellEntry(SurfaceCellMask& mask,
                         const SurfacePtr& surface,
                         int row,
                         int col);
    void insertCells(const std::vector<std::pair<CellKey, CellEntry>>& cells);
    void removeCells(const SurfacePtr& surface,
                     int rowStart,
                     int rowEnd,
                     int colStart,
                     int colEnd);

    bool replaceSurfaceEntries(const SurfacePtr& surface,
                               std::vector<std::pair<CellKey, CellEntry>>&& newCells);

    bool removeSurfaceEntries(const SurfacePtr& surface);

    void removeSurfaceEntriesFromTree(const SurfacePtr& surface, SurfaceCellMask& mask);

    bool flushPendingSurface(const SurfacePtr& surface, SurfaceCellMask& mask);

    // Evaluates a tile (rec spans tileStride×tileStride source cells).
    // Sub-iterates at triStride to find the closest sub-quad, returning a
    // PatchHit with u,v expressed in tile-local source-mesh-cell units
    // (range [0, tileStride], not bary over the full tile).
    // earlyOutDistSq >= 0 returns as soon as any sub-quad hit is at least that
    // close — for callers that only need "within tolerance", not the minimum.
    static PatchHit evaluatePatch(const PatchRecord& rec,
                                  int tileStride,
                                  int triStride,
                                  const cv::Vec3f& point,
                                  float earlyOutDistSq = -1.0f) {
        PatchHit best;
        triStride = std::max(1, triStride);
        tileStride = std::max(triStride, tileStride);

        for (int subJ = 0; subJ < tileStride; subJ += triStride) {
            for (int subI = 0; subI < tileStride; subI += triStride) {
                const PatchRecord subRec{rec.surface, rec.mapped, rec.i + subI, rec.j + subJ};
                std::array<cv::Vec3f, 4> corners;
                if (!loadPatchCorners(subRec, triStride, corners)) {
                    continue;
                }

                const auto& p00 = corners[0];
                const auto& p10 = corners[1];
                const auto& p11 = corners[2];
                const auto& p01 = corners[3];

                auto recordHit = [&](float subU, float subV, float distSq) {
                    if (distSq >= best.distSq) return;
                    best.valid = true;
                    best.distSq = distSq;
                    best.u = static_cast<float>(subI) + subU * static_cast<float>(triStride);
                    best.v = static_cast<float>(subJ) + subV * static_cast<float>(triStride);
                };

                // Triangle 0: (p00, p10, p01)
                {
                    TriangleHit tri = closestPointOnTriangle(point, p00, p10, p01);
                    recordHit(clamp01(tri.bary[1]), clamp01(tri.bary[2]), tri.distSq);
                }
                // Triangle 1: (p10, p11, p01)
                {
                    TriangleHit tri = closestPointOnTriangle(point, p10, p11, p01);
                    float u = clamp01(tri.bary[0] + tri.bary[1]);
                    float v = clamp01(tri.bary[1] + tri.bary[2]);
                    recordHit(u, v, tri.distSq);
                }

                if (earlyOutDistSq >= 0.0f && best.valid &&
                    best.distSq <= earlyOutDistSq) {
                    return best;
                }
            }
        }

        return best;
    }

    // Per-surface metadata resolved once per query and shared across every
    // tile of that surface (center/scale lookups are not free at ~100k-tile
    // query volumes).
    struct TileSurfaceCache {
        float cx = 0.0f;
        float cy = 0.0f;
        int rows = 0;
        int cols = 0;
        SurfacePtr ownedPtr;  // Shared ownership for TriangleCandidate
    };

    // Emits the two fine triangles of every triStride sub-cell of the tile
    // anchored at (rec.i, rec.j), bbox-culled against `bounds`. Shared by
    // forEachTriangleImpl and the parallel plane-intersection path; must
    // only touch read-only surface data so it can run concurrently from
    // several worker threads while a reader holds the index lock.
    template <typename Visitor>
    static void emitTileTriangles(const PatchRecord& rec,
                                  int triStride,
                                  int tile,
                                  const TileSurfaceCache& cache,
                                  const Rect3D& bounds,
                                  Visitor&& visitor)
    {
        // Sub-iteration bounds: stop at the tile edge AND the surface edge.
        // loadPatchCorners requires col+stride and row+stride to be < cols/rows.
        const int subColLimit = std::min(rec.i + tile, cache.cols - 1);
        const int subRowLimit = std::min(rec.j + tile, cache.rows - 1);

        for (int subJ = rec.j; subJ < subRowLimit; subJ += triStride) {
            for (int subI = rec.i; subI < subColLimit; subI += triStride) {
                const PatchRecord subRec{rec.surface, rec.mapped, subI, subJ};
                std::array<cv::Vec3f, 4> corners;
                if (!loadPatchCorners(subRec, triStride, corners)) {
                    continue;
                }

                const float baseX = static_cast<float>(subI);
                const float baseY = static_cast<float>(subJ);
                const float effectiveStrideX = static_cast<float>(std::min(triStride, cache.cols - 1 - subI));
                const float effectiveStrideY = static_cast<float>(std::min(triStride, cache.rows - 1 - subJ));

                const std::array<cv::Vec3f, 4> params = {
                    cv::Vec3f(baseX - cache.cx, baseY - cache.cy, 0.0f),
                    cv::Vec3f(baseX + effectiveStrideX - cache.cx, baseY - cache.cy, 0.0f),
                    cv::Vec3f(baseX + effectiveStrideX - cache.cx, baseY + effectiveStrideY - cache.cy, 0.0f),
                    cv::Vec3f(baseX - cache.cx, baseY + effectiveStrideY - cache.cy, 0.0f)
                };

                for (int triIdx = 0; triIdx < 2; ++triIdx) {
                    TriangleCandidate candidate;
                    candidate.surface = cache.ownedPtr;
                    candidate.i = subI;
                    candidate.j = subJ;
                    candidate.triangleIndex = triIdx;

                    if (triIdx == 0) {
                        candidate.world = {corners[0], corners[1], corners[3]};
                        candidate.surfaceParams = {params[0], params[1], params[3]};
                    } else {
                        candidate.world = {corners[1], corners[2], corners[3]};
                        candidate.surfaceParams = {params[1], params[2], params[3]};
                    }

                    const auto& w0 = candidate.world[0];
                    const auto& w1 = candidate.world[1];
                    const auto& w2 = candidate.world[2];
                    // Short-circuited bbox cull: `max(a,b,c) < lo` == "all <
                    // lo", `min(a,b,c) > hi` == "all > hi". Written out
                    // this way to skip the std::max<initializer_list>
                    // allocation+iterate dance the compiler can't optimise
                    // away — this test fires once per triangle (~3M calls
                    // per flattened-view intersection pass on the heavy
                    // workload) and was ~1% of total CPU on its own.
                    if ((w0[0] < bounds.low[0]  && w1[0] < bounds.low[0]  && w2[0] < bounds.low[0])  ||
                        (w0[0] > bounds.high[0] && w1[0] > bounds.high[0] && w2[0] > bounds.high[0]) ||
                        (w0[1] < bounds.low[1]  && w1[1] < bounds.low[1]  && w2[1] < bounds.low[1])  ||
                        (w0[1] > bounds.high[1] && w1[1] > bounds.high[1] && w2[1] > bounds.high[1]) ||
                        (w0[2] < bounds.low[2]  && w1[2] < bounds.low[2]  && w2[2] < bounds.low[2])  ||
                        (w0[2] > bounds.high[2] && w1[2] > bounds.high[2] && w2[2] > bounds.high[2])) {
                        continue;
                    }

                    visitor(candidate);
                }
            }
        }
    }
};

SurfacePatchIndex::SurfacePatchIndex()
    : impl_(std::make_unique<Impl>())
{}

SurfacePatchIndex::~SurfacePatchIndex() = default;
SurfacePatchIndex::SurfacePatchIndex(SurfacePatchIndex&& other) noexcept
{
    std::unique_lock<std::shared_mutex> lock(other.mutex_);
    impl_ = std::move(other.impl_);
    globalGeneration_.store(other.globalGeneration_.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
}

SurfacePatchIndex& SurfacePatchIndex::operator=(SurfacePatchIndex&& other) noexcept
{
    if (this == &other) {
        return *this;
    }

    std::unique_lock<std::shared_mutex> selfLock(mutex_, std::defer_lock);
    std::unique_lock<std::shared_mutex> otherLock(other.mutex_, std::defer_lock);
    std::lock(selfLock, otherLock);
    impl_ = std::move(other.impl_);
    // Guarantee observers of *this see a generation change even if the
    // incoming index happens to share the counter value.
    globalGeneration_.store(
        std::max(globalGeneration_.load(std::memory_order_relaxed),
                 other.globalGeneration_.load(std::memory_order_relaxed)) + 1,
        std::memory_order_relaxed);
    return *this;
}

SurfacePatchIndex::Impl::SurfaceRecord* SurfacePatchIndex::Impl::getRecord(QuadSurface* raw)
{
    auto it = surfaceRecords.find(raw);
    return it != surfaceRecords.end() ? &it->second : nullptr;
}

std::vector<std::pair<CellKey, SurfacePatchIndex::Impl::CellEntry>>
SurfacePatchIndex::Impl::collectEntriesForSurface(const SurfacePtr& surface,
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

    // Estimate capacity to avoid repeated reallocations
    const int rowSpan = rowEnd - rowStart;
    const int colSpan = colEnd - colStart;
    const size_t estimatedCells =
        static_cast<size_t>((rowSpan + samplingStride - 1) / samplingStride) *
        static_cast<size_t>((colSpan + samplingStride - 1) / samplingStride);
    result.reserve(estimatedCells);

    // Step by stride, creating cells that span 'stride' vertices
    for (int j = rowStart; j < rowEnd; j += samplingStride) {
        for (int i = colStart; i < colEnd; i += samplingStride) {
            CellEntry entry;
            if (!buildCellEntry(surface, *points, i, j, samplingStride, bboxPadding, entry)) {
                continue;
            }

            result.emplace_back(CellKey(surface, j, i), std::move(entry));
        }
    }

    return result;
}

std::vector<SurfacePatchIndex::Impl::Entry>
SurfacePatchIndex::Impl::collectEntriesForRebuild(QuadSurface* surface,
                                                  const cv::Mat_<cv::Vec3f>& points,
                                                  SurfaceCellMask& mask,
                                                  float bboxPadding,
                                                  int samplingStride)
{
    std::vector<Entry> entries;
    if (!surface || points.empty()) {
        return entries;
    }

    const int cellRowCount = points.rows - 1;
    const int cellColCount = points.cols - 1;
    if (cellRowCount <= 0 || cellColCount <= 0) {
        return entries;
    }

    samplingStride = std::max(1, samplingStride);
    const size_t estimatedCells =
        static_cast<size_t>((cellRowCount + samplingStride - 1) / samplingStride) *
        static_cast<size_t>((cellColCount + samplingStride - 1) / samplingStride);
    entries.reserve(estimatedCells);

    for (int row = 0; row < cellRowCount; row += samplingStride) {
        for (int col = 0; col < cellColCount; col += samplingStride) {
            Entry entry{Box3{}, PatchRecord{}};
            if (!buildEntryForCell(surface, points, col, row, samplingStride, bboxPadding, entry)) {
                continue;
            }
            mask.setActive(row, col, true);
            mask.storeBox(row, col, entry.first);
            entries.push_back(std::move(entry));
        }
    }

    return entries;
}

bool SurfacePatchIndex::Impl::buildMappedEntryForCell(QuadSurface* surface,
                                                      const TifxyzBands& points,
                                                      int col,
                                                      int row,
                                                      int stride,
                                                      float bboxPadding,
                                                      Entry& outEntry)
{
    if (!surface) {
        return false;
    }

    const int maxColStride = points.cols() - 1 - col;
    const int maxRowStride = points.rows() - 1 - row;
    const int effectiveColStride = std::min(stride, maxColStride);
    const int effectiveRowStride = std::min(stride, maxRowStride);
    if (effectiveColStride <= 0 || effectiveRowStride <= 0) {
        return false;
    }

    PatchRecord rec;
    rec.surface = surface;
    rec.mapped = &points;
    rec.i = col;
    rec.j = row;

    cv::Vec3f low{std::numeric_limits<float>::max(),
                  std::numeric_limits<float>::max(),
                  std::numeric_limits<float>::max()};
    cv::Vec3f high{std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest()};
    bool hasValidPoint = false;
    auto extend = [&](const std::optional<cv::Vec3f>& vertex) {
        if (!vertex || !vc::geometry::validSurfacePoint(*vertex)) return;
        const cv::Vec3f& p = *vertex;
        hasValidPoint = true;
        low[0] = std::min(low[0], p[0]);
        low[1] = std::min(low[1], p[1]);
        low[2] = std::min(low[2], p[2]);
        high[0] = std::max(high[0], p[0]);
        high[1] = std::max(high[1], p[1]);
        high[2] = std::max(high[2], p[2]);
    };
    for (int dr = 0; dr <= effectiveRowStride; ++dr) {
        for (int dc = 0; dc <= effectiveColStride; ++dc) {
            extend(points.at(row + dr, col + dc));
        }
    }
    if (!hasValidPoint || !isValidBounds(low, high)) {
        return false;
    }

    outEntry = buildEntryFromBbox(rec, low, high, bboxPadding);
    return true;
}

std::vector<SurfacePatchIndex::Impl::Entry>
SurfacePatchIndex::Impl::collectMappedEntriesForRebuild(QuadSurface* surface,
                                                        const TifxyzBands& points,
                                                        SurfaceCellMask& mask,
                                                        float bboxPadding,
                                                        int samplingStride)
{
    std::vector<Entry> entries;
    if (!surface || points.rows() < 2 || points.cols() < 2) {
        return entries;
    }

    const int cellRowCount = points.rows() - 1;
    const int cellColCount = points.cols() - 1;
    samplingStride = std::max(1, samplingStride);
    const size_t estimatedCells =
        static_cast<size_t>((cellRowCount + samplingStride - 1) / samplingStride) *
        static_cast<size_t>((cellColCount + samplingStride - 1) / samplingStride);
    entries.reserve(estimatedCells);

    for (int row = 0; row < cellRowCount; row += samplingStride) {
        for (int col = 0; col < cellColCount; col += samplingStride) {
            Entry entry{Box3{}, PatchRecord{}};
            if (!buildMappedEntryForCell(surface, points, col, row, samplingStride, bboxPadding, entry)) {
                continue;
            }
            mask.setActive(row, col, true);
            mask.storeBox(row, col, entry.first);
            entries.push_back(std::move(entry));
        }
    }

    return entries;
}

void SurfacePatchIndex::rebuild(const std::vector<SurfacePtr>& surfaces,
                                 float bboxPadding,
                                 bool useMappedSurfaces)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    impl_->bboxPadding = bboxPadding;
    impl_->surfaceRecords.clear();
    impl_->patchCount = 0;

    std::vector<SurfaceForIndex> indexSurfaces;
    indexSurfaces.reserve(surfaces.size());
    for (const auto& surface : surfaces) {
        if (!surface) {
            continue;
        }
        std::shared_ptr<TifxyzBands> mapped;
        if (useMappedSurfaces && !surface->path.empty()) {
            try {
                mapped = std::make_shared<TifxyzBands>(surface->path);
            } catch (const std::exception& e) {
                std::cerr << "[SurfacePatchIndex] cannot read the tifxyz files of "
                          << surface->id << ": " << e.what()
                          << " -- falling back to loaded points" << std::endl;
            }
        }

        int rows = 0;
        int cols = 0;
        if (mapped) {
            rows = mapped->rows();
            cols = mapped->cols();
        } else {
            try {
                const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
                if (points) {
                    rows = points->rows;
                    cols = points->cols;
                }
            } catch (const std::exception& e) {
                std::cerr << "[SurfacePatchIndex] failed to load fallback points for "
                          << surface->id << ": " << e.what()
                          << " -- skipping surface" << std::endl;
                continue;
            }
        }

        if (rows < 2 || cols < 2) {
            std::cerr << "[SurfacePatchIndex] skipping surface with insufficient points: "
                      << surface->id << " rows=" << rows << " cols=" << cols << std::endl;
            continue;
        }
        indexSurfaces.push_back(SurfaceForIndex{surface, std::move(mapped), rows, cols});
    }

    const size_t surfaceCount = indexSurfaces.size();
    if (DebugLoggingEnabled()) {
        std::cout << "[SurfacePatchIndex] rebuild: " << surfaceCount
                  << " of " << surfaces.size() << " surfaces available" << std::endl;
    }
    if (surfaceCount == 0) {
        impl_->tree.reset();
        impl_->samplingStride = std::max(1, impl_->samplingStride);
        impl_->tileStride = Impl::computeTileStride(impl_->samplingStride);
        return;
    }

    impl_->samplingStride = std::max(1, impl_->samplingStride);
    impl_->tileStride = Impl::computeTileStride(impl_->samplingStride);
    const int stride = impl_->tileStride;
    const float padding = bboxPadding;

    const bool debugLogging = DebugLoggingEnabled();

    // Pre-create all masks (sequential, enables thread-safe parallel access)
    if (debugLogging) {
        std::cout << "[SurfacePatchIndex] creating masks for " << surfaceCount << " surfaces" << std::endl;
    }
    for (const auto& item : indexSurfaces) {
        Impl::SurfaceRecord& rec = impl_->surfaceRecords[item.surface.get()];
        rec.surface = item.surface;
        rec.mapped = item.mapped;
        rec.mask.ensureSize(std::max(0, item.rows - 1),
                            std::max(0, item.cols - 1),
                            stride);
    }

    // Per-surface entries for parallel collection. Rebuild does not need
    // CellKey/CellEntry wrappers; masks are updated immediately per surface.
    std::vector<std::vector<Impl::Entry>> perSurfaceEntries(surfaceCount);
    std::atomic_size_t indexedSurfaces{0};
    const auto collectStart = std::chrono::steady_clock::now();

    // Parallel phase: collect entries and update masks for each surface
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < surfaceCount; ++i) {
        const SurfacePtr& surface = indexSurfaces[i].surface;
        auto* rec = impl_->getRecord(surface.get());
        if (rec) {
            if (indexSurfaces[i].mapped) {
                perSurfaceEntries[i] = Impl::collectMappedEntriesForRebuild(
                    surface.get(), *indexSurfaces[i].mapped, rec->mask, padding, stride);
            } else {
                try {
                    if (const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr()) {
                        perSurfaceEntries[i] = Impl::collectEntriesForRebuild(
                            surface.get(), *points, rec->mask, padding, stride);
                    }
                } catch (const std::exception& e) {
                    #pragma omp critical(surface_patch_index_fallback_warning)
                    {
                        std::cerr << "[SurfacePatchIndex] failed to index fallback points for "
                                  << surface->id << ": " << e.what() << std::endl;
                    }
                }
            }
        }

        const size_t done = indexedSurfaces.fetch_add(1, std::memory_order_relaxed) + 1;
        if (debugLogging && (done == surfaceCount || done % 1000 == 0)) {
            #pragma omp critical(surface_patch_index_progress)
            {
                const double seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - collectStart).count();
                std::cout << "[SurfacePatchIndex] indexed surfaces "
                          << done << "/" << surfaceCount
                          << " seconds=" << seconds << std::endl;
            }
        }
    }

    // Merge entries from all surfaces
    if (debugLogging) {
        std::cout << "[SurfacePatchIndex] merging entries" << std::endl;
    }
    size_t totalEntries = 0;
    for (const auto& entriesForSurface : perSurfaceEntries) {
        totalEntries += entriesForSurface.size();
    }

    std::vector<Impl::Entry> entries;
    entries.reserve(totalEntries);

    for (auto& entriesForSurface : perSurfaceEntries) {
        entries.insert(entries.end(),
                       std::make_move_iterator(entriesForSurface.begin()),
                       std::make_move_iterator(entriesForSurface.end()));
        entriesForSurface.clear();
        entriesForSurface.shrink_to_fit();
    }

    impl_->patchCount = entries.size();
    if (entries.empty()) {
        impl_->tree.reset();
    } else {
        if (debugLogging) {
            std::cout << "[SurfacePatchIndex] building R-tree with " << entries.size() << " entries" << std::endl;
        }
        impl_->tree = std::make_unique<Impl::PatchTree>(entries.begin(), entries.end());
    }
    if (debugLogging) {
        std::cout << "[SurfacePatchIndex] rebuild complete: surfaces=" << surfaceCount
                  << " patches=" << impl_->patchCount << std::endl;
    }
    globalGeneration_.fetch_add(1, std::memory_order_relaxed);
}

void SurfacePatchIndex::clear()
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (impl_) {
        impl_->tree.reset();
        impl_->patchCount = 0;
        impl_->bboxPadding = 0.0f;
        impl_->surfaceRecords.clear();
        impl_->samplingStride = 1;
        impl_->tileStride = Impl::computeTileStride(1);
        globalGeneration_.fetch_add(1, std::memory_order_relaxed);
    }
}

bool SurfacePatchIndex::empty() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return !impl_ || !impl_->tree || impl_->patchCount == 0;
}

size_t SurfacePatchIndex::patchCount() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return impl_ ? impl_->patchCount : 0;
}

size_t SurfacePatchIndex::surfaceCount() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return impl_ ? impl_->surfaceRecords.size() : 0;
}

bool SurfacePatchIndex::containsSurface(const SurfacePtr& surface) const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return impl_ && surface && impl_->surfaceRecords.find(surface.get()) != impl_->surfaceRecords.end();
}

std::optional<SurfacePatchIndex::LookupResult>
SurfacePatchIndex::locate(const PointQuery& pointQuery) const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!impl_ || !impl_->tree || pointQuery.tolerance <= 0.0f ||
        !isFinitePoint(pointQuery.worldPoint)) {
        return std::nullopt;
    }

    const float tol = std::max(pointQuery.tolerance, 0.0f);
    Impl::Point3 min_pt(pointQuery.worldPoint[0] - tol,
                        pointQuery.worldPoint[1] - tol,
                        pointQuery.worldPoint[2] - tol);
    Impl::Point3 max_pt(pointQuery.worldPoint[0] + tol,
                        pointQuery.worldPoint[1] + tol,
                        pointQuery.worldPoint[2] + tol);
    Impl::Box3 query(min_pt, max_pt);

    const float toleranceSq = tol * tol;
    const RawSurfaceFilter surfaceFilter = makeRawSurfaceFilter(pointQuery.surfaces);
    if (surfaceFilter.rejectAllIncludes) {
        return std::nullopt;
    }
    SurfacePatchIndex::LookupResult best;
    QuadSurface* bestSurfaceRaw = nullptr;
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
        if (!surfaceFilter.accepts(rec.surface)) {
            return;
        }

        Impl::PatchHit hit = Impl::evaluatePatch(rec, impl_->tileStride,
                                                  impl_->samplingStride, pointQuery.worldPoint);
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

        bestSurfaceRaw = rec.surface;
        best.ptr = ptr;
        bestDistSq = hit.distSq;
        found = true;
    };

    try {
        impl_->tree->query(
            bgi::intersects(query),
            boost::make_function_output_iterator(processEntry));
    } catch (const std::exception& e) {
        std::cerr << "[SurfacePatchIndex] locate query failed: " << e.what() << std::endl;
        return std::nullopt;
    } catch (...) {
        std::cerr << "[SurfacePatchIndex] locate query failed: unknown exception" << std::endl;
        return std::nullopt;
    }

    if (!found) {
        return std::nullopt;
    }

    if (auto srIt = impl_->surfaceRecords.find(bestSurfaceRaw);
        srIt != impl_->surfaceRecords.end()) {
        best.surface = srIt->second.surface;
    }
    best.distance = std::sqrt(bestDistSq);
    return best;
}

std::optional<SurfacePatchIndex::LookupResult>
SurfacePatchIndex::locateAny(const PointQuery& pointQuery) const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!impl_ || !impl_->tree || pointQuery.tolerance <= 0.0f ||
        !isFinitePoint(pointQuery.worldPoint)) {
        return std::nullopt;
    }

    const float tol = std::max(pointQuery.tolerance, 0.0f);
    Impl::Point3 min_pt(pointQuery.worldPoint[0] - tol,
                        pointQuery.worldPoint[1] - tol,
                        pointQuery.worldPoint[2] - tol);
    Impl::Point3 max_pt(pointQuery.worldPoint[0] + tol,
                        pointQuery.worldPoint[1] + tol,
                        pointQuery.worldPoint[2] + tol);
    Impl::Box3 query(min_pt, max_pt);

    const float toleranceSq = tol * tol;

    // Filter directly against the caller's sets. makeRawSurfaceFilter copies
    // the include/exclude sets per call, which is prohibitive when this runs
    // per sample point with thousands of included surfaces.
    const SurfaceFilter& filter = pointQuery.surfaces;
    QuadSurface* const only = filter.only ? filter.only.get() : nullptr;
    if ((filter.includeRaw && filter.includeRaw->empty()) ||
        (!filter.includeRaw && filter.include && filter.include->empty())) {
        return std::nullopt;
    }
    // Non-owning aliasing key for lookups in shared_ptr-keyed sets.
    const auto sharedKey = [](QuadSurface* s) {
        return std::shared_ptr<QuadSurface>(std::shared_ptr<QuadSurface>(), s);
    };
    const auto accepts = [&](QuadSurface* s) -> bool {
        if (!s || (only && s != only)) {
            return false;
        }
        if (filter.includeRaw) {
            if (filter.includeRaw->find(s) == filter.includeRaw->end()) {
                return false;
            }
        } else if (filter.include) {
            if (filter.include->find(sharedKey(s)) == filter.include->end()) {
                return false;
            }
        }
        if (filter.excludeRaw &&
            filter.excludeRaw->find(s) != filter.excludeRaw->end()) {
            return false;
        }
        if (filter.exclude &&
            filter.exclude->find(sharedKey(s)) != filter.exclude->end()) {
            return false;
        }
        return true;
    };

    try {
        for (auto it = impl_->tree->qbegin(bgi::intersects(query));
             it != impl_->tree->qend(); ++it) {
            const Impl::PatchRecord& rec = it->second;
            if (!accepts(rec.surface)) {
                continue;
            }

            Impl::PatchHit hit = Impl::evaluatePatch(rec, impl_->tileStride,
                                                     impl_->samplingStride,
                                                     pointQuery.worldPoint,
                                                     toleranceSq);
            if (!hit.valid || hit.distSq > toleranceSq) {
                continue;
            }

            SurfacePatchIndex::LookupResult result;
            const cv::Vec3f center = rec.surface->center();
            const cv::Vec2f scale = rec.surface->scale();
            const float absX = static_cast<float>(rec.i) + hit.u;
            const float absY = static_cast<float>(rec.j) + hit.v;
            result.ptr = {
                absX - center[0] * scale[0],
                absY - center[1] * scale[1],
                0.0f
            };
            if (auto srIt = impl_->surfaceRecords.find(rec.surface);
                srIt != impl_->surfaceRecords.end()) {
                result.surface = srIt->second.surface;
            }
            result.distance = std::sqrt(hit.distSq);
            return result;
        }
    } catch (const std::exception& e) {
        std::cerr << "[SurfacePatchIndex] locateAny query failed: " << e.what() << std::endl;
        return std::nullopt;
    } catch (...) {
        std::cerr << "[SurfacePatchIndex] locateAny query failed: unknown exception" << std::endl;
        return std::nullopt;
    }

    return std::nullopt;
}

std::vector<SurfacePatchIndex::LookupResult>
SurfacePatchIndex::locateAll(const PointQuery& pointQuery) const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!impl_ || !impl_->tree || pointQuery.tolerance <= 0.0f ||
        !isFinitePoint(pointQuery.worldPoint)) {
        return {};
    }

    const float tol = std::max(pointQuery.tolerance, 0.0f);
    Impl::Point3 min_pt(pointQuery.worldPoint[0] - tol,
                        pointQuery.worldPoint[1] - tol,
                        pointQuery.worldPoint[2] - tol);
    Impl::Point3 max_pt(pointQuery.worldPoint[0] + tol,
                        pointQuery.worldPoint[1] + tol,
                        pointQuery.worldPoint[2] + tol);
    Impl::Box3 query(min_pt, max_pt);

    const float toleranceSq = tol * tol;
    const RawSurfaceFilter surfaceFilter = makeRawSurfaceFilter(pointQuery.surfaces);
    if (surfaceFilter.rejectAllIncludes) {
        return {};
    }

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

    struct SurfaceHit {
        LookupResult result;
        float distSq = std::numeric_limits<float>::infinity();
    };
    std::unordered_map<QuadSurface*, SurfaceHit> hits;
    hits.reserve(8);

    auto processEntry = [&](const Impl::Entry& entry) {
        const Impl::PatchRecord& rec = entry.second;
        if (!surfaceFilter.accepts(rec.surface)) {
            return;
        }

        Impl::PatchHit hit = Impl::evaluatePatch(rec, impl_->tileStride,
                                                  impl_->samplingStride, pointQuery.worldPoint);
        if (!hit.valid || hit.distSq > toleranceSq) {
            return;
        }

        auto& best = hits[rec.surface];
        if (hit.distSq > best.distSq) {
            return;
        }

        const SurfaceInfo& info = ensureSurfaceInfo(rec.surface);
        const float absX = static_cast<float>(rec.i) + hit.u;
        const float absY = static_cast<float>(rec.j) + hit.v;
        best.result.ptr = {
            absX - info.center[0] * info.scale[0],
            absY - info.center[1] * info.scale[1],
            0.0f
        };
        best.distSq = hit.distSq;
    };

    try {
        impl_->tree->query(
            bgi::intersects(query),
            boost::make_function_output_iterator(processEntry));
    } catch (const std::exception& e) {
        std::cerr << "[SurfacePatchIndex] locateAll query failed: " << e.what() << std::endl;
        return {};
    } catch (...) {
        std::cerr << "[SurfacePatchIndex] locateAll query failed: unknown exception" << std::endl;
        return {};
    }

    std::vector<LookupResult> results;
    results.reserve(hits.size());
    for (auto& [surfaceRaw, hit] : hits) {
        auto srIt = impl_->surfaceRecords.find(surfaceRaw);
        if (srIt == impl_->surfaceRecords.end()) {
            continue;
        }
        hit.result.surface = srIt->second.surface;
        hit.result.distance = std::sqrt(hit.distSq);
        results.push_back(std::move(hit.result));
    }
    return results;
}

std::vector<SurfacePatchIndex::SurfacePtr>
SurfacePatchIndex::locateSurfaces(const PointQuery& pointQuery) const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!impl_ || !impl_->tree || pointQuery.tolerance <= 0.0f ||
        !isFinitePoint(pointQuery.worldPoint)) {
        return {};
    }

    const float tol = std::max(pointQuery.tolerance, 0.0f);
    Impl::Point3 min_pt(pointQuery.worldPoint[0] - tol,
                        pointQuery.worldPoint[1] - tol,
                        pointQuery.worldPoint[2] - tol);
    Impl::Point3 max_pt(pointQuery.worldPoint[0] + tol,
                        pointQuery.worldPoint[1] + tol,
                        pointQuery.worldPoint[2] + tol);
    Impl::Box3 query(min_pt, max_pt);
    const float toleranceSq = tol * tol;
    const RawSurfaceFilter surfaceFilter = makeRawSurfaceFilter(pointQuery.surfaces);
    if (surfaceFilter.rejectAllIncludes) {
        return {};
    }

    std::vector<SurfacePtr> surfaces;
    surfaces.reserve(8);
    std::unordered_set<QuadSurface*> emitted;
    emitted.reserve(8);

    auto processEntry = [&](const Impl::Entry& entry) {
        const Impl::PatchRecord& rec = entry.second;
        if (!surfaceFilter.accepts(rec.surface) || emitted.find(rec.surface) != emitted.end()) {
            return;
        }

        Impl::PatchHit hit = Impl::evaluatePatch(rec, impl_->tileStride,
                                                  impl_->samplingStride, pointQuery.worldPoint);
        if (!hit.valid || hit.distSq > toleranceSq) {
            return;
        }

        auto srIt = impl_->surfaceRecords.find(rec.surface);
        if (srIt == impl_->surfaceRecords.end()) {
            return;
        }
        emitted.insert(rec.surface);
        surfaces.push_back(srIt->second.surface);
    };

    try {
        impl_->tree->query(
            bgi::intersects(query),
            boost::make_function_output_iterator(processEntry));
    } catch (const std::exception& e) {
        std::cerr << "[SurfacePatchIndex] locateSurfaces query failed: "
                  << e.what() << std::endl;
        return {};
    } catch (...) {
        std::cerr << "[SurfacePatchIndex] locateSurfaces query failed: unknown exception"
                  << std::endl;
        return {};
    }

    return surfaces;
}

void SurfacePatchIndex::forEachTriangle(
    const TriangleQuery& query,
    const std::function<void(const TriangleCandidate&)>& visitor) const
{
    if (!visitor) {
        return;
    }

    auto patchFilter = [&](const Impl::Box3& box) {
        if (!query.patchFilter) {
            return true;
        }
        const PatchBounds bounds{
            cv::Vec3f(box.min_corner().get<0>(),
                      box.min_corner().get<1>(),
                      box.min_corner().get<2>()),
            cv::Vec3f(box.max_corner().get<0>(),
                      box.max_corner().get<1>(),
                      box.max_corner().get<2>())
        };
        return query.patchFilter(bounds);
    };

    forEachTriangleImpl(query.bounds,
                        query.surfaces,
                        visitor,
                        patchFilter);
}

void SurfacePatchIndex::forEachTriangle(
    const RayQuery& query,
    const std::function<void(const TriangleCandidate&)>& visitor) const
{
    if (!visitor || !isFinitePoint(query.src) || !isFinitePoint(query.end) ||
        !std::isfinite(query.minT) || !std::isfinite(query.bboxPadding)) {
        return;
    }

    const cv::Vec3f ray = query.end - query.src;
    const float maxT = cv::norm(ray);
    if (!std::isfinite(maxT) || maxT <= 1e-6f || query.minT > maxT) {
        return;
    }

    const cv::Vec3f dir = ray * (1.0f / maxT);
    Rect3D bounds;
    for (int ax = 0; ax < 3; ++ax) {
        bounds.low[ax] = std::min(query.src[ax], query.end[ax]) - query.bboxPadding;
        bounds.high[ax] = std::max(query.src[ax], query.end[ax]) + query.bboxPadding;
    }

    TriangleQuery triangleQuery;
    triangleQuery.bounds = bounds;
    triangleQuery.surfaces = query.surfaces;
    triangleQuery.patchFilter = [src = query.src, dir, minT = query.minT, maxT](
        const PatchBounds& patchBounds) {
        float t0 = minT;
        float t1 = maxT;
        for (int ax = 0; ax < 3; ++ax) {
            const float d = dir[ax];
            if (std::abs(d) <= 1e-8f) {
                if (src[ax] < patchBounds.low[ax] || src[ax] > patchBounds.high[ax]) {
                    return false;
                }
                continue;
            }

            const float invD = 1.0f / d;
            float nearT = (patchBounds.low[ax] - src[ax]) * invD;
            float farT = (patchBounds.high[ax] - src[ax]) * invD;
            if (nearT > farT) {
                std::swap(nearT, farT);
            }
            t0 = std::max(t0, nearT);
            t1 = std::min(t1, farT);
            if (t0 > t1) {
                return false;
            }
        }
        return true;
    };

    forEachTriangle(triangleQuery, visitor);
}

template <typename Visitor, typename PatchFilter>
void SurfacePatchIndex::forEachTriangleImpl(
    const Rect3D& bounds,
    const SurfaceFilter& surfaces,
    Visitor&& visitor,
    PatchFilter&& patchFilter) const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!impl_ || !impl_->tree) {
        return;
    }

    Impl::Point3 min_pt(bounds.low[0], bounds.low[1], bounds.low[2]);
    Impl::Point3 max_pt(bounds.high[0], bounds.high[1], bounds.high[2]);
    if (!isValidBounds(bounds.low, bounds.high)) {
        return;
    }
    Impl::Box3 query(min_pt, max_pt);

    // Cache surface metadata to avoid redundant lookups across patches
    std::unordered_map<QuadSurface*, Impl::TileSurfaceCache> surfaceCacheMap;
    surfaceCacheMap.reserve(std::min<std::size_t>(
        surfaces.include ? surfaces.include->size() : 4, 4096));

    const RawSurfaceFilter surfaceFilter = makeRawSurfaceFilter(surfaces);
    if (surfaceFilter.rejectAllIncludes) {
        return;
    }

    auto emitFromPatch = [&](const Impl::Entry& entry) {
        const Impl::PatchRecord& rec = entry.second;
        if (!surfaceFilter.accepts(rec.surface)) {
            return;
        }
        // Optional caller-supplied bbox-level reject (e.g. plane-vs-bbox).
        // Runs before the expensive loadPatchCorners call. With the default
        // NoPatchFilter the compiler folds this away.
        if (!patchFilter(entry.first)) {
            return;
        }

        // The rtree entry covers a tileStride×tileStride source-mesh tile
        // anchored at (rec.i, rec.j). Sub-iterate at the (finer) triangulation
        // stride, emitting fine triangles per sub-cell. This keeps the rtree
        // small while letting the visual triangulation be as fine as the user
        // wants.
        const int triStride = std::max(1, impl_->samplingStride);
        const int tile = std::max(triStride, impl_->tileStride);

        // Surface cache lookup (once per tile)
        auto cacheIt = surfaceCacheMap.find(rec.surface);
        if (cacheIt == surfaceCacheMap.end()) {
            auto srIt = impl_->surfaceRecords.find(rec.surface);
            if (srIt == impl_->surfaceRecords.end()) {
                return;
            }
            const cv::Vec3f center = rec.surface->center();
            const cv::Vec2f scale = rec.surface->scale();
            int rows = 0;
            int cols = 0;
            if (rec.mapped) {
                rows = rec.mapped->rows();
                cols = rec.mapped->cols();
            } else {
                const cv::Mat_<cv::Vec3f>* points = rec.surface->rawPointsPtr();
                rows = points ? points->rows : 0;
                cols = points ? points->cols : 0;
            }
            cacheIt = surfaceCacheMap.emplace(rec.surface,
                Impl::TileSurfaceCache{center[0] * scale[0], center[1] * scale[1],
                                       rows, cols, srIt->second.surface}).first;
        }
        Impl::emitTileTriangles(rec, triStride, tile, cacheIt->second, bounds, visitor);
    };

    try {
        impl_->tree->query(bgi::intersects(query),
                           boost::make_function_output_iterator(emitFromPatch));
    } catch (const std::exception& e) {
        std::cerr << "[SurfacePatchIndex] triangle query failed: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[SurfacePatchIndex] triangle query failed: unknown exception" << std::endl;
    }
}

bool SurfacePatchIndex::Impl::removeSurfaceEntries(const SurfacePtr& surface)
{
    if (!surface) {
        return false;
    }

    auto it = surfaceRecords.find(surface.get());
    if (it == surfaceRecords.end() || it->second.mask.empty()) {
        // Short-circuit: the surface was never actually indexed (or was
        // already removed). Avoid the full mask walk and log nothing.
        return false;
    }

    SurfaceCellMask& mask = it->second.mask;

    // Walk active tile slots via states bitset, use compact cachedBoxes for each.
    if (tree && mask.activeCount > 0 && !mask.cachedBoxes.empty()) {
        for (std::size_t tileIdx = 0; tileIdx < mask.cachedBoxes.size(); ++tileIdx) {
            const auto [row, col] = mask.nativeAnchor(tileIdx);
            if (!mask.isActive(row, col)) continue;
            const QBox& qb = mask.cachedBoxes[tileIdx];
            const auto mapped = it->second.mapped.get();
            PatchRecord rec{surface.get(), mapped, col, row};
            if (tree->remove(Entry(boxDequantize(qb), rec)) && patchCount > 0) {
                --patchCount;
            }
        }
    }

    // Clear the mask entirely (faster than individual eraseBox calls)
    mask.clear();
    surfaceRecords.erase(it);

    if (tree && patchCount == 0) {
        tree.reset();
    }

    return true;
}

void SurfacePatchIndex::Impl::removeSurfaceEntriesFromTree(const SurfacePtr& surface, SurfaceCellMask& mask)
{
    if (!surface || mask.cachedBoxes.empty() || mask.activeCount == 0) {
        return;
    }

    if (tree) {
        const TifxyzBands* mapped = nullptr;
        if (auto it = surfaceRecords.find(surface.get()); it != surfaceRecords.end()) {
            mapped = it->second.mapped.get();
        }
        for (std::size_t tileIdx = 0; tileIdx < mask.cachedBoxes.size(); ++tileIdx) {
            const auto [row, col] = mask.nativeAnchor(tileIdx);
            if (!mask.isActive(row, col)) continue;
            const QBox& qb = mask.cachedBoxes[tileIdx];
            PatchRecord rec{surface.get(), mapped, col, row};
            if (tree->remove(Entry(boxDequantize(qb), rec)) && patchCount > 0) {
                --patchCount;
            }
        }
    }

    if (tree && patchCount == 0) {
        tree.reset();
    }
}

bool SurfacePatchIndex::Impl::replaceSurfaceEntries(
    const SurfacePtr& surface,
    std::vector<std::pair<CellKey, CellEntry>>&& newCells)
{
    if (!surface) {
        return false;
    }

    removeSurfaceEntries(surface);
    insertCells(newCells);
    // Return true even if newCells is empty - the surface was successfully processed.
    // An empty surface (all invalid points) is still a valid state, not an error.
    // Returning false would incorrectly trigger a global index rebuild.
    return true;
}

namespace {
struct IntersectionEndpoint {
    cv::Vec3f world;
    cv::Vec3f param;
};

bool pointsApproximatelyEqual(const cv::Vec3f& a, const cv::Vec3f& b, float epsilon)
{
    // Use squared distance to avoid expensive sqrt
    return cv::norm(a - b, cv::NORM_L2SQR) <= epsilon * epsilon;
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
    } else if ((positive == 0 || negative == 0) && onPlane == 0) {
        // Triangle is fully on one side of plane (no vertices near it).
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
        // Use squared distance for comparison to avoid sqrt
        float bestDistSq = -1.0f;
        std::pair<size_t, size_t> bestPair = {0, 1};
        for (size_t a = 0; a < endpointCount; ++a) {
            for (size_t b = a + 1; b < endpointCount; ++b) {
                float distSq = cv::norm(endpoints[a].world - endpoints[b].world, cv::NORM_L2SQR);
                if (distSq > bestDistSq) {
                    bestDistSq = distSq;
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

bool SurfacePatchIndex::updateSurface(const SurfacePtr& surface)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!impl_ || !surface) {
        return false;
    }
    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->rows < 2 || points->cols < 2) {
        return false;
    }

    auto cells = Impl::collectEntriesForSurface(surface,
                                                impl_->bboxPadding,
                                                impl_->tileStride,
                                                0,
                                                points->rows - 1,
                                                0,
                                                points->cols - 1);
    const bool updated = impl_->replaceSurfaceEntries(surface, std::move(cells));
    if (updated) {
        ++impl_->surfaceGenerations[surface.get()];
        globalGeneration_.fetch_add(1, std::memory_order_relaxed);
    }
    return updated;
}

bool SurfacePatchIndex::updateSurfaceRegion(const SurfacePtr& surface,
                                            int rowStart,
                                            int rowEnd,
                                            int colStart,
                                            int colEnd)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
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

    const int stride = impl_->tileStride;

    // Entries are always keyed by the index-wide tile stride. PatchRecord
    // does not store a per-entry stride, so region updates must replace the
    // same stride-aligned cells created by rebuild().
    const int alignedRowStart = (rowStart / stride) * stride;
    const int alignedColStart = (colStart / stride) * stride;

    impl_->removeCells(surface, alignedRowStart, rowEnd, alignedColStart, colEnd);

    auto cells = Impl::collectEntriesForSurface(surface,
                                                impl_->bboxPadding,
                                                stride,
                                                alignedRowStart,
                                                rowEnd,
                                                alignedColStart,
                                                colEnd);
    impl_->insertCells(cells);
    if (!cells.empty()) {
        ++impl_->surfaceGenerations[surface.get()];
        globalGeneration_.fetch_add(1, std::memory_order_relaxed);
    }
    return !cells.empty();
}

bool SurfacePatchIndex::removeSurface(const SurfacePtr& surface)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!impl_ || !surface) {
        return false;
    }
    const bool removed = impl_->removeSurfaceEntries(surface);
    if (removed) {
        globalGeneration_.fetch_add(1, std::memory_order_relaxed);
    }
    return removed;
}

bool SurfacePatchIndex::setSamplingStride(int stride)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    stride = std::max(1, stride);
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    if (impl_->samplingStride == stride) {
        return false;
    }
    impl_->samplingStride = stride;
    impl_->tileStride = Impl::computeTileStride(stride);
    impl_->tree.reset();
    impl_->surfaceRecords.clear();
    impl_->patchCount = 0;
    globalGeneration_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

int SurfacePatchIndex::samplingStride() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!impl_) {
        return 1;
    }
    return impl_->samplingStride;  // Invariant: always >= 1 (enforced by setter)
}

void SurfacePatchIndex::setReadOnly(bool readOnly)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    if (impl_->readOnly == readOnly) {
        return;
    }
    impl_->readOnly = readOnly;
    impl_->tree.reset();
    impl_->surfaceRecords.clear();
    impl_->patchCount = 0;
}

std::optional<SurfacePatchIndex::Impl::Entry>
SurfacePatchIndex::Impl::makePatchEntry(const CellKey& key) const
{
    if (!key.surface) {
        return std::nullopt;
    }

    PatchRecord rec;
    rec.surface = key.surface.get();
    if (auto it = surfaceRecords.find(key.surface.get()); it != surfaceRecords.end()) {
        rec.mapped = it->second.mapped.get();
    }
    rec.i = key.colIndex();
    rec.j = key.rowIndex();

    std::array<cv::Vec3f, 4> corners;
    if (!loadPatchCorners(rec, tileStride, corners)) {
        return std::nullopt;
    }

    return buildEntryFromCorners(rec, corners, bboxPadding);
}

// Static helper to build an R-tree Entry from 4 corners
SurfacePatchIndex::Impl::Entry SurfacePatchIndex::Impl::buildEntryFromCorners(
    const PatchRecord& rec,
    const std::array<cv::Vec3f, 4>& corners,
    float bboxPadding)
{
    // Unrolled min/max (avoids loop overhead for 4 corners)
    cv::Vec3f low{
        std::min({corners[0][0], corners[1][0], corners[2][0], corners[3][0]}),
        std::min({corners[0][1], corners[1][1], corners[2][1], corners[3][1]}),
        std::min({corners[0][2], corners[1][2], corners[2][2], corners[3][2]})
    };
    cv::Vec3f high{
        std::max({corners[0][0], corners[1][0], corners[2][0], corners[3][0]}),
        std::max({corners[0][1], corners[1][1], corners[2][1], corners[3][1]}),
        std::max({corners[0][2], corners[1][2], corners[2][2], corners[3][2]})
    };

    return buildEntryFromBbox(rec, low, high, bboxPadding);
}

SurfacePatchIndex::Impl::Entry SurfacePatchIndex::Impl::buildEntryFromBbox(
    const PatchRecord& rec,
    cv::Vec3f low,
    cv::Vec3f high,
    float bboxPadding)
{
    if (bboxPadding > 0.0f) {
        low -= cv::Vec3f(bboxPadding, bboxPadding, bboxPadding);
        high += cv::Vec3f(bboxPadding, bboxPadding, bboxPadding);
    }

    Point3 min_pt(low[0], low[1], low[2]);
    Point3 max_pt(high[0], high[1], high[2]);
    // Snap through the cachedBoxes quantization grid so this Box3 matches
    // (bit-for-bit) what we'll later look up via bboxAt() in tree->remove.
    return Entry(boxSnap(Box3(min_pt, max_pt)), rec);
}

SurfacePatchIndex::Impl::SurfaceCellMask&
SurfacePatchIndex::Impl::ensureMask(const SurfacePtr& surface)
{
    const cv::Mat_<cv::Vec3f>* points = surface ? surface->rawPointsPtr() : nullptr;
    const int rowCount = points ? std::max(0, points->rows - 1) : 0;
    const int colCount = points ? std::max(0, points->cols - 1) : 0;

    auto it = surfaceRecords.find(surface.get());
    if (it != surfaceRecords.end()) {
        SurfaceCellMask& mask = it->second.mask;
        // Check if dimensions are changing for an existing mask
        const bool dimensionsChanging = (mask.rows > 0 || mask.cols > 0) &&
                                        (mask.rows != rowCount || mask.cols != colCount);
        if (dimensionsChanging) {
            // Remove old R-tree entries BEFORE clearing the mask
            // This prevents orphaned entries when surface grows/shrinks
            removeSurfaceEntriesFromTree(surface, mask);
        }
        mask.ensureSize(rowCount, colCount, tileStride);
        return mask;
    }

    // New surface - create fresh record
    SurfaceRecord& rec = surfaceRecords[surface.get()];
    rec.surface = surface;  // Keep the surface alive
    rec.mask.ensureSize(rowCount, colCount, tileStride);
    return rec.mask;
}

bool SurfacePatchIndex::Impl::loadPatchCorners(const PatchRecord& rec,
                                               int stride,
                                               std::array<cv::Vec3f, 4>& outCorners)
{
    if (!rec.surface) {
        return false;
    }
    const int rows = rec.mapped ? rec.mapped->rows() : [&]() {
        const cv::Mat_<cv::Vec3f>* points = rec.surface->rawPointsPtr();
        return points ? points->rows : 0;
    }();
    const int cols = rec.mapped ? rec.mapped->cols() : [&]() {
        const cv::Mat_<cv::Vec3f>* points = rec.surface->rawPointsPtr();
        return points ? points->cols : 0;
    }();
    if (rows < 2 || cols < 2) {
        return false;
    }

    const int row = rec.j;
    const int col = rec.i;
    stride = std::max(1, stride);

    // Clamp stride to not exceed bounds
    const int effectiveColStride = std::min(stride, cols - 1 - col);
    const int effectiveRowStride = std::min(stride, rows - 1 - row);

    if (row < 0 || col < 0 || effectiveColStride <= 0 || effectiveRowStride <= 0) {
        return false;
    }

    cv::Vec3f p00;
    cv::Vec3f p10;
    cv::Vec3f p01;
    cv::Vec3f p11;
    if (rec.mapped) {
        const auto v00 = rec.mapped->at(row, col);
        const auto v10 = rec.mapped->at(row, col + effectiveColStride);
        const auto v01 = rec.mapped->at(row + effectiveRowStride, col);
        const auto v11 = rec.mapped->at(row + effectiveRowStride, col + effectiveColStride);
        if (!v00 || !v10 || !v01 || !v11) {
            return false;
        }
        p00 = *v00;
        p10 = *v10;
        p01 = *v01;
        p11 = *v11;
    } else {
        const cv::Mat_<cv::Vec3f>* points = rec.surface->rawPointsPtr();
        if (!points) {
            return false;
        }
        p00 = (*points)(row, col);
        p10 = (*points)(row, col + effectiveColStride);
        p01 = (*points)(row + effectiveRowStride, col);
        p11 = (*points)(row + effectiveRowStride, col + effectiveColStride);
    }

    if (!vc::geometry::validSurfacePoint(p00) || !vc::geometry::validSurfacePoint(p10) ||
        !vc::geometry::validSurfacePoint(p01) || !vc::geometry::validSurfacePoint(p11)) {
        return false;
    }

    outCorners = {p00, p10, p11, p01};
    return true;
}

bool SurfacePatchIndex::Impl::buildEntryForCell(QuadSurface* surface,
                                                const cv::Mat_<cv::Vec3f>& points,
                                                int col,
                                                int row,
                                                int stride,
                                                float bboxPadding,
                                                Entry& outEntry)
{
    if (!surface) {
        return false;
    }

    // Clamp stride to not exceed bounds
    const int maxColStride = points.cols - 1 - col;
    const int maxRowStride = points.rows - 1 - row;
    const int effectiveColStride = std::min(stride, maxColStride);
    const int effectiveRowStride = std::min(stride, maxRowStride);
    if (effectiveColStride <= 0 || effectiveRowStride <= 0) {
        return false;
    }

    PatchRecord rec;
    rec.surface = surface;
    rec.mapped = nullptr;
    rec.i = col;
    rec.j = row;

    // True-bbox from every interior source point. The visitor sub-iterates
    // inside this tile at the (finer) triangulation stride and emits
    // triangles built from arbitrary interior source points. Computing the
    // bbox from only the 4 tile corners would miss interior bulge on a
    // curved surface, which would let the bbox-vs-plane early reject drop
    // tiles whose interior actually crosses the plane. Scanning all
    // (effectiveStride+1)² points (≤81 reads per tile at tileStride 8)
    // is paid once at index-build time on the background thread.
    cv::Vec3f low{std::numeric_limits<float>::max(),
                  std::numeric_limits<float>::max(),
                  std::numeric_limits<float>::max()};
    cv::Vec3f high{std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest()};
    bool hasValidPoint = false;
    auto extend = [&](const std::optional<cv::Vec3f>& vertex) {
        if (!vertex || !vc::geometry::validSurfacePoint(*vertex)) return;
        const cv::Vec3f& p = *vertex;
        hasValidPoint = true;
        low[0] = std::min(low[0], p[0]);
        low[1] = std::min(low[1], p[1]);
        low[2] = std::min(low[2], p[2]);
        high[0] = std::max(high[0], p[0]);
        high[1] = std::max(high[1], p[1]);
        high[2] = std::max(high[2], p[2]);
    };
    for (int dr = 0; dr <= effectiveRowStride; ++dr) {
        const cv::Vec3f* rowPtr = &points(row + dr, col);
        for (int dc = 0; dc <= effectiveColStride; ++dc) {
            extend(rowPtr[dc]);
        }
    }
    if (!hasValidPoint || !isValidBounds(low, high)) {
        return false;
    }

    outEntry = buildEntryFromBbox(rec, low, high, bboxPadding);

    return true;
}

bool SurfacePatchIndex::Impl::buildCellEntry(const SurfacePtr& surface,
                                             const cv::Mat_<cv::Vec3f>& points,
                                             int col,
                                             int row,
                                             int stride,
                                             float bboxPadding,
                                             CellEntry& outEntry)
{
    Entry entry{Box3{}, PatchRecord{}};
    if (!buildEntryForCell(surface.get(), points, col, row, stride, bboxPadding, entry)) {
        return false;
    }

    outEntry.patch = std::move(entry);
    outEntry.hasPatch = true;
    return true;
}

void SurfacePatchIndex::Impl::removeCellEntry(SurfaceCellMask& mask,
                                              const SurfacePtr& surface,
                                              int row,
                                              int col)
{
    if (!surface || !mask.isActive(row, col)) {
        mask.eraseBox(row, col);
        return;
    }

    bool removed = false;
    if (tree) {
        if (auto cachedBox = mask.bboxAt(row, col)) {
            const TifxyzBands* mapped = nullptr;
            if (auto it = surfaceRecords.find(surface.get()); it != surfaceRecords.end()) {
                mapped = it->second.mapped.get();
            }
            PatchRecord rec{surface.get(), mapped, col, row};
            removed = tree->remove(Entry(*cachedBox, rec));
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
    mask.eraseBox(row, col);
}

void SurfacePatchIndex::Impl::insertCells(const std::vector<std::pair<CellKey, CellEntry>>& cells)
{
    // Collect entries for batch insertion (more efficient than one-by-one)
    std::vector<Entry> toInsert;
    toInsert.reserve(cells.size());

    for (const auto& cell : cells) {
        const SurfacePtr& surface = cell.first.surface;
        if (!surface) {
            continue;
        }
        auto& mask = ensureMask(surface);
        const int row = cell.first.rowIndex();
        const int col = cell.first.colIndex();

        if (cell.second.hasPatch) {
            toInsert.push_back(cell.second.patch);
            mask.setActive(row, col, true);
            mask.storeBox(row, col, cell.second.patch.first);
        } else {
            mask.setActive(row, col, false);
            mask.eraseBox(row, col);
        }
    }

    // Batch insert into R-tree
    if (!toInsert.empty()) {
        if (!tree) {
            // Use range constructor for optimal packing when tree is empty
            tree = std::make_unique<PatchTree>(toInsert.begin(), toInsert.end());
        } else {
            // Range insert is still more efficient than individual inserts
            tree->insert(toInsert.begin(), toInsert.end());
        }
        patchCount += toInsert.size();
    }
}

void SurfacePatchIndex::Impl::removeCells(const SurfacePtr& surface,
                                          int rowStart,
                                          int rowEnd,
                                          int colStart,
                                          int colEnd)
{
    if (!surface) {
        return;
    }
    auto surfaceIt = surfaceRecords.find(surface.get());
    if (surfaceIt == surfaceRecords.end() || surfaceIt->second.mask.empty()) {
        return;
    }

    SurfaceCellMask& mask = surfaceIt->second.mask;
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
        surfaceRecords.erase(surfaceIt);
    }
}

// ============================================================================
// Pending update tracking implementation
// ============================================================================

void SurfacePatchIndex::queueCellUpdateForVertex(const SurfacePtr& surface, int vertexRow, int vertexCol)
{
    if (!surface) {
        return;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->rows < 2 || points->cols < 2) {
        return;
    }

    // A vertex at (row, col) affects cells at:
    // (row-1, col-1), (row-1, col), (row, col-1), (row, col)
    // Cells are indexed by their top-left vertex
    const int cellRowCount = points->rows - 1;
    const int cellColCount = points->cols - 1;

    const int rowStart = std::max(0, vertexRow - 1);
    const int rowEnd = std::min(cellRowCount, vertexRow + 1);
    const int colStart = std::max(0, vertexCol - 1);
    const int colEnd = std::min(cellColCount, vertexCol + 1);

    queueCellRangeUpdate(surface, rowStart, rowEnd, colStart, colEnd);
}

void SurfacePatchIndex::queueCellRangeUpdate(const SurfacePtr& surface,
                                           int rowStart,
                                           int rowEnd,
                                           int colStart,
                                           int colEnd)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!impl_ || !surface) {
        return;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->rows < 2 || points->cols < 2) {
        return;
    }

    const int cellRowCount = points->rows - 1;
    const int cellColCount = points->cols - 1;

    // Clamp to valid cell range
    rowStart = std::max(0, rowStart);
    rowEnd = std::min(cellRowCount, rowEnd);
    colStart = std::max(0, colStart);
    colEnd = std::min(cellColCount, colEnd);

    if (rowStart >= rowEnd || colStart >= colEnd) {
        return;
    }

    // Queue only the stride-aligned entries that can overlap this changed cell
    // range. PatchRecord has no per-entry stride, so queuing every cell and
    // flushing with stride-1 entries would mix incompatible entry sizes and
    // leave duplicate-looking intersection geometry.
    const int stride = impl_->tileStride;
    const int alignedRowStart = (rowStart / stride) * stride;
    const int alignedColStart = (colStart / stride) * stride;

    auto& mask = impl_->ensureMask(surface);
    for (int row = alignedRowStart; row < rowEnd; row += stride) {
        for (int col = alignedColStart; col < colEnd; col += stride) {
            mask.queueUpdate(row, col);
        }
    }
}

bool SurfacePatchIndex::flushPendingUpdates(const SurfacePtr& surface)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!impl_) {
        return false;
    }

    bool anyFlushed = false;

    if (surface) {
        // Flush single surface
        auto it = impl_->surfaceRecords.find(surface.get());
        if (it != impl_->surfaceRecords.end() && it->second.mask.hasPending()) {
            if (impl_->flushPendingSurface(it->second.surface, it->second.mask)) {
                anyFlushed = true;
                // Increment generation after successful flush
                ++impl_->surfaceGenerations[surface.get()];
            }
        }
    } else {
        // Flush all surfaces
        for (auto& [raw, rec] : impl_->surfaceRecords) {
            if (rec.mask.hasPending()) {
                if (impl_->flushPendingSurface(rec.surface, rec.mask)) {
                    anyFlushed = true;
                    // Increment generation after successful flush
                    ++impl_->surfaceGenerations[raw];
                }
            }
        }
    }

    if (anyFlushed) {
        globalGeneration_.fetch_add(1, std::memory_order_relaxed);
    }
    return anyFlushed;
}

bool SurfacePatchIndex::Impl::flushPendingSurface(const SurfacePtr& surface, SurfaceCellMask& mask)
{
    if (!surface || !mask.hasPending()) {
        return false;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->rows < 2 || points->cols < 2) {
        mask.clearAllPending();
        return false;
    }

    // Collect pending cells and process them
    std::vector<Entry> toRemove;
    std::vector<std::pair<CellKey, CellEntry>> toInsert;
    toRemove.reserve(mask.pendingCells.size());
    toInsert.reserve(mask.pendingCells.size());

    const int stride = tileStride;

    for (std::size_t idx : mask.pendingCells) {
        const auto [row, col] = mask.nativeAnchor(idx);

        // Remove old entry if it exists
        if (mask.isActive(row, col)) {
            if (auto cachedBox = mask.bboxAt(row, col)) {
                const TifxyzBands* mapped = nullptr;
                if (auto it = surfaceRecords.find(surface.get()); it != surfaceRecords.end()) {
                    mapped = it->second.mapped.get();
                }
                PatchRecord rec{surface.get(), mapped, col, row};
                toRemove.emplace_back(*cachedBox, rec);
            }
            mask.setActive(row, col, false);
            mask.eraseBox(row, col);
            if (patchCount > 0) {
                --patchCount;
            }
        }

        // Build new entry
        CellEntry entry;
        if (buildCellEntry(surface, *points, col, row, stride, bboxPadding, entry)) {
            toInsert.emplace_back(CellKey(surface, row, col), std::move(entry));
        }
    }

    // Batch remove from R-tree
    if (tree && !toRemove.empty()) {
        for (const auto& entry : toRemove) {
            tree->remove(entry);
        }
    }

    // Batch insert into R-tree
    if (!toInsert.empty()) {
        insertCells(toInsert);
    }

    mask.clearAllPending();
    return !toInsert.empty() || !toRemove.empty();
}

bool SurfacePatchIndex::hasPendingUpdates(const SurfacePtr& surface) const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!impl_) {
        return false;
    }

    if (surface) {
        auto it = impl_->surfaceRecords.find(surface.get());
        return it != impl_->surfaceRecords.end() && it->second.mask.hasPending();
    }

    // Check all surfaces
    for (const auto& [raw, rec] : impl_->surfaceRecords) {
        if (rec.mask.hasPending()) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// Generation tracking for undo/redo detection
// ============================================================================

uint64_t SurfacePatchIndex::generation(const SurfacePtr& surface) const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!impl_ || !surface) {
        return 0;
    }
    auto it = impl_->surfaceGenerations.find(surface.get());
    return it != impl_->surfaceGenerations.end() ? it->second : 0;
}

uint64_t SurfacePatchIndex::globalGeneration() const
{
    return globalGeneration_.load(std::memory_order_relaxed);
}

std::unordered_map<SurfacePatchIndex::SurfacePtr, std::vector<SurfacePatchIndex::TriangleSegment>>
SurfacePatchIndex::computePlaneIntersections(
    const PlaneSurface& plane,
    const cv::Rect& planeRoi,
    const std::unordered_set<SurfacePtr>& targets,
    float clipTolerance) const
{
    std::unordered_map<SurfacePtr, std::vector<TriangleSegment>> result;
    if (empty() || targets.empty()) {
        return result;
    }

    // Build 3D bounding box from the plane ROI with padding
    constexpr int kPadding = 8;
    cv::Rect roi = planeRoi;
    roi.x -= kPadding;
    roi.y -= kPadding;
    roi.width += kPadding * 2;
    roi.height += kPadding * 2;

    cv::Vec3f corner = plane.coord(
        cv::Vec3f(0, 0, 0),
        {static_cast<float>(roi.x), static_cast<float>(roi.y), 0.0f});
    Rect3D viewBbox = {corner, corner};
    viewBbox = expand_rect(
        viewBbox,
        plane.coord(cv::Vec3f(0, 0, 0),
                    {static_cast<float>(roi.br().x), static_cast<float>(roi.y), 0.0f}));
    viewBbox = expand_rect(
        viewBbox,
        plane.coord(cv::Vec3f(0, 0, 0),
                    {static_cast<float>(roi.x), static_cast<float>(roi.br().y), 0.0f}));
    viewBbox = expand_rect(
        viewBbox,
        plane.coord(cv::Vec3f(0, 0, 0),
                    {static_cast<float>(roi.br().x), static_cast<float>(roi.br().y), 0.0f}));
    if (!isValidBounds(viewBbox.low, viewBbox.high)) {
        return result;
    }

    // The four ROI corners all lie on the plane, so viewBbox is flat along
    // the plane normal. bgi::intersects with this (near-)zero-thickness box
    // already returns exactly the patch bboxes that straddle or touch the
    // plane inside the ROI; the only slack needed is the R-tree box
    // quantization (~1.54 units/axis, see boxSnap) plus the clip tolerance.
    // Padding by a fraction of the viewport extent instead (as this used to)
    // turns the query into a thousands-of-units-thick slab at zoomed-out
    // views and makes the R-tree enumerate a large share of ALL patches.
    constexpr float kQueryPad = 4.0f;
    viewBbox.low -= cv::Vec3f(kQueryPad, kQueryPad, kQueryPad);
    viewBbox.high += cv::Vec3f(kQueryPad, kQueryPad, kQueryPad);

    // Map raw target pointers back to their shared_ptrs. Also serves as the
    // include filter, so the (possibly ~100k-entry) target set is not copied
    // into a second raw-pointer set per query. Do not pre-create result
    // vectors for every visible surface: large VC3D sessions can have
    // 100k+ targets, while a single viewport/plane usually intersects only a
    // small fraction of them.
    std::unordered_map<const QuadSurface*, SurfacePtr> targetByRaw;
    targetByRaw.reserve(targets.size());
    for (const auto& t : targets) {
        if (t) {
            targetByRaw.emplace(t.get(), t);
        }
    }
    if (targetByRaw.empty()) {
        return result;
    }

    // Recover the plane normal via scalarp so `plane` can stay const
    // (Surface::normal() is non-const): scalarp(p) = p·n - o·n with |n| = 1.
    const float planeD0 = plane.scalarp({0.0f, 0.0f, 0.0f});
    const cv::Vec3f planeNormalAbs(
        std::abs(plane.scalarp({1.0f, 0.0f, 0.0f}) - planeD0),
        std::abs(plane.scalarp({0.0f, 1.0f, 0.0f}) - planeD0),
        std::abs(plane.scalarp({0.0f, 0.0f, 1.0f}) - planeD0));

    // Patch-level reject: skip patches whose bbox lies entirely on one side
    // of the plane, via the standard center-vs-projected-radius box/plane
    // test (one scalarp instead of eight corner evaluations).
    //
    // R-tree boxes are quantized through boxSnap (16-bit per axis over
    // ~101000 units → ~1.54 units/step). The stored lo/hi can each shift by
    // up to half a step inward, shrinking the box by up to one full step per
    // axis vs the true patch bbox; kQuantPad widens the half-extents so a
    // genuinely-intersecting patch is never wrongly rejected.
    constexpr float kQuantPad = 1.6f;  // > one quant step (~1.541)
    auto bboxStraddlesPlane = [&](const Impl::Box3& box) {
        const auto& lo = box.min_corner();
        const auto& hi = box.max_corner();
        const float ex = 0.5f * (hi.get<0>() - lo.get<0>()) + kQuantPad;
        const float ey = 0.5f * (hi.get<1>() - lo.get<1>()) + kQuantPad;
        const float ez = 0.5f * (hi.get<2>() - lo.get<2>()) + kQuantPad;
        const float radius = ex * planeNormalAbs[0] +
                             ey * planeNormalAbs[1] +
                             ez * planeNormalAbs[2];
        const float centerDist = plane.scalarp({0.5f * (lo.get<0>() + hi.get<0>()),
                                                0.5f * (lo.get<1>() + hi.get<1>()),
                                                0.5f * (lo.get<2>() + hi.get<2>())});
        return std::abs(centerDist) <= radius + clipTolerance;
    };

    // Hold the reader lock across BOTH phases below: phase 2's workers read
    // surface/tile data without taking the lock themselves, relying on this
    // thread to keep writers out until they are joined.
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!impl_ || !impl_->tree) {
        return result;
    }

    const int triStride = std::max(1, impl_->samplingStride);
    const int tile = std::max(triStride, impl_->tileStride);

    // Phase 1 (serial): drain the R-tree query, keeping only tiles that
    // belong to a target surface and actually straddle the plane, and
    // resolve each surviving surface's metadata once.
    std::vector<Impl::PatchRecord> tiles;
    std::unordered_map<QuadSurface*, Impl::TileSurfaceCache> surfaceCacheMap;
    auto collectTile = [&](const Impl::Entry& entry) {
        const Impl::PatchRecord& rec = entry.second;
        if (targetByRaw.find(rec.surface) == targetByRaw.end()) {
            return;
        }
        if (!bboxStraddlesPlane(entry.first)) {
            return;
        }
        auto cacheIt = surfaceCacheMap.find(rec.surface);
        if (cacheIt == surfaceCacheMap.end()) {
            auto srIt = impl_->surfaceRecords.find(rec.surface);
            if (srIt == impl_->surfaceRecords.end()) {
                return;
            }
            const cv::Vec3f center = rec.surface->center();
            const cv::Vec2f scale = rec.surface->scale();
            int rows = 0;
            int cols = 0;
            if (rec.mapped) {
                rows = rec.mapped->rows();
                cols = rec.mapped->cols();
            } else {
                const cv::Mat_<cv::Vec3f>* points = rec.surface->rawPointsPtr();
                rows = points ? points->rows : 0;
                cols = points ? points->cols : 0;
            }
            surfaceCacheMap.emplace(rec.surface,
                Impl::TileSurfaceCache{center[0] * scale[0], center[1] * scale[1],
                                       rows, cols, srIt->second.surface});
        }
        tiles.push_back(rec);
    };

    Impl::Point3 minPt(viewBbox.low[0], viewBbox.low[1], viewBbox.low[2]);
    Impl::Point3 maxPt(viewBbox.high[0], viewBbox.high[1], viewBbox.high[2]);
    try {
        impl_->tree->query(bgi::intersects(Impl::Box3(minPt, maxPt)),
                           boost::make_function_output_iterator(collectTile));
    } catch (const std::exception& e) {
        std::cerr << "[SurfacePatchIndex] plane intersection query failed: "
                  << e.what() << std::endl;
        return result;
    } catch (...) {
        std::cerr << "[SurfacePatchIndex] plane intersection query failed: unknown exception"
                  << std::endl;
        return result;
    }
    if (tiles.empty()) {
        return result;
    }

    // Phase 2: sub-triangulate + clip every surviving tile, fanned out over
    // std::async workers into per-worker buckets (cv::parallel_for_/OMP are
    // deliberately pinned to one thread in VC3D, so fan out manually). The
    // per-tile work is independent and reads only const data.
    using RawBuckets = std::unordered_map<const QuadSurface*, std::vector<TriangleSegment>>;
    auto processRange = [&](std::size_t begin, std::size_t end, RawBuckets& out) {
        for (std::size_t k = begin; k < end; ++k) {
            const Impl::PatchRecord& rec = tiles[k];
            const Impl::TileSurfaceCache& cache = surfaceCacheMap.find(rec.surface)->second;
            Impl::emitTileTriangles(rec, triStride, tile, cache, viewBbox,
                [&](const TriangleCandidate& tri) {
                    auto seg = clipTriangleToPlane(tri, plane, clipTolerance);
                    if (!seg) {
                        return;
                    }
                    out[rec.surface].push_back(std::move(*seg));
                });
        }
    };

    // Below ~a few hundred tiles the clip work is cheaper than thread spawn.
    constexpr std::size_t kMinTilesPerWorker = 256;
    const std::size_t hw = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t workerCount = std::min<std::size_t>(
        {hw, 16, std::max<std::size_t>(1, tiles.size() / kMinTilesPerWorker)});

    std::vector<RawBuckets> buckets(workerCount);
    if (workerCount <= 1) {
        processRange(0, tiles.size(), buckets[0]);
    } else {
        const std::size_t chunk = (tiles.size() + workerCount - 1) / workerCount;
        std::vector<std::future<void>> futures;
        futures.reserve(workerCount - 1);
        try {
            for (std::size_t w = 1; w < workerCount; ++w) {
                const std::size_t begin = w * chunk;
                const std::size_t end = std::min(tiles.size(), begin + chunk);
                if (begin >= end) {
                    break;
                }
                futures.push_back(std::async(std::launch::async,
                    [&processRange, &buckets, begin, end, w]() {
                        processRange(begin, end, buckets[w]);
                    }));
            }
            processRange(0, std::min(tiles.size(), chunk), buckets[0]);
            for (auto& f : futures) {
                f.get();
            }
        } catch (const std::exception& e) {
            // std::async futures join in their destructor, so remaining
            // workers are safely drained before the lock is released.
            std::cerr << "[SurfacePatchIndex] plane intersection clip failed: "
                      << e.what() << std::endl;
            return result;
        }
    }

    for (auto& bucket : buckets) {
        for (auto& [raw, segments] : bucket) {
            auto it = targetByRaw.find(raw);
            if (it == targetByRaw.end()) {
                continue;
            }
            auto& out = result[it->second];
            if (out.empty()) {
                out = std::move(segments);
            } else {
                out.insert(out.end(),
                           std::make_move_iterator(segments.begin()),
                           std::make_move_iterator(segments.end()));
            }
        }
    }

    return result;
}
