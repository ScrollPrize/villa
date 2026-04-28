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
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>
#include <unordered_map>

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/iterator/function_output_iterator.hpp>

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

inline bool isValidSurfacePoint(const cv::Vec3f& p) noexcept
{
    return p[0] != -1.0f && isFinitePoint(p);
}

inline bool isValidBounds(const cv::Vec3f& low, const cv::Vec3f& high) noexcept
{
    return isFinitePoint(low) && isFinitePoint(high)
        && low[0] <= high[0] && low[1] <= high[1] && low[2] <= high[2];
}

std::vector<SurfacePtr> loadSurfacesInBatches(const std::vector<SurfacePtr>& surfaces)
{
    constexpr size_t kBatchSize = 5000;
    constexpr size_t kMaxWorkers = 8;

    std::vector<SurfacePtr> loaded;
    loaded.reserve(surfaces.size());

    for (size_t batchStart = 0; batchStart < surfaces.size(); batchStart += kBatchSize) {
        const size_t batchEnd = std::min(batchStart + kBatchSize, surfaces.size());
        const size_t batchSize = batchEnd - batchStart;
        const auto batchTimeStart = std::chrono::steady_clock::now();
        std::cout << "[SurfacePatchIndex] loading surfaces "
                  << batchStart << "-" << (batchEnd == 0 ? 0 : batchEnd - 1)
                  << " / " << surfaces.size() << std::endl;
        std::vector<SurfacePtr> batchLoaded(batchSize);

        const unsigned hw = std::thread::hardware_concurrency();
        const size_t workerCount = std::max<size_t>(
            1, std::min<size_t>({batchSize, kMaxWorkers, hw == 0 ? 4 : hw}));
        std::atomic_size_t next{0};
        std::vector<std::future<void>> workers;
        workers.reserve(workerCount);

        for (size_t worker = 0; worker < workerCount; ++worker) {
            workers.emplace_back(std::async(std::launch::async, [&]() {
                while (true) {
                    const size_t localIndex = next.fetch_add(1, std::memory_order_relaxed);
                    if (localIndex >= batchSize) {
                        break;
                    }

                    const auto& s = surfaces[batchStart + localIndex];
                    if (!s) {
                        continue;
                    }
                    if (!s->isLoaded()) {
                        if (DebugLoggingEnabled()) {
                            std::cout << "[SurfacePatchIndex] Loading surface: " << s->id << std::endl;
                        }
                        try {
                            s->rawPointsPtr();
                        } catch (const std::exception& e) {
                            std::cout << "[SurfacePatchIndex] Skipping surface "
                                      << s->id << ": " << e.what() << std::endl;
                        }
                    }
                    if (s->isLoaded()) {
                        batchLoaded[localIndex] = s;
                        if (DebugLoggingEnabled()) {
                            std::cout << "[SurfacePatchIndex] Indexed surface: " << s->id << std::endl;
                        }
                    } else if (DebugLoggingEnabled()) {
                        std::cout << "[SurfacePatchIndex] Failed to load surface: " << s->id << std::endl;
                    }
                }
            }));
        }

        for (auto& worker : workers) {
            worker.get();
        }

        for (auto& surface : batchLoaded) {
            if (surface) {
                loaded.push_back(std::move(surface));
            }
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - batchTimeStart).count();
        std::cout << "[SurfacePatchIndex] loaded batch "
                  << batchEnd << "/" << surfaces.size()
                  << " cumulative_loaded=" << loaded.size()
                  << " seconds=" << seconds << std::endl;
    }

    return loaded;
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

    std::uint64_t packedIndex() const noexcept
    {
        return packed;
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
        int i = 0;
        int j = 0;
        // NOTE: stride is not stored — it is always equal to
        // SurfacePatchIndex::Impl::samplingStride (global per-index).

        bool operator==(const PatchRecord& other) const noexcept {
            return surface == other.surface &&
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

    static uint16_t qEnc(float v) noexcept {
        if (!std::isfinite(v)) return 0;
        float t = (v - kQLow) * (65535.0f / kQRange);
        if (t < 0.0f) return 0;
        if (t > 65535.0f) return 65535;
        return static_cast<uint16_t>(t + 0.5f);
    }
    static float qDec(uint16_t c) noexcept {
        return kQLow + float(c) * (kQRange / 65535.0f);
    }
    static QBox boxQuantize(const Box3& b) noexcept {
        QBox q;
        q.lo[0] = qEnc(b.min_corner().get<0>());
        q.lo[1] = qEnc(b.min_corner().get<1>());
        q.lo[2] = qEnc(b.min_corner().get<2>());
        q.hi[0] = qEnc(b.max_corner().get<0>());
        q.hi[1] = qEnc(b.max_corner().get<1>());
        q.hi[2] = qEnc(b.max_corner().get<2>());
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
        int activeCount = 0;
        std::vector<uint8_t> states;
        // Dense vector of each cell's quantized R-tree Box3 (keyed by flat cell
        // index). Dense because after the on-load trim, >95% of cells are
        // active, making the map overhead (~52B/entry) far more costly than
        // the empty-slot cost of a dense array (12B/cell).
        // Validity is determined by states[] bits, not by a sentinel QBox.
        std::vector<QBox> cachedBoxes;
        std::unordered_set<std::size_t> pendingCells;  // Cells needing R-tree update

        void clear()
        {
            rows = 0;
            cols = 0;
            activeCount = 0;
            states.clear();
            cachedBoxes.clear();
            cachedBoxes.shrink_to_fit();
            pendingCells.clear();
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
            const std::size_t requiredBytes = (required + 7u) / 8u;
            if (rows == rowCount && cols == colCount && states.size() == requiredBytes
                && cachedBoxes.size() == required) {
                return;
            }
            rows = rowCount;
            cols = colCount;
            activeCount = 0;
            states.assign(requiredBytes, 0);
            cachedBoxes.assign(required, QBox{});
            pendingCells.clear();
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
            const std::size_t idx = index(row, col);
            return (states[idx >> 3] >> (idx & 7u)) & 1u;
        }

        void setActive(int row, int col, bool active)
        {
            if (!validIndex(row, col)) {
                return;
            }
            const std::size_t idx = index(row, col);
            const std::size_t byte = idx >> 3;
            const uint8_t bit = static_cast<uint8_t>(1u << (idx & 7u));
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
            if (!validIndex(row, col) || !isActive(row, col)) {
                return std::nullopt;
            }
            return boxDequantize(cachedBoxes[index(row, col)]);
        }

        void storeBox(int row, int col, const Box3& box)
        {
            if (!validIndex(row, col)) {
                return;
            }
            cachedBoxes[index(row, col)] = boxQuantize(box);
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
            if (!validIndex(row, col)) {
                return;
            }
            pendingCells.insert(index(row, col));
        }

        void clearPending(int row, int col)
        {
            if (!validIndex(row, col)) {
                return;
            }
            pendingCells.erase(index(row, col));
        }

        bool isPending(int row, int col) const
        {
            if (!validIndex(row, col)) {
                return false;
            }
            return pendingCells.count(index(row, col)) > 0;
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
    static PatchHit evaluatePatch(const PatchRecord& rec,
                                  int tileStride,
                                  int triStride,
                                  const cv::Vec3f& point) {
        PatchHit best;
        triStride = std::max(1, triStride);
        tileStride = std::max(triStride, tileStride);

        for (int subJ = 0; subJ < tileStride; subJ += triStride) {
            for (int subI = 0; subI < tileStride; subI += triStride) {
                const PatchRecord subRec{rec.surface, rec.i + subI, rec.j + subJ};
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
            }
        }

        return best;
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
    return *this;
}

std::string SurfacePatchIndex::cacheKeyForSurfaces(const std::vector<SurfacePtr>& surfaces,
                                                   int samplingStride,
                                                   float bboxPadding)
{
    struct SurfaceIdentity {
        std::string path;
        std::string id;
    };

    std::vector<SurfaceIdentity> identities;
    identities.reserve(surfaces.size());
    for (const auto& surface : surfaces) {
        if (!surface) {
            continue;
        }
        identities.push_back({normalized_surface_path(surface.get()), surface->id});
    }
    std::sort(identities.begin(), identities.end(),
              [](const SurfaceIdentity& a, const SurfaceIdentity& b) {
                  if (a.path != b.path) {
                      return a.path < b.path;
                  }
                  return a.id < b.id;
              });

    std::uint64_t h = 1469598103934665603ULL;
    h = fnv1a_update(h, kCacheMagic, sizeof(kCacheMagic));
    h = fnv1a_update(h, &kCacheVersion, sizeof(kCacheVersion));
    samplingStride = std::max(1, samplingStride);
    h = fnv1a_update(h, &samplingStride, sizeof(samplingStride));
    h = fnv1a_update(h, &bboxPadding, sizeof(bboxPadding));
    const std::uint64_t surfaceCount = identities.size();
    h = fnv1a_update(h, &surfaceCount, sizeof(surfaceCount));

    for (const auto& identity : identities) {
        h = fnv1a_update_string(h, identity.id);
        h = fnv1a_update_string(h, identity.path);
        if (!identity.path.empty()) {
            const std::filesystem::path base(identity.path);
            hash_path_identity(h, base / "meta.json");
            hash_path_identity(h, base / "x.tif");
            hash_path_identity(h, base / "y.tif");
            hash_path_identity(h, base / "z.tif");
            hash_path_identity(h, base / "overlapping.json");
        }
    }

    std::ostringstream os;
    os << std::hex << std::setfill('0') << std::setw(16) << h;
    return os.str();
}

bool SurfacePatchIndex::saveCache(const std::filesystem::path& cachePath,
                                  const std::string& cacheKey) const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!impl_) {
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(cachePath.parent_path(), ec);
    if (ec) {
        std::cerr << "[SurfacePatchIndex] cache save: failed to create "
                  << cachePath.parent_path() << ": " << ec.message() << std::endl;
        return false;
    }

    const std::filesystem::path tmpPath = cachePath.string() + ".tmp";
    std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "[SurfacePatchIndex] cache save: failed to open " << tmpPath << std::endl;
        return false;
    }

    std::vector<QuadSurface*> surfaces;
    surfaces.reserve(impl_->surfaceRecords.size());
    for (const auto& it : impl_->surfaceRecords) {
        if (it.first) {
            surfaces.push_back(it.first);
        }
    }
    std::sort(surfaces.begin(), surfaces.end(), [](QuadSurface* a, QuadSurface* b) {
        const std::string ap = normalized_surface_path(a);
        const std::string bp = normalized_surface_path(b);
        if (ap != bp) {
            return ap < bp;
        }
        return a->id < b->id;
    });

    std::unordered_map<QuadSurface*, std::uint32_t> surfaceIndex;
    surfaceIndex.reserve(surfaces.size());
    for (std::uint32_t i = 0; i < surfaces.size(); ++i) {
        surfaceIndex[surfaces[i]] = i;
    }

    std::vector<Impl::Entry> entries;
    if (impl_->tree) {
        entries.reserve(impl_->patchCount);
        for (const auto& entry : *impl_->tree) {
            if (surfaceIndex.find(entry.second.surface) != surfaceIndex.end()) {
                entries.push_back(entry);
            }
        }
    }

    out.write(kCacheMagic, sizeof(kCacheMagic));
    if (!write_pod(out, kCacheVersion) ||
        !write_string(out, cacheKey) ||
        !write_pod(out, impl_->samplingStride) ||
        !write_pod(out, impl_->bboxPadding)) {
        return false;
    }

    const std::uint64_t surfaceCount = surfaces.size();
    if (!write_pod(out, surfaceCount)) {
        return false;
    }
    for (QuadSurface* surface : surfaces) {
        const auto recIt = impl_->surfaceRecords.find(surface);
        if (recIt == impl_->surfaceRecords.end()) {
            return false;
        }
        const Impl::SurfaceCellMask& mask = recIt->second.mask;
        const std::int32_t rows = mask.rows;
        const std::int32_t cols = mask.cols;
        if (!write_string(out, surface->id) ||
            !write_string(out, normalized_surface_path(surface)) ||
            !write_pod(out, rows) ||
            !write_pod(out, cols)) {
            return false;
        }
    }

    const std::uint64_t entryCount = entries.size();
    if (!write_pod(out, entryCount)) {
        return false;
    }
    for (const auto& entry : entries) {
        const auto idxIt = surfaceIndex.find(entry.second.surface);
        if (idxIt == surfaceIndex.end()) {
            return false;
        }
        const std::uint32_t idx = idxIt->second;
        const std::int32_t i = entry.second.i;
        const std::int32_t j = entry.second.j;
        const Impl::QBox qbox = Impl::boxQuantize(entry.first);
        if (!write_pod(out, idx) ||
            !write_pod(out, i) ||
            !write_pod(out, j) ||
            !write_pod(out, qbox)) {
            return false;
        }
    }

    out.close();
    if (!out) {
        return false;
    }
    std::filesystem::remove(cachePath, ec);
    ec.clear();
    std::filesystem::rename(tmpPath, cachePath, ec);
    if (ec) {
        std::cerr << "[SurfacePatchIndex] cache save: failed to publish "
                  << cachePath << ": " << ec.message() << std::endl;
        return false;
    }
    return true;
}

bool SurfacePatchIndex::loadCache(const std::filesystem::path& cachePath,
                                  const std::vector<SurfacePtr>& surfaces,
                                  const std::string& expectedKey)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    std::ifstream in(cachePath, std::ios::binary);
    if (!in) {
        return false;
    }

    char magic[sizeof(kCacheMagic)]{};
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, kCacheMagic, sizeof(kCacheMagic)) != 0) {
        return false;
    }

    std::uint32_t version = 0;
    std::string storedKey;
    int storedStride = 1;
    float storedPadding = 0.0f;
    if (!read_pod(in, version) ||
        version != kCacheVersion ||
        !read_string(in, storedKey) ||
        storedKey != expectedKey ||
        !read_pod(in, storedStride) ||
        !read_pod(in, storedPadding)) {
        return false;
    }

    std::unordered_map<std::string, SurfacePtr> byPath;
    std::unordered_map<std::string, SurfacePtr> byId;
    byPath.reserve(surfaces.size());
    byId.reserve(surfaces.size());
    for (const auto& surface : surfaces) {
        if (!surface) {
            continue;
        }
        const std::string path = normalized_surface_path(surface.get());
        if (!path.empty()) {
            byPath.emplace(path, surface);
        }
        byId.emplace(surface->id, surface);
    }

    std::uint64_t surfaceCount64 = 0;
    if (!read_pod(in, surfaceCount64) ||
        surfaceCount64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        return false;
    }

    struct CachedSurface {
        SurfacePtr surface;
        std::int32_t rows = 0;
        std::int32_t cols = 0;
    };
    std::vector<CachedSurface> cachedSurfaces;
    cachedSurfaces.reserve(static_cast<std::size_t>(surfaceCount64));

    for (std::uint64_t idx = 0; idx < surfaceCount64; ++idx) {
        std::string id;
        std::string path;
        std::int32_t rows = 0;
        std::int32_t cols = 0;
        if (!read_string(in, id) ||
            !read_string(in, path) ||
            !read_pod(in, rows) ||
            !read_pod(in, cols) ||
            rows < 0 ||
            cols < 0) {
            return false;
        }

        SurfacePtr surface;
        auto pathIt = byPath.find(path);
        if (pathIt != byPath.end()) {
            surface = pathIt->second;
        } else {
            auto idIt = byId.find(id);
            if (idIt != byId.end()) {
                surface = idIt->second;
            }
        }
        if (!surface) {
            return false;
        }
        cachedSurfaces.push_back({surface, rows, cols});
    }

    std::uint64_t entryCount64 = 0;
    if (!read_pod(in, entryCount64) ||
        entryCount64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return false;
    }

    auto newImpl = std::make_unique<Impl>();
    newImpl->bboxPadding = storedPadding;
    newImpl->samplingStride = std::max(1, storedStride);
    newImpl->tileStride = Impl::computeTileStride(newImpl->samplingStride);
    newImpl->readOnly = impl_ ? impl_->readOnly : false;

    for (const auto& cached : cachedSurfaces) {
        Impl::SurfaceRecord rec;
        rec.surface = cached.surface;
        rec.mask.ensureSize(cached.rows, cached.cols);
        newImpl->surfaceRecords[cached.surface.get()] = std::move(rec);
    }

    std::vector<Impl::Entry> entries;
    entries.reserve(static_cast<std::size_t>(entryCount64));
    for (std::uint64_t entryIdx = 0; entryIdx < entryCount64; ++entryIdx) {
        std::uint32_t surfaceIdx = 0;
        std::int32_t i = 0;
        std::int32_t j = 0;
        Impl::QBox qbox{};
        if (!read_pod(in, surfaceIdx) ||
            !read_pod(in, i) ||
            !read_pod(in, j) ||
            !read_pod(in, qbox) ||
            surfaceIdx >= cachedSurfaces.size()) {
            return false;
        }

        const SurfacePtr& surface = cachedSurfaces[surfaceIdx].surface;
        auto recIt = newImpl->surfaceRecords.find(surface.get());
        if (recIt == newImpl->surfaceRecords.end() ||
            !recIt->second.mask.validIndex(j, i)) {
            return false;
        }

        Impl::PatchRecord rec;
        rec.surface = surface.get();
        rec.i = i;
        rec.j = j;
        entries.emplace_back(Impl::boxDequantize(qbox), rec);

        Impl::SurfaceCellMask& mask = recIt->second.mask;
        mask.setActive(j, i, true);
        mask.cachedBoxes[mask.index(j, i)] = qbox;
    }

    newImpl->patchCount = entries.size();
    if (!entries.empty()) {
        newImpl->tree = std::make_unique<Impl::PatchTree>(entries.begin(), entries.end());
    }
    impl_ = std::move(newImpl);
    return true;
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

void SurfacePatchIndex::rebuild(const std::vector<SurfacePtr>& surfaces, float bboxPadding)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    impl_->bboxPadding = bboxPadding;
    impl_->surfaceRecords.clear();
    impl_->patchCount = 0;

    // Eagerly load any surfaces whose TIFF data hasn't been loaded yet.
    // This is safe because rebuild() typically runs on a background thread.
    std::vector<SurfacePtr> loaded = loadSurfacesInBatches(surfaces);
    const size_t surfaceCount = loaded.size();
    if (DebugLoggingEnabled()) {
        std::cout << "[SurfacePatchIndex] rebuild: " << surfaceCount << " of " << surfaces.size() << " surfaces loaded" << std::endl;
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

    // Pre-create all masks (sequential, enables thread-safe parallel access)
    std::cout << "[SurfacePatchIndex] creating masks for " << surfaceCount << " surfaces" << std::endl;
    for (const SurfacePtr& surface : loaded) {
        impl_->ensureMask(surface);
    }

    // Per-surface results for parallel collection
    using CellResult = std::vector<std::pair<CellKey, Impl::CellEntry>>;
    std::vector<CellResult> perSurfaceCells(surfaceCount);
    std::atomic_size_t indexedSurfaces{0};
    const auto collectStart = std::chrono::steady_clock::now();

    // Parallel phase: collect entries and update masks for each surface
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < surfaceCount; ++i) {
        perSurfaceCells[i] = Impl::collectEntriesForSurface(
            loaded[i],
            padding,
            stride,
            0,
            std::numeric_limits<int>::max(),
            0,
            std::numeric_limits<int>::max());

        // Update mask for this surface (each surface has its own mask, no contention)
        auto* rec = impl_->getRecord(loaded[i].get());
        if (rec) {
            for (auto& cell : perSurfaceCells[i]) {
                rec->mask.setActive(cell.first.rowIndex(), cell.first.colIndex(), cell.second.hasPatch);
                if (cell.second.hasPatch) {
                    rec->mask.storeBox(cell.first.rowIndex(), cell.first.colIndex(),
                                       cell.second.patch.first);
                }
            }
        }
        const size_t done = indexedSurfaces.fetch_add(1, std::memory_order_relaxed) + 1;
        if (done == surfaceCount || done % 1000 == 0) {
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
    std::cout << "[SurfacePatchIndex] merging entries" << std::endl;
    size_t totalEntries = 0;
    for (const auto& cells : perSurfaceCells) {
        for (const auto& cell : cells) {
            if (cell.second.hasPatch) {
                ++totalEntries;
            }
        }
    }

    std::vector<Impl::Entry> entries;
    entries.reserve(totalEntries);

    for (auto& cells : perSurfaceCells) {
        for (auto& cell : cells) {
            if (cell.second.hasPatch) {
                entries.push_back(std::move(cell.second.patch));
            }
        }
    }

    impl_->patchCount = entries.size();
    if (entries.empty()) {
        impl_->tree.reset();
    } else {
        std::cout << "[SurfacePatchIndex] building R-tree with " << entries.size() << " entries" << std::endl;
        impl_->tree = std::make_unique<Impl::PatchTree>(entries.begin(), entries.end());
    }
    std::cout << "[SurfacePatchIndex] rebuild complete: surfaces=" << surfaceCount
              << " patches=" << impl_->patchCount << std::endl;
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
SurfacePatchIndex::locate(const cv::Vec3f& worldPoint, float tolerance, const SurfacePtr& targetSurface) const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!impl_ || !impl_->tree || tolerance <= 0.0f || !isFinitePoint(worldPoint)) {
        return std::nullopt;
    }

    const float tol = std::max(tolerance, 0.0f);
    Impl::Point3 min_pt(worldPoint[0] - tol, worldPoint[1] - tol, worldPoint[2] - tol);
    Impl::Point3 max_pt(worldPoint[0] + tol, worldPoint[1] + tol, worldPoint[2] + tol);
    Impl::Box3 query(min_pt, max_pt);

    const float toleranceSq = tol * tol;
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
        if (targetSurface && rec.surface != targetSurface.get()) {
            return;
        }

        Impl::PatchHit hit = Impl::evaluatePatch(rec, impl_->tileStride,
                                                  impl_->samplingStride, worldPoint);
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

void SurfacePatchIndex::queryTriangles(const Rect3D& bounds,
                                       const SurfacePtr& targetSurface,
                                       std::vector<TriangleCandidate>& outCandidates) const
{
    outCandidates.clear();
    // Rough upper bound: keep whatever capacity the vector already had.
    // A typical viewport has a few thousand triangle candidates; a small
    // reserve avoids 3-4 reallocations during the push_back loop.
    if (outCandidates.capacity() < 2048) {
        outCandidates.reserve(2048);
    }
    forEachTriangleImpl(bounds, targetSurface, nullptr, [&](const TriangleCandidate& candidate) {
        outCandidates.push_back(candidate);
    });
}

void SurfacePatchIndex::queryTriangles(const Rect3D& bounds,
                                       const std::unordered_set<SurfacePtr>& targetSurfaces,
                                       std::vector<TriangleCandidate>& outCandidates) const
{
    outCandidates.clear();
    if (targetSurfaces.empty()) {
        return;
    }
    if (outCandidates.capacity() < 2048) {
        outCandidates.reserve(2048);
    }
    forEachTriangleImpl(bounds, nullptr, &targetSurfaces, [&](const TriangleCandidate& candidate) {
        outCandidates.push_back(candidate);
    });
}

void SurfacePatchIndex::forEachTriangle(const Rect3D& bounds,
                                        const SurfacePtr& targetSurface,
                                        const std::function<void(const TriangleCandidate&)>& visitor) const
{
    if (!visitor) return;
    forEachTriangleImpl(bounds, targetSurface, nullptr, visitor);
}

void SurfacePatchIndex::forEachTriangleIntersectingRay(
    const Rect3D& bounds,
    const SurfacePtr& targetSurface,
    const cv::Vec3f& origin,
    const cv::Vec3f& dir,
    float minT,
    float maxT,
    const std::function<void(const TriangleCandidate&)>& visitor) const
{
    if (!visitor || minT > maxT) {
        return;
    }

    auto rayIntersectsBox = [&](const Impl::Box3& box) {
        float t0 = minT;
        float t1 = maxT;
        const float lows[3] = {
            box.min_corner().get<0>(),
            box.min_corner().get<1>(),
            box.min_corner().get<2>()
        };
        const float highs[3] = {
            box.max_corner().get<0>(),
            box.max_corner().get<1>(),
            box.max_corner().get<2>()
        };

        for (int ax = 0; ax < 3; ++ax) {
            const float d = dir[ax];
            if (std::abs(d) <= 1e-8f) {
                if (origin[ax] < lows[ax] || origin[ax] > highs[ax]) {
                    return false;
                }
                continue;
            }

            const float invD = 1.0f / d;
            float nearT = (lows[ax] - origin[ax]) * invD;
            float farT = (highs[ax] - origin[ax]) * invD;
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

    forEachTriangleImpl(bounds, targetSurface, nullptr, visitor, rayIntersectsBox);
}

void SurfacePatchIndex::forEachTriangle(const Rect3D& bounds,
                                        const std::unordered_set<SurfacePtr>& targetSurfaces,
                                        const std::function<void(const TriangleCandidate&)>& visitor) const
{
    if (!visitor || targetSurfaces.empty()) {
        return;
    }
    forEachTriangleImpl(bounds, nullptr, &targetSurfaces, visitor);
}

template <typename Visitor, typename PatchFilter>
void SurfacePatchIndex::forEachTriangleImpl(
    const Rect3D& bounds,
    const SurfacePtr& targetSurface,
    const std::unordered_set<SurfacePtr>* filterSurfaces,
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
    struct SurfaceCache {
        float cx;
        float cy;
        int rows;
        int cols;
        SurfacePtr ownedPtr;  // Shared ownership for TriangleCandidate
    };
    std::unordered_map<QuadSurface*, SurfaceCache> surfaceCacheMap;
    surfaceCacheMap.reserve(filterSurfaces ? filterSurfaces->size() : 4);

    auto emitFromPatch = [&](const Impl::Entry& entry) {
        const Impl::PatchRecord& rec = entry.second;
        if (targetSurface && rec.surface != targetSurface.get()) {
            return;
        }
        if (filterSurfaces) {
            bool found = false;
            for (const auto& s : *filterSurfaces) {
                if (s.get() == rec.surface) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return;
            }
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
            const cv::Mat_<cv::Vec3f>* points = rec.surface->rawPointsPtr();
            const int rows = points ? points->rows : 0;
            const int cols = points ? points->cols : 0;
            cacheIt = surfaceCacheMap.emplace(rec.surface,
                SurfaceCache{center[0] * scale[0], center[1] * scale[1],
                             rows, cols, srIt->second.surface}).first;
        }
        const SurfaceCache& cache = cacheIt->second;

        // Sub-iteration bounds: stop at the tile edge AND the surface edge.
        // loadPatchCorners requires col+stride and row+stride to be < cols/rows.
        const int subColLimit = std::min(rec.i + tile, cache.cols - 1);
        const int subRowLimit = std::min(rec.j + tile, cache.rows - 1);

        for (int subJ = rec.j; subJ < subRowLimit; subJ += triStride) {
            for (int subI = rec.i; subI < subColLimit; subI += triStride) {
                const Impl::PatchRecord subRec{rec.surface, subI, subJ};
                std::array<cv::Vec3f, 4> corners;
                if (!Impl::loadPatchCorners(subRec, triStride, corners)) {
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

    // Walk active cells via states bitset, use dense cachedBoxes for each.
    if (tree && mask.activeCount > 0 && !mask.cachedBoxes.empty()) {
        for (int row = 0; row < mask.rows; ++row) {
            for (int col = 0; col < mask.cols; ++col) {
                if (!mask.isActive(row, col)) continue;
                const QBox& qb = mask.cachedBoxes[mask.index(row, col)];
                PatchRecord rec{surface.get(), col, row};
                if (tree->remove(Entry(boxDequantize(qb), rec)) && patchCount > 0) {
                    --patchCount;
                }
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
        for (int row = 0; row < mask.rows; ++row) {
            for (int col = 0; col < mask.cols; ++col) {
                if (!mask.isActive(row, col)) continue;
                const QBox& qb = mask.cachedBoxes[mask.index(row, col)];
                PatchRecord rec{surface.get(), col, row};
                if (tree->remove(Entry(boxDequantize(qb), rec)) && patchCount > 0) {
                    --patchCount;
                }
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
    }
    return !cells.empty();
}

bool SurfacePatchIndex::removeSurface(const SurfacePtr& surface)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!impl_ || !surface) {
        return false;
    }
    return impl_->removeSurfaceEntries(surface);
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

bool SurfacePatchIndex::readOnly() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return impl_ && impl_->readOnly;
}

std::optional<SurfacePatchIndex::Impl::Entry>
SurfacePatchIndex::Impl::makePatchEntry(const CellKey& key) const
{
    if (!key.surface) {
        return std::nullopt;
    }

    PatchRecord rec;
    rec.surface = key.surface.get();
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
        mask.ensureSize(rowCount, colCount);
        return mask;
    }

    // New surface - create fresh record
    SurfaceRecord& rec = surfaceRecords[surface.get()];
    rec.surface = surface;  // Keep the surface alive
    rec.mask.ensureSize(rowCount, colCount);
    return rec.mask;
}

bool SurfacePatchIndex::Impl::loadPatchCorners(const PatchRecord& rec,
                                               int stride,
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
    stride = std::max(1, stride);

    // Clamp stride to not exceed bounds
    const int effectiveColStride = std::min(stride, cols - 1 - col);
    const int effectiveRowStride = std::min(stride, rows - 1 - row);

    if (row < 0 || col < 0 || effectiveColStride <= 0 || effectiveRowStride <= 0) {
        return false;
    }

    const cv::Vec3f& p00 = (*points)(row, col);
    const cv::Vec3f& p10 = (*points)(row, col + effectiveColStride);
    const cv::Vec3f& p01 = (*points)(row + effectiveRowStride, col);
    const cv::Vec3f& p11 = (*points)(row + effectiveRowStride, col + effectiveColStride);

    if (!isValidSurfacePoint(p00) || !isValidSurfacePoint(p10)
        || !isValidSurfacePoint(p01) || !isValidSurfacePoint(p11)) {
        return false;
    }

    outCorners = {p00, p10, p11, p01};
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
    // Clamp stride to not exceed bounds
    const int maxColStride = points.cols - 1 - col;
    const int maxRowStride = points.rows - 1 - row;
    const int effectiveColStride = std::min(stride, maxColStride);
    const int effectiveRowStride = std::min(stride, maxRowStride);
    if (effectiveColStride <= 0 || effectiveRowStride <= 0) {
        return false;
    }

    // Tile corners must be valid — that's the contract for visitors that
    // load corners at tile-stride (e.g. evaluatePatch). Skip the tile
    // entirely if any corner is the -1.0f sentinel.
    const cv::Vec3f& p00 = points(row, col);
    const cv::Vec3f& p10 = points(row, col + effectiveColStride);
    const cv::Vec3f& p01 = points(row + effectiveRowStride, col);
    const cv::Vec3f& p11 = points(row + effectiveRowStride, col + effectiveColStride);
    if (!isValidSurfacePoint(p00) || !isValidSurfacePoint(p10)
        || !isValidSurfacePoint(p01) || !isValidSurfacePoint(p11)) {
        return false;
    }

    PatchRecord rec;
    rec.surface = surface.get();
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
    cv::Vec3f low{p00};
    cv::Vec3f high{p00};
    auto extend = [&](const cv::Vec3f& p) {
        if (!isValidSurfacePoint(p)) return;
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
    if (!isValidBounds(low, high)) {
        return false;
    }

    outEntry.patch = buildEntryFromBbox(rec, low, high, bboxPadding);
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
            PatchRecord rec{surface.get(), col, row};
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
        const int row = static_cast<int>(idx / mask.cols);
        const int col = static_cast<int>(idx % mask.cols);

        // Remove old entry if it exists
        if (mask.isActive(row, col)) {
            if (auto cachedBox = mask.bboxAt(row, col)) {
                PatchRecord rec{surface.get(), col, row};
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

void SurfacePatchIndex::incrementGeneration(const SurfacePtr& surface)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!impl_ || !surface) {
        return;
    }
    ++impl_->surfaceGenerations[surface.get()];
}

uint64_t SurfacePatchIndex::generation(const SurfacePtr& surface) const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!impl_ || !surface) {
        return 0;
    }
    auto it = impl_->surfaceGenerations.find(surface.get());
    return it != impl_->surfaceGenerations.end() ? it->second : 0;
}

void SurfacePatchIndex::setGeneration(const SurfacePtr& surface, uint64_t gen)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!impl_ || !surface) {
        return;
    }
    impl_->surfaceGenerations[surface.get()] = gen;
}

bool SurfacePatchIndex::segmentsEqual(const std::vector<TriangleSegment>& a,
                                      const std::vector<TriangleSegment>& b,
                                      float epsilon)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        for (int j = 0; j < 2; ++j) {
            const auto& wa = a[i].world[j];
            const auto& wb = b[i].world[j];
            if (std::abs(wa[0] - wb[0]) > epsilon ||
                std::abs(wa[1] - wb[1]) > epsilon ||
                std::abs(wa[2] - wb[2]) > epsilon) {
                return false;
            }
        }
    }
    return true;
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

    // Pad the bbox by 25% of the max extent (minimum 64 units)
    const cv::Vec3f extent = viewBbox.high - viewBbox.low;
    const float maxExtent = std::max(
        std::abs(extent[0]), std::max(std::abs(extent[1]), std::abs(extent[2])));
    const float padding = std::max(64.0f, maxExtent * 0.25f);
    viewBbox.low -= cv::Vec3f(padding, padding, padding);
    viewBbox.high += cv::Vec3f(padding, padding, padding);

    // Pre-create per-target buckets so the visitor lookup is a single
    // pointer-keyed find (no allocation, no rehash) per triangle.
    std::unordered_map<const QuadSurface*, std::vector<TriangleSegment>*> buckets;
    buckets.reserve(targets.size());
    for (const auto& t : targets) {
        if (t) buckets.emplace(t.get(), &result[t]);
    }

    // Patch-level reject: skip patches whose bbox lies entirely on one
    // side of the plane. plane.scalarp(p) returns signed distance from
    // plane to point; if all 8 bbox corners share the same side of the
    // plane (and clear the tolerance), the patch can't intersect.
    //
    // R-tree boxes are quantized through boxSnap (16-bit per axis over
    // ~101000 units → ~1.54 units/step). The stored lo/hi can each
    // shift by up to half a step inward, shrinking the box by up to
    // one full step per axis vs the true patch bbox. We expand the
    // box by one step on every side before the corner test so a
    // genuinely-intersecting patch is never wrongly rejected.
    constexpr float kQuantPad = 1.6f;  // > one quant step (~1.541)
    auto bboxStraddlesPlane = [&](const Impl::Box3& box) {
        const auto& lo = box.min_corner();
        const auto& hi = box.max_corner();
        const float lox = lo.get<0>() - kQuantPad;
        const float loy = lo.get<1>() - kQuantPad;
        const float loz = lo.get<2>() - kQuantPad;
        const float hix = hi.get<0>() + kQuantPad;
        const float hiy = hi.get<1>() + kQuantPad;
        const float hiz = hi.get<2>() + kQuantPad;
        const float d000 = plane.scalarp({lox, loy, loz});
        const float d100 = plane.scalarp({hix, loy, loz});
        const float d010 = plane.scalarp({lox, hiy, loz});
        const float d110 = plane.scalarp({hix, hiy, loz});
        const float d001 = plane.scalarp({lox, loy, hiz});
        const float d101 = plane.scalarp({hix, loy, hiz});
        const float d011 = plane.scalarp({lox, hiy, hiz});
        const float d111 = plane.scalarp({hix, hiy, hiz});
        const float dmin = std::min({d000, d100, d010, d110, d001, d101, d011, d111});
        const float dmax = std::max({d000, d100, d010, d110, d001, d101, d011, d111});
        return dmin <= clipTolerance && dmax >= -clipTolerance;
    };

    // Fused visitor: for every triangle the R-tree spits out, clip it
    // against the plane right there and append the resulting segment to
    // the per-surface bucket. Skips the intermediate TriangleCandidate
    // vector and the bySurface grouping pass entirely.
    forEachTriangleImpl(viewBbox, nullptr, &targets,
        [&](const TriangleCandidate& tri) {
            auto seg = clipTriangleToPlane(tri, plane, clipTolerance);
            if (!seg) return;
            auto it = buckets.find(tri.surface.get());
            if (it == buckets.end()) return;
            it->second->push_back(std::move(*seg));
        },
        bboxStraddlesPlane);

    // Drop empty entries that were pre-created but never received a segment.
    for (auto it = result.begin(); it != result.end(); ) {
        if (it->second.empty()) it = result.erase(it);
        else ++it;
    }

    return result;
}
