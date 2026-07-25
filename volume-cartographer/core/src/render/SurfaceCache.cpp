#include "vc/core/render/SurfaceCache.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <utils/thread_pool.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <list>
#include <mutex>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vc::render {
namespace {

constexpr int kTileSize = SurfaceCache::kTileSize;

// These are hard metadata bounds, independent of the decoded cache. A
// discontinuity inside a surface tile used to expand an 8x8 sample AABB into
// hundreds of millions of ChunkKeys before any cache limit was involved.
constexpr std::size_t kMaxDependencyMetadataKeys = 8192;
constexpr std::size_t kMaxDependencyBoxExpansion = 256;
// Eight fill workers share a 2-GiB decoded pool by default. Keeping each
// blocking batch near 256 MiB lets all eight make progress without forcing the
// first worker's freshly-prefetched chunks out before it samples them.
constexpr std::size_t kDependencyDecodedBytesPerBatch = 256ULL << 20;


// How many times a tile that came back incomplete is refilled before it is left
// alone until new chunk data arrives.
constexpr unsigned kMaxFillAttempts = 3;

int floorDiv(int value, int divisor)
{
    const int quotient = value / divisor;
    return (value % divisor != 0 && ((value < 0) != (divisor < 0))) ? quotient - 1 : quotient;
}

std::uint64_t packTile(int tu, int tv)
{
    return (std::uint64_t(std::uint32_t(tv)) << 32) | std::uint32_t(tu);
}

std::size_t envSize(const char* name, std::size_t fallback)
{
    if (const char* value = std::getenv(name)) {
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        if (end != value && parsed > 0)
            return std::size_t(parsed);
    }
    return fallback;
}

// Fill workers spend most of their time blocked on a chunk prefetch rather than
// sampling, so some oversubscription pays. It is capped because every concurrent
// fill has a live chunk working set: raising this multiplies decoded-byte
// residency as well as throughput. VC_SURFACE_FILL_WORKERS overrides for tuning.
std::size_t defaultWorkerCount()
{
    const unsigned hardware = std::thread::hardware_concurrency();
    const std::size_t cores = hardware == 0 ? 4 : std::size_t(hardware);
    return envSize("VC_SURFACE_FILL_WORKERS",
                   std::clamp<std::size_t>(cores > 2 ? cores - 2 : 1, 1, 8));
}

// One pool for tile fills. Fills block on ChunkCache prefetches (the
// sanctioned pattern for non-UI callers), so they must not share a pool with
// the resample path below.
//
// Deliberately never destroyed, on every platform: a fill worker waits on a
// chunk that a ChunkCache I/O worker resolves, and joining this pool from a
// static destructor would depend on that other pool still running. Leaking it
// removes the destruction-order coupling. This matches ChunkCache's own
// contract that the chunk worker pool outlives individual viewers.
utils::ThreadPool& surfaceFillPool()
{
    static auto* pool = new utils::ThreadPool(defaultWorkerCount());
    return *pool;
}

// Pure CPU resampling of already-resident tiles; never blocks on I/O, so it
// cannot be queued behind a blocking fill.
utils::ThreadPool& surfaceResamplePool()
{
#if defined(_WIN32)
    static auto* pool = new utils::ThreadPool(defaultWorkerCount());
    return *pool;
#else
    static utils::ThreadPool pool(defaultWorkerCount());
    return pool;
#endif
}

// ---------------------------------------------------------------------------
// Volume access
//
// Mirrors the private helpers of ChunkedPlaneSampler (toLevelCoord /
// inLevelBounds / readVoxel / sampleNearest / sampleTrilinear) so a stored
// sample is the value the legacy render path would have produced for the same
// coordinate. Kept local rather than shared because the fill kernel iterates
// `w` innermost over a chunk set that a blocking prefetch has already
// resolved, which wants a different chunk-pinning policy than the interactive
// sampler's deliberately bounded window.
// ---------------------------------------------------------------------------

struct LevelAccess {
    std::array<int, 3> shape{};
    std::array<int, 3> chunkShape{};
    IChunkedArray::LevelTransform transform;
    uint8_t fill = 0;
};

LevelAccess makeLevelAccess(IChunkedArray& array, int level)
{
    LevelAccess access;
    access.shape = array.shape(level);
    access.chunkShape = array.chunkShape(level);
    access.transform = array.levelTransform(level);
    access.fill = static_cast<uint8_t>(std::clamp(std::lround(array.fillValue()), 0L, 255L));
    return access;
}

bool hasSampleableLevel(const LevelAccess& access)
{
    return access.shape[0] > 0 && access.shape[1] > 0 && access.shape[2] > 0 &&
           access.chunkShape[0] > 0 && access.chunkShape[1] > 0 && access.chunkShape[2] > 0;
}

bool finiteCoord(const cv::Vec3f& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

bool surfaceSentinel(const cv::Vec3f& p)
{
    return !finiteCoord(p) || p[0] == -1.0f || p[1] == -1.0f || p[2] == -1.0f ||
           (p[0] == 0.0f && p[1] == 0.0f && p[2] == 0.0f);
}

cv::Vec3f toLevelCoord(const LevelAccess& access, const cv::Vec3f& p0)
{
    const auto& t = access.transform;
    return {
        float(double(p0[0]) * t.scaleFromLevel0[0] + t.offsetFromLevel0[0]),
        float(double(p0[1]) * t.scaleFromLevel0[1] + t.offsetFromLevel0[1]),
        float(double(p0[2]) * t.scaleFromLevel0[2] + t.offsetFromLevel0[2]),
    };
}

bool inLevelBounds(const std::array<int, 3>& shape, float z, float y, float x)
{
    return z >= 0.0f && y >= 0.0f && x >= 0.0f && z < float(shape[0]) &&
           y < float(shape[1]) && x < float(shape[2]);
}

// Chunks resolved for one tile fill.
//
// A ChunkResult owns a shared_ptr to the decoded bytes, so retaining every
// result a fill touches keeps that memory alive even after the backing pool has
// evicted it -- the pool then reports itself at capacity while the process holds
// far more. This is the same trap ChunkedPlaneSampler::LocalChunkCache documents,
// and it scales with the fill worker count, so an unbounded version here turned
// every concurrent fill into another decoded-byte cache with no ceiling.
//
// A small window keeps the hot consecutive-lookup fast path -- the `w`-innermost
// kernel walks one voxel at a time along the normal, so successive taps
// overwhelmingly hit the same chunk -- while bounding what one fill can pin. The
// blocking prefetch has already made the tile's whole dependency set resident, so
// a lookup that falls out of the window is a cheap re-read, not a refetch.
struct FillChunkSet {
    static constexpr std::size_t kMaxPinnedChunks = 8;

    explicit FillChunkSet(IChunkedArray& array_) : array(array_) {}

    const ChunkResult& get(const ChunkKey& key)
    {
        if (lastResult && lastKey == key)
            return *lastResult;
        auto it = chunks.find(key);
        if (it == chunks.end()) {
            if (chunks.size() >= kMaxPinnedChunks) {
                lastResult = nullptr;
                chunks.clear();
            }
            it = chunks
                     .emplace(key,
                              array.getChunkIfCached(key.level, key.iz, key.iy, key.ix))
                     .first;
        }
        lastKey = key;
        lastResult = &it->second;
        return it->second;
    }

    IChunkedArray& array;
    std::unordered_map<ChunkKey, ChunkResult, ChunkKeyHash> chunks;
    ChunkKey lastKey{};
    const ChunkResult* lastResult = nullptr;
};

bool readVoxel(FillChunkSet& chunks,
               const LevelAccess& access,
               int level,
               int iz,
               int iy,
               int ix,
               uint8_t& out)
{
    const auto& shape = access.shape;
    if (unsigned(iz) >= unsigned(shape[0]) || unsigned(iy) >= unsigned(shape[1]) ||
        unsigned(ix) >= unsigned(shape[2])) {
        out = access.fill;
        return true;
    }

    const auto& chunkShape = access.chunkShape;
    const int cz = iz / chunkShape[0];
    const int cy = iy / chunkShape[1];
    const int cx = ix / chunkShape[2];
    const ChunkResult& result = chunks.get({level, cz, cy, cx});
    if (result.status == ChunkStatus::MissQueued || result.status == ChunkStatus::Missing ||
        result.status == ChunkStatus::Error)
        return false;
    if (result.status == ChunkStatus::AllFill) {
        out = access.fill;
        return true;
    }
    if (result.status != ChunkStatus::Data || !result.bytes)
        return false;

    const int lz = iz - cz * chunkShape[0];
    const int ly = iy - cy * chunkShape[1];
    const int lx = ix - cx * chunkShape[2];
    const std::size_t offset =
        (std::size_t(lz) * std::size_t(chunkShape[1]) + std::size_t(ly)) *
            std::size_t(chunkShape[2]) +
        std::size_t(lx);
    if (offset >= result.bytes->size())
        return false;
    out = std::to_integer<uint8_t>((*result.bytes)[offset]);
    return true;
}

bool sampleNearestLevel(FillChunkSet& chunks,
                        const LevelAccess& access,
                        int level,
                        const cv::Vec3f& p,
                        uint8_t& out)
{
    const auto& shape = access.shape;
    const float x = p[0], y = p[1], z = p[2];
    if (!inLevelBounds(shape, z, y, x)) {
        out = access.fill;
        return true;
    }
    const int ix = std::clamp(int(x + 0.5f), 0, shape[2] - 1);
    const int iy = std::clamp(int(y + 0.5f), 0, shape[1] - 1);
    const int iz = std::clamp(int(z + 0.5f), 0, shape[0] - 1);
    return readVoxel(chunks, access, level, iz, iy, ix, out);
}

bool sampleTrilinearLevel(FillChunkSet& chunks,
                          const LevelAccess& access,
                          int level,
                          const cv::Vec3f& p,
                          uint8_t& out)
{
    const auto& shape = access.shape;
    const float x = p[0], y = p[1], z = p[2];
    if (!inLevelBounds(shape, z, y, x)) {
        out = access.fill;
        return true;
    }

    const int ix = int(x);
    const int iy = int(y);
    const int iz = int(z);
    const float fx = x - float(ix);
    const float fy = y - float(iy);
    const float fz = z - float(iz);

    uint8_t v000 = 0, v001 = 0, v010 = 0, v011 = 0;
    uint8_t v100 = 0, v101 = 0, v110 = 0, v111 = 0;
    bool ready = true;
    ready = readVoxel(chunks, access, level, iz, iy, ix, v000) && ready;
    ready = readVoxel(chunks, access, level, iz, iy, ix + 1, v001) && ready;
    ready = readVoxel(chunks, access, level, iz, iy + 1, ix, v010) && ready;
    ready = readVoxel(chunks, access, level, iz, iy + 1, ix + 1, v011) && ready;
    ready = readVoxel(chunks, access, level, iz + 1, iy, ix, v100) && ready;
    ready = readVoxel(chunks, access, level, iz + 1, iy, ix + 1, v101) && ready;
    ready = readVoxel(chunks, access, level, iz + 1, iy + 1, ix, v110) && ready;
    ready = readVoxel(chunks, access, level, iz + 1, iy + 1, ix + 1, v111) && ready;
    if (!ready)
        return false;

    const float c00 = std::fma(fx, float(v001) - float(v000), float(v000));
    const float c01 = std::fma(fx, float(v011) - float(v010), float(v010));
    const float c10 = std::fma(fx, float(v101) - float(v100), float(v100));
    const float c11 = std::fma(fx, float(v111) - float(v110), float(v110));
    const float c0 = std::fma(fy, c01 - c00, c00);
    const float c1 = std::fma(fy, c11 - c10, c10);
    out = static_cast<uint8_t>(std::clamp(std::fma(fz, c1 - c0, c0), 0.0f, 255.0f));
    return true;
}

bool sampleLevelPoint(FillChunkSet& chunks,
                      const LevelAccess& access,
                      int level,
                      const cv::Vec3f& p0,
                      vc::Sampling sampling,
                      uint8_t& out)
{
    const cv::Vec3f p = toLevelCoord(access, p0);
    if (sampling == vc::Sampling::Nearest)
        return sampleNearestLevel(chunks, access, level, p, out);
    return sampleTrilinearLevel(chunks, access, level, p, out);
}

// ---------------------------------------------------------------------------
// Tile identity
// ---------------------------------------------------------------------------

struct TileKey {
    int level = 0;
    int tu = 0;
    int tv = 0;
    friend bool operator==(const TileKey&, const TileKey&) = default;
};

struct TileKeyHash {
    std::size_t operator()(const TileKey& key) const noexcept
    {
        std::size_t seed = 0;
        auto combine = [&seed](int value) {
            seed ^= std::hash<int>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        };
        combine(key.level);
        combine(key.tu);
        combine(key.tv);
        return seed;
    }
};

struct SurfaceTile {
    // (i * kTileSize + j) * wCount + k -- `w` innermost so both the fill
    // kernel and the composite reduction walk the band contiguously.
    std::vector<uint8_t> data;
    std::vector<uint8_t> valid;        // coord is a surface point
    std::vector<uint8_t> validOffset;  // coord *and* normal are usable
    bool complete = false;
};

using SurfaceTilePtr = std::shared_ptr<const SurfaceTile>;

// Nominal (u, v) rectangle a tile covers.
cv::Rect2d tileNominalRect(int level, int tu, int tv)
{
    const double span = double(kTileSize) * std::ldexp(1.0, level);
    return {double(tu) * span, double(tv) * span, span, span};
}

// Per-layer band indices. `w` is constant across the frame, so the two band
// slices each layer blends and their weight are computed once per view.
struct BandTap {
    int k0 = 0;
    int k1 = 0;
    float t = 0.0f;
};

// The reduction sampleView applies across the band.
struct Reduction {
    int numLayers = 1;
    double firstOffset = 0.0;
    double layerStep = 1.0;
    bool reduce = false;
    CompositeParams params;
};

} // namespace

// ===========================================================================
// SurfaceGeometryTileCache
// ===========================================================================

struct SurfaceGeometryTileCache::State {
    std::shared_ptr<QuadSurface> surface;
    std::size_t maxTiles = 192;

    struct Slot {
        std::shared_ptr<const Tile> tile;
        bool ready = false;
    };

    mutable std::mutex mutex;
    std::condition_variable ready;
    std::unordered_map<TileKey, std::shared_ptr<Slot>, TileKeyHash> slots;
    std::list<TileKey> lru;  // front = most recently used
    std::unordered_map<TileKey, std::list<TileKey>::iterator, TileKeyHash> lruPos;
    std::uint64_t epoch = 0;

    void touchLocked(const TileKey& key)
    {
        if (auto it = lruPos.find(key); it != lruPos.end())
            lru.erase(it->second);
        lru.push_front(key);
        lruPos[key] = lru.begin();
    }

    void eraseLocked(const TileKey& key)
    {
        slots.erase(key);
        if (auto pos = lruPos.find(key); pos != lruPos.end()) {
            lru.erase(pos->second);
            lruPos.erase(pos);
        }
    }

    void trimLocked()
    {
        auto it = lru.end();
        while (lru.size() > maxTiles && it != lru.begin()) {
            --it;
            const TileKey victim = *it;
            auto slot = slots.find(victim);
            // A slot another thread is still awaiting must stay put.
            if (slot != slots.end() && !slot->second->ready)
                continue;
            it = lru.erase(it);
            lruPos.erase(victim);
            slots.erase(victim);
        }
    }
};

SurfaceGeometryTileCache::SurfaceGeometryTileCache(std::shared_ptr<QuadSurface> surface,
                                                   std::size_t maxTiles)
    : _state(std::make_shared<State>())
{
    _state->surface = std::move(surface);
    _state->maxTiles = std::max<std::size_t>(8, maxTiles);
}

SurfaceGeometryTileCache::~SurfaceGeometryTileCache() = default;

std::shared_ptr<const SurfaceGeometryTileCache::Tile>
SurfaceGeometryTileCache::get(int level, int tu, int tv)
{
    if (!_state->surface)
        return nullptr;

    const TileKey key{level, tu, tv};
    std::shared_ptr<State::Slot> slot;
    std::uint64_t epoch = 0;
    {
        std::unique_lock lock(_state->mutex);
        epoch = _state->epoch;
        if (auto it = _state->slots.find(key); it != _state->slots.end()) {
            slot = it->second;
            _state->touchLocked(key);
            if (!slot->ready) {
                // Another thread is computing this tile; wait for it instead
                // of duplicating the gen().
                _state->ready.wait(lock, [&] { return slot->ready; });
            }
            return slot->tile;
        }
        slot = std::make_shared<State::Slot>();
        _state->slots.emplace(key, slot);
        _state->touchLocked(key);
    }

    // gen() hands back views into thread_local scratch that this thread's next
    // call reuses, so the tile must own copies of both mats.
    std::shared_ptr<const Tile> tile;
    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<cv::Vec3f> normals;
    const double step = std::ldexp(1.0, level);  // nominal units per sample
    const float genScale = static_cast<float>(1.0 / step);
    const double u0 = double(tu) * double(kTileSize) * step;
    const double v0 = double(tv) * double(kTileSize) * step;
    // gen() derives its upper-left grid position from offset/scale + center, so
    // offset = (u0, v0) * genScale puts sample (0, 0) at nominal (u0, v0) and
    // steps `step` nominal units per sample.
    const cv::Vec3f offset(static_cast<float>(u0 * double(genScale)),
                           static_cast<float>(v0 * double(genScale)), 0.0f);
    try {
        _state->surface->gen(&coords, &normals, cv::Size(kTileSize, kTileSize), {0, 0, 0},
                             genScale, offset);
    } catch (const std::exception&) {
        coords.release();
        normals.release();
    }
    if (!coords.empty() && coords.rows >= kTileSize && coords.cols >= kTileSize) {
        auto built = std::make_shared<Tile>();
        built->coords = coords(cv::Rect(0, 0, kTileSize, kTileSize)).clone();
        if (!normals.empty() && normals.rows >= kTileSize && normals.cols >= kTileSize)
            built->normals = normals(cv::Rect(0, 0, kTileSize, kTileSize)).clone();
        else
            built->normals =
                cv::Mat_<cv::Vec3f>(kTileSize, kTileSize, cv::Vec3f(NAN, NAN, NAN));
        built->valid.create(kTileSize, kTileSize);
        built->validOffset.create(kTileSize, kTileSize);
        for (int y = 0; y < kTileSize; ++y) {
            const cv::Vec3f* coordRow = built->coords.ptr<cv::Vec3f>(y);
            const cv::Vec3f* normalRow = built->normals.ptr<cv::Vec3f>(y);
            uint8_t* validRow = built->valid.ptr<uint8_t>(y);
            uint8_t* offsetRow = built->validOffset.ptr<uint8_t>(y);
            for (int x = 0; x < kTileSize; ++x) {
                const bool coordValid = !surfaceSentinel(coordRow[x]);
                validRow[x] = coordValid ? 1 : 0;
                offsetRow[x] = (coordValid && finiteCoord(normalRow[x])) ? 1 : 0;
            }
        }
        tile = std::move(built);
    }

    {
        std::lock_guard lock(_state->mutex);
        slot->tile = tile;
        slot->ready = true;
        if (epoch != _state->epoch) {
            // Invalidated while generating: hand this caller what it computed
            // but do not publish stale geometry for anyone else.
            _state->eraseLocked(key);
        } else {
            _state->trimLocked();
        }
        _state->ready.notify_all();
    }
    return tile;
}

void SurfaceGeometryTileCache::invalidateAll()
{
    std::lock_guard lock(_state->mutex);
    ++_state->epoch;
    std::vector<TileKey> victims;
    victims.reserve(_state->slots.size());
    for (const auto& [key, slot] : _state->slots) {
        if (slot->ready)
            victims.push_back(key);
    }
    for (const TileKey& key : victims)
        _state->eraseLocked(key);
}

cv::Rect2d SurfaceGeometryTileCache::gridCellsToNominal(const cv::Rect& gridCells) const
{
    if (!_state->surface || gridCells.empty())
        return {};
    // gen() bilinearly warps the grid and a normal differences its cell's
    // neighbours, so an edited cell influences samples up to two cells away.
    const cv::Rect padded(gridCells.x - 2, gridCells.y - 2, gridCells.width + 4,
                          gridCells.height + 4);
    const cv::Vec2d lo = _state->surface->gridToSurface({double(padded.x), double(padded.y)});
    const cv::Vec2d hi = _state->surface->gridToSurface(
        {double(padded.x + padded.width), double(padded.y + padded.height)});
    if (!std::isfinite(lo[0]) || !std::isfinite(lo[1]) || !std::isfinite(hi[0]) ||
        !std::isfinite(hi[1]))
        return {};
    return {std::min(lo[0], hi[0]), std::min(lo[1], hi[1]), std::abs(hi[0] - lo[0]),
            std::abs(hi[1] - lo[1])};
}

void SurfaceGeometryTileCache::invalidateSurfaceRegion(const cv::Rect& gridCells)
{
    const cv::Rect2d region = gridCellsToNominal(gridCells);
    if (region.width <= 0.0 || region.height <= 0.0) {
        invalidateAll();
        return;
    }

    std::lock_guard lock(_state->mutex);
    ++_state->epoch;
    std::vector<TileKey> victims;
    for (const auto& [key, slot] : _state->slots) {
        if (!slot->ready)
            continue;
        if ((tileNominalRect(key.level, key.tu, key.tv) & region).area() > 0.0)
            victims.push_back(key);
    }
    for (const TileKey& key : victims)
        _state->eraseLocked(key);
}

std::size_t SurfaceGeometryTileCache::size() const
{
    std::lock_guard lock(_state->mutex);
    return _state->slots.size();
}

// ===========================================================================
// SurfaceCache
// ===========================================================================

struct SurfaceCache::State {
    std::shared_ptr<IChunkedArray> volume;
    std::shared_ptr<QuadSurface> surface;
    std::shared_ptr<SurfaceGeometryTileCache> geometry;
    Options options;
    int levels = 1;
    int wMin = -16;
    int wCount = 32;
    // Band index of w == 0, or -1 when the configured band excludes it.
    int zeroIndex = 16;
    std::size_t tileBytes = 0;
    std::size_t maxOutstandingFills = 8;

    mutable std::mutex mutex;
    std::unordered_map<TileKey, SurfaceTilePtr, TileKeyHash> tiles;
    std::list<TileKey> lru;  // front = most recently used
    std::unordered_map<TileKey, std::list<TileKey>::iterator, TileKeyHash> lruPos;
    std::size_t bytes = 0;
    std::size_t capacity = 0;
    std::size_t incomplete = 0;
    std::unordered_set<TileKey, TileKeyHash> outstanding;
    std::unordered_set<TileKey, TileKeyHash> viewTiles;
    // Fills that stored an incomplete tile, per tile. A tile can be incomplete
    // because a chunk was evicted mid-fill or is genuinely missing; re-queueing
    // it unconditionally would spin forever, since publishing a tile wakes the
    // render loop, which calls requestView again. Bounded here and reset when
    // the backing array reports new chunk data -- the only event that can
    // actually change the outcome.
    std::unordered_map<TileKey, unsigned, TileKeyHash> failedAttempts;
    IChunkedArray::ChunkReadyCallbackId chunkReadyId = 0;
    std::uint64_t viewGeneration = 0;
    std::uint64_t epoch = 0;
    bool shuttingDown = false;
    std::unordered_map<TileReadyCallbackId, std::function<void()>> listeners;
    TileReadyCallbackId nextListenerId = 1;

    void touchLocked(const TileKey& key)
    {
        if (auto it = lruPos.find(key); it != lruPos.end())
            lru.erase(it->second);
        lru.push_front(key);
        lruPos[key] = lru.begin();
    }

    void dropLocked(const TileKey& key)
    {
        auto it = tiles.find(key);
        if (it != tiles.end()) {
            if (it->second && !it->second->complete && incomplete > 0)
                --incomplete;
            bytes -= std::min(bytes, tileBytes);
            // Eviction only drops this map's reference; a render already
            // reading the tile keeps it alive through its own shared_ptr.
            tiles.erase(it);
        }
        if (auto pos = lruPos.find(key); pos != lruPos.end()) {
            lru.erase(pos->second);
            lruPos.erase(pos);
        }
    }

    void enforceCapacityLocked()
    {
        while (bytes > capacity && !lru.empty())
            dropLocked(lru.back());
    }

    static void runFill(const std::shared_ptr<State>& self, TileKey key, std::uint64_t epoch);
    static void notifyTileReady(const std::shared_ptr<State>& self);
    static SampleStats sampleReduced(const State& self,
                                     int startLevel,
                                     double uMin,
                                     double vMin,
                                     double scale,
                                     double w,
                                     const Reduction& reduction,
                                     cv::Mat_<uint8_t>& out,
                                     cv::Mat_<uint8_t>& coverage);
};

namespace {

struct DependencyBatch {
    std::vector<ChunkKey> keys;
    bool overflow = false;
};

struct ChunkBox {
    int cx0 = 0;
    int cx1 = -1;
    int cy0 = 0;
    int cy1 = -1;
    int cz0 = 0;
    int cz1 = -1;
};

std::size_t chunkBoxCountCapped(const ChunkBox& box, std::size_t cap)
{
    if (box.cx1 < box.cx0 || box.cy1 < box.cy0 || box.cz1 < box.cz0)
        return 0;
    std::size_t count = 1;
    for (const std::size_t extent :
         {std::size_t(box.cx1 - box.cx0 + 1),
          std::size_t(box.cy1 - box.cy0 + 1),
          std::size_t(box.cz1 - box.cz0 + 1)}) {
        if (extent > cap / count)
            return cap + 1;
        count *= extent;
    }
    return count;
}

// Collect one region's source chunks without allowing a surface discontinuity
// to materialize its full 3-D AABB. Regions whose box is too large are
// subdivided down to individual pixels; a single pixel then contributes only
// the exact integer normal samples the filler will read.
DependencyBatch collectRegionDependencies(
    const SurfaceGeometryTileCache::Tile& geometry,
    const LevelAccess& access,
    int level,
    int wMin,
    int wCount,
    int xBegin,
    int xEnd,
    int yBegin,
    int yEnd,
    std::size_t maxKeys)
{
    std::unordered_set<ChunkKey, ChunkKeyHash> keys;
    keys.reserve(std::min<std::size_t>(maxKeys, 1024));
    const auto& chunkShape = access.chunkShape;
    const auto& shape = access.shape;
    const float wLow = float(wMin);
    const float wHigh = float(wMin + wCount - 1);

    DependencyBatch result;
    auto addKey = [&](const ChunkKey& key) {
        if (keys.contains(key))
            return true;
        if (keys.size() >= maxKeys) {
            result.overflow = true;
            return false;
        }
        keys.insert(key);
        return true;
    };

    auto addPoint = [&](const cv::Vec3f& p0) {
        const cv::Vec3f p = toLevelCoord(access, p0);
        if (!finiteCoord(p) || !inLevelBounds(shape, p[2], p[1], p[0]))
            return true;
        const int vx0 = std::clamp(int(std::floor(p[0])), 0, shape[2] - 1);
        const int vy0 = std::clamp(int(std::floor(p[1])), 0, shape[1] - 1);
        const int vz0 = std::clamp(int(std::floor(p[2])), 0, shape[0] - 1);
        const int vx1 = std::min(vx0 + 1, shape[2] - 1);
        const int vy1 = std::min(vy0 + 1, shape[1] - 1);
        const int vz1 = std::min(vz0 + 1, shape[0] - 1);
        for (const int vz : {vz0, vz1}) {
            for (const int vy : {vy0, vy1}) {
                for (const int vx : {vx0, vx1}) {
                    if (!addKey({level, vz / chunkShape[0],
                                 vy / chunkShape[1], vx / chunkShape[2]})) {
                        return false;
                    }
                }
            }
        }
        return true;
    };

    auto clampedVoxel = [](double value, int extent, bool high) {
        double voxel = std::floor(value);
        if (high)
            voxel += 1.0; // trilinear neighbour
        if (voxel <= 0.0)
            return 0;
        if (voxel >= double(extent - 1))
            return extent - 1;
        return static_cast<int>(voxel);
    };

    std::function<void(int, int, int, int)> collect;
    collect = [&](int x0, int x1, int y0, int y1) {
        if (result.overflow || x0 >= x1 || y0 >= y1)
            return;

        double lo[3] = {std::numeric_limits<double>::max(),
                        std::numeric_limits<double>::max(),
                        std::numeric_limits<double>::max()};
        double hi[3] = {std::numeric_limits<double>::lowest(),
                        std::numeric_limits<double>::lowest(),
                        std::numeric_limits<double>::lowest()};
        bool any = false;
        for (int y = y0; y < y1; ++y) {
            const cv::Vec3f* coordRow = geometry.coords.ptr<cv::Vec3f>(y);
            const cv::Vec3f* normalRow = geometry.normals.ptr<cv::Vec3f>(y);
            const uint8_t* validRow = geometry.valid.ptr<uint8_t>(y);
            const uint8_t* offsetRow = geometry.validOffset.ptr<uint8_t>(y);
            for (int x = x0; x < x1; ++x) {
                if (!validRow[x])
                    continue;
                const cv::Vec3f& coord = coordRow[x];
                cv::Vec3f ends[2] = {coord, coord};
                if (offsetRow[x]) {
                    ends[0] = coord + normalRow[x] * wLow;
                    ends[1] = coord + normalRow[x] * wHigh;
                }
                for (const cv::Vec3f& end : ends) {
                    const cv::Vec3f p = toLevelCoord(access, end);
                    if (!finiteCoord(p))
                        continue;
                    for (int axis = 0; axis < 3; ++axis) {
                        lo[axis] = std::min(lo[axis], double(p[axis]));
                        hi[axis] = std::max(hi[axis], double(p[axis]));
                    }
                    any = true;
                }
            }
        }
        if (!any)
            return;

        const ChunkBox box{
            clampedVoxel(lo[0], shape[2], false) / chunkShape[2],
            clampedVoxel(hi[0], shape[2], true) / chunkShape[2],
            clampedVoxel(lo[1], shape[1], false) / chunkShape[1],
            clampedVoxel(hi[1], shape[1], true) / chunkShape[1],
            clampedVoxel(lo[2], shape[0], false) / chunkShape[0],
            clampedVoxel(hi[2], shape[0], true) / chunkShape[0],
        };
        if (chunkBoxCountCapped(box, kMaxDependencyBoxExpansion) <=
            kMaxDependencyBoxExpansion) {
            for (int cz = box.cz0; cz <= box.cz1 && !result.overflow; ++cz) {
                for (int cy = box.cy0; cy <= box.cy1 && !result.overflow; ++cy) {
                    for (int cx = box.cx0; cx <= box.cx1; ++cx) {
                        if (!addKey({level, cz, cy, cx}))
                            break;
                    }
                }
            }
            return;
        }

        const int width = x1 - x0;
        const int height = y1 - y0;
        if (width > 1 || height > 1) {
            if (width >= height && width > 1) {
                const int middle = x0 + width / 2;
                collect(x0, middle, y0, y1);
                collect(middle, x1, y0, y1);
            } else {
                const int middle = y0 + height / 2;
                collect(x0, x1, y0, middle);
                collect(x0, x1, middle, y1);
            }
            return;
        }

        // A single malformed or extremely oblique normal must not expand a
        // diagonal bounding box. Queue only the points the sampling loop reads.
        const cv::Vec3f& coord = geometry.coords(y0, x0);
        if (!geometry.validOffset(y0, x0)) {
            (void)addPoint(coord);
            return;
        }
        const cv::Vec3f& normal = geometry.normals(y0, x0);
        for (int k = 0; k < wCount && !result.overflow; ++k)
            (void)addPoint(coord + normal * float(wMin + k));
    };

    collect(std::clamp(xBegin, 0, kTileSize),
            std::clamp(xEnd, 0, kTileSize),
            std::clamp(yBegin, 0, kTileSize),
            std::clamp(yEnd, 0, kTileSize));

    result.keys.assign(keys.begin(), keys.end());
    std::sort(result.keys.begin(), result.keys.end(),
              [](const ChunkKey& lhs, const ChunkKey& rhs) {
                  return std::tie(lhs.level, lhs.iz, lhs.iy, lhs.ix) <
                         std::tie(rhs.level, rhs.iz, rhs.iy, rhs.ix);
              });
    return result;
}

} // namespace

void SurfaceCache::State::notifyTileReady(const std::shared_ptr<State>& self)
{
    std::vector<std::function<void()>> callbacks;
    {
        std::lock_guard lock(self->mutex);
        callbacks.reserve(self->listeners.size());
        for (const auto& [id, callback] : self->listeners)
            callbacks.push_back(callback);
    }
    for (const auto& callback : callbacks) {
        if (callback)
            callback();
    }
}

void SurfaceCache::State::runFill(const std::shared_ptr<State>& self,
                                  TileKey key,
                                  std::uint64_t epoch)
{
#if defined(_OPENMP)
    // QuadSurface::gen() has OpenMP loops, but runFill is already parallel at
    // tile granularity. Avoid building a full nested team for every fill.
    omp_set_num_threads(1);
#endif

    bool published = false;
    {
        std::lock_guard lock(self->mutex);
        if (self->shuttingDown || epoch != self->epoch || !self->viewTiles.count(key)) {
            self->outstanding.erase(key);
            return;
        }
    }

    auto release = [&]() {
        std::lock_guard lock(self->mutex);
        self->outstanding.erase(key);
    };

    std::shared_ptr<const SurfaceGeometryTileCache::Tile> geometry;
    if (self->geometry)
        geometry = self->geometry->get(key.level, key.tu, key.tv);
    if (!geometry || !self->volume) {
        release();
        return;
    }

    const LevelAccess access = makeLevelAccess(*self->volume, key.level);
    if (!hasSampleableLevel(access)) {
        release();
        return;
    }

    const int wCount = self->wCount;
    const int wMin = self->wMin;
    const std::size_t samples = std::size_t(kTileSize) * std::size_t(kTileSize);
    auto tile = std::make_shared<SurfaceTile>();
    tile->data.assign(samples * std::size_t(wCount), 0);
    tile->valid.assign(samples, 0);
    tile->validOffset.assign(samples, 0);
    for (int y = 0; y < kTileSize; ++y) {
        std::memcpy(tile->valid.data() + std::size_t(y) * kTileSize,
                    geometry->valid.ptr<uint8_t>(y), kTileSize);
        std::memcpy(tile->validOffset.data() + std::size_t(y) * kTileSize,
                    geometry->validOffset.ptr<uint8_t>(y), kTileSize);
    }
    if (self->zeroIndex < 0)
        tile->valid = tile->validOffset;
    const vc::Sampling sampling = self->options.sampling;
    std::size_t decodedChunkBytes =
        self->volume->dtype() == ChunkDtype::UInt16 ? 2 : 1;
    for (const int dimension : access.chunkShape) {
        if (dimension <= 0 ||
            decodedChunkBytes >
                std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(dimension)) {
            decodedChunkBytes = std::numeric_limits<std::size_t>::max();
            break;
        }
        decodedChunkBytes *= static_cast<std::size_t>(dimension);
    }
    const std::size_t dependencyKeyLimit = std::min(
        kMaxDependencyMetadataKeys,
        std::max<std::size_t>(
            1, kDependencyDecodedBytesPerBatch /
                   std::max<std::size_t>(1, decodedChunkBytes)));

    bool complete = true;
    auto stillCurrent = [&]() {
        std::lock_guard lock(self->mutex);
        return !self->shuttingDown && epoch == self->epoch &&
               self->viewTiles.count(key);
    };
    // The normal case remains one parallel prefetch for the whole tile.
    // Pathological/discontinuous geometry is adaptively split only when its
    // exact dependency set exceeds the hard metadata bound.
    std::function<bool(int, int, int, int)> fillRegion;
    fillRegion = [&](int x0, int x1, int y0, int y1) {
        if (!stillCurrent())
            return false;

        DependencyBatch dependencies = collectRegionDependencies(
            *geometry, access, key.level, wMin, wCount, x0, x1, y0, y1,
            dependencyKeyLimit);
        if (dependencies.overflow) {
            const int width = x1 - x0;
            const int height = y1 - y0;
            if (width <= 1 && height <= 1) {
                complete = false;
                return true;
            }
            std::vector<ChunkKey>().swap(dependencies.keys);
            if (width >= height && width > 1) {
                const int middle = x0 + width / 2;
                return fillRegion(x0, middle, y0, y1) &&
                       fillRegion(middle, x1, y0, y1);
            }
            const int middle = y0 + height / 2;
            return fillRegion(x0, x1, y0, middle) &&
                   fillRegion(x0, x1, middle, y1);
        }

        if (!dependencies.keys.empty()) {
            self->volume->prefetchChunks(dependencies.keys, /*wait=*/true);
        }
        if (!stillCurrent())
            return false;
        std::vector<ChunkKey>().swap(dependencies.keys);

        FillChunkSet chunks(*self->volume);
        for (int y = y0; y < y1; ++y) {
            const cv::Vec3f* coordRow = geometry->coords.ptr<cv::Vec3f>(y);
            const cv::Vec3f* normalRow = geometry->normals.ptr<cv::Vec3f>(y);
            for (int x = x0; x < x1; ++x) {
                const std::size_t index =
                    std::size_t(y) * kTileSize + std::size_t(x);
                if (!tile->valid[index] && !tile->validOffset[index])
                    continue;
                uint8_t* out =
                    tile->data.data() + index * std::size_t(wCount);
                if (!tile->validOffset[index]) {
                    if (self->zeroIndex >= 0 &&
                        !sampleLevelPoint(chunks, access, key.level,
                                          coordRow[x], sampling,
                                          out[self->zeroIndex])) {
                        complete = false;
                    }
                    continue;
                }
                const cv::Vec3f& coord = coordRow[x];
                const cv::Vec3f& normal = normalRow[x];
                for (int k = 0; k < wCount; ++k) {
                    const cv::Vec3f p =
                        coord + normal * float(wMin + k);
                    if (!sampleLevelPoint(chunks, access, key.level, p,
                                          sampling, out[k])) {
                        complete = false;
                    }
                }
            }
        }
        return true;
    };

    if (!fillRegion(0, kTileSize, 0, kTileSize)) {
        release();
        return;
    }
    tile->complete = complete;

    {
        std::lock_guard lock(self->mutex);
        self->outstanding.erase(key);
        if (epoch == self->epoch) {
            // Never discard computed work: an incomplete tile is stored and
            // used anyway, and requestView re-queues it.
            self->dropLocked(key);
            self->tiles.emplace(key, SurfaceTilePtr(std::move(tile)));
            self->bytes += self->tileBytes;
            if (complete) {
                self->failedAttempts.erase(key);
            } else {
                ++self->incomplete;
                ++self->failedAttempts[key];
            }
            self->touchLocked(key);
            self->enforceCapacityLocked();
            published = true;
        }
    }

    if (published)
        notifyTileReady(self);
}

SurfaceCache::SurfaceCache(std::shared_ptr<IChunkedArray> volume,
                           std::shared_ptr<QuadSurface> surface,
                           Options options)
    : _state(std::make_shared<State>())
{
    _state->volume = std::move(volume);
    _state->surface = std::move(surface);
    _state->options = options;
    _state->wMin = options.wMin;
    _state->wCount = std::max(1, options.wCount);
    _state->zeroIndex = (-_state->wMin >= 0 && -_state->wMin < _state->wCount)
                            ? -_state->wMin
                            : -1;
    const int arrayLevels = _state->volume ? std::max(1, _state->volume->numLevels()) : 1;
    _state->levels = options.levels > 0 ? std::min(options.levels, arrayLevels) : arrayLevels;
    _state->tileBytes = std::size_t(kTileSize) * std::size_t(kTileSize) *
                            std::size_t(_state->wCount) +
                        2 * std::size_t(kTileSize) * std::size_t(kTileSize);
    _state->capacity = std::max(options.byteCapacity, _state->tileBytes * 4);
    _state->geometry = options.geometry
                           ? options.geometry
                           : std::make_shared<SurfaceGeometryTileCache>(_state->surface);
    // Keep at most one running and one queued tile per fill worker. The worker
    // pool is the consumer; admitting hundreds of tile producers merely lets
    // repeated pans build a large stale-work queue ahead of it.
    const std::size_t fillWorkers =
        options.fillWorkers > 0 ? options.fillWorkers
                                : surfaceFillPool().worker_count();
    _state->maxOutstandingFills =
        envSize("VC_SURFACE_MAX_OUTSTANDING_FILLS",
                std::max<std::size_t>(1, 2 * fillWorkers));

    // New chunk data is the one event that can turn an incomplete tile
    // complete, so it is what re-arms the retry bound. Captured weakly: this
    // fires on I/O threads and may already be running when the listener is
    // removed.
    if (_state->volume) {
        std::weak_ptr<State> weak = _state;
        _state->chunkReadyId = _state->volume->addChunkReadyListener([weak]() {
            if (auto state = weak.lock()) {
                std::lock_guard lock(state->mutex);
                state->failedAttempts.clear();
            }
        });
    }
}

SurfaceCache::~SurfaceCache()
{
    shutdown();
}

void SurfaceCache::shutdown()
{
    // Does not wait for in-flight fills, and must not: a fill blocks on a
    // chunk prefetch, and shutdown() is called from the UI thread when a
    // workspace closes or a surface is replaced. Every fill holds a shared_ptr
    // to this State (and through it the volume, surface and geometry tiles), so
    // a task that is still running simply finishes against state nobody reads
    // and then releases the last reference. Clearing the listeners here is what
    // guarantees no callback fires after this returns.
    IChunkedArray::ChunkReadyCallbackId chunkReadyId = 0;
    {
        std::lock_guard lock(_state->mutex);
        _state->shuttingDown = true;
        _state->viewTiles.clear();
        _state->listeners.clear();
        chunkReadyId = _state->chunkReadyId;
        _state->chunkReadyId = 0;
    }
    if (chunkReadyId != 0 && _state->volume)
        _state->volume->removeChunkReadyListener(chunkReadyId);
}

std::size_t SurfaceCache::fillWorkerCount() { return surfaceFillPool().worker_count(); }

int SurfaceCache::levels() const { return _state->levels; }
int SurfaceCache::wMin() const { return _state->wMin; }
int SurfaceCache::wCount() const { return _state->wCount; }
std::size_t SurfaceCache::tileBytes() const { return _state->tileBytes; }

std::shared_ptr<SurfaceGeometryTileCache> SurfaceCache::geometryTiles() const
{
    return _state->geometry;
}

bool SurfaceCache::bandCovers(double wLow, double wHigh) const
{
    if (!(std::isfinite(wLow) && std::isfinite(wHigh)))
        return false;
    return wLow >= double(_state->wMin) && wHigh <= double(_state->wMin + _state->wCount - 1);
}

SurfaceCache::Stats SurfaceCache::stats() const
{
    std::lock_guard lock(_state->mutex);
    Stats stats;
    stats.bytes = _state->bytes;
    stats.capacity = _state->capacity;
    stats.tiles = _state->tiles.size();
    stats.tilesInFlight = _state->outstanding.size();
    stats.tilesIncomplete = _state->incomplete;
    return stats;
}

void SurfaceCache::setByteCapacity(std::size_t bytes)
{
    std::lock_guard lock(_state->mutex);
    _state->capacity = std::max(bytes, _state->tileBytes * 4);
    _state->enforceCapacityLocked();
}

void SurfaceCache::invalidateAll()
{
    {
        std::lock_guard lock(_state->mutex);
        ++_state->epoch;
        _state->tiles.clear();
        _state->lru.clear();
        _state->lruPos.clear();
        _state->bytes = 0;
        _state->incomplete = 0;
        _state->failedAttempts.clear();
        _state->viewTiles.clear();
    }
    if (_state->geometry)
        _state->geometry->invalidateAll();
}

void SurfaceCache::invalidateSurfaceRegion(const cv::Rect& gridCells)
{
    if (!_state->geometry) {
        invalidateAll();
        return;
    }
    const cv::Rect2d region = _state->geometry->gridCellsToNominal(gridCells);
    if (region.width <= 0.0 || region.height <= 0.0) {
        invalidateAll();
        return;
    }
    _state->geometry->invalidateSurfaceRegion(gridCells);

    std::lock_guard lock(_state->mutex);
    ++_state->epoch;
    std::vector<TileKey> victims;
    for (const auto& [key, tile] : _state->tiles) {
        if ((tileNominalRect(key.level, key.tu, key.tv) & region).area() > 0.0)
            victims.push_back(key);
    }
    for (const TileKey& key : victims) {
        _state->dropLocked(key);
        _state->failedAttempts.erase(key);
    }
}

SurfaceCache::TileReadyCallbackId
SurfaceCache::addTileReadyListener(std::function<void()> callback)
{
    std::lock_guard lock(_state->mutex);
    const auto id = _state->nextListenerId++;
    _state->listeners.emplace(id, std::move(callback));
    return id;
}

void SurfaceCache::removeTileReadyListener(TileReadyCallbackId id)
{
    std::lock_guard lock(_state->mutex);
    _state->listeners.erase(id);
}

// ---------------------------------------------------------------------------
// Scheduling
// ---------------------------------------------------------------------------

void SurfaceCache::requestView(int startLevel,
                               double uMin,
                               double vMin,
                               double scale,
                               int fbW,
                               int fbH,
                               std::uint64_t viewGeneration)
{
    if (!_state->volume || !_state->surface || fbW <= 0 || fbH <= 0 || !(scale > 0.0))
        return;
    if (!std::isfinite(uMin) || !std::isfinite(vMin))
        return;

    const int level = std::clamp(startLevel, 0, _state->levels - 1);
    const double step = std::ldexp(1.0, level);
    const double span = double(kTileSize) * step;
    const double uMax = uMin + double(fbW) / scale;
    const double vMax = vMin + double(fbH) / scale;
    // One extra sample on the high side for the bilinear neighbour tap.
    const int tu0 = int(std::floor(uMin / span));
    const int tu1 = int(std::floor((uMax + step) / span));
    const int tv0 = int(std::floor(vMin / span));
    const int tv1 = int(std::floor((vMax + step) / span));
    if (tu1 < tu0 || tv1 < tv0)
        return;

    const double uCenter = 0.5 * (uMin + uMax);
    const double vCenter = 0.5 * (vMin + vMax);

    struct Candidate {
        TileKey key;
        double distance = 0.0;
    };
    const std::size_t tileCount = std::size_t(tu1 - tu0 + 1) * std::size_t(tv1 - tv0 + 1);
    std::vector<Candidate> candidates;
    std::unordered_set<TileKey, TileKeyHash> viewTiles;
    viewTiles.reserve(tileCount);
    candidates.reserve(tileCount);

    bool viewChanged = false;
    {
        std::lock_guard lock(_state->mutex);
        if (_state->shuttingDown)
            return;
        for (int tv = tv0; tv <= tv1; ++tv) {
            for (int tu = tu0; tu <= tu1; ++tu) {
                const TileKey key{level, tu, tv};
                viewTiles.insert(key);
                auto it = _state->tiles.find(key);
                const bool haveTile = it != _state->tiles.end() && it->second;
                const bool haveComplete = haveTile && it->second->complete;
                if (haveComplete || _state->outstanding.count(key))
                    continue;
                // Incomplete tiles are re-queued so pixels a mid-fill eviction
                // lost eventually resolve, but only while retrying can still
                // change anything.
                if (haveTile) {
                    auto attempt = _state->failedAttempts.find(key);
                    if (attempt != _state->failedAttempts.end() &&
                        attempt->second >= kMaxFillAttempts)
                        continue;
                }
                const double tileU = (double(tu) + 0.5) * span;
                const double tileV = (double(tv) + 0.5) * span;
                candidates.push_back({key, std::hypot(tileU - uCenter, tileV - vCenter)});
            }
        }
        viewChanged = _state->viewTiles != viewTiles;
        _state->viewTiles = std::move(viewTiles);
        _state->viewGeneration = viewGeneration;
    }

    // The base Spiral surface cache has an exclusive chunk pool. Smaller
    // volume chunks make a tile's dependency list large, so keeping unresolved
    // batches from old pans would otherwise grow the fetch queue without bound.
    if (viewChanged && _state->options.supersedeChunkRequests)
        _state->volume->beginViewRequest(/*discardPending=*/true);

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.distance < b.distance; });

    const std::size_t pageTiles = std::max<std::size_t>(
        1, envSize("VC_SURFACE_ADMISSION_PAGE_BYTES", _state->options.admissionPageBytes) /
               std::max<std::size_t>(1, _state->tileBytes));
    std::size_t admitted = 0;
    for (const Candidate& candidate : candidates) {
        if (admitted >= pageTiles)
            break;
        std::uint64_t epoch = 0;
        {
            std::lock_guard lock(_state->mutex);
            if (_state->shuttingDown)
                return;
            if (_state->outstanding.size() >= _state->maxOutstandingFills)
                break;
            if (!_state->outstanding.insert(candidate.key).second)
                continue;
            epoch = _state->epoch;
        }
        ++admitted;
        auto state = _state;
        const TileKey key = candidate.key;
        surfaceFillPool().enqueue(
            [state, key, epoch]() { State::runFill(state, key, epoch); });
    }
}

// ---------------------------------------------------------------------------
// Render path
// ---------------------------------------------------------------------------

namespace {

// Resolves a (u, v) sample index to the sample run for its tile. Tiles are
// 128 samples wide, so consecutive pixels and all four bilinear taps
// overwhelmingly hit the same tile; a one-entry memo removes the map probe.
class TileResolver {
public:
    TileResolver(const std::unordered_map<std::uint64_t, SurfaceTilePtr>& tiles,
                 int wCount,
                 bool requiresOffset)
        : _tiles(tiles), _wCount(wCount), _requiresOffset(requiresOffset)
    {
    }

    const uint8_t* run(int i, int j)
    {
        const int tu = floorDiv(j, kTileSize);
        const int tv = floorDiv(i, kTileSize);
        const std::uint64_t key = packTile(tu, tv);
        if (key != _lastKey || !_haveLast) {
            auto it = _tiles.find(key);
            _lastTile = (it == _tiles.end() || !it->second) ? nullptr : it->second.get();
            _lastKey = key;
            _haveLast = true;
        }
        if (!_lastTile)
            return nullptr;
        const std::size_t index = std::size_t(i - tv * kTileSize) * kTileSize +
                                  std::size_t(j - tu * kTileSize);
        const auto& mask = _requiresOffset ? _lastTile->validOffset : _lastTile->valid;
        if (!mask[index])
            return nullptr;
        return _lastTile->data.data() + index * std::size_t(_wCount);
    }

private:
    const std::unordered_map<std::uint64_t, SurfaceTilePtr>& _tiles;
    int _wCount;
    bool _requiresOffset;
    std::uint64_t _lastKey = 0;
    bool _haveLast = false;
    const SurfaceTile* _lastTile = nullptr;
};

float sampleBand(const uint8_t* run, const BandTap& tap)
{
    const float v0 = float(run[tap.k0]);
    if (tap.t == 0.0f)
        return v0;
    return std::fma(tap.t, float(run[tap.k1]) - v0, v0);
}

Reduction reductionFor(const CompositeRenderSettings& composite)
{
    Reduction reduction;
    if (!composite.enabled)
        return reduction;
    const int front = std::max(0, composite.layersFront);
    const int behind = std::max(0, composite.layersBehind);
    reduction.numLayers = front + behind + 1;
    reduction.layerStep = composite.reverseDirection ? -1.0 : 1.0;
    reduction.firstOffset = -double(behind) * reduction.layerStep;
    reduction.reduce = true;
    reduction.params = composite.params;
    return reduction;
}

Reduction reductionFor(const OverlayCompositeSettings& composite, double zStep)
{
    Reduction reduction;
    if (!composite.enabled)
        return reduction;
    const int front = std::max(0, composite.layersFront);
    const int behind = std::max(0, composite.layersBehind);
    if (front == 0 && behind == 0)
        return reduction;
    reduction.numLayers = front + behind + 1;
    reduction.layerStep = zStep < 0.0 ? -1.0 : 1.0;
    reduction.firstOffset = -double(behind) * reduction.layerStep;
    reduction.reduce = true;
    reduction.params.method = composite.method;
    reduction.params.isoCutoff = 0;
    return reduction;
}

void reductionWRange(const Reduction& reduction, double w, double& lo, double& hi)
{
    const double a = w + reduction.firstOffset;
    const double b = a + double(reduction.numLayers - 1) * reduction.layerStep;
    lo = std::min(a, b);
    hi = std::max(a, b);
}

} // namespace

void SurfaceCache::compositeWRange(const CompositeRenderSettings& composite,
                                   double w,
                                   double& wLow,
                                   double& wHigh)
{
    reductionWRange(reductionFor(composite), w, wLow, wHigh);
}

void SurfaceCache::overlayCompositeWRange(const OverlayCompositeSettings& composite,
                                          double w,
                                          double zStep,
                                          double& wLow,
                                          double& wHigh)
{
    reductionWRange(reductionFor(composite, zStep), w, wLow, wHigh);
}

SurfaceCache::SampleStats SurfaceCache::State::sampleReduced(const State& self,
                                                             int startLevel,
                                                             double uMin,
                                                             double vMin,
                                                             double scale,
                                                             double w,
                                                             const Reduction& reduction,
                                                             cv::Mat_<uint8_t>& out,
                                                             cv::Mat_<uint8_t>& coverage)
{
    SampleStats stats;
    if (out.empty() || coverage.empty() || !(scale > 0.0))
        return stats;
    if (!std::isfinite(uMin) || !std::isfinite(vMin) || !std::isfinite(w))
        return stats;

    const int fbH = std::min(out.rows, coverage.rows);
    const int fbW = std::min(out.cols, coverage.cols);
    if (fbH <= 0 || fbW <= 0)
        return stats;

    const int wCount = self.wCount;
    const int wMin = self.wMin;
    double wLow = 0.0, wHigh = 0.0;
    reductionWRange(reduction, w, wLow, wHigh);
    if (wLow < double(wMin) || wHigh > double(wMin + wCount - 1))
        return stats;
    const bool requiresOffset = !(wLow == 0.0 && wHigh == 0.0);

    // `w` is constant across the frame, so each layer's two band slices and
    // their blend weight are resolved once.
    std::vector<BandTap> layerTaps(std::size_t(reduction.numLayers));
    for (int layer = 0; layer < reduction.numLayers; ++layer) {
        const double wl = w + reduction.firstOffset + double(layer) * reduction.layerStep;
        double rel = wl - double(wMin);
        int k0 = int(std::floor(rel));
        float t = float(rel - double(k0));
        if (k0 < 0) {
            k0 = 0;
            t = 0.0f;
        }
        if (k0 >= wCount - 1) {
            k0 = wCount - 1;
            t = 0.0f;
        }
        layerTaps[std::size_t(layer)] = {k0, std::min(k0 + 1, wCount - 1), t};
    }

    const int startClamped = std::clamp(startLevel, 0, self.levels - 1);
    for (int level = startClamped; level < self.levels; ++level) {
        const double step = std::ldexp(1.0, level);
        const double span = double(kTileSize) * step;
        const double uMax = uMin + double(fbW) / scale;
        const double vMax = vMin + double(fbH) / scale;
        const int tu0 = int(std::floor(uMin / span));
        const int tu1 = int(std::floor((uMax + step) / span));
        const int tv0 = int(std::floor(vMin / span));
        const int tv1 = int(std::floor((vMax + step) / span));

        std::unordered_map<std::uint64_t, SurfaceTilePtr> levelTiles;
        int missing = 0;
        {
            std::lock_guard lock(self.mutex);
            levelTiles.reserve(std::size_t(std::max(0, tu1 - tu0 + 1)) *
                               std::size_t(std::max(0, tv1 - tv0 + 1)));
            for (int tv = tv0; tv <= tv1; ++tv) {
                for (int tu = tu0; tu <= tu1; ++tu) {
                    auto it = self.tiles.find({level, tu, tv});
                    if (it == self.tiles.end() || !it->second) {
                        ++missing;
                        continue;
                    }
                    // Holding the shared_ptr keeps the tile alive for the whole
                    // resample even if the LRU evicts it meanwhile.
                    levelTiles.emplace(packTile(tu, tv), it->second);
                    const_cast<State&>(self).touchLocked({level, tu, tv});
                }
            }
        }
        if (level == startClamped)
            stats.missingTiles = missing;
        if (levelTiles.empty())
            continue;

        const double invStep = 1.0 / step;
        auto sampleRows = [&](int yBegin, int yEnd) {
            TileResolver resolver(levelTiles, wCount, requiresOffset);
            std::vector<float> layerValues(std::size_t(reduction.numLayers));
            LayerStack stack;
            stack.values.resize(std::size_t(reduction.numLayers));
            int covered = 0;
            const uint8_t* runs[4] = {nullptr, nullptr, nullptr, nullptr};
            for (int y = yBegin; y < yEnd; ++y) {
                const double v = vMin + double(y) / scale;
                const double fv = v * invStep;
                const int i0 = int(std::floor(fv));
                const float b = float(fv - double(i0));
                const float ib = 1.0f - b;
                uint8_t* outRow = out.ptr<uint8_t>(y);
                uint8_t* coverageRow = coverage.ptr<uint8_t>(y);
                for (int x = 0; x < fbW; ++x) {
                    if (coverageRow[x])
                        continue;
                    const double u = uMin + double(x) / scale;
                    const double fu = u * invStep;
                    const int j0 = int(std::floor(fu));
                    const float a = float(fu - double(j0));
                    const float ia = 1.0f - a;

                    runs[0] = resolver.run(i0, j0);
                    if (!runs[0])
                        continue;
                    runs[1] = resolver.run(i0, j0 + 1);
                    if (!runs[1])
                        continue;
                    runs[2] = resolver.run(i0 + 1, j0);
                    if (!runs[2])
                        continue;
                    runs[3] = resolver.run(i0 + 1, j0 + 1);
                    if (!runs[3])
                        continue;
                    const float weights[4] = {ia * ib, a * ib, ia * b, a * b};

                    if (!reduction.reduce) {
                        const BandTap& tap = layerTaps[0];
                        float value = 0.0f;
                        for (int t = 0; t < 4; ++t)
                            value += weights[t] * sampleBand(runs[t], tap);
                        outRow[x] = static_cast<uint8_t>(
                            std::clamp(std::lround(value), 0L, 255L));
                        coverageRow[x] = 1;
                        ++covered;
                        continue;
                    }

                    stack.validCount = 0;
                    for (int layer = 0; layer < reduction.numLayers; ++layer) {
                        const BandTap& tap = layerTaps[std::size_t(layer)];
                        float value = 0.0f;
                        for (int t = 0; t < 4; ++t)
                            value += weights[t] * sampleBand(runs[t], tap);
                        if (value < float(reduction.params.isoCutoff))
                            continue;
                        stack.values[std::size_t(stack.validCount++)] = value;
                    }
                    if (stack.validCount > 0) {
                        outRow[x] = static_cast<uint8_t>(std::clamp(
                            compositeLayerStack(stack, reduction.params), 0.0f, 255.0f));
                        coverageRow[x] = 1;
                        ++covered;
                    }
                }
            }
            return covered;
        };

        const std::size_t workers = surfaceResamplePool().worker_count();
        int covered = 0;
        if (workers <= 1 || fbH < 64) {
            covered = sampleRows(0, fbH);
        } else {
            const int bands = int(std::min<std::size_t>(workers, std::size_t(fbH)));
            const int rowsPerBand = (fbH + bands - 1) / bands;
            std::vector<std::future<int>> futures;
            futures.reserve(std::size_t(bands));
            for (int band = 0; band < bands; ++band) {
                const int yBegin = band * rowsPerBand;
                const int yEnd = std::min(yBegin + rowsPerBand, fbH);
                if (yBegin >= yEnd)
                    break;
                futures.push_back(surfaceResamplePool().submit(
                    [&sampleRows, yBegin, yEnd]() { return sampleRows(yBegin, yEnd); }));
            }
            for (auto& future : futures)
                covered += future.get();
        }
        stats.coveredPixels += covered;
        if (level > startClamped && covered > 0)
            stats.usedCoarseFallback = true;
    }

    return stats;
}

SurfaceCache::SampleStats SurfaceCache::sampleView(int startLevel,
                                                   double uMin,
                                                   double vMin,
                                                   double scale,
                                                   double w,
                                                   const CompositeRenderSettings& composite,
                                                   cv::Mat_<uint8_t>& out,
                                                   cv::Mat_<uint8_t>& coverage) const
{
    return State::sampleReduced(*_state, startLevel, uMin, vMin, scale, w,
                                reductionFor(composite), out, coverage);
}

SurfaceCache::SampleStats SurfaceCache::sampleViewOverlay(
    int startLevel,
    double uMin,
    double vMin,
    double scale,
    double w,
    const OverlayCompositeSettings& composite,
    double zStep,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage) const
{
    return State::sampleReduced(*_state, startLevel, uMin, vMin, scale, w,
                                reductionFor(composite, zStep), out, coverage);
}

} // namespace vc::render
