#include "vc/core/render/ChunkedPlaneSampler.hpp"

#include <utils/thread_pool.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vc::render {
namespace {

struct LocalChunkCache {
    // ChunkResult owns a shared_ptr to decoded chunk bytes. Keeping every
    // result touched by a frame pins the whole working set even after the
    // shared ChunkCache evicts it. A small window preserves the hot consecutive
    // lookup fast path without allowing each render worker to become another
    // unbounded decoded-byte cache.
    static constexpr std::size_t kMaxPinnedChunks = 8;

    explicit LocalChunkCache(
        IChunkedArray& a, std::size_t expectedChunks = 0, bool queueMisses_ = true)
        : array(a)
        , queueMisses(queueMisses_)
    {
        if (expectedChunks > 0) {
            chunks.reserve(std::min(expectedChunks, kMaxPinnedChunks));
            requestedKeys.reserve(expectedChunks);
            errorKeys.reserve(expectedChunks);
        }
    }

    const ChunkResult& get(const ChunkKey& key, int& requested, int& errors)
    {
        // Trilinear sampling reads 8 voxels per pixel and adjacent pixels share
        // chunks, so consecutive lookups overwhelmingly hit the same key. Skip
        // the hash-map probe in that case.
        if (lastResult && lastKey == key)
            return *lastResult;

        auto it = chunks.find(key);
        if (it == chunks.end()) {
            if (chunks.size() >= kMaxPinnedChunks) {
                lastResult = nullptr;
                chunks.clear();
            }
            ChunkResult result = queueMisses
                ? array.tryGetChunk(key.level, key.iz, key.iy, key.ix)
                : array.getChunkIfCached(key.level, key.iz, key.iy, key.ix);
            if (queueMisses && result.status == ChunkStatus::MissQueued &&
                requestedKeys.insert(key).second)
                ++requested;
            if (result.status == ChunkStatus::Error && errorKeys.insert(key).second)
                ++errors;
            it = chunks.emplace(key, std::move(result)).first;
        }

        lastKey = key;
        lastResult = &it->second;
        return it->second;
    }

    IChunkedArray& array;
    bool queueMisses = true;
    std::unordered_map<ChunkKey, ChunkResult, ChunkKeyHash> chunks;
    std::unordered_set<ChunkKey, ChunkKeyHash> requestedKeys;
    std::unordered_set<ChunkKey, ChunkKeyHash> errorKeys;
    ChunkKey lastKey{};
    const ChunkResult* lastResult = nullptr;
};

struct PinnedChunkCache {
    explicit PinnedChunkCache(std::size_t expectedChunks = 0)
    {
        if (expectedChunks > 0)
            chunks.reserve(expectedChunks);
    }

    const ChunkResult* get(const ChunkKey& key) const
    {
        auto it = chunks.find(key);
        if (it == chunks.end())
            return nullptr;
        return &it->second;
    }

    std::unordered_map<ChunkKey, ChunkResult, ChunkKeyHash> chunks;
};

enum class StrictSampleStatus {
    Ready,
    Missing,
    Error
};

constexpr int kParallelMinPixels = 128 * 128;
constexpr int kMaxRenderSamplerWorkers = 8;
constexpr int kMaxCornerBatchWorkers = 64;

int renderSamplerWorkerCount()
{
    const unsigned hc = std::thread::hardware_concurrency();
    if (hc <= 2)
        return 1;
    return std::clamp(static_cast<int>(hc) - 2, 1, kMaxRenderSamplerWorkers);
}

utils::ThreadPool& renderSamplerPool()
{
#if defined(_WIN32)
    // Avoid joining DLL-owned worker threads from a static destructor while
    // Windows holds the loader lock during vc_core.dll shutdown.
    static auto* pool = new utils::ThreadPool(
        static_cast<std::size_t>(renderSamplerWorkerCount()));
    return *pool;
#else
    static utils::ThreadPool pool(static_cast<std::size_t>(renderSamplerWorkerCount()));
    return pool;
#endif
}

int cornerBatchWorkerCount()
{
    const unsigned hc = std::thread::hardware_concurrency();
    if (hc <= 2)
        return 1;
    return std::clamp(static_cast<int>(hc) - 2, 1, kMaxCornerBatchWorkers);
}

utils::ThreadPool& cornerBatchPool()
{
#if defined(_WIN32)
    static auto* pool = new utils::ThreadPool(
        static_cast<std::size_t>(cornerBatchWorkerCount()));
    return *pool;
#else
    static utils::ThreadPool pool(static_cast<std::size_t>(cornerBatchWorkerCount()));
    return pool;
#endif
}

bool shouldParallelizeSamples(int rows, int cols)
{
    return renderSamplerWorkerCount() > 1 &&
           rows > 0 && cols > 0 &&
           rows * cols >= kParallelMinPixels;
}

struct LevelAccess {
    std::array<int, 3> shape{};
    std::array<int, 3> chunkShape{};
    IChunkedArray::LevelTransform transform;
    uint8_t fill = 0;
};

struct LevelPlane {
    cv::Vec3f origin;
    cv::Vec3f vxStep;
    cv::Vec3f vyStep;
};

struct SampleTile {
    int tx = 0;
    int ty = 0;
    int xEnd = 0;
    int yEnd = 0;
};

struct MissingLevelContext {
    MissingLevelContext(IChunkedArray& array_,
                        int level_,
                        LevelAccess access_,
                        bool queueMisses_,
                        LevelPlane plane_ = {})
        : level(level_)
        , access(access_)
        , plane(plane_)
        , cache(array_, 64, queueMisses_)
    {
    }

    int level = 0;
    LevelAccess access;
    LevelPlane plane;
    LocalChunkCache cache;
};

enum class ChunkDependencyState {
    Ready,
    KnownMissing,
    Transient
};

bool finiteCoord(const cv::Vec3f& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

bool surfaceSentinel(const cv::Vec3f& p)
{
    return !finiteCoord(p)
        || p[0] == -1.0f || p[1] == -1.0f || p[2] == -1.0f
        || (p[0] == 0.0f && p[1] == 0.0f && p[2] == 0.0f);
}

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
    return access.shape[0] > 0 && access.shape[1] > 0 && access.shape[2] > 0
        && access.chunkShape[0] > 0 && access.chunkShape[1] > 0 && access.chunkShape[2] > 0;
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

cv::Vec3f toLevelVector(const LevelAccess& access, const cv::Vec3f& v0)
{
    const auto& t = access.transform;
    return {
        float(double(v0[0]) * t.scaleFromLevel0[0]),
        float(double(v0[1]) * t.scaleFromLevel0[1]),
        float(double(v0[2]) * t.scaleFromLevel0[2]),
    };
}

LevelPlane toLevelPlane(const LevelAccess& access,
                        const cv::Vec3f& origin,
                        const cv::Vec3f& vxStep,
                        const cv::Vec3f& vyStep)
{
    return {toLevelCoord(access, origin),
            toLevelVector(access, vxStep),
            toLevelVector(access, vyStep)};
}

bool inLevelBounds(const std::array<int, 3>& shape, float z, float y, float x)
{
    return z >= 0.0f && y >= 0.0f && x >= 0.0f
        && z < float(shape[0]) && y < float(shape[1]) && x < float(shape[2]);
}

bool readVoxel(IChunkedArray& array,
               LocalChunkCache& cache,
               const LevelAccess& access,
               int level,
               int iz,
               int iy,
               int ix,
               uint8_t& out,
               int& requested,
               int& errors)
{
    const auto& shape = access.shape;
    if (unsigned(iz) >= unsigned(shape[0])
        || unsigned(iy) >= unsigned(shape[1])
        || unsigned(ix) >= unsigned(shape[2])) {
        out = access.fill;
        return true;
    }

    const auto& chunkShape = access.chunkShape;
    if (chunkShape[0] <= 0 || chunkShape[1] <= 0 || chunkShape[2] <= 0)
        return false;

    const int cz = iz / chunkShape[0];
    const int cy = iy / chunkShape[1];
    const int cx = ix / chunkShape[2];
    const ChunkResult& result = cache.get({level, cz, cy, cx}, requested, errors);
    if (result.status == ChunkStatus::MissQueued ||
        result.status == ChunkStatus::Missing ||
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
    const std::size_t offset = (std::size_t(lz) * std::size_t(chunkShape[1])
                              + std::size_t(ly)) * std::size_t(chunkShape[2])
                              + std::size_t(lx);
    if (offset >= result.bytes->size())
        return false;

    out = std::to_integer<uint8_t>((*result.bytes)[offset]);
    return true;
}

StrictSampleStatus readPinnedVoxel(const PinnedChunkCache& cache,
                                   const LevelAccess& access,
                                   int level,
                                   int iz,
                                   int iy,
                                   int ix,
                                   uint8_t& out,
                                   std::string& error)
{
    const auto& shape = access.shape;
    if (unsigned(iz) >= unsigned(shape[0])
        || unsigned(iy) >= unsigned(shape[1])
        || unsigned(ix) >= unsigned(shape[2])) {
        out = access.fill;
        return StrictSampleStatus::Ready;
    }

    const auto& chunkShape = access.chunkShape;
    if (chunkShape[0] <= 0 || chunkShape[1] <= 0 || chunkShape[2] <= 0) {
        error = "invalid chunk shape while sampling pinned chunk";
        return StrictSampleStatus::Error;
    }

    const int cz = iz / chunkShape[0];
    const int cy = iy / chunkShape[1];
    const int cx = ix / chunkShape[2];
    const ChunkKey key{level, cz, cy, cx};
    const ChunkResult* result = cache.get(key);
    if (!result) {
        error = "required chunk was not pinned before requested-level sampling";
        return StrictSampleStatus::Error;
    }
    if (result->status == ChunkStatus::Missing)
        return StrictSampleStatus::Missing;
    if (result->status == ChunkStatus::MissQueued) {
        error = "blocking requested-level sampling encountered an unresolved chunk";
        return StrictSampleStatus::Error;
    }
    if (result->status == ChunkStatus::Error) {
        error = result->error.empty() ? "chunk fetch failed" : result->error;
        return StrictSampleStatus::Error;
    }

    if (result->status == ChunkStatus::AllFill) {
        out = access.fill;
        return StrictSampleStatus::Ready;
    }

    if (result->status != ChunkStatus::Data || !result->bytes) {
        error = "chunk resolved without payload";
        return StrictSampleStatus::Error;
    }

    const int lz = iz - cz * chunkShape[0];
    const int ly = iy - cy * chunkShape[1];
    const int lx = ix - cx * chunkShape[2];
    const std::size_t offset = (std::size_t(lz) * std::size_t(chunkShape[1])
                              + std::size_t(ly)) * std::size_t(chunkShape[2])
                              + std::size_t(lx);
    if (offset >= result->bytes->size()) {
        error = "chunk payload is smaller than expected";
        return StrictSampleStatus::Error;
    }

    out = std::to_integer<uint8_t>((*result->bytes)[offset]);
    return StrictSampleStatus::Ready;
}

bool addVoxelDependency(IChunkedArray& array,
                        const LevelAccess& access,
                        int level,
                        int iz,
                        int iy,
                        int ix,
                        std::unordered_set<ChunkKey, ChunkKeyHash>& keys)
{
    const auto& shape = access.shape;
    if (unsigned(iz) >= unsigned(shape[0])
        || unsigned(iy) >= unsigned(shape[1])
        || unsigned(ix) >= unsigned(shape[2]))
        return true;

    const auto& chunkShape = access.chunkShape;
    if (chunkShape[0] <= 0 || chunkShape[1] <= 0 || chunkShape[2] <= 0)
        return false;

    keys.insert({level, iz / chunkShape[0], iy / chunkShape[1], ix / chunkShape[2]});
    return true;
}

bool addRequiredVoxelChunk(const LevelAccess& access,
                           int level,
                           int iz,
                           int iy,
                           int ix,
                           std::vector<ChunkKey>& keys)
{
    const auto& shape = access.shape;
    if (unsigned(iz) >= unsigned(shape[0])
        || unsigned(iy) >= unsigned(shape[1])
        || unsigned(ix) >= unsigned(shape[2]))
        return true;

    const auto& chunkShape = access.chunkShape;
    if (chunkShape[0] <= 0 || chunkShape[1] <= 0 || chunkShape[2] <= 0)
        return false;

    const ChunkKey key{level, iz / chunkShape[0], iy / chunkShape[1], ix / chunkShape[2]};
    if (std::find(keys.begin(), keys.end(), key) == keys.end())
        keys.push_back(key);
    return true;
}

bool collectRequiredLevelChunks(const LevelAccess& access,
                                int level,
                                const cv::Vec3f& p,
                                vc::Sampling sampling,
                                std::vector<ChunkKey>& keys)
{
    keys.clear();
    if (!finiteCoord(p))
        return true;

    const auto& shape = access.shape;
    const float x = p[0], y = p[1], z = p[2];
    if (!inLevelBounds(shape, z, y, x))
        return true;

    if (sampling == vc::Sampling::Nearest) {
        int ix = int(x + 0.5f);
        int iy = int(y + 0.5f);
        int iz = int(z + 0.5f);
        ix = std::clamp(ix, 0, shape[2] - 1);
        iy = std::clamp(iy, 0, shape[1] - 1);
        iz = std::clamp(iz, 0, shape[0] - 1);
        return addRequiredVoxelChunk(access, level, iz, iy, ix, keys);
    }

    const int ix = int(std::floor(x));
    const int iy = int(std::floor(y));
    const int iz = int(std::floor(z));
    bool ok = true;
    ok = addRequiredVoxelChunk(access, level, iz,     iy,     ix,     keys) && ok;
    ok = addRequiredVoxelChunk(access, level, iz,     iy,     ix + 1, keys) && ok;
    ok = addRequiredVoxelChunk(access, level, iz,     iy + 1, ix,     keys) && ok;
    ok = addRequiredVoxelChunk(access, level, iz,     iy + 1, ix + 1, keys) && ok;
    ok = addRequiredVoxelChunk(access, level, iz + 1, iy,     ix,     keys) && ok;
    ok = addRequiredVoxelChunk(access, level, iz + 1, iy,     ix + 1, keys) && ok;
    ok = addRequiredVoxelChunk(access, level, iz + 1, iy + 1, ix,     keys) && ok;
    ok = addRequiredVoxelChunk(access, level, iz + 1, iy + 1, ix + 1, keys) && ok;
    return ok;
}

ChunkDependencyState classifyRequiredChunks(LocalChunkCache& cache,
                                            const std::vector<ChunkKey>& keys)
{
    if (keys.empty())
        return ChunkDependencyState::Ready;

    bool sawMissing = false;
    int requested = 0;
    int errors = 0;
    for (const ChunkKey& key : keys) {
        const ChunkResult& result = cache.get(key, requested, errors);
        if (result.status == ChunkStatus::MissQueued ||
            result.status == ChunkStatus::Error)
            return ChunkDependencyState::Transient;
        if (result.status == ChunkStatus::Missing)
            sawMissing = true;
    }

    return sawMissing ? ChunkDependencyState::KnownMissing
                      : ChunkDependencyState::Ready;
}

bool collectPointDependencies(IChunkedArray& array,
                              const LevelAccess& access,
                              int level,
                              const cv::Vec3f& p0,
                              vc::Sampling sampling,
                              bool zeroIsSentinel,
                              std::unordered_set<ChunkKey, ChunkKeyHash>& keys)
{
    if (!finiteCoord(p0) || (zeroIsSentinel && surfaceSentinel(p0)))
        return true;

    const cv::Vec3f p = toLevelCoord(access, p0);
    const auto& shape = access.shape;
    const float x = p[0], y = p[1], z = p[2];
    if (!inLevelBounds(shape, z, y, x))
        return true;

    if (sampling == vc::Sampling::Nearest) {
        int ix = int(x + 0.5f);
        int iy = int(y + 0.5f);
        int iz = int(z + 0.5f);
        ix = std::clamp(ix, 0, shape[2] - 1);
        iy = std::clamp(iy, 0, shape[1] - 1);
        iz = std::clamp(iz, 0, shape[0] - 1);
        return addVoxelDependency(array, access, level, iz, iy, ix, keys);
    }

    const int ix = int(std::floor(x));
    const int iy = int(std::floor(y));
    const int iz = int(std::floor(z));
    bool ok = true;
    ok = addVoxelDependency(array, access, level, iz,     iy,     ix,     keys) && ok;
    ok = addVoxelDependency(array, access, level, iz,     iy,     ix + 1, keys) && ok;
    ok = addVoxelDependency(array, access, level, iz,     iy + 1, ix,     keys) && ok;
    ok = addVoxelDependency(array, access, level, iz,     iy + 1, ix + 1, keys) && ok;
    ok = addVoxelDependency(array, access, level, iz + 1, iy,     ix,     keys) && ok;
    ok = addVoxelDependency(array, access, level, iz + 1, iy,     ix + 1, keys) && ok;
    ok = addVoxelDependency(array, access, level, iz + 1, iy + 1, ix,     keys) && ok;
    ok = addVoxelDependency(array, access, level, iz + 1, iy + 1, ix + 1, keys) && ok;
    return ok;
}

bool collectLevelPointDependencies(IChunkedArray& array,
                                   const LevelAccess& access,
                                   int level,
                                   const cv::Vec3f& p,
                                   vc::Sampling sampling,
                                   std::unordered_set<ChunkKey, ChunkKeyHash>& keys)
{
    if (!finiteCoord(p))
        return true;

    const auto& shape = access.shape;
    const float x = p[0], y = p[1], z = p[2];
    if (!inLevelBounds(shape, z, y, x))
        return true;

    if (sampling == vc::Sampling::Nearest) {
        int ix = int(x + 0.5f);
        int iy = int(y + 0.5f);
        int iz = int(z + 0.5f);
        ix = std::clamp(ix, 0, shape[2] - 1);
        iy = std::clamp(iy, 0, shape[1] - 1);
        iz = std::clamp(iz, 0, shape[0] - 1);
        return addVoxelDependency(array, access, level, iz, iy, ix, keys);
    }

    const int ix = int(std::floor(x));
    const int iy = int(std::floor(y));
    const int iz = int(std::floor(z));
    bool ok = true;
    ok = addVoxelDependency(array, access, level, iz,     iy,     ix,     keys) && ok;
    ok = addVoxelDependency(array, access, level, iz,     iy,     ix + 1, keys) && ok;
    ok = addVoxelDependency(array, access, level, iz,     iy + 1, ix,     keys) && ok;
    ok = addVoxelDependency(array, access, level, iz,     iy + 1, ix + 1, keys) && ok;
    ok = addVoxelDependency(array, access, level, iz + 1, iy,     ix,     keys) && ok;
    ok = addVoxelDependency(array, access, level, iz + 1, iy,     ix + 1, keys) && ok;
    ok = addVoxelDependency(array, access, level, iz + 1, iy + 1, ix,     keys) && ok;
    ok = addVoxelDependency(array, access, level, iz + 1, iy + 1, ix + 1, keys) && ok;
    return ok;
}

void requestDependencies(LocalChunkCache& cache,
                         const std::unordered_set<ChunkKey, ChunkKeyHash>& keys,
                         ChunkedPlaneSampler::Stats& stats)
{
    for (const ChunkKey& key : keys)
        (void)cache.get(key, stats.requestedChunks, stats.errorChunks);
}

bool sampleNearest(IChunkedArray& array,
                   LocalChunkCache& cache,
                   const LevelAccess& access,
                   int level,
                   const cv::Vec3f& p,
                   uint8_t& out,
                   int& requested,
                   int& errors)
{
    const auto& shape = access.shape;
    const float x = p[0], y = p[1], z = p[2];
    if (!inLevelBounds(shape, z, y, x)) {
        out = access.fill;
        return true;
    }

    int ix = int(x + 0.5f);
    int iy = int(y + 0.5f);
    int iz = int(z + 0.5f);
    ix = std::clamp(ix, 0, shape[2] - 1);
    iy = std::clamp(iy, 0, shape[1] - 1);
    iz = std::clamp(iz, 0, shape[0] - 1);
    return readVoxel(array, cache, access, level, iz, iy, ix, out, requested, errors);
}

bool sampleTrilinear(IChunkedArray& array,
                     LocalChunkCache& cache,
                     const LevelAccess& access,
                     int level,
                     const cv::Vec3f& p,
                     uint8_t& out,
                     int& requested,
                     int& errors)
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

    const auto& chunkShape = access.chunkShape;
    if (chunkShape[0] > 0 && chunkShape[1] > 0 && chunkShape[2] > 0 &&
        ix + 1 < shape[2] && iy + 1 < shape[1] && iz + 1 < shape[0]) {
        const int cz = iz / chunkShape[0];
        const int cy = iy / chunkShape[1];
        const int cx = ix / chunkShape[2];
        const int lz = iz - cz * chunkShape[0];
        const int ly = iy - cy * chunkShape[1];
        const int lx = ix - cx * chunkShape[2];
        if (lx + 1 < chunkShape[2] && ly + 1 < chunkShape[1] && lz + 1 < chunkShape[0]) {
            const ChunkResult& result = cache.get({level, cz, cy, cx}, requested, errors);
            if (result.status == ChunkStatus::MissQueued ||
                result.status == ChunkStatus::Missing ||
                result.status == ChunkStatus::Error)
                return false;

            if (result.status == ChunkStatus::AllFill) {
                out = access.fill;
                return true;
            }

            if (result.status == ChunkStatus::Data && result.bytes) {
                const std::size_t strideX = 1;
                const std::size_t strideY = std::size_t(chunkShape[2]);
                const std::size_t strideZ = std::size_t(chunkShape[1]) * std::size_t(chunkShape[2]);
                const std::size_t offset000 = std::size_t(lz) * strideZ
                                            + std::size_t(ly) * strideY
                                            + std::size_t(lx);
                const std::size_t offset111 = offset000 + strideZ + strideY + strideX;
                if (offset111 < result.bytes->size()) {
                    const auto* bytes = result.bytes->data();
                    const uint8_t v000 = std::to_integer<uint8_t>(bytes[offset000]);
                    const uint8_t v001 = std::to_integer<uint8_t>(bytes[offset000 + strideX]);
                    const uint8_t v010 = std::to_integer<uint8_t>(bytes[offset000 + strideY]);
                    const uint8_t v011 = std::to_integer<uint8_t>(bytes[offset000 + strideY + strideX]);
                    const uint8_t v100 = std::to_integer<uint8_t>(bytes[offset000 + strideZ]);
                    const uint8_t v101 = std::to_integer<uint8_t>(bytes[offset000 + strideZ + strideX]);
                    const uint8_t v110 = std::to_integer<uint8_t>(bytes[offset000 + strideZ + strideY]);
                    const uint8_t v111 = std::to_integer<uint8_t>(bytes[offset111]);

                    const float c00 = std::fma(fx, float(v001) - float(v000), float(v000));
                    const float c01 = std::fma(fx, float(v011) - float(v010), float(v010));
                    const float c10 = std::fma(fx, float(v101) - float(v100), float(v100));
                    const float c11 = std::fma(fx, float(v111) - float(v110), float(v110));
                    const float c0 = std::fma(fy, c01 - c00, c00);
                    const float c1 = std::fma(fy, c11 - c10, c10);
                    const float value = std::clamp(std::fma(fz, c1 - c0, c0), 0.0f, 255.0f);
                    out = static_cast<uint8_t>(value);
                    return true;
                }
            }
        }
    }

    uint8_t v000 = 0, v001 = 0, v010 = 0, v011 = 0;
    uint8_t v100 = 0, v101 = 0, v110 = 0, v111 = 0;
    bool ready = true;
    ready = readVoxel(array, cache, access, level, iz,     iy,     ix,     v000, requested, errors) && ready;
    ready = readVoxel(array, cache, access, level, iz,     iy,     ix + 1, v001, requested, errors) && ready;
    ready = readVoxel(array, cache, access, level, iz,     iy + 1, ix,     v010, requested, errors) && ready;
    ready = readVoxel(array, cache, access, level, iz,     iy + 1, ix + 1, v011, requested, errors) && ready;
    ready = readVoxel(array, cache, access, level, iz + 1, iy,     ix,     v100, requested, errors) && ready;
    ready = readVoxel(array, cache, access, level, iz + 1, iy,     ix + 1, v101, requested, errors) && ready;
    ready = readVoxel(array, cache, access, level, iz + 1, iy + 1, ix,     v110, requested, errors) && ready;
    ready = readVoxel(array, cache, access, level, iz + 1, iy + 1, ix + 1, v111, requested, errors) && ready;
    if (!ready)
        return false;

    const float c00 = std::fma(fx, float(v001) - float(v000), float(v000));
    const float c01 = std::fma(fx, float(v011) - float(v010), float(v010));
    const float c10 = std::fma(fx, float(v101) - float(v100), float(v100));
    const float c11 = std::fma(fx, float(v111) - float(v110), float(v110));
    const float c0 = std::fma(fy, c01 - c00, c00);
    const float c1 = std::fma(fy, c11 - c10, c10);
    const float value = std::clamp(std::fma(fz, c1 - c0, c0), 0.0f, 255.0f);
    out = static_cast<uint8_t>(value);
    return true;
}

StrictSampleStatus samplePinnedNearest(const PinnedChunkCache& cache,
                                       const LevelAccess& access,
                                       int level,
                                       const cv::Vec3f& p,
                                       uint8_t& out,
                                       std::string& error)
{
    const auto& shape = access.shape;
    const float x = p[0], y = p[1], z = p[2];
    if (!inLevelBounds(shape, z, y, x)) {
        out = access.fill;
        return StrictSampleStatus::Ready;
    }

    int ix = int(x + 0.5f);
    int iy = int(y + 0.5f);
    int iz = int(z + 0.5f);
    ix = std::clamp(ix, 0, shape[2] - 1);
    iy = std::clamp(iy, 0, shape[1] - 1);
    iz = std::clamp(iz, 0, shape[0] - 1);
    return readPinnedVoxel(cache, access, level, iz, iy, ix, out, error);
}

StrictSampleStatus samplePinnedTrilinear(const PinnedChunkCache& cache,
                                         const LevelAccess& access,
                                         int level,
                                         const cv::Vec3f& p,
                                         uint8_t& out,
                                         std::string& error)
{
    const auto& shape = access.shape;
    const float x = p[0], y = p[1], z = p[2];
    if (!inLevelBounds(shape, z, y, x)) {
        out = access.fill;
        return StrictSampleStatus::Ready;
    }

    const int ix = int(std::floor(x));
    const int iy = int(std::floor(y));
    const int iz = int(std::floor(z));
    const float fx = x - float(ix);
    const float fy = y - float(iy);
    const float fz = z - float(iz);

    uint8_t v000 = 0, v001 = 0, v010 = 0, v011 = 0;
    uint8_t v100 = 0, v101 = 0, v110 = 0, v111 = 0;
    std::array<StrictSampleStatus, 8> statuses{};
    statuses[0] = readPinnedVoxel(cache, access, level, iz,     iy,     ix,     v000, error);
    statuses[1] = readPinnedVoxel(cache, access, level, iz,     iy,     ix + 1, v001, error);
    statuses[2] = readPinnedVoxel(cache, access, level, iz,     iy + 1, ix,     v010, error);
    statuses[3] = readPinnedVoxel(cache, access, level, iz,     iy + 1, ix + 1, v011, error);
    statuses[4] = readPinnedVoxel(cache, access, level, iz + 1, iy,     ix,     v100, error);
    statuses[5] = readPinnedVoxel(cache, access, level, iz + 1, iy,     ix + 1, v101, error);
    statuses[6] = readPinnedVoxel(cache, access, level, iz + 1, iy + 1, ix,     v110, error);
    statuses[7] = readPinnedVoxel(cache, access, level, iz + 1, iy + 1, ix + 1, v111, error);
    for (StrictSampleStatus status : statuses) {
        if (status == StrictSampleStatus::Error)
            return StrictSampleStatus::Error;
        if (status == StrictSampleStatus::Missing)
            return StrictSampleStatus::Missing;
    }

    const float c00 = std::fma(fx, float(v001) - float(v000), float(v000));
    const float c01 = std::fma(fx, float(v011) - float(v010), float(v010));
    const float c10 = std::fma(fx, float(v101) - float(v100), float(v100));
    const float c11 = std::fma(fx, float(v111) - float(v110), float(v110));
    const float c0 = std::fma(fy, c01 - c00, c00);
    const float c1 = std::fma(fy, c11 - c10, c10);
    const float value = std::clamp(std::fma(fz, c1 - c0, c0), 0.0f, 255.0f);
    out = static_cast<uint8_t>(value);
    return StrictSampleStatus::Ready;
}

StrictSampleStatus samplePinnedPoint(const PinnedChunkCache& cache,
                                     const LevelAccess& access,
                                     int level,
                                     const cv::Vec3f& p0,
                                     vc::Sampling sampling,
                                     bool zeroIsSentinel,
                                     uint8_t& out,
                                     std::string& error)
{
    if (!finiteCoord(p0) || (zeroIsSentinel && surfaceSentinel(p0))) {
        if (zeroIsSentinel)
            return StrictSampleStatus::Error;
        out = access.fill;
        return StrictSampleStatus::Ready;
    }

    const cv::Vec3f p = toLevelCoord(access, p0);
    if (sampling == vc::Sampling::Nearest)
        return samplePinnedNearest(cache, access, level, p, out, error);

    return samplePinnedTrilinear(cache, access, level, p, out, error);
}

bool samplePoint(IChunkedArray& array,
                 LocalChunkCache& cache,
                 const LevelAccess& access,
                 int level,
                 const cv::Vec3f& p0,
                 vc::Sampling sampling,
                 bool zeroIsSentinel,
                 uint8_t& out,
                 int& requested,
                 int& errors)
{
    if (!finiteCoord(p0) || (zeroIsSentinel && surfaceSentinel(p0))) {
        if (zeroIsSentinel)
            return false;
        out = access.fill;
        return true;
    }

    const cv::Vec3f p = toLevelCoord(access, p0);
    if (sampling == vc::Sampling::Nearest)
        return sampleNearest(array, cache, access, level, p, out, requested, errors);

    return sampleTrilinear(array, cache, access, level, p, out, requested, errors);
}

bool sampleLevelPoint(IChunkedArray& array,
                      LocalChunkCache& cache,
                      const LevelAccess& access,
                      int level,
                      const cv::Vec3f& p,
                      vc::Sampling sampling,
                      uint8_t& out,
                      int& requested,
                      int& errors)
{
    // Non-finite coords fail inLevelBounds (NaN compares false) and return
    // fill, identical to an explicit finiteCoord check.
    if (sampling == vc::Sampling::Nearest)
        return sampleNearest(array, cache, access, level, p, out, requested, errors);

    return sampleTrilinear(array, cache, access, level, p, out, requested, errors);
}

void addStats(ChunkedPlaneSampler::Stats& dst, const ChunkedPlaneSampler::Stats& src)
{
    dst.coveredPixels += src.coveredPixels;
    dst.requestedChunks += src.requestedChunks;
    dst.errorChunks += src.errorChunks;
    dst.missingChunks += src.missingChunks;
    dst.fallbackLevels += src.fallbackLevels;
    dst.requestedLevelOnly = dst.requestedLevelOnly || src.requestedLevelOnly;
    dst.cornerPrepareSeconds += src.cornerPrepareSeconds;
    dst.cornerLayoutSeconds += src.cornerLayoutSeconds;
    dst.cornerPinSeconds += src.cornerPinSeconds;
    dst.cornerGatherSeconds += src.cornerGatherSeconds;
    dst.cornerLayoutChunkRuns += src.cornerLayoutChunkRuns;
    dst.cornerBoundaryPoints += src.cornerBoundaryPoints;
    dst.cornerDependencies += src.cornerDependencies;
}

int countUncovered(const cv::Mat_<uint8_t>& coverage)
{
    int count = 0;
    for (int y = 0; y < coverage.rows; ++y) {
        const uint8_t* row = coverage.ptr<uint8_t>(y);
        for (int x = 0; x < coverage.cols; ++x)
            if (!row[x])
                ++count;
    }
    return count;
}

int countSampleableCoords(const cv::Mat_<uint8_t>& coverage,
                          const cv::Mat_<cv::Vec3f>& coords)
{
    const int h = std::min(coverage.rows, coords.rows);
    const int w = std::min(coverage.cols, coords.cols);
    int count = 0;
    for (int y = 0; y < h; ++y) {
        const uint8_t* coverageRow = coverage.ptr<uint8_t>(y);
        const cv::Vec3f* coordRow = coords.ptr<cv::Vec3f>(y);
        for (int x = 0; x < w; ++x)
            if (!coverageRow[x] && !surfaceSentinel(coordRow[x]))
                ++count;
    }
    return count;
}

ChunkedPlaneSampler::Stats markKnownMissingPlanePixels(
    IChunkedArray& array,
    int startLevel,
    const cv::Vec3f& origin,
    const cv::Vec3f& vxStep,
    const cv::Vec3f& vyStep,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const ChunkedPlaneSampler::Options& options)
{
    ChunkedPlaneSampler::Stats stats;
    const int firstLevel = std::max(0, startLevel);
    std::vector<MissingLevelContext> levels;
    levels.reserve(std::max(0, array.numLevels() - firstLevel));
    for (int level = firstLevel; level < array.numLevels(); ++level) {
        const LevelAccess access = makeLevelAccess(array, level);
        if (!hasSampleableLevel(access))
            continue;
        const bool queueMisses =
            options.queueMisses &&
            (options.queuedFallbackLevels < 0 ||
             level - firstLevel <= options.queuedFallbackLevels);
        levels.emplace_back(array, level, access, queueMisses,
                            toLevelPlane(access, origin, vxStep, vyStep));
    }
    std::vector<ChunkKey> keys;
    keys.reserve(8);

    for (int y = 0; y < out.rows; ++y) {
        uint8_t* outRow = out.ptr<uint8_t>(y);
        uint8_t* coverageRow = coverage.ptr<uint8_t>(y);
        for (int x = 0; x < out.cols; ++x) {
            if (coverageRow[x])
                continue;

            bool sawKnownMissing = false;
            bool sawReady = false;
            bool sawTransient = false;
            for (auto& levelCtx : levels) {
                const cv::Vec3f p = levelCtx.plane.origin
                                  + levelCtx.plane.vyStep * float(y)
                                  + levelCtx.plane.vxStep * float(x);
                if (!collectRequiredLevelChunks(levelCtx.access, levelCtx.level, p,
                                                 options.sampling, keys)) {
                    sawTransient = true;
                    break;
                }
                const ChunkDependencyState state = classifyRequiredChunks(levelCtx.cache, keys);
                if (state == ChunkDependencyState::Transient) {
                    sawTransient = true;
                    break;
                }
                if (state == ChunkDependencyState::KnownMissing) {
                    sawKnownMissing = true;
                } else {
                    sawReady = true;
                    break;
                }
            }

            if (!sawTransient && sawKnownMissing && !sawReady) {
                outRow[x] = 0;
                coverageRow[x] = 1;
                ++stats.coveredPixels;
            }
        }
    }
    return stats;
}

ChunkedPlaneSampler::Stats markKnownMissingCoordsPixels(
    IChunkedArray& array,
    int startLevel,
    const cv::Mat_<cv::Vec3f>& coords,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const ChunkedPlaneSampler::Options& options)
{
    ChunkedPlaneSampler::Stats stats;
    const int firstLevel = std::max(0, startLevel);
    std::vector<MissingLevelContext> levels;
    levels.reserve(std::max(0, array.numLevels() - firstLevel));
    for (int level = firstLevel; level < array.numLevels(); ++level) {
        const LevelAccess access = makeLevelAccess(array, level);
        if (!hasSampleableLevel(access))
            continue;
        const bool queueMisses =
            options.queueMisses &&
            (options.queuedFallbackLevels < 0 ||
             level - firstLevel <= options.queuedFallbackLevels);
        levels.emplace_back(array, level, access, queueMisses);
    }
    std::vector<ChunkKey> keys;
    keys.reserve(8);

    const int h = std::min({coords.rows, out.rows, coverage.rows});
    const int w = std::min({coords.cols, out.cols, coverage.cols});
    for (int y = 0; y < h; ++y) {
        const cv::Vec3f* coordRow = coords.ptr<cv::Vec3f>(y);
        uint8_t* outRow = out.ptr<uint8_t>(y);
        uint8_t* coverageRow = coverage.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            if (coverageRow[x] || surfaceSentinel(coordRow[x]))
                continue;

            bool sawKnownMissing = false;
            bool sawReady = false;
            bool sawTransient = false;
            for (auto& levelCtx : levels) {
                const cv::Vec3f p = toLevelCoord(levelCtx.access, coordRow[x]);
                if (!collectRequiredLevelChunks(levelCtx.access, levelCtx.level, p,
                                                 options.sampling, keys)) {
                    sawTransient = true;
                    break;
                }
                const ChunkDependencyState state = classifyRequiredChunks(levelCtx.cache, keys);
                if (state == ChunkDependencyState::Transient) {
                    sawTransient = true;
                    break;
                }
                if (state == ChunkDependencyState::KnownMissing) {
                    sawKnownMissing = true;
                } else {
                    sawReady = true;
                    break;
                }
            }

            if (!sawTransient && sawKnownMissing && !sawReady) {
                outRow[x] = 0;
                coverageRow[x] = 1;
                ++stats.coveredPixels;
            }
        }
    }
    return stats;
}

} // namespace

std::vector<ChunkKey> ChunkedPlaneSampler::collectPlaneDependencies(
    IChunkedArray& array,
    int level,
    const cv::Vec3f& origin,
    const cv::Vec3f& vxStep,
    const cv::Vec3f& vyStep,
    const cv::Mat_<uint8_t>& coverage,
    const Options& options)
{
    std::vector<ChunkKey> result;
    if (level < 0 || level >= array.numLevels() || coverage.empty())
        return result;

    const LevelAccess access = makeLevelAccess(array, level);
    if (!hasSampleableLevel(access))
        return result;
    const LevelPlane levelPlane = toLevelPlane(access, origin, vxStep, vyStep);
    const int tile = std::max(1, options.tileSize);
    std::unordered_set<ChunkKey, ChunkKeyHash> keys;
    keys.reserve(std::size_t(coverage.rows / tile + 2) *
                 std::size_t(coverage.cols / tile + 2) * 4);
    for (int ty = 0; ty < coverage.rows; ty += tile) {
        const int yEnd = std::min(ty + tile, coverage.rows);
        for (int tx = 0; tx < coverage.cols; tx += tile) {
            const int xEnd = std::min(tx + tile, coverage.cols);
            for (int y = ty; y < yEnd; ++y) {
                const uint8_t* coverageRow = coverage.ptr<uint8_t>(y);
                const cv::Vec3f rowBase = levelPlane.origin + levelPlane.vyStep * float(y);
                for (int x = tx; x < xEnd; ++x) {
                    if (coverageRow[x])
                        continue;
                    (void)collectLevelPointDependencies(
                        array, access, level, rowBase + levelPlane.vxStep * float(x),
                        options.sampling, keys);
                }
            }
        }
    }

    result.reserve(keys.size());
    for (const ChunkKey& key : keys)
        result.push_back(key);
    return result;
}

std::vector<ChunkKey> ChunkedPlaneSampler::collectCoordsDependencies(
    IChunkedArray& array,
    int level,
    const cv::Mat_<cv::Vec3f>& coords,
    const cv::Mat_<uint8_t>& coverage,
    const Options& options)
{
    std::vector<ChunkKey> result;
    if (level < 0 || level >= array.numLevels() || coords.empty() || coverage.empty())
        return result;

    const LevelAccess access = makeLevelAccess(array, level);
    if (!hasSampleableLevel(access))
        return result;
    const int tile = std::max(1, options.tileSize);
    const int h = std::min(coords.rows, coverage.rows);
    const int w = std::min(coords.cols, coverage.cols);
    std::unordered_set<ChunkKey, ChunkKeyHash> keys;
    keys.reserve(std::size_t(h / tile + 2) * std::size_t(w / tile + 2) * 4);
    for (int ty = 0; ty < h; ty += tile) {
        const int yEnd = std::min(ty + tile, h);
        for (int tx = 0; tx < w; tx += tile) {
            const int xEnd = std::min(tx + tile, w);
            for (int y = ty; y < yEnd; ++y) {
                const cv::Vec3f* coordRow = coords.ptr<cv::Vec3f>(y);
                const uint8_t* coverageRow = coverage.ptr<uint8_t>(y);
                for (int x = tx; x < xEnd; ++x) {
                    if (coverageRow[x])
                        continue;
                    (void)collectPointDependencies(array, access, level, coordRow[x],
                                                   options.sampling, true, keys);
                }
            }
        }
    }

    result.reserve(keys.size());
    for (const ChunkKey& key : keys)
        result.push_back(key);
    return result;
}

ChunkedPlaneSampler::Stats ChunkedPlaneSampler::requestPlaneDependencies(
    IChunkedArray& array,
    int level,
    const cv::Vec3f& origin,
    const cv::Vec3f& vxStep,
    const cv::Vec3f& vyStep,
    const cv::Mat_<uint8_t>& coverage,
    const Options& options)
{
    Stats stats;
    if (level < 0 || level >= array.numLevels() || coverage.empty())
        return stats;

    const LevelAccess access = makeLevelAccess(array, level);
    if (!hasSampleableLevel(access))
        return stats;
    const LevelPlane levelPlane = toLevelPlane(access, origin, vxStep, vyStep);
    LocalChunkCache chunkCache(array, 64, options.queueMisses);
    const int tile = std::max(1, options.tileSize);
    std::unordered_set<ChunkKey, ChunkKeyHash> tileKeys;
    tileKeys.reserve(std::size_t(tile) * std::size_t(tile) * 2);
    for (int ty = 0; ty < coverage.rows; ty += tile) {
        const int yEnd = std::min(ty + tile, coverage.rows);
        for (int tx = 0; tx < coverage.cols; tx += tile) {
            const int xEnd = std::min(tx + tile, coverage.cols);
            tileKeys.clear();
            for (int y = ty; y < yEnd; ++y) {
                const uint8_t* coverageRow = coverage.ptr<uint8_t>(y);
                const cv::Vec3f rowBase = levelPlane.origin + levelPlane.vyStep * float(y);
                for (int x = tx; x < xEnd; ++x) {
                    if (coverageRow[x])
                        continue;
                    (void)collectLevelPointDependencies(
                        array, access, level, rowBase + levelPlane.vxStep * float(x),
                        options.sampling, tileKeys);
                }
            }
            requestDependencies(chunkCache, tileKeys, stats);
        }
    }
    return stats;
}

ChunkedPlaneSampler::Stats ChunkedPlaneSampler::requestCoordsDependencies(
    IChunkedArray& array,
    int level,
    const cv::Mat_<cv::Vec3f>& coords,
    const cv::Mat_<uint8_t>& coverage,
    const Options& options)
{
    Stats stats;
    if (level < 0 || level >= array.numLevels() || coords.empty() || coverage.empty())
        return stats;

    const LevelAccess access = makeLevelAccess(array, level);
    if (!hasSampleableLevel(access))
        return stats;
    LocalChunkCache chunkCache(array, 64, options.queueMisses);
    const int tile = std::max(1, options.tileSize);
    const int h = std::min(coords.rows, coverage.rows);
    const int w = std::min(coords.cols, coverage.cols);
    std::unordered_set<ChunkKey, ChunkKeyHash> tileKeys;
    tileKeys.reserve(std::size_t(tile) * std::size_t(tile) * 2);
    for (int ty = 0; ty < h; ty += tile) {
        const int yEnd = std::min(ty + tile, h);
        for (int tx = 0; tx < w; tx += tile) {
            const int xEnd = std::min(tx + tile, w);
            tileKeys.clear();
            for (int y = ty; y < yEnd; ++y) {
                const cv::Vec3f* coordRow = coords.ptr<cv::Vec3f>(y);
                const uint8_t* coverageRow = coverage.ptr<uint8_t>(y);
                for (int x = tx; x < xEnd; ++x) {
                    if (coverageRow[x])
                        continue;
                    (void)collectPointDependencies(array, access, level, coordRow[x],
                                                   options.sampling, true, tileKeys);
                }
            }
            requestDependencies(chunkCache, tileKeys, stats);
        }
    }
    return stats;
}

ChunkedPlaneSampler::Stats samplePlaneLevelImpl(
    IChunkedArray& array,
    int level,
    const cv::Vec3f& origin,
    const cv::Vec3f& vxStep,
    const cv::Vec3f& vyStep,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const ChunkedPlaneSampler::Options& options,
    bool overwriteCovered)
{
    ChunkedPlaneSampler::Stats stats;
    if (level < 0 || level >= array.numLevels() || out.empty() || coverage.empty())
        return stats;

    const LevelAccess access = makeLevelAccess(array, level);
    if (!hasSampleableLevel(access))
        return stats;
    const LevelPlane levelPlane = toLevelPlane(access, origin, vxStep, vyStep);
    const int tile = std::max(1, options.tileSize);
    std::vector<SampleTile> tiles;
    tiles.reserve(std::size_t((out.rows + tile - 1) / tile) *
                  std::size_t((out.cols + tile - 1) / tile));
    for (int ty = 0; ty < out.rows; ty += tile) {
        const int yEnd = std::min(ty + tile, out.rows);
        for (int tx = 0; tx < out.cols; tx += tile) {
            const int xEnd = std::min(tx + tile, out.cols);
            tiles.push_back({tx, ty, xEnd, yEnd});
        }
    }

    auto processTileRange = [&](std::size_t begin, std::size_t end) {
        ChunkedPlaneSampler::Stats localStats;
        LocalChunkCache chunkCache(
            array, std::max<std::size_t>(16, (end - begin) * 4), options.queueMisses);
        for (std::size_t i = begin; i < end; ++i) {
            const SampleTile& sampleTile = tiles[i];
            for (int y = sampleTile.ty; y < sampleTile.yEnd; ++y) {
                uint8_t* outRow = out.ptr<uint8_t>(y);
                uint8_t* coverageRow = coverage.ptr<uint8_t>(y);
                const cv::Vec3f rowBase = levelPlane.origin + levelPlane.vyStep * float(y);
                for (int x = sampleTile.tx; x < sampleTile.xEnd; ++x) {
                    if (!overwriteCovered && coverageRow[x])
                        continue;

                    uint8_t value = 0;
                    if (sampleLevelPoint(array, chunkCache, access, level,
                                         rowBase + levelPlane.vxStep * float(x),
                                         options.sampling, value,
                                         localStats.requestedChunks, localStats.errorChunks)) {
                        const bool wasCovered = coverageRow[x] != 0;
                        outRow[x] = value;
                        coverageRow[x] = 1;
                        if (!wasCovered)
                            ++localStats.coveredPixels;
                    }
                }
            }
        }
        return localStats;
    };

    if (!shouldParallelizeSamples(out.rows, out.cols) || tiles.size() <= 1)
        return processTileRange(0, tiles.size());

    const std::size_t workerCount = std::min<std::size_t>(
        renderSamplerPool().worker_count(), tiles.size());
    const std::size_t tilesPerWorker = (tiles.size() + workerCount - 1) / workerCount;
    std::vector<std::future<ChunkedPlaneSampler::Stats>> futures;
    futures.reserve(workerCount);
    for (std::size_t worker = 0; worker < workerCount; ++worker) {
        const std::size_t begin = worker * tilesPerWorker;
        const std::size_t end = std::min(begin + tilesPerWorker, tiles.size());
        if (begin >= end)
            break;
        futures.push_back(renderSamplerPool().submit([&, begin, end] {
            return processTileRange(begin, end);
        }));
    }
    for (auto& future : futures) {
        addStats(stats, future.get());
    }
    return stats;
}

ChunkedPlaneSampler::Stats sampleCoordsLevelImpl(
    IChunkedArray& array,
    int level,
    const cv::Mat_<cv::Vec3f>& coords,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const ChunkedPlaneSampler::Options& options,
    bool overwriteCovered)
{
    ChunkedPlaneSampler::Stats stats;
    if (level < 0 || level >= array.numLevels() || coords.empty() || out.empty() || coverage.empty())
        return stats;

    const LevelAccess access = makeLevelAccess(array, level);
    if (!hasSampleableLevel(access))
        return stats;
    const int tile = std::max(1, options.tileSize);
    const int h = std::min({coords.rows, out.rows, coverage.rows});
    const int w = std::min({coords.cols, out.cols, coverage.cols});
    std::vector<SampleTile> tiles;
    tiles.reserve(std::size_t((h + tile - 1) / tile) *
                  std::size_t((w + tile - 1) / tile));
    for (int ty = 0; ty < h; ty += tile) {
        const int yEnd = std::min(ty + tile, h);
        for (int tx = 0; tx < w; tx += tile) {
            const int xEnd = std::min(tx + tile, w);
            tiles.push_back({tx, ty, xEnd, yEnd});
        }
    }

    auto processTileRange = [&](std::size_t begin, std::size_t end) {
        ChunkedPlaneSampler::Stats localStats;
        LocalChunkCache chunkCache(
            array, std::max<std::size_t>(16, (end - begin) * 4), options.queueMisses);
        for (std::size_t i = begin; i < end; ++i) {
            const SampleTile& sampleTile = tiles[i];
            for (int y = sampleTile.ty; y < sampleTile.yEnd; ++y) {
                const cv::Vec3f* coordRow = coords.ptr<cv::Vec3f>(y);
                uint8_t* outRow = out.ptr<uint8_t>(y);
                uint8_t* coverageRow = coverage.ptr<uint8_t>(y);
                for (int x = sampleTile.tx; x < sampleTile.xEnd; ++x) {
                    if (!overwriteCovered && coverageRow[x])
                        continue;

                    uint8_t value = 0;
                    if (samplePoint(array, chunkCache, access, level, coordRow[x], options.sampling,
                                    true, value, localStats.requestedChunks, localStats.errorChunks)) {
                        const bool wasCovered = coverageRow[x] != 0;
                        outRow[x] = value;
                        coverageRow[x] = 1;
                        if (!wasCovered)
                            ++localStats.coveredPixels;
                    }
                }
            }
        }
        return localStats;
    };

    if (!shouldParallelizeSamples(h, w) || tiles.size() <= 1)
        return processTileRange(0, tiles.size());

    const std::size_t workerCount = std::min<std::size_t>(
        renderSamplerPool().worker_count(), tiles.size());
    const std::size_t tilesPerWorker = (tiles.size() + workerCount - 1) / workerCount;
    std::vector<std::future<ChunkedPlaneSampler::Stats>> futures;
    futures.reserve(workerCount);
    for (std::size_t worker = 0; worker < workerCount; ++worker) {
        const std::size_t begin = worker * tilesPerWorker;
        const std::size_t end = std::min(begin + tilesPerWorker, tiles.size());
        if (begin >= end)
            break;
        futures.push_back(renderSamplerPool().submit([&, begin, end] {
            return processTileRange(begin, end);
        }));
    }
    for (auto& future : futures) {
        addStats(stats, future.get());
    }
    return stats;
}

ChunkedPlaneSampler::Stats sampleCoordsLevelBlockingRequestedLevelImpl(
    IChunkedArray& array,
    int level,
    const cv::Mat_<cv::Vec3f>& coords,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const ChunkedPlaneSampler::Options& options)
{
    ChunkedPlaneSampler::Stats stats;
    stats.requestedLevelOnly = true;
    stats.fallbackLevels = 0;
    if (level < 0 || level >= array.numLevels() || coords.empty() || out.empty() || coverage.empty())
        return stats;

    const LevelAccess access = makeLevelAccess(array, level);
    if (!hasSampleableLevel(access))
        return stats;

    const std::vector<ChunkKey> keys = ChunkedPlaneSampler::collectCoordsDependencies(
        array, level, coords, coverage, options);
    stats.requestedChunks = static_cast<int>(keys.size());

    if (!keys.empty())
        array.prefetchChunks(keys, false);

    PinnedChunkCache pinned(keys.size());
    for (const ChunkKey& key : keys) {
        ChunkResult result = array.getChunkBlocking(key.level, key.iz, key.iy, key.ix);
        if (result.status == ChunkStatus::Error) {
            ++stats.errorChunks;
            const std::string message = result.error.empty()
                ? "chunk fetch failed during blocking requested-level sampling"
                : result.error;
            throw std::runtime_error(message);
        }
        if (result.status == ChunkStatus::MissQueued) {
            ++stats.errorChunks;
            throw std::runtime_error(
                "blocking requested-level sampling received unresolved chunk after getChunkBlocking");
        }
        if (result.status == ChunkStatus::Missing)
            ++stats.missingChunks;
        pinned.chunks.emplace(key, std::move(result));
    }

    const int tile = std::max(1, options.tileSize);
    const int h = std::min({coords.rows, out.rows, coverage.rows});
    const int w = std::min({coords.cols, out.cols, coverage.cols});
    std::vector<SampleTile> tiles;
    tiles.reserve(std::size_t((h + tile - 1) / tile) *
                  std::size_t((w + tile - 1) / tile));
    for (int ty = 0; ty < h; ty += tile) {
        const int yEnd = std::min(ty + tile, h);
        for (int tx = 0; tx < w; tx += tile) {
            const int xEnd = std::min(tx + tile, w);
            tiles.push_back({tx, ty, xEnd, yEnd});
        }
    }

    auto processTileRange = [&](std::size_t begin, std::size_t end) {
        ChunkedPlaneSampler::Stats localStats;
        localStats.requestedLevelOnly = true;
        for (std::size_t i = begin; i < end; ++i) {
            const SampleTile& sampleTile = tiles[i];
            for (int y = sampleTile.ty; y < sampleTile.yEnd; ++y) {
                const cv::Vec3f* coordRow = coords.ptr<cv::Vec3f>(y);
                uint8_t* outRow = out.ptr<uint8_t>(y);
                uint8_t* coverageRow = coverage.ptr<uint8_t>(y);
                for (int x = sampleTile.tx; x < sampleTile.xEnd; ++x) {
                    if (coverageRow[x] || surfaceSentinel(coordRow[x]))
                        continue;

                    uint8_t value = 0;
                    std::string error;
                    const StrictSampleStatus status = samplePinnedPoint(
                        pinned, access, level, coordRow[x], options.sampling,
                        false, value, error);
                    if (status == StrictSampleStatus::Error) {
                        throw std::runtime_error(
                            error.empty()
                                ? "blocking requested-level sampling failed"
                                : error);
                    }

                    const bool wasCovered = coverageRow[x] != 0;
                    outRow[x] = status == StrictSampleStatus::Missing ? uint8_t{0} : value;
                    coverageRow[x] = 1;
                    if (!wasCovered)
                        ++localStats.coveredPixels;
                }
            }
        }
        return localStats;
    };

    if (!shouldParallelizeSamples(h, w) || tiles.size() <= 1) {
        addStats(stats, processTileRange(0, tiles.size()));
        return stats;
    }

    const std::size_t workerCount = std::min<std::size_t>(
        renderSamplerPool().worker_count(), tiles.size());
    const std::size_t tilesPerWorker = (tiles.size() + workerCount - 1) / workerCount;
    std::vector<std::future<ChunkedPlaneSampler::Stats>> futures;
    futures.reserve(workerCount);
    for (std::size_t worker = 0; worker < workerCount; ++worker) {
        const std::size_t begin = worker * tilesPerWorker;
        const std::size_t end = std::min(begin + tilesPerWorker, tiles.size());
        if (begin >= end)
            break;
        futures.push_back(renderSamplerPool().submit([&, begin, end] {
            return processTileRange(begin, end);
        }));
    }
    for (auto& future : futures)
        addStats(stats, future.get());
    return stats;
}

ChunkedPlaneSampler::Stats ChunkedPlaneSampler::samplePlaneLevel(
    IChunkedArray& array,
    int level,
    const cv::Vec3f& origin,
    const cv::Vec3f& vxStep,
    const cv::Vec3f& vyStep,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const Options& options)
{
    return samplePlaneLevelImpl(array, level, origin, vxStep, vyStep, out, coverage,
                                options, false);
}

ChunkedPlaneSampler::Stats ChunkedPlaneSampler::sampleCoordsLevel(
    IChunkedArray& array,
    int level,
    const cv::Mat_<cv::Vec3f>& coords,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const Options& options)
{
    return sampleCoordsLevelImpl(array, level, coords, out, coverage, options, false);
}

ChunkedPlaneSampler::Stats ChunkedPlaneSampler::sampleCoordsLevelBlockingRequestedLevel(
    IChunkedArray& array,
    int level,
    const cv::Mat_<cv::Vec3f>& coords,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const Options& options)
{
    return sampleCoordsLevelBlockingRequestedLevelImpl(
        array, level, coords, out, coverage, options);
}

ChunkedPlaneSampler::Stats
ChunkedPlaneSampler::sampleTrilinearCornersLevelBlockingRequestedLevel(
    const std::vector<IChunkedArray*>& arrays,
    int level,
    const std::vector<cv::Vec3f>& levelCoords,
    std::vector<std::vector<std::array<uint8_t, 8>>>& values,
    std::vector<cv::Vec3f>& fractionsXYZ,
    std::vector<uint8_t>& valid,
    int parallelThreads)
{
    using Clock = std::chrono::steady_clock;
    const auto prepareStart = Clock::now();
    Stats stats;
    stats.requestedLevelOnly = true;
    values.resize(arrays.size());
    for (auto& volumeValues : values)
        volumeValues.resize(levelCoords.size());
    fractionsXYZ.resize(levelCoords.size());
    valid.resize(levelCoords.size());
    if (arrays.empty() || levelCoords.empty())
        return stats;
    if (arrays.front() == nullptr || level < 0 || level >= arrays.front()->numLevels())
        throw std::invalid_argument("invalid requested-level corner batch source");

    const LevelAccess firstAccess = makeLevelAccess(*arrays.front(), level);
    if (!hasSampleableLevel(firstAccess))
        throw std::invalid_argument("requested-level corner batch has an invalid grid");
    std::vector<LevelAccess> accesses;
    accesses.reserve(arrays.size());
    for (IChunkedArray* array : arrays) {
        if (array == nullptr || level >= array->numLevels() ||
            array->shape(level) != firstAccess.shape ||
            array->dtype() != ChunkDtype::UInt8) {
            throw std::invalid_argument(
                "requested-level corner batch arrays must share one uint8 shape");
        }
        LevelAccess access = makeLevelAccess(*array, level);
        if (!hasSampleableLevel(access))
            throw std::invalid_argument("requested-level corner batch has an invalid grid");
        accesses.push_back(access);
    }

    struct VoxelCube {
        int x0 = 0;
        int y0 = 0;
        int z0 = 0;
        int x1 = 0;
        int y1 = 0;
        int z1 = 0;
    };
    constexpr uint32_t kNoBoundary = std::numeric_limits<uint32_t>::max();
    struct PointCorners {
        uint32_t dependencyIndex = 0;
        uint32_t baseByteOffset = 0;
        uint32_t boundaryIndex = std::numeric_limits<uint32_t>::max();
        uint8_t clampedAxes = 0;

        [[nodiscard]] bool singleDependency() const noexcept
        {
            return boundaryIndex == std::numeric_limits<uint32_t>::max();
        }
    };
    struct BoundaryCorners {
        std::array<uint32_t, 8> dependencyIndex{};
        std::array<uint32_t, 8> byteOffset{};
    };
    std::vector<VoxelCube> voxelCubes(levelCoords.size());
    const auto prepareRange = [&](size_t begin, size_t end) {
        for (size_t pointIndex = begin; pointIndex < end; ++pointIndex) {
            const cv::Vec3f point = levelCoords[pointIndex];
            if (!finiteCoord(point) ||
                !inLevelBounds(firstAccess.shape, point[2], point[1], point[0])) {
                fractionsXYZ[pointIndex] = {0.0f, 0.0f, 0.0f};
                valid[pointIndex] = 0;
                continue;
            }
            VoxelCube& cube = voxelCubes[pointIndex];
            cube.x0 = static_cast<int>(std::floor(point[0]));
            cube.y0 = static_cast<int>(std::floor(point[1]));
            cube.z0 = static_cast<int>(std::floor(point[2]));
            cube.x1 = std::min(cube.x0 + 1, firstAccess.shape[2] - 1);
            cube.y1 = std::min(cube.y0 + 1, firstAccess.shape[1] - 1);
            cube.z1 = std::min(cube.z0 + 1, firstAccess.shape[0] - 1);
            fractionsXYZ[pointIndex] = {
                point[0] - static_cast<float>(cube.x0),
                point[1] - static_cast<float>(cube.y0),
                point[2] - static_cast<float>(cube.z0)};
            valid[pointIndex] = 1;
        }
    };
    prepareRange(0, levelCoords.size());
    const auto prepareEnd = Clock::now();
    stats.cornerPrepareSeconds =
        std::chrono::duration<double>(prepareEnd - prepareStart).count();

    struct CornerLayout {
        std::array<int, 3> chunkShape{};
        std::vector<PointCorners> points;
        std::vector<ChunkKey> dependencies;
        std::vector<BoundaryCorners> boundaryCorners;
        std::vector<size_t> volumes;
    };
    std::vector<CornerLayout> layouts;
    std::vector<size_t> volumeLayout(arrays.size(), 0);
    for (size_t volumeIndex = 0; volumeIndex < arrays.size(); ++volumeIndex) {
        const auto chunkShape = accesses[volumeIndex].chunkShape;
        auto it = std::find_if(
            layouts.begin(), layouts.end(),
            [&](const CornerLayout& layout) { return layout.chunkShape == chunkShape; });
        if (it == layouts.end()) {
            layouts.push_back({chunkShape, {}, {}, {}, {volumeIndex}});
            volumeLayout[volumeIndex] = layouts.size() - 1;
        } else {
            volumeLayout[volumeIndex] =
                static_cast<size_t>(std::distance(layouts.begin(), it));
            it->volumes.push_back(volumeIndex);
        }
    }

    for (auto& layout : layouts) {
        layout.points.resize(levelCoords.size());
        layout.boundaryCorners.reserve(levelCoords.size() / 8 + 8);
        constexpr size_t kLinearDependencyLimit = 16;
        layout.dependencies.reserve(kLinearDependencyLimit);
        std::unordered_map<ChunkKey, size_t, ChunkKeyHash> dependencyIndices;
        bool useDependencyMap = false;
        ChunkKey currentKey{};
        size_t currentDependencyIndex = 0;
        std::array<int, 3> currentBegin{};
        std::array<int, 3> currentEnd{};
        bool haveCurrentChunk = false;
        const auto resolveDependency = [&](const ChunkKey& key) {
            if (!useDependencyMap) {
                const auto it = std::find(
                    layout.dependencies.begin(), layout.dependencies.end(), key);
                if (it != layout.dependencies.end()) {
                    return static_cast<size_t>(
                        std::distance(layout.dependencies.begin(), it));
                }
                const size_t index = layout.dependencies.size();
                layout.dependencies.push_back(key);
                if (layout.dependencies.size() > kLinearDependencyLimit) {
                    dependencyIndices.reserve(layout.dependencies.size() * 2);
                    for (size_t dependencyIndex = 0;
                         dependencyIndex < layout.dependencies.size();
                         ++dependencyIndex) {
                        dependencyIndices.emplace(
                            layout.dependencies[dependencyIndex], dependencyIndex);
                    }
                    useDependencyMap = true;
                }
                return index;
            }
            const auto [it, inserted] = dependencyIndices.try_emplace(
                key, layout.dependencies.size());
            if (inserted)
                layout.dependencies.push_back(key);
            return it->second;
        };
        for (size_t pointIndex = 0; pointIndex < voxelCubes.size(); ++pointIndex) {
            if (valid[pointIndex] == 0)
                continue;
            const VoxelCube& cube = voxelCubes[pointIndex];
            const bool withinCurrentChunk =
                haveCurrentChunk &&
                cube.x0 >= currentBegin[2] && cube.x0 < currentEnd[2] &&
                cube.y0 >= currentBegin[1] && cube.y0 < currentEnd[1] &&
                cube.z0 >= currentBegin[0] && cube.z0 < currentEnd[0];
            if (!withinCurrentChunk) {
                ++stats.cornerLayoutChunkRuns;
                currentKey = {
                    level,
                    cube.z0 / layout.chunkShape[0],
                    cube.y0 / layout.chunkShape[1],
                    cube.x0 / layout.chunkShape[2]};
                currentDependencyIndex = resolveDependency(currentKey);
                currentBegin = {
                    currentKey.iz * layout.chunkShape[0],
                    currentKey.iy * layout.chunkShape[1],
                    currentKey.ix * layout.chunkShape[2]};
                currentEnd = {
                    currentBegin[0] + layout.chunkShape[0],
                    currentBegin[1] + layout.chunkShape[1],
                    currentBegin[2] + layout.chunkShape[2]};
                haveCurrentChunk = true;
            }
            PointCorners& point = layout.points[pointIndex];
            const bool singleDependency =
                cube.x1 < currentEnd[2] &&
                cube.y1 < currentEnd[1] &&
                cube.z1 < currentEnd[0];
            if (currentDependencyIndex >
                static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
                throw std::overflow_error(
                    "corner batch has too many chunk dependencies");
            }
            point.dependencyIndex = static_cast<uint32_t>(currentDependencyIndex);
            if (!singleDependency) {
                ++stats.cornerBoundaryPoints;
                if (layout.boundaryCorners.size() >=
                    static_cast<size_t>(kNoBoundary)) {
                    throw std::overflow_error(
                        "corner batch has too many boundary points");
                }
                point.boundaryIndex = static_cast<uint32_t>(
                    layout.boundaryCorners.size());
                layout.boundaryCorners.emplace_back();
            } else {
                const int lz = cube.z0 - currentBegin[0];
                const int ly = cube.y0 - currentBegin[1];
                const int lx = cube.x0 - currentBegin[2];
                const size_t byteOffset =
                    (static_cast<size_t>(lz) *
                         static_cast<size_t>(layout.chunkShape[1]) +
                     static_cast<size_t>(ly)) *
                        static_cast<size_t>(layout.chunkShape[2]) +
                    static_cast<size_t>(lx);
                if (byteOffset >
                    static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
                    throw std::overflow_error(
                        "corner batch chunk exceeds uint32 byte offsets");
                }
                point.baseByteOffset = static_cast<uint32_t>(byteOffset);
                point.clampedAxes = static_cast<uint8_t>(
                    (cube.x1 == cube.x0 ? 1 : 0) |
                    (cube.y1 == cube.y0 ? 2 : 0) |
                    (cube.z1 == cube.z0 ? 4 : 0));
                continue;
            }
            BoundaryCorners& boundary =
                layout.boundaryCorners[point.boundaryIndex];
            std::array<uint32_t, 8> dependencyByChunkMask{};
            dependencyByChunkMask[0] =
                static_cast<uint32_t>(currentDependencyIndex);
            uint8_t resolvedChunkMasks = 1;
            const bool crossesX = cube.x1 >= currentEnd[2];
            const bool crossesY = cube.y1 >= currentEnd[1];
            const bool crossesZ = cube.z1 >= currentEnd[0];
            const std::array<int, 2> localX{
                cube.x0 - currentBegin[2],
                cube.x1 - (crossesX ? currentEnd[2] : currentBegin[2])};
            const std::array<int, 2> localY{
                cube.y0 - currentBegin[1],
                cube.y1 - (crossesY ? currentEnd[1] : currentBegin[1])};
            const std::array<int, 2> localZ{
                cube.z0 - currentBegin[0],
                cube.z1 - (crossesZ ? currentEnd[0] : currentBegin[0])};
            size_t corner = 0;
            for (int dz = 0; dz <= 1; ++dz) {
                for (int dy = 0; dy <= 1; ++dy) {
                    for (int dx = 0; dx <= 1; ++dx) {
                        const uint8_t chunkMask = static_cast<uint8_t>(
                            (dx != 0 && crossesX ? 1 : 0) |
                            (dy != 0 && crossesY ? 2 : 0) |
                            (dz != 0 && crossesZ ? 4 : 0));
                        if ((resolvedChunkMasks & (uint8_t{1} << chunkMask)) == 0) {
                            const ChunkKey adjacentKey{
                                level,
                                currentKey.iz + ((chunkMask & 4) != 0 ? 1 : 0),
                                currentKey.iy + ((chunkMask & 2) != 0 ? 1 : 0),
                                currentKey.ix + ((chunkMask & 1) != 0 ? 1 : 0)};
                            const size_t dependencyIndex =
                                resolveDependency(adjacentKey);
                            if (dependencyIndex > static_cast<size_t>(
                                    std::numeric_limits<uint32_t>::max())) {
                                throw std::overflow_error(
                                    "corner batch has too many chunk dependencies");
                            }
                            dependencyByChunkMask[chunkMask] =
                                static_cast<uint32_t>(dependencyIndex);
                            resolvedChunkMasks |= uint8_t{1} << chunkMask;
                        }
                        const size_t dependencyIndex =
                            dependencyByChunkMask[chunkMask];
                        if (dependencyIndex >
                            static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
                            throw std::overflow_error(
                                "corner batch has too many chunk dependencies");
                        }
                        boundary.dependencyIndex[corner] =
                            static_cast<uint32_t>(dependencyIndex);
                        const size_t byteOffset =
                            (static_cast<size_t>(localZ[dz]) *
                                 static_cast<size_t>(layout.chunkShape[1]) +
                             static_cast<size_t>(localY[dy])) *
                                static_cast<size_t>(layout.chunkShape[2]) +
                            static_cast<size_t>(localX[dx]);
                        if (byteOffset >
                            static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
                            throw std::overflow_error(
                                "corner batch chunk exceeds uint32 byte offsets");
                        }
                        boundary.byteOffset[corner] =
                            static_cast<uint32_t>(byteOffset);
                        ++corner;
                    }
                }
            }
        }
        stats.cornerDependencies += layout.dependencies.size();
    }
    const auto layoutEnd = Clock::now();
    stats.cornerLayoutSeconds =
        std::chrono::duration<double>(layoutEnd - prepareEnd).count();

    for (size_t volumeIndex = 0; volumeIndex < arrays.size(); ++volumeIndex) {
        const auto& dependencies = layouts[volumeLayout[volumeIndex]].dependencies;
        stats.requestedChunks += static_cast<int>(dependencies.size());
        if (!dependencies.empty())
            arrays[volumeIndex]->prefetchChunks(dependencies, false);
    }

    struct PinnedVolume {
        std::vector<ChunkResult> chunks;
        std::vector<const std::byte*> data;
        std::vector<size_t> size;
    };
    std::vector<PinnedVolume> pinned(arrays.size());
    for (size_t volumeIndex = 0; volumeIndex < pinned.size(); ++volumeIndex) {
        auto& volume = pinned[volumeIndex];
        const size_t dependencyCount =
            layouts[volumeLayout[volumeIndex]].dependencies.size();
        volume.chunks.resize(dependencyCount);
        volume.data.resize(dependencyCount, nullptr);
        volume.size.resize(dependencyCount, 0);
    }
    size_t volumeIndex = 0;
    for (IChunkedArray* array : arrays) {
        const auto& dependencies = layouts[volumeLayout[volumeIndex]].dependencies;
        for (size_t dependencyIndex = 0;
             dependencyIndex < dependencies.size();
             ++dependencyIndex) {
            const ChunkKey& key = dependencies[dependencyIndex];
            ChunkResult result = array->getChunkBlocking(
                key.level, key.iz, key.iy, key.ix);
            if (result.status == ChunkStatus::Error) {
                ++stats.errorChunks;
                throw std::runtime_error(
                    result.error.empty() ? "corner batch chunk fetch failed" : result.error);
            }
            if (result.status == ChunkStatus::MissQueued) {
                ++stats.errorChunks;
                throw std::runtime_error(
                    "corner batch received unresolved chunk after blocking fetch");
            }
            if (result.status == ChunkStatus::Missing)
                ++stats.missingChunks;
            pinned[volumeIndex].chunks[dependencyIndex] = std::move(result);
            const auto& stored = pinned[volumeIndex].chunks[dependencyIndex];
            if (stored.status == ChunkStatus::Data && stored.bytes) {
                pinned[volumeIndex].data[dependencyIndex] = stored.bytes->data();
                pinned[volumeIndex].size[dependencyIndex] = stored.bytes->size();
            }
        }
        ++volumeIndex;
    }
    const auto pinEnd = Clock::now();
    stats.cornerPinSeconds =
        std::chrono::duration<double>(pinEnd - layoutEnd).count();

    const size_t requestedWorkers = parallelThreads > 0
        ? static_cast<size_t>(parallelThreads)
        : cornerBatchPool().worker_count();
    const size_t workerCount = std::min({
        cornerBatchPool().worker_count(), requestedWorkers, levelCoords.size()});
    const size_t pointsPerWorker =
        (levelCoords.size() + workerCount - 1) / workerCount;
    std::vector<std::future<void>> sampleFutures;
    sampleFutures.reserve(layouts.size() * workerCount);
    for (size_t layoutIndex = 0; layoutIndex < layouts.size(); ++layoutIndex) {
        for (size_t worker = 0; worker < workerCount; ++worker) {
            const size_t begin = worker * pointsPerWorker;
            const size_t end = std::min(begin + pointsPerWorker, levelCoords.size());
            if (begin >= end)
                break;
            sampleFutures.push_back(cornerBatchPool().submit(
                [&, layoutIndex, begin, end] {
        const CornerLayout& layout = layouts[layoutIndex];
        for (size_t pointIndex = begin; pointIndex < end; ++pointIndex) {
            if (valid[pointIndex] == 0)
                continue;
            const PointCorners& point = layout.points[pointIndex];
            std::array<uint32_t, 8> commonOffsets{};
            uint32_t commonMaxOffset = 0;
            const BoundaryCorners* boundary = nullptr;
            if (point.singleDependency()) {
                const uint32_t xStride = (point.clampedAxes & 1) != 0 ? 0U : 1U;
                const uint32_t yStride = (point.clampedAxes & 2) != 0
                    ? 0U
                    : static_cast<uint32_t>(layout.chunkShape[2]);
                const uint32_t zStride = (point.clampedAxes & 4) != 0
                    ? 0U
                    : static_cast<uint32_t>(layout.chunkShape[1]) *
                          static_cast<uint32_t>(layout.chunkShape[2]);
                const uint32_t base = point.baseByteOffset;
                commonOffsets = {
                    base,
                    base + xStride,
                    base + yStride,
                    base + yStride + xStride,
                    base + zStride,
                    base + zStride + xStride,
                    base + zStride + yStride,
                    base + zStride + yStride + xStride};
                commonMaxOffset = commonOffsets[7];
            } else {
                boundary = &layout.boundaryCorners[point.boundaryIndex];
            }
            for (const size_t volumeIndex : layout.volumes) {
                const PinnedVolume& volume = pinned[volumeIndex];
                const uint8_t fill = accesses[volumeIndex].fill;
                if (point.singleDependency()) {
                    const size_t dependencyIndex = point.dependencyIndex;
                    const std::byte* data = volume.data[dependencyIndex];
                    if (data == nullptr) {
                        values[volumeIndex][pointIndex].fill(fill);
                        continue;
                    }
                    if (commonMaxOffset >= volume.size[dependencyIndex]) {
                        throw std::runtime_error(
                            "corner batch decoded chunk has invalid byte extent");
                    }
                    auto& pointValues = values[volumeIndex][pointIndex];
                    for (size_t corner = 0; corner < 8; ++corner) {
                        pointValues[corner] =
                            std::to_integer<uint8_t>(data[commonOffsets[corner]]);
                    }
                    continue;
                }
                for (size_t corner = 0; corner < 8; ++corner) {
                    uint8_t value = fill;
                    const size_t dependencyIndex =
                        boundary->dependencyIndex[corner];
                    const std::byte* data = volume.data[dependencyIndex];
                    if (data != nullptr) {
                        if (boundary->byteOffset[corner] >=
                            volume.size[dependencyIndex]) {
                            throw std::runtime_error(
                                "corner batch decoded chunk has invalid byte extent");
                        }
                        value = std::to_integer<uint8_t>(
                            data[boundary->byteOffset[corner]]);
                    }
                    values[volumeIndex][pointIndex][corner] = value;
                }
            }
        }
                }));
        }
    }
    for (auto& future : sampleFutures)
        future.get();
    const auto gatherEnd = Clock::now();
    stats.cornerGatherSeconds =
        std::chrono::duration<double>(gatherEnd - pinEnd).count();
    const size_t validPoints = static_cast<size_t>(
        std::count(valid.begin(), valid.end(), uint8_t{1}));
    const size_t covered = validPoints * arrays.size() * 8;
    stats.coveredPixels = static_cast<int>(std::min<size_t>(
        covered, static_cast<size_t>(std::numeric_limits<int>::max())));
    return stats;
}

ChunkedPlaneSampler::Stats ChunkedPlaneSampler::samplePlaneFineToCoarse(
    IChunkedArray& array,
    int startLevel,
    const cv::Vec3f& origin,
    const cv::Vec3f& vxStep,
    const cv::Vec3f& vyStep,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const Options& options)
{
    Stats total;
    int remaining = countUncovered(coverage);
    const int firstLevel = std::max(0, startLevel);
    for (int level = firstLevel; level < array.numLevels(); ++level) {
        Options levelOptions = options;
        levelOptions.queueMisses =
            options.queueMisses &&
            (options.queuedFallbackLevels < 0 ||
             level - firstLevel <= options.queuedFallbackLevels);
        Stats stats = samplePlaneLevel(array, level, origin, vxStep, vyStep,
                                       out, coverage, levelOptions);
        addStats(total, stats);
        remaining -= stats.coveredPixels;
        if (remaining <= 0)
            break;
    }
    if (remaining > 0) {
        Stats stats = markKnownMissingPlanePixels(
            array, startLevel, origin, vxStep, vyStep, out, coverage, options);
        addStats(total, stats);
    }
    return total;
}

ChunkedPlaneSampler::Stats ChunkedPlaneSampler::sampleCoordsFineToCoarse(
    IChunkedArray& array,
    int startLevel,
    const cv::Mat_<cv::Vec3f>& coords,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const Options& options)
{
    Stats total;
    int remaining = countSampleableCoords(coverage, coords);
    const int firstLevel = std::max(0, startLevel);
    for (int level = firstLevel; level < array.numLevels(); ++level) {
        Options levelOptions = options;
        levelOptions.queueMisses =
            options.queueMisses &&
            (options.queuedFallbackLevels < 0 ||
             level - firstLevel <= options.queuedFallbackLevels);
        Stats stats = sampleCoordsLevel(
            array, level, coords, out, coverage, levelOptions);
        addStats(total, stats);
        remaining -= stats.coveredPixels;
        if (remaining <= 0)
            break;
    }
    if (remaining > 0) {
        Stats stats = markKnownMissingCoordsPixels(
            array, startLevel, coords, out, coverage, options);
        addStats(total, stats);
    }
    return total;
}

ChunkedPlaneSampler::Stats ChunkedPlaneSampler::samplePlaneCoarseToFine(
    IChunkedArray& array,
    int finestLevel,
    const cv::Vec3f& origin,
    const cv::Vec3f& vxStep,
    const cv::Vec3f& vyStep,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const Options& options)
{
    Stats total;
    if (array.numLevels() <= 0)
        return total;

    const int firstLevel = std::clamp(finestLevel, 0, array.numLevels() - 1);
    for (int level = array.numLevels() - 1; level >= firstLevel; --level) {
        Stats stats = samplePlaneLevelImpl(array, level, origin, vxStep, vyStep,
                                           out, coverage, options, true);
        addStats(total, stats);
    }
    return total;
}

ChunkedPlaneSampler::Stats ChunkedPlaneSampler::sampleCoordsCoarseToFine(
    IChunkedArray& array,
    int finestLevel,
    const cv::Mat_<cv::Vec3f>& coords,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const Options& options)
{
    Stats total;
    if (array.numLevels() <= 0)
        return total;

    const int firstLevel = std::clamp(finestLevel, 0, array.numLevels() - 1);
    for (int level = array.numLevels() - 1; level >= firstLevel; --level) {
        Stats stats = sampleCoordsLevelImpl(array, level, coords, out, coverage,
                                            options, true);
        addStats(total, stats);
    }
    return total;
}

} // namespace vc::render
