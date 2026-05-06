#include "vc/core/render/ChunkedPlaneSampler.hpp"

#include <utils/thread_pool.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <future>
#include <limits>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vc::render {
namespace {

struct LocalChunkCache {
    explicit LocalChunkCache(IChunkedArray& a, std::size_t expectedChunks = 0)
        : array(a)
    {
        if (expectedChunks > 0) {
            chunks.reserve(expectedChunks);
            requestedKeys.reserve(expectedChunks);
            errorKeys.reserve(expectedChunks);
        }
    }

    ChunkResult get(const ChunkKey& key, int& requested, int& errors)
    {
        auto it = chunks.find(key);
        if (it != chunks.end())
            return it->second;

        ChunkResult result = array.tryGetChunk(key.level, key.iz, key.iy, key.ix);
        chunks.emplace(key, result);
        if (result.status == ChunkStatus::MissQueued && requestedKeys.insert(key).second)
            ++requested;
        if (result.status == ChunkStatus::Error && errorKeys.insert(key).second)
            ++errors;
        return result;
    }

    IChunkedArray& array;
    std::unordered_map<ChunkKey, ChunkResult, ChunkKeyHash> chunks;
    std::unordered_set<ChunkKey, ChunkKeyHash> requestedKeys;
    std::unordered_set<ChunkKey, ChunkKeyHash> errorKeys;
};

constexpr int kParallelMinPixels = 128 * 128;
constexpr int kMaxRenderSamplerWorkers = 8;

int renderSamplerWorkerCount()
{
    const unsigned hc = std::thread::hardware_concurrency();
    if (hc <= 2)
        return 1;
    return std::clamp(static_cast<int>(hc) - 2, 1, kMaxRenderSamplerWorkers);
}

utils::ThreadPool& renderSamplerPool()
{
    static utils::ThreadPool pool(static_cast<std::size_t>(renderSamplerWorkerCount()));
    return pool;
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
    ChunkResult result = cache.get({level, cz, cy, cx}, requested, errors);
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

    const int ix = int(std::floor(x));
    const int iy = int(std::floor(y));
    const int iz = int(std::floor(z));
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
            ChunkResult result = cache.get({level, cz, cy, cx}, requested, errors);
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
    if (!finiteCoord(p)) {
        out = access.fill;
        return true;
    }

    if (sampling == vc::Sampling::Nearest)
        return sampleNearest(array, cache, access, level, p, out, requested, errors);

    return sampleTrilinear(array, cache, access, level, p, out, requested, errors);
}

void addStats(ChunkedPlaneSampler::Stats& dst, const ChunkedPlaneSampler::Stats& src)
{
    dst.coveredPixels += src.coveredPixels;
    dst.requestedChunks += src.requestedChunks;
    dst.errorChunks += src.errorChunks;
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
                const cv::Vec3f rowBase = origin + vyStep * float(y);
                for (int x = tx; x < xEnd; ++x) {
                    if (coverageRow[x])
                        continue;
                    (void)collectPointDependencies(array, access, level, rowBase + vxStep * float(x),
                                                   options.sampling, false, keys);
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
    LocalChunkCache chunkCache(array, 64);
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
                const cv::Vec3f rowBase = origin + vyStep * float(y);
                for (int x = tx; x < xEnd; ++x) {
                    if (coverageRow[x])
                        continue;
                    (void)collectPointDependencies(array, access, level, rowBase + vxStep * float(x),
                                                   options.sampling, false, tileKeys);
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
    LocalChunkCache chunkCache(array, 64);
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
        LocalChunkCache chunkCache(array, std::max<std::size_t>(16, (end - begin) * 4));
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
        LocalChunkCache chunkCache(array, std::max<std::size_t>(16, (end - begin) * 4));
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
    for (int level = std::max(0, startLevel); level < array.numLevels(); ++level) {
        Stats stats = samplePlaneLevel(array, level, origin, vxStep, vyStep,
                                       out, coverage, options);
        addStats(total, stats);
        remaining -= stats.coveredPixels;
        if (remaining <= 0)
            break;
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
    for (int level = std::max(0, startLevel); level < array.numLevels(); ++level) {
        Stats stats = sampleCoordsLevel(array, level, coords, out, coverage, options);
        addStats(total, stats);
        remaining -= stats.coveredPixels;
        if (remaining <= 0)
            break;
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
