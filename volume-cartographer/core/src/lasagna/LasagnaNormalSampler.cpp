#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include "utils/zarr.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vc::lasagna {
namespace {

constexpr double kEpsilon = 1.0e-12;

[[nodiscard]] double length(const cv::Vec3d& v)
{
    return std::sqrt(v.dot(v));
}

[[nodiscard]] cv::Vec3d normalizedOrZero(const cv::Vec3d& v)
{
    const double len = length(v);
    if (!(len > kEpsilon) || !std::isfinite(len)) {
        return {0.0, 0.0, 0.0};
    }
    return v / len;
}

[[nodiscard]] double decodeNormalComponent(double raw)
{
    return (raw - 128.0) / 127.0;
}

[[nodiscard]] cv::Vec3d decodedNormalFromRaw(double rawNx, double rawNy)
{
    const double nx = decodeNormalComponent(rawNx);
    const double ny = decodeNormalComponent(rawNy);
    const double nzSq = std::max(0.0, 1.0 - nx * nx - ny * ny);
    return normalizedOrZero({nx, ny, std::sqrt(nzSq)});
}

struct ChunkKey {
    uint32_t arrayId = 0;
    uint32_t channelIndex = 0;
    uint32_t z = 0;
    uint32_t y = 0;
    uint32_t x = 0;

    [[nodiscard]] bool operator==(const ChunkKey& other) const noexcept
    {
        return arrayId == other.arrayId &&
               channelIndex == other.channelIndex &&
               z == other.z &&
               y == other.y &&
               x == other.x;
    }
};

struct ChunkKeyHash {
    [[nodiscard]] size_t operator()(const ChunkKey& key) const noexcept
    {
        size_t hash = key.arrayId;
        hash ^= key.channelIndex + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        hash ^= key.z + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        hash ^= key.y + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        hash ^= key.x + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        return hash;
    }
};

struct CachedChunk {
    std::vector<uint8_t> values;
    std::array<size_t, 3> dimsZYX{0, 0, 0};
};

struct ChannelBinding {
    const LasagnaChannelGroup* group = nullptr;
    uint32_t arrayId = 0;
    size_t channelIndex = 0;
    std::filesystem::path path;
    std::shared_ptr<utils::ZarrArray> array;
    bool hasChannelDimension = false;
    std::array<size_t, 3> shapeZYX{0, 0, 0};
    std::array<size_t, 3> chunksZYX{0, 0, 0};
    double spacing = 1.0;
};

[[nodiscard]] uint32_t checkedChunkIndex(size_t value)
{
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Lasagna chunk index exceeds compact cache key range");
    }
    return static_cast<uint32_t>(value);
}

[[nodiscard]] std::vector<size_t> chunkPathIndicesForKey(
    const ChannelBinding& binding,
    const ChunkKey& key)
{
    if (binding.hasChannelDimension) {
        const auto& chunks = binding.array->metadata().chunks;
        return {
            binding.channelIndex / chunks[0],
            key.z,
            key.y,
            key.x,
        };
    }
    return {key.z, key.y, key.x};
}

[[nodiscard]] ChunkKey chunkKeyForVoxel(
    const ChannelBinding& binding,
    size_t z,
    size_t y,
    size_t x)
{
    return {
        binding.arrayId,
        static_cast<uint32_t>(binding.channelIndex),
        checkedChunkIndex(z / binding.chunksZYX[0]),
        checkedChunkIndex(y / binding.chunksZYX[1]),
        checkedChunkIndex(x / binding.chunksZYX[2]),
    };
}

[[nodiscard]] std::string chunkKeyToString(const ChunkKey& key)
{
    std::ostringstream out;
    out << "array=" << key.arrayId
        << " channel=" << key.channelIndex
        << " zyx=" << key.z << "," << key.y << "," << key.x;
    return out.str();
}

[[nodiscard]] size_t originalChunkOffset(
    const ChannelBinding& binding,
    size_t localZ,
    size_t localY,
    size_t localX)
{
    const auto& chunks = binding.array->metadata().chunks;
    if (binding.hasChannelDimension) {
        const size_t chunkC = chunks[0];
        const size_t chunkZ = chunks[1];
        const size_t chunkY = chunks[2];
        const size_t chunkX = chunks[3];
        return (((binding.channelIndex % chunkC) * chunkZ + localZ) * chunkY + localY) * chunkX + localX;
    }
    return (localZ * binding.chunksZYX[1] + localY) * binding.chunksZYX[2] + localX;
}

[[nodiscard]] std::shared_ptr<const CachedChunk> buildHaloChunk(
    const ChannelBinding& binding,
    const utils::ZarrArray& array,
    const ChunkKey& key)
{
    const size_t chunkZIndex = key.z;
    const size_t chunkYIndex = key.y;
    const size_t chunkXIndex = key.x;
    const size_t originZ = chunkZIndex * binding.chunksZYX[0];
    const size_t originY = chunkYIndex * binding.chunksZYX[1];
    const size_t originX = chunkXIndex * binding.chunksZYX[2];
    if (originZ >= binding.shapeZYX[0] ||
        originY >= binding.shapeZYX[1] ||
        originX >= binding.shapeZYX[2]) {
        return nullptr;
    }

    auto cached = std::make_shared<CachedChunk>();
    cached->dimsZYX = {
        binding.chunksZYX[0] + 2,
        binding.chunksZYX[1] + 2,
        binding.chunksZYX[2] + 2,
    };
    cached->values.resize(cached->dimsZYX[0] * cached->dimsZYX[1] * cached->dimsZYX[2], 0);

    std::unordered_map<ChunkKey, std::optional<std::vector<std::byte>>, ChunkKeyHash> sourceChunks;
    sourceChunks.reserve(27);
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int64_t nz = static_cast<int64_t>(chunkZIndex) + dz;
                const int64_t ny = static_cast<int64_t>(chunkYIndex) + dy;
                const int64_t nx = static_cast<int64_t>(chunkXIndex) + dx;
                if (nz < 0 || ny < 0 || nx < 0) {
                    continue;
                }
                const size_t neighborOriginZ = static_cast<size_t>(nz) * binding.chunksZYX[0];
                const size_t neighborOriginY = static_cast<size_t>(ny) * binding.chunksZYX[1];
                const size_t neighborOriginX = static_cast<size_t>(nx) * binding.chunksZYX[2];
                if (neighborOriginZ >= binding.shapeZYX[0] ||
                    neighborOriginY >= binding.shapeZYX[1] ||
                    neighborOriginX >= binding.shapeZYX[2]) {
                    continue;
                }
                std::vector<size_t> indices;
                if (binding.hasChannelDimension) {
                    const auto& chunks = binding.array->metadata().chunks;
                    indices = {
                        binding.channelIndex / chunks[0],
                        static_cast<size_t>(nz),
                        static_cast<size_t>(ny),
                        static_cast<size_t>(nx),
                    };
                } else {
                    indices = {
                        static_cast<size_t>(nz),
                        static_cast<size_t>(ny),
                        static_cast<size_t>(nx),
                    };
                }
                ChunkKey neighborKey{
                    binding.arrayId,
                    static_cast<uint32_t>(binding.channelIndex),
                    checkedChunkIndex(static_cast<size_t>(nz)),
                    checkedChunkIndex(static_cast<size_t>(ny)),
                    checkedChunkIndex(static_cast<size_t>(nx)),
                };
                sourceChunks.emplace(neighborKey, array.read_chunk(indices));
            }
        }
    }

    const auto readSourceVoxel = [&](size_t gz, size_t gy, size_t gx) -> uint8_t {
        gz = std::min(gz, binding.shapeZYX[0] - 1);
        gy = std::min(gy, binding.shapeZYX[1] - 1);
        gx = std::min(gx, binding.shapeZYX[2] - 1);
        const ChunkKey sourceKey = chunkKeyForVoxel(binding, gz, gy, gx);
        auto it = sourceChunks.find(sourceKey);
        if (it == sourceChunks.end() || !it->second.has_value()) {
            return 0;
        }
        const size_t localZ = gz % binding.chunksZYX[0];
        const size_t localY = gy % binding.chunksZYX[1];
        const size_t localX = gx % binding.chunksZYX[2];
        const size_t offset = originalChunkOffset(binding, localZ, localY, localX);
        if (offset >= it->second->size()) {
            throw std::runtime_error("Lasagna zarr chunk is smaller than expected at chunk " +
                                     chunkKeyToString(sourceKey));
        }
        return static_cast<uint8_t>((*it->second)[offset]);
    };

    for (size_t hz = 0; hz < cached->dimsZYX[0]; ++hz) {
        const int64_t gzSigned = static_cast<int64_t>(originZ) + static_cast<int64_t>(hz) - 1;
        const size_t gz = gzSigned < 0 ? 0 : static_cast<size_t>(gzSigned);
        for (size_t hy = 0; hy < cached->dimsZYX[1]; ++hy) {
            const int64_t gySigned = static_cast<int64_t>(originY) + static_cast<int64_t>(hy) - 1;
            const size_t gy = gySigned < 0 ? 0 : static_cast<size_t>(gySigned);
            for (size_t hx = 0; hx < cached->dimsZYX[2]; ++hx) {
                const int64_t gxSigned = static_cast<int64_t>(originX) + static_cast<int64_t>(hx) - 1;
                const size_t gx = gxSigned < 0 ? 0 : static_cast<size_t>(gxSigned);
                cached->values[(hz * cached->dimsZYX[1] + hy) * cached->dimsZYX[2] + hx] =
                    readSourceVoxel(gz, gy, gx);
            }
        }
    }

    return cached;
}

class ChunkCache {
public:
    using ResolvedChunkMap =
        std::unordered_map<ChunkKey, std::shared_ptr<const CachedChunk>, ChunkKeyHash>;

    explicit ChunkCache(size_t capacityBytes)
        : capacityBytes_(std::max<size_t>(1, capacityBytes))
    {
    }

    [[nodiscard]] std::shared_ptr<const CachedChunk> get(
        const ChannelBinding& binding,
        const utils::ZarrArray& array,
        const ChunkKey& key) const
    {
        {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            if (auto it = entries_.find(key); it != entries_.end()) {
                lru_.splice(lru_.begin(), lru_, it->second.lruIt);
                return it->second.bytes;
            }
        }

        store(key, buildHaloChunk(binding, array, key));
        return getCached(key);
    }

    [[nodiscard]] NormalPrefetchReport prefetchResolved(
        const ChannelBinding& binding,
        const utils::ZarrArray& array,
        const std::vector<ChunkKey>& keys,
        size_t maxWorkers,
        ResolvedChunkMap& resolved) const
    {
        NormalPrefetchReport report;
        resolved.clear();
        std::unordered_set<ChunkKey, ChunkKeyHash> uniqueKeys;
        uniqueKeys.reserve(keys.size());
        for (const auto& key : keys) {
            uniqueKeys.insert(key);
        }
        report.requestedChunks = uniqueKeys.size();

        std::vector<ChunkKey> missing;
        missing.reserve(uniqueKeys.size());
        {
            std::shared_lock<std::shared_mutex> lock(mutex_);
            for (const auto& key : uniqueKeys) {
                if (auto it = entries_.find(key); it != entries_.end()) {
                    resolved.emplace(key, it->second.bytes);
                } else {
                    missing.push_back(key);
                }
            }
        }
        report.chunksRead = missing.size();
        if (!missing.empty()) {
            maxWorkers = std::clamp<size_t>(maxWorkers, 1, missing.size());
            std::vector<std::future<void>> futures;
            futures.reserve(maxWorkers);
            std::atomic<size_t> next{0};
            for (size_t worker = 0; worker < maxWorkers; ++worker) {
                futures.push_back(std::async(std::launch::async, [this, &binding, &array, &missing, &next]() {
                    while (true) {
                        const size_t index = next.fetch_add(1);
                        if (index >= missing.size()) {
                            return;
                        }
                        const ChunkKey key = missing[index];
                        if (getCached(key)) {
                            continue;
                        }
                        store(key, buildHaloChunk(binding, array, key));
                    }
                }));
            }
            for (auto& future : futures) {
                future.get();
            }
            std::shared_lock<std::shared_mutex> lock(mutex_);
            for (const auto& key : missing) {
                if (auto it = entries_.find(key); it != entries_.end()) {
                    resolved.emplace(key, it->second.bytes);
                }
            }
        }
        return report;
    }

    [[nodiscard]] NormalPrefetchReport prefetch(
        const ChannelBinding& binding,
        const utils::ZarrArray& array,
        const std::vector<ChunkKey>& keys,
        size_t maxWorkers) const
    {
        NormalPrefetchReport report;
        std::vector<ChunkKey> missing;
        missing.reserve(keys.size());
        std::unordered_set<ChunkKey, ChunkKeyHash> seen;
        seen.reserve(keys.size());
        for (const auto& key : keys) {
            if (seen.insert(key).second) {
                ++report.requestedChunks;
            }
        }
        {
            std::shared_lock<std::shared_mutex> lock(mutex_);
            for (const auto& key : seen) {
                if (entries_.find(key) != entries_.end()) {
                    continue;
                }
                missing.push_back(key);
            }
        }
        report.chunksRead = missing.size();

        if (missing.empty()) {
            return report;
        }

        maxWorkers = std::clamp<size_t>(maxWorkers, 1, missing.size());
        std::vector<std::future<void>> futures;
        futures.reserve(maxWorkers);
        std::atomic<size_t> next{0};
        for (size_t worker = 0; worker < maxWorkers; ++worker) {
            futures.push_back(std::async(std::launch::async, [this, &binding, &array, &missing, &next]() {
                while (true) {
                    const size_t index = next.fetch_add(1);
                    if (index >= missing.size()) {
                        return;
                    }
                    const ChunkKey key = missing[index];
                    if (getCached(key)) {
                        continue;
                    }
                    store(key, buildHaloChunk(binding, array, key));
                }
            }));
        }
        for (auto& future : futures) {
            future.get();
        }
        return report;
    }

private:
    [[nodiscard]] std::shared_ptr<const CachedChunk> getCached(const ChunkKey& key) const
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        auto it = entries_.find(key);
        if (it == entries_.end()) {
            return nullptr;
        }
        lru_.splice(lru_.begin(), lru_, it->second.lruIt);
        return it->second.bytes;
    }

    void store(ChunkKey key, std::shared_ptr<const CachedChunk> bytes) const
    {
        const size_t byteSize = bytes ? bytes->values.size() : 0;
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (auto it = entries_.find(key); it != entries_.end()) {
            lru_.splice(lru_.begin(), lru_, it->second.lruIt);
            if (it->second.bytes) {
                cachedBytes_ -= it->second.bytes->values.size();
            }
            it->second.bytes = std::move(bytes);
            cachedBytes_ += byteSize;
            trim();
            return;
        }

        lru_.push_front(key);
        entries_.emplace(std::move(key), Entry{std::move(bytes), lru_.begin()});
        cachedBytes_ += byteSize;
        trim();
    }

    struct Entry {
        std::shared_ptr<const CachedChunk> bytes;
        std::list<ChunkKey>::iterator lruIt;
    };

    void trim() const
    {
        while (cachedBytes_ > capacityBytes_ && !lru_.empty()) {
            const ChunkKey evicted = lru_.back();
            lru_.pop_back();
            auto it = entries_.find(evicted);
            if (it != entries_.end()) {
                if (it->second.bytes) {
                    cachedBytes_ -= it->second.bytes->values.size();
                }
                entries_.erase(it);
            }
        }
    }

    size_t capacityBytes_ = 512ULL * 1024ULL * 1024ULL;
    mutable size_t cachedBytes_ = 0;
    mutable std::shared_mutex mutex_;
    mutable std::list<ChunkKey> lru_;
    mutable std::unordered_map<ChunkKey, Entry, ChunkKeyHash> entries_;
};

[[nodiscard]] std::shared_ptr<ChunkCache> sharedNormalChunkCache(size_t capacityBytes)
{
    static std::mutex mutex;
    static std::shared_ptr<ChunkCache> cache;
    std::lock_guard<std::mutex> lock(mutex);
    if (!cache) {
        cache = std::make_shared<ChunkCache>(capacityBytes);
    }
    return cache;
}

[[nodiscard]] uint32_t arrayIdForPath(const std::filesystem::path& path)
{
    static std::mutex mutex;
    static std::unordered_map<std::string, uint32_t> ids;
    static uint32_t nextId = 1;
    const std::string key = path.lexically_normal().string();
    std::lock_guard<std::mutex> lock(mutex);
    if (auto it = ids.find(key); it != ids.end()) {
        return it->second;
    }
    const uint32_t id = nextId++;
    ids.emplace(key, id);
    return id;
}

[[nodiscard]] ChannelBinding bindChannel(
    const LasagnaDatasetManifest& manifest,
    std::string_view channel)
{
    const LasagnaChannelGroup* group = manifest.groupForChannel(channel);
    if (group == nullptr) {
        throw std::runtime_error("Lasagna dataset missing required channel '" + std::string(channel) + "'");
    }

    const auto channelIndex = group->channelIndex(channel);
    if (!channelIndex.has_value()) {
        throw std::runtime_error("Internal Lasagna channel lookup failure");
    }

    ChannelBinding binding;
    binding.group = group;
    binding.channelIndex = *channelIndex;
    binding.path = group->zarrPath;
    binding.arrayId = arrayIdForPath(binding.path);
    binding.array = std::make_shared<utils::ZarrArray>(
        utils::ZarrArray::open(group->zarrPath, vc::buildZarrCodecRegistry(1)));
    binding.spacing = static_cast<double>(group->scaleFactor()) * manifest.sourceToBase;

    const auto& meta = binding.array->metadata();
    if (meta.dtype != utils::ZarrDtype::uint8) {
        throw std::runtime_error("Lasagna channel '" + std::string(channel) + "' must be uint8");
    }
    if (meta.shape.size() == 3) {
        if (meta.chunks.size() != 3) {
            throw std::runtime_error("Lasagna channel '" + std::string(channel) + "' zarr has invalid chunks");
        }
        binding.hasChannelDimension = false;
        binding.shapeZYX = {meta.shape[0], meta.shape[1], meta.shape[2]};
        binding.chunksZYX = {meta.chunks[0], meta.chunks[1], meta.chunks[2]};
    } else if (meta.shape.size() == 4) {
        if (meta.chunks.size() != 4) {
            throw std::runtime_error("Lasagna channel '" + std::string(channel) + "' zarr has invalid chunks");
        }
        if (*channelIndex >= meta.shape[0]) {
            throw std::runtime_error("Lasagna channel index is outside zarr channel dimension for '" +
                                     std::string(channel) + "'");
        }
        binding.hasChannelDimension = true;
        binding.shapeZYX = {meta.shape[1], meta.shape[2], meta.shape[3]};
        binding.chunksZYX = {meta.chunks[1], meta.chunks[2], meta.chunks[3]};
    } else {
        throw std::runtime_error("Lasagna channel '" + std::string(channel) +
                                 "' zarr must have shape (Z,Y,X) or (C,Z,Y,X)");
    }

    if (binding.spacing <= 0.0 || !std::isfinite(binding.spacing)) {
        throw std::runtime_error("Lasagna channel '" + std::string(channel) + "' has invalid spacing");
    }
    if (binding.shapeZYX[0] == 0 || binding.shapeZYX[1] == 0 || binding.shapeZYX[2] == 0 ||
        binding.chunksZYX[0] == 0 || binding.chunksZYX[1] == 0 || binding.chunksZYX[2] == 0) {
        throw std::runtime_error("Lasagna channel '" + std::string(channel) + "' has empty zarr shape/chunks");
    }
    return binding;
}

[[nodiscard]] std::optional<uint8_t> readVoxel(
    const ChannelBinding& binding,
    const ChunkCache& cache,
    size_t z,
    size_t y,
    size_t x)
{
    if (z >= binding.shapeZYX[0] || y >= binding.shapeZYX[1] || x >= binding.shapeZYX[2]) {
        return std::nullopt;
    }

    const ChunkKey key = chunkKeyForVoxel(binding, z, y, x);
    const auto chunk = cache.get(binding, *binding.array, key);
    if (chunk == nullptr) {
        return std::nullopt;
    }

    const size_t localZ = (z % binding.chunksZYX[0]) + 1;
    const size_t localY = (y % binding.chunksZYX[1]) + 1;
    const size_t localX = (x % binding.chunksZYX[2]) + 1;
    const size_t offset = (localZ * chunk->dimsZYX[1] + localY) * chunk->dimsZYX[2] + localX;
    if (offset >= chunk->values.size()) {
        throw std::runtime_error("Lasagna cached halo chunk is smaller than expected at chunk " +
                                 chunkKeyToString(key));
    }
    return chunk->values[offset];
}

struct CubeValues {
    double c000 = 0.0;
    double c001 = 0.0;
    double c010 = 0.0;
    double c011 = 0.0;
    double c100 = 0.0;
    double c101 = 0.0;
    double c110 = 0.0;
    double c111 = 0.0;
};

struct CubeRequest {
    bool valid = false;
    size_t z0 = 0;
    size_t y0 = 0;
    size_t x0 = 0;
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    ChunkKey key{};
    std::shared_ptr<const CachedChunk> chunk;
};

struct PreparedNormalPoint {
    CubeRequest gradMag;
    CubeRequest nx;
    CubeRequest ny;
};

[[nodiscard]] CubeRequest prepareCubeRequest(
    const ChannelBinding& binding,
    const cv::Vec3d& volumePoint)
{
    CubeRequest request;
    const double x = volumePoint[0] / binding.spacing;
    const double y = volumePoint[1] / binding.spacing;
    const double z = volumePoint[2] / binding.spacing;
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        return request;
    }
    if (x < 0.0 || y < 0.0 || z < 0.0 ||
        x > static_cast<double>(binding.shapeZYX[2] - 1) ||
        y > static_cast<double>(binding.shapeZYX[1] - 1) ||
        z > static_cast<double>(binding.shapeZYX[0] - 1)) {
        return request;
    }

    request.x0 = static_cast<size_t>(std::floor(x));
    request.y0 = static_cast<size_t>(std::floor(y));
    request.z0 = static_cast<size_t>(std::floor(z));
    request.fx = x - static_cast<double>(request.x0);
    request.fy = y - static_cast<double>(request.y0);
    request.fz = z - static_cast<double>(request.z0);
    request.key = chunkKeyForVoxel(binding, request.z0, request.y0, request.x0);
    request.valid = true;
    return request;
}

[[nodiscard]] std::optional<CubeValues> readInterpolationCube(
    const ChannelBinding& binding,
    const ChunkKey& key,
    const std::shared_ptr<const CachedChunk>& chunk,
    size_t z0,
    size_t y0,
    size_t x0)
{
    if (chunk == nullptr) {
        return std::nullopt;
    }

    const size_t localZ = (z0 % binding.chunksZYX[0]) + 1;
    const size_t localY = (y0 % binding.chunksZYX[1]) + 1;
    const size_t localX = (x0 % binding.chunksZYX[2]) + 1;
    const auto value = [&](size_t dz, size_t dy, size_t dx) -> double {
        const size_t offset = ((localZ + dz) * chunk->dimsZYX[1] + (localY + dy)) *
                                  chunk->dimsZYX[2] +
                              (localX + dx);
        if (offset >= chunk->values.size()) {
            throw std::runtime_error("Lasagna cached halo chunk is smaller than expected at chunk " +
                                     chunkKeyToString(key));
        }
        return static_cast<double>(chunk->values[offset]);
    };

    return CubeValues{
        value(0, 0, 0),
        value(0, 0, 1),
        value(0, 1, 0),
        value(0, 1, 1),
        value(1, 0, 0),
        value(1, 0, 1),
        value(1, 1, 0),
        value(1, 1, 1),
    };
}

[[nodiscard]] std::optional<CubeValues> readInterpolationCube(
    const ChannelBinding& binding,
    const ChunkCache& cache,
    size_t z0,
    size_t y0,
    size_t x0)
{
    const ChunkKey key = chunkKeyForVoxel(binding, z0, y0, x0);
    return readInterpolationCube(binding, key, cache.get(binding, *binding.array, key), z0, y0, x0);
}

void appendInterpolationChunkKeys(
    const ChannelBinding& binding,
    const cv::Vec3d& volumePoint,
    std::vector<ChunkKey>& keys)
{
    const double x = volumePoint[0] / binding.spacing;
    const double y = volumePoint[1] / binding.spacing;
    const double z = volumePoint[2] / binding.spacing;
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        return;
    }
    if (x < 0.0 || y < 0.0 || z < 0.0 ||
        x > static_cast<double>(binding.shapeZYX[2] - 1) ||
        y > static_cast<double>(binding.shapeZYX[1] - 1) ||
        z > static_cast<double>(binding.shapeZYX[0] - 1)) {
        return;
    }

    const size_t x0 = static_cast<size_t>(std::floor(x));
    const size_t y0 = static_cast<size_t>(std::floor(y));
    const size_t z0 = static_cast<size_t>(std::floor(z));
    keys.push_back(chunkKeyForVoxel(binding, z0, y0, x0));
}

[[nodiscard]] std::optional<double> sampleChannel(
    const ChannelBinding& binding,
    const ChunkCache& cache,
    const cv::Vec3d& volumePoint)
{
    const double x = volumePoint[0] / binding.spacing;
    const double y = volumePoint[1] / binding.spacing;
    const double z = volumePoint[2] / binding.spacing;
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        return std::nullopt;
    }
    if (x < 0.0 || y < 0.0 || z < 0.0 ||
        x > static_cast<double>(binding.shapeZYX[2] - 1) ||
        y > static_cast<double>(binding.shapeZYX[1] - 1) ||
        z > static_cast<double>(binding.shapeZYX[0] - 1)) {
        return std::nullopt;
    }

    const size_t x0 = static_cast<size_t>(std::floor(x));
    const size_t y0 = static_cast<size_t>(std::floor(y));
    const size_t z0 = static_cast<size_t>(std::floor(z));
    const double fx = x - static_cast<double>(x0);
    const double fy = y - static_cast<double>(y0);
    const double fz = z - static_cast<double>(z0);

    const auto cube = readInterpolationCube(binding, cache, z0, y0, x0);
    if (!cube.has_value()) {
        return std::nullopt;
    }

    const double c00 = cube->c000 * (1.0 - fx) + cube->c001 * fx;
    const double c01 = cube->c010 * (1.0 - fx) + cube->c011 * fx;
    const double c10 = cube->c100 * (1.0 - fx) + cube->c101 * fx;
    const double c11 = cube->c110 * (1.0 - fx) + cube->c111 * fx;
    const double c0 = c00 * (1.0 - fy) + c01 * fy;
    const double c1 = c10 * (1.0 - fy) + c11 * fy;
    return c0 * (1.0 - fz) + c1 * fz;
}

[[nodiscard]] std::optional<double> sampleChannel(
    const ChannelBinding& binding,
    const CubeRequest& request)
{
    if (!request.valid) {
        return std::nullopt;
    }
    const auto cube = readInterpolationCube(
        binding, request.key, request.chunk, request.z0, request.y0, request.x0);
    if (!cube.has_value()) {
        return std::nullopt;
    }

    const double c00 = cube->c000 * (1.0 - request.fx) + cube->c001 * request.fx;
    const double c01 = cube->c010 * (1.0 - request.fx) + cube->c011 * request.fx;
    const double c10 = cube->c100 * (1.0 - request.fx) + cube->c101 * request.fx;
    const double c11 = cube->c110 * (1.0 - request.fx) + cube->c111 * request.fx;
    const double c0 = c00 * (1.0 - request.fy) + c01 * request.fy;
    const double c1 = c10 * (1.0 - request.fy) + c11 * request.fy;
    return c0 * (1.0 - request.fz) + c1 * request.fz;
}

[[nodiscard]] cv::Vec3d tensorTimesVector(const cv::Matx33d& tensor, const cv::Vec3d& vector)
{
    return {
        tensor(0, 0) * vector[0] + tensor(0, 1) * vector[1] + tensor(0, 2) * vector[2],
        tensor(1, 0) * vector[0] + tensor(1, 1) * vector[1] + tensor(1, 2) * vector[2],
        tensor(2, 0) * vector[0] + tensor(2, 1) * vector[1] + tensor(2, 2) * vector[2],
    };
}

[[nodiscard]] cv::Vec3d fallbackTensorAxis(const cv::Matx33d& tensor)
{
    int axis = 0;
    double value = tensor(0, 0);
    if (tensor(1, 1) > value) {
        axis = 1;
        value = tensor(1, 1);
    }
    if (tensor(2, 2) > value) {
        axis = 2;
    }
    cv::Vec3d result{0.0, 0.0, 0.0};
    result[axis] = 1.0;
    return result;
}

[[nodiscard]] cv::Vec3d principalTensorAxis(const cv::Matx33d& tensor, const cv::Vec3d& hint)
{
    cv::Vec3d axis = normalizedOrZero(hint);
    if (length(axis) <= kEpsilon) {
        axis = fallbackTensorAxis(tensor);
    }
    for (int iteration = 0; iteration < 16; ++iteration) {
        const cv::Vec3d next = normalizedOrZero(tensorTimesVector(tensor, axis));
        if (length(next) <= kEpsilon) {
            break;
        }
        axis = next;
    }
    if (length(axis) <= kEpsilon) {
        return {0.0, 0.0, 0.0};
    }
    const cv::Vec3d normalizedHint = normalizedOrZero(hint);
    if (length(normalizedHint) > kEpsilon) {
        if (axis.dot(normalizedHint) < 0.0) {
            axis *= -1.0;
        }
    } else if (axis[2] < 0.0) {
        axis *= -1.0;
    }
    return axis;
}

[[nodiscard]] std::optional<cv::Vec3d> sampleNormalTensor(
    const ChannelBinding& nxBinding,
    const ChannelBinding& nyBinding,
    const ChunkCache& cache,
    const cv::Vec3d& volumePoint)
{
    const double x = volumePoint[0] / nxBinding.spacing;
    const double y = volumePoint[1] / nxBinding.spacing;
    const double z = volumePoint[2] / nxBinding.spacing;
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        return std::nullopt;
    }
    if (x < 0.0 || y < 0.0 || z < 0.0 ||
        x > static_cast<double>(nxBinding.shapeZYX[2] - 1) ||
        y > static_cast<double>(nxBinding.shapeZYX[1] - 1) ||
        z > static_cast<double>(nxBinding.shapeZYX[0] - 1)) {
        return std::nullopt;
    }

    const size_t x0 = static_cast<size_t>(std::floor(x));
    const size_t y0 = static_cast<size_t>(std::floor(y));
    const size_t z0 = static_cast<size_t>(std::floor(z));
    const double fx = x - static_cast<double>(x0);
    const double fy = y - static_cast<double>(y0);
    const double fz = z - static_cast<double>(z0);

    const auto nxCube = readInterpolationCube(nxBinding, cache, z0, y0, x0);
    const auto nyCube = readInterpolationCube(nyBinding, cache, z0, y0, x0);
    if (!nxCube.has_value() || !nyCube.has_value()) {
        return std::nullopt;
    }

    const auto cubeValue = [](const CubeValues& cube, int dz, int dy, int dx) -> double {
        if (dz == 0 && dy == 0 && dx == 0) {
            return cube.c000;
        }
        if (dz == 0 && dy == 0 && dx == 1) {
            return cube.c001;
        }
        if (dz == 0 && dy == 1 && dx == 0) {
            return cube.c010;
        }
        if (dz == 0 && dy == 1 && dx == 1) {
            return cube.c011;
        }
        if (dz == 1 && dy == 0 && dx == 0) {
            return cube.c100;
        }
        if (dz == 1 && dy == 0 && dx == 1) {
            return cube.c101;
        }
        if (dz == 1 && dy == 1 && dx == 0) {
            return cube.c110;
        }
        return cube.c111;
    };

    cv::Matx33d tensor = cv::Matx33d::zeros();
    cv::Vec3d hint{0.0, 0.0, 0.0};
    double totalWeight = 0.0;
    for (int dz = 0; dz <= 1; ++dz) {
        const double wz = dz == 0 ? (1.0 - fz) : fz;
        for (int dy = 0; dy <= 1; ++dy) {
            const double wy = dy == 0 ? (1.0 - fy) : fy;
            for (int dx = 0; dx <= 1; ++dx) {
                const double wx = dx == 0 ? (1.0 - fx) : fx;
                const double weight = wx * wy * wz;
                if (weight <= 0.0) {
                    continue;
                }
                const cv::Vec3d normal = decodedNormalFromRaw(
                    cubeValue(*nxCube, dz, dy, dx),
                    cubeValue(*nyCube, dz, dy, dx));
                if (length(normal) <= kEpsilon) {
                    continue;
                }
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        tensor(row, col) += weight * normal[row] * normal[col];
                    }
                }
                hint += normal * weight;
                totalWeight += weight;
            }
        }
    }
    if (totalWeight <= kEpsilon) {
        return std::nullopt;
    }
    const cv::Vec3d normal = principalTensorAxis(tensor, hint);
    if (length(normal) <= kEpsilon) {
        return std::nullopt;
    }
    return normal;
}

[[nodiscard]] std::optional<cv::Vec3d> sampleNormalTensor(
    const ChannelBinding& nxBinding,
    const ChannelBinding& nyBinding,
    const CubeRequest& nxRequest,
    const CubeRequest& nyRequest)
{
    if (!nxRequest.valid || !nyRequest.valid) {
        return std::nullopt;
    }
    const auto nxCube = readInterpolationCube(
        nxBinding, nxRequest.key, nxRequest.chunk, nxRequest.z0, nxRequest.y0, nxRequest.x0);
    const auto nyCube = readInterpolationCube(
        nyBinding, nyRequest.key, nyRequest.chunk, nyRequest.z0, nyRequest.y0, nyRequest.x0);
    if (!nxCube.has_value() || !nyCube.has_value()) {
        return std::nullopt;
    }

    const auto cubeValue = [](const CubeValues& cube, int dz, int dy, int dx) -> double {
        if (dz == 0 && dy == 0 && dx == 0) {
            return cube.c000;
        }
        if (dz == 0 && dy == 0 && dx == 1) {
            return cube.c001;
        }
        if (dz == 0 && dy == 1 && dx == 0) {
            return cube.c010;
        }
        if (dz == 0 && dy == 1 && dx == 1) {
            return cube.c011;
        }
        if (dz == 1 && dy == 0 && dx == 0) {
            return cube.c100;
        }
        if (dz == 1 && dy == 0 && dx == 1) {
            return cube.c101;
        }
        if (dz == 1 && dy == 1 && dx == 0) {
            return cube.c110;
        }
        return cube.c111;
    };

    cv::Matx33d tensor = cv::Matx33d::zeros();
    cv::Vec3d hint{0.0, 0.0, 0.0};
    double totalWeight = 0.0;
    for (int dz = 0; dz <= 1; ++dz) {
        const double wz = dz == 0 ? (1.0 - nxRequest.fz) : nxRequest.fz;
        for (int dy = 0; dy <= 1; ++dy) {
            const double wy = dy == 0 ? (1.0 - nxRequest.fy) : nxRequest.fy;
            for (int dx = 0; dx <= 1; ++dx) {
                const double wx = dx == 0 ? (1.0 - nxRequest.fx) : nxRequest.fx;
                const double weight = wx * wy * wz;
                if (weight <= 0.0) {
                    continue;
                }
                const cv::Vec3d normal = decodedNormalFromRaw(
                    cubeValue(*nxCube, dz, dy, dx),
                    cubeValue(*nyCube, dz, dy, dx));
                if (length(normal) <= kEpsilon) {
                    continue;
                }
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        tensor(row, col) += weight * normal[row] * normal[col];
                    }
                }
                hint += normal * weight;
                totalWeight += weight;
            }
        }
    }
    if (totalWeight <= kEpsilon) {
        return std::nullopt;
    }
    const cv::Vec3d normal = principalTensorAxis(tensor, hint);
    if (length(normal) <= kEpsilon) {
        return std::nullopt;
    }
    return normal;
}

} // namespace

class LasagnaNormalSampler::Impl {
public:
    Impl(const LasagnaDataset& dataset, LasagnaNormalSamplerOptions options)
        : nx_(bindChannel(dataset.manifest(), "nx"))
        , ny_(bindChannel(dataset.manifest(), "ny"))
        , gradMag_(bindChannel(dataset.manifest(), "grad_mag"))
        , options_(options)
        , cache_(sharedNormalChunkCache(options.maxCachedBytes))
    {
        if (nx_.shapeZYX != ny_.shapeZYX) {
            throw std::runtime_error("Lasagna nx and ny channels must have matching spatial shapes");
        }
    }

    [[nodiscard]] NormalSample sampleNormal(const cv::Vec3d& volumePoint) const
    {
        const auto gradMag = sampleChannel(gradMag_, *cache_, volumePoint);
        if (!gradMag.has_value()) {
            return {{0.0, 0.0, 0.0}, false, "missing Lasagna grad_mag sample"};
        }
        if (*gradMag <= 0.0) {
            return {{0.0, 0.0, 0.0}, false, "Lasagna grad_mag sample is zero"};
        }

        const auto normal = sampleNormalTensor(nx_, ny_, *cache_, volumePoint);
        if (!normal.has_value()) {
            return {{0.0, 0.0, 0.0}, false, "missing Lasagna nx/ny sample"};
        }
        if (length(*normal) <= kEpsilon) {
            return {{0.0, 0.0, 0.0}, false, "degenerate Lasagna normal sample"};
        }
        return {*normal, true, {}};
    }

    [[nodiscard]] NormalSampleWithDerivative sampleNormalWithDerivative(const cv::Vec3d& volumePoint) const
    {
        const auto gradMag = sampleChannel(gradMag_, *cache_, volumePoint);
        if (!gradMag.has_value()) {
            return {{{0.0, 0.0, 0.0}, false, "missing Lasagna grad_mag sample"},
                    cv::Matx33d::zeros(),
                    false};
        }
        if (*gradMag <= 0.0) {
            return {{{0.0, 0.0, 0.0}, false, "Lasagna grad_mag sample is zero"},
                    cv::Matx33d::zeros(),
                    false};
        }

        const auto normal = sampleNormalTensor(nx_, ny_, *cache_, volumePoint);
        if (!normal.has_value()) {
            return {{{0.0, 0.0, 0.0}, false, "missing Lasagna nx/ny sample"},
                    cv::Matx33d::zeros(),
                    false};
        }
        if (length(*normal) <= kEpsilon) {
            return {{{0.0, 0.0, 0.0}, false, "degenerate Lasagna normal sample"},
                    cv::Matx33d::zeros(),
                    false};
        }
        return {{*normal, true, {}}, cv::Matx33d::zeros(), false};
    }

    [[nodiscard]] NormalPrefetchReport prefetchNormalSamples(
        const std::vector<cv::Vec3d>& volumePoints,
        bool withDerivative) const
    {
        if (volumePoints.empty()) {
            return {};
        }

        std::vector<ChunkKey> gradMagKeys;
        std::vector<ChunkKey> nxKeys;
        std::vector<ChunkKey> nyKeys;
        gradMagKeys.reserve(volumePoints.size() * 8);
        nxKeys.reserve(volumePoints.size() * 8);
        nyKeys.reserve(volumePoints.size() * 8);

        for (const auto& point : volumePoints) {
            appendInterpolationChunkKeys(gradMag_, point, gradMagKeys);
            appendInterpolationChunkKeys(nx_, point, nxKeys);
            appendInterpolationChunkKeys(ny_, point, nyKeys);
        }

        const unsigned hardwareThreads = std::thread::hardware_concurrency();
        const size_t workers = std::clamp<size_t>(
            hardwareThreads == 0 ? 4 : static_cast<size_t>(hardwareThreads),
            1,
            8);

        auto gradMagPrefetch = std::async(std::launch::async, [&]() {
            return cache_->prefetch(gradMag_, *gradMag_.array, gradMagKeys, workers);
        });
        auto nxPrefetch = std::async(std::launch::async, [&]() {
            return cache_->prefetch(nx_, *nx_.array, nxKeys, workers);
        });
        auto nyPrefetch = std::async(std::launch::async, [&]() {
            return cache_->prefetch(ny_, *ny_.array, nyKeys, workers);
        });
        NormalPrefetchReport report;
        const NormalPrefetchReport gradMagReport = gradMagPrefetch.get();
        const NormalPrefetchReport nxReport = nxPrefetch.get();
        const NormalPrefetchReport nyReport = nyPrefetch.get();
        report.requestedChunks = gradMagReport.requestedChunks +
                                 nxReport.requestedChunks +
                                 nyReport.requestedChunks;
        report.chunksRead = gradMagReport.chunksRead +
                            nxReport.chunksRead +
                            nyReport.chunksRead;
        (void)withDerivative;
        return report;
    }

    [[nodiscard]] NormalBatchReport sampleNormalBatch(
        const std::vector<cv::Vec3d>& volumePoints,
        bool withDerivative,
        std::vector<NormalSampleWithDerivative>& samples) const
    {
        using Clock = std::chrono::steady_clock;
        NormalBatchReport report;
        samples.clear();
        samples.resize(volumePoints.size());
        if (volumePoints.empty()) {
            return report;
        }

        const auto prefetchStart = Clock::now();
        std::vector<PreparedNormalPoint> prepared(volumePoints.size());
        std::vector<ChunkKey> gradMagKeys;
        std::vector<ChunkKey> nxKeys;
        std::vector<ChunkKey> nyKeys;
        gradMagKeys.reserve(volumePoints.size());
        nxKeys.reserve(volumePoints.size());
        nyKeys.reserve(volumePoints.size());
        for (size_t index = 0; index < volumePoints.size(); ++index) {
            auto& point = prepared[index];
            point.gradMag = prepareCubeRequest(gradMag_, volumePoints[index]);
            point.nx = prepareCubeRequest(nx_, volumePoints[index]);
            point.ny = prepareCubeRequest(ny_, volumePoints[index]);
            if (point.gradMag.valid) {
                gradMagKeys.push_back(point.gradMag.key);
            }
            if (point.nx.valid) {
                nxKeys.push_back(point.nx.key);
            }
            if (point.ny.valid) {
                nyKeys.push_back(point.ny.key);
            }
        }

        const unsigned hardwareThreads = std::thread::hardware_concurrency();
        const size_t workers = std::clamp<size_t>(
            hardwareThreads == 0 ? 4 : static_cast<size_t>(hardwareThreads),
            1,
            8);

        ChunkCache::ResolvedChunkMap gradMagChunks;
        ChunkCache::ResolvedChunkMap nxChunks;
        ChunkCache::ResolvedChunkMap nyChunks;
        auto gradMagPrefetch = std::async(std::launch::async, [&]() {
            return cache_->prefetchResolved(
                gradMag_, *gradMag_.array, gradMagKeys, workers, gradMagChunks);
        });
        auto nxPrefetch = std::async(std::launch::async, [&]() {
            return cache_->prefetchResolved(nx_, *nx_.array, nxKeys, workers, nxChunks);
        });
        auto nyPrefetch = std::async(std::launch::async, [&]() {
            return cache_->prefetchResolved(ny_, *ny_.array, nyKeys, workers, nyChunks);
        });
        const NormalPrefetchReport gradMagReport = gradMagPrefetch.get();
        const NormalPrefetchReport nxReport = nxPrefetch.get();
        const NormalPrefetchReport nyReport = nyPrefetch.get();
        report.prefetch.requestedChunks = gradMagReport.requestedChunks +
                                          nxReport.requestedChunks +
                                          nyReport.requestedChunks;
        report.prefetch.chunksRead = gradMagReport.chunksRead +
                                     nxReport.chunksRead +
                                     nyReport.chunksRead;

        const auto assignChunks = [](std::vector<PreparedNormalPoint>& points,
                                     const ChunkCache::ResolvedChunkMap& chunks,
                                     CubeRequest PreparedNormalPoint::*member) {
            ChunkKey lastKey{};
            std::shared_ptr<const CachedChunk> lastChunk;
            bool hasLast = false;
            for (auto& point : points) {
                CubeRequest& request = point.*member;
                if (!request.valid) {
                    continue;
                }
                if (hasLast && request.key == lastKey) {
                    request.chunk = lastChunk;
                    continue;
                }
                auto it = chunks.find(request.key);
                if (it != chunks.end()) {
                    request.chunk = it->second;
                }
                lastKey = request.key;
                lastChunk = request.chunk;
                hasLast = true;
            }
        };
        assignChunks(prepared, gradMagChunks, &PreparedNormalPoint::gradMag);
        assignChunks(prepared, nxChunks, &PreparedNormalPoint::nx);
        assignChunks(prepared, nyChunks, &PreparedNormalPoint::ny);
        const auto prefetchEnd = Clock::now();

        const size_t materializeWorkers = std::min(volumePoints.size(), workers);
        const auto materializeOne = [&](size_t index) {
            const PreparedNormalPoint& point = prepared[index];
            const auto gradMag = sampleChannel(gradMag_, point.gradMag);
            if (!gradMag.has_value()) {
                samples[index] = {{{0.0, 0.0, 0.0}, false, "missing Lasagna grad_mag sample"},
                                  cv::Matx33d::zeros(),
                                  false};
                return;
            }
            if (*gradMag <= 0.0) {
                samples[index] = {{{0.0, 0.0, 0.0}, false, "Lasagna grad_mag sample is zero"},
                                  cv::Matx33d::zeros(),
                                  false};
                return;
            }

            const auto normal = sampleNormalTensor(nx_, ny_, point.nx, point.ny);
            if (!normal.has_value()) {
                samples[index] = {{{0.0, 0.0, 0.0}, false, "missing Lasagna nx/ny sample"},
                                  cv::Matx33d::zeros(),
                                  false};
                return;
            }
            if (length(*normal) <= kEpsilon) {
                samples[index] = {{{0.0, 0.0, 0.0}, false, "degenerate Lasagna normal sample"},
                                  cv::Matx33d::zeros(),
                                  false};
                return;
            }
            samples[index] = {{*normal, true, {}}, cv::Matx33d::zeros(), false};
        };

        if (materializeWorkers <= 1) {
            for (size_t index = 0; index < volumePoints.size(); ++index) {
                materializeOne(index);
            }
        } else {
            std::vector<std::future<void>> futures;
            futures.reserve(materializeWorkers);
            std::atomic<size_t> next{0};
            for (size_t worker = 0; worker < materializeWorkers; ++worker) {
                futures.push_back(std::async(std::launch::async, [&]() {
                    while (true) {
                        const size_t index = next.fetch_add(1);
                        if (index >= volumePoints.size()) {
                            return;
                        }
                        materializeOne(index);
                    }
                }));
            }
            for (auto& future : futures) {
                future.get();
            }
        }
        const auto materializeEnd = Clock::now();
        report.prefetchMs = std::chrono::duration<double, std::milli>(
            prefetchEnd - prefetchStart).count();
        report.materializeMs = std::chrono::duration<double, std::milli>(
            materializeEnd - prefetchEnd).count();
        (void)withDerivative;
        return report;
    }

private:
    ChannelBinding nx_;
    ChannelBinding ny_;
    ChannelBinding gradMag_;
    LasagnaNormalSamplerOptions options_;
    std::shared_ptr<ChunkCache> cache_;
};

LasagnaNormalSampler::LasagnaNormalSampler(
    const LasagnaDataset& dataset,
    LasagnaNormalSamplerOptions options)
    : impl_(std::make_unique<Impl>(dataset, options))
{
}

LasagnaNormalSampler::~LasagnaNormalSampler() = default;

LasagnaNormalSampler::LasagnaNormalSampler(LasagnaNormalSampler&&) noexcept = default;

LasagnaNormalSampler& LasagnaNormalSampler::operator=(LasagnaNormalSampler&&) noexcept = default;

NormalSample LasagnaNormalSampler::sampleNormal(const cv::Vec3d& volumePoint) const
{
    return impl_->sampleNormal(volumePoint);
}

NormalSampleWithDerivative LasagnaNormalSampler::sampleNormalWithDerivative(
    const cv::Vec3d& volumePoint) const
{
    return impl_->sampleNormalWithDerivative(volumePoint);
}

NormalPrefetchReport LasagnaNormalSampler::prefetchNormalSamples(
    const std::vector<cv::Vec3d>& volumePoints,
    bool withDerivative) const
{
    return impl_->prefetchNormalSamples(volumePoints, withDerivative);
}

NormalBatchReport LasagnaNormalSampler::sampleNormalBatch(
    const std::vector<cv::Vec3d>& volumePoints,
    bool withDerivative,
    std::vector<NormalSampleWithDerivative>& samples) const
{
    return impl_->sampleNormalBatch(volumePoints, withDerivative, samples);
}

} // namespace vc::lasagna
