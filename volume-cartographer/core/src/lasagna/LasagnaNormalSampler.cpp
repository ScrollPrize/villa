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
#include <list>
#include <mutex>
#include <optional>
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

[[nodiscard]] std::string indicesToString(const std::vector<size_t>& indices)
{
    std::ostringstream out;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << indices[i];
    }
    return out.str();
}

struct ChunkKey {
    std::filesystem::path path;
    std::vector<size_t> indices;

    [[nodiscard]] bool operator==(const ChunkKey& other) const noexcept
    {
        return path == other.path && indices == other.indices;
    }
};

struct ChunkKeyHash {
    [[nodiscard]] size_t operator()(const ChunkKey& key) const noexcept
    {
        size_t hash = std::filesystem::hash_value(key.path);
        for (const size_t index : key.indices) {
            hash ^= index + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        }
        return hash;
    }
};

struct CachedChunk {
    std::vector<uint8_t> values;
    std::array<size_t, 3> dimsZYX{0, 0, 0};
};

struct ChannelBinding {
    const LasagnaChannelGroup* group = nullptr;
    size_t channelIndex = 0;
    std::filesystem::path path;
    std::shared_ptr<utils::ZarrArray> array;
    bool hasChannelDimension = false;
    std::array<size_t, 3> shapeZYX{0, 0, 0};
    std::array<size_t, 3> chunksZYX{0, 0, 0};
    double spacing = 1.0;
};

[[nodiscard]] std::vector<size_t> chunkIndicesForVoxel(
    const ChannelBinding& binding,
    size_t z,
    size_t y,
    size_t x)
{
    if (binding.hasChannelDimension) {
        const auto& chunks = binding.array->metadata().chunks;
        return {
            binding.channelIndex / chunks[0],
            z / binding.chunksZYX[0],
            y / binding.chunksZYX[1],
            x / binding.chunksZYX[2],
        };
    }
    return {
        z / binding.chunksZYX[0],
        y / binding.chunksZYX[1],
        x / binding.chunksZYX[2],
    };
}

[[nodiscard]] ChunkKey chunkKeyForVoxel(
    const ChannelBinding& binding,
    size_t z,
    size_t y,
    size_t x)
{
    return {binding.path, chunkIndicesForVoxel(binding, z, y, x)};
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
    if (key.indices.size() < 3) {
        return nullptr;
    }

    const size_t spatialOffset = binding.hasChannelDimension ? 1 : 0;
    const size_t chunkZIndex = key.indices[spatialOffset + 0];
    const size_t chunkYIndex = key.indices[spatialOffset + 1];
    const size_t chunkXIndex = key.indices[spatialOffset + 2];
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
                    indices = {
                        key.indices[0],
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
                ChunkKey neighborKey{binding.path, std::move(indices)};
                sourceChunks.emplace(neighborKey, array.read_chunk(neighborKey.indices));
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
                                     indicesToString(sourceKey.indices));
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
            std::lock_guard<std::mutex> lock(mutex_);
            if (auto it = entries_.find(key); it != entries_.end()) {
                lru_.splice(lru_.begin(), lru_, it->second.lruIt);
                return it->second.bytes;
            }
        }

        store(key, buildHaloChunk(binding, array, key));
        return getCached(key);
    }

    void prefetch(
        const ChannelBinding& binding,
        const utils::ZarrArray& array,
        const std::vector<ChunkKey>& keys,
        size_t maxWorkers) const
    {
        std::vector<ChunkKey> missing;
        missing.reserve(keys.size());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::unordered_set<ChunkKey, ChunkKeyHash> seen;
            seen.reserve(keys.size());
            for (const auto& key : keys) {
                if (!seen.insert(key).second || entries_.find(key) != entries_.end()) {
                    continue;
                }
                missing.push_back(key);
            }
        }

        if (missing.empty()) {
            return;
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
    }

private:
    [[nodiscard]] std::shared_ptr<const CachedChunk> getCached(const ChunkKey& key) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
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
        std::lock_guard<std::mutex> lock(mutex_);
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
    mutable std::mutex mutex_;
    mutable std::list<ChunkKey> lru_;
    mutable std::unordered_map<ChunkKey, Entry, ChunkKeyHash> entries_;
};

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
                                 indicesToString(key.indices));
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

[[nodiscard]] std::optional<CubeValues> readInterpolationCube(
    const ChannelBinding& binding,
    const ChunkCache& cache,
    size_t z0,
    size_t y0,
    size_t x0)
{
    const ChunkKey key = chunkKeyForVoxel(binding, z0, y0, x0);
    struct LastChunk {
        ChunkKey key;
        std::shared_ptr<const CachedChunk> chunk;
        bool valid = false;
    };
    thread_local std::array<LastChunk, 8> lastChunks;
    const size_t slot =
        (std::filesystem::hash_value(binding.path) + binding.channelIndex) % lastChunks.size();
    auto& last = lastChunks[slot];
    std::shared_ptr<const CachedChunk> chunk;
    if (last.valid && last.key == key) {
        chunk = last.chunk;
    } else {
        chunk = cache.get(binding, *binding.array, key);
        last.key = key;
        last.chunk = chunk;
        last.valid = true;
    }
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
                                     indicesToString(key.indices));
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

struct ChannelSampleWithGradient {
    double value = 0.0;
    cv::Vec3d dValueDVolume{0.0, 0.0, 0.0};
};

[[nodiscard]] std::optional<ChannelSampleWithGradient> sampleChannelWithGradient(
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

    const double dcDGridX =
        ((cube->c001 - cube->c000) * (1.0 - fy) + (cube->c011 - cube->c010) * fy) * (1.0 - fz) +
        ((cube->c101 - cube->c100) * (1.0 - fy) + (cube->c111 - cube->c110) * fy) * fz;
    const double dcDGridY =
        ((c01 - c00) * (1.0 - fz)) +
        ((c11 - c10) * fz);
    const double dcDGridZ = c1 - c0;

    return ChannelSampleWithGradient{
        c0 * (1.0 - fz) + c1 * fz,
        {dcDGridX / binding.spacing, dcDGridY / binding.spacing, dcDGridZ / binding.spacing},
    };
}

} // namespace

class LasagnaNormalSampler::Impl {
public:
    Impl(const LasagnaDataset& dataset, LasagnaNormalSamplerOptions options)
        : nx_(bindChannel(dataset.manifest(), "nx"))
        , ny_(bindChannel(dataset.manifest(), "ny"))
        , gradMag_(bindChannel(dataset.manifest(), "grad_mag"))
        , options_(options)
        , cache_(options.maxCachedBytes)
    {
        if (nx_.shapeZYX != ny_.shapeZYX) {
            throw std::runtime_error("Lasagna nx and ny channels must have matching spatial shapes");
        }
    }

    [[nodiscard]] NormalSample sampleNormal(const cv::Vec3d& volumePoint) const
    {
        const auto gradMag = sampleChannel(gradMag_, cache_, volumePoint);
        if (!gradMag.has_value()) {
            return {{0.0, 0.0, 0.0}, false, "missing Lasagna grad_mag sample"};
        }
        if (*gradMag <= 0.0) {
            return {{0.0, 0.0, 0.0}, false, "Lasagna grad_mag sample is zero"};
        }

        const auto rawNx = sampleChannel(nx_, cache_, volumePoint);
        const auto rawNy = sampleChannel(ny_, cache_, volumePoint);
        if (!rawNx.has_value() || !rawNy.has_value()) {
            return {{0.0, 0.0, 0.0}, false, "missing Lasagna nx/ny sample"};
        }

        const double nx = decodeNormalComponent(*rawNx);
        const double ny = decodeNormalComponent(*rawNy);
        const double nzSq = std::max(0.0, 1.0 - nx * nx - ny * ny);
        const cv::Vec3d normal = normalizedOrZero({nx, ny, std::sqrt(nzSq)});
        if (length(normal) <= kEpsilon) {
            return {{0.0, 0.0, 0.0}, false, "degenerate Lasagna normal sample"};
        }
        return {normal, true, {}};
    }

    [[nodiscard]] NormalSampleWithDerivative sampleNormalWithDerivative(const cv::Vec3d& volumePoint) const
    {
        const auto gradMag = sampleChannel(gradMag_, cache_, volumePoint);
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

        const auto rawNx = sampleChannelWithGradient(nx_, cache_, volumePoint);
        const auto rawNy = sampleChannelWithGradient(ny_, cache_, volumePoint);
        if (!rawNx.has_value() || !rawNy.has_value()) {
            return {{{0.0, 0.0, 0.0}, false, "missing Lasagna nx/ny sample"},
                    cv::Matx33d::zeros(),
                    false};
        }

        const double nx = decodeNormalComponent(rawNx->value);
        const double ny = decodeNormalComponent(rawNy->value);
        const cv::Vec3d dNx = rawNx->dValueDVolume * (1.0 / 127.0);
        const cv::Vec3d dNy = rawNy->dValueDVolume * (1.0 / 127.0);

        const double nzSq = std::max(0.0, 1.0 - nx * nx - ny * ny);
        const double nz = std::sqrt(nzSq);
        cv::Vec3d dNz{0.0, 0.0, 0.0};
        if (nz > kEpsilon) {
            dNz = -(dNx * nx + dNy * ny) * (1.0 / nz);
        }

        const cv::Vec3d rawNormal{nx, ny, nz};
        const double normalLength = length(rawNormal);
        if (normalLength <= kEpsilon) {
            return {{{0.0, 0.0, 0.0}, false, "degenerate Lasagna normal sample"},
                    cv::Matx33d::zeros(),
                    false};
        }

        const cv::Vec3d normal = rawNormal * (1.0 / normalLength);
        cv::Matx33d dRawDVolume(
            dNx[0], dNx[1], dNx[2],
            dNy[0], dNy[1], dNy[2],
            dNz[0], dNz[1], dNz[2]);
        cv::Matx33d projection = cv::Matx33d::eye() - normal * normal.t();
        cv::Matx33d dNormalDVolume = (1.0 / normalLength) * projection * dRawDVolume;
        return {{normal, true, {}}, dNormalDVolume, true};
    }

    void prefetchNormalSamples(const std::vector<cv::Vec3d>& volumePoints, bool withDerivative) const
    {
        if (volumePoints.empty()) {
            return;
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
            cache_.prefetch(gradMag_, *gradMag_.array, gradMagKeys, workers);
        });
        auto nxPrefetch = std::async(std::launch::async, [&]() {
            cache_.prefetch(nx_, *nx_.array, nxKeys, workers);
        });
        auto nyPrefetch = std::async(std::launch::async, [&]() {
            cache_.prefetch(ny_, *ny_.array, nyKeys, workers);
        });
        gradMagPrefetch.get();
        nxPrefetch.get();
        nyPrefetch.get();
        (void)withDerivative;
    }

private:
    ChannelBinding nx_;
    ChannelBinding ny_;
    ChannelBinding gradMag_;
    LasagnaNormalSamplerOptions options_;
    ChunkCache cache_;
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

void LasagnaNormalSampler::prefetchNormalSamples(
    const std::vector<cv::Vec3d>& volumePoints,
    bool withDerivative) const
{
    impl_->prefetchNormalSamples(volumePoints, withDerivative);
}

} // namespace vc::lasagna
