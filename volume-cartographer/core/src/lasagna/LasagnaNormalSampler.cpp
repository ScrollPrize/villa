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

class ChunkCache {
public:
    explicit ChunkCache(size_t capacityBytes)
        : capacityBytes_(std::max<size_t>(1, capacityBytes))
    {
    }

    [[nodiscard]] std::shared_ptr<const std::vector<std::byte>> get(
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

        std::optional<std::vector<std::byte>> bytes = array.read_chunk(key.indices);
        auto sharedBytes = bytes
            ? std::make_shared<const std::vector<std::byte>>(std::move(*bytes))
            : std::shared_ptr<const std::vector<std::byte>>{};
        store(key, std::move(sharedBytes));
        return getCached(key);
    }

    void prefetch(
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
            futures.push_back(std::async(std::launch::async, [this, &array, &missing, &next]() {
                while (true) {
                    const size_t index = next.fetch_add(1);
                    if (index >= missing.size()) {
                        return;
                    }
                    const ChunkKey key = missing[index];
                    if (getCached(key)) {
                        continue;
                    }
                    std::optional<std::vector<std::byte>> bytes = array.read_chunk(key.indices);
                    auto sharedBytes = bytes
                        ? std::make_shared<const std::vector<std::byte>>(std::move(*bytes))
                        : std::shared_ptr<const std::vector<std::byte>>{};
                    store(key, std::move(sharedBytes));
                }
            }));
        }
        for (auto& future : futures) {
            future.get();
        }
    }

private:
    [[nodiscard]] std::shared_ptr<const std::vector<std::byte>> getCached(const ChunkKey& key) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(key);
        if (it == entries_.end()) {
            return nullptr;
        }
        lru_.splice(lru_.begin(), lru_, it->second.lruIt);
        return it->second.bytes;
    }

    void store(ChunkKey key, std::shared_ptr<const std::vector<std::byte>> bytes) const
    {
        const size_t byteSize = bytes ? bytes->size() : 0;
        std::lock_guard<std::mutex> lock(mutex_);
        if (auto it = entries_.find(key); it != entries_.end()) {
            lru_.splice(lru_.begin(), lru_, it->second.lruIt);
            if (it->second.bytes) {
                cachedBytes_ -= it->second.bytes->size();
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
        std::shared_ptr<const std::vector<std::byte>> bytes;
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
                    cachedBytes_ -= it->second.bytes->size();
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

[[nodiscard]] size_t localChunkOffset(
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

    std::vector<size_t> chunkIndices;
    size_t localZ = z % binding.chunksZYX[0];
    size_t localY = y % binding.chunksZYX[1];
    size_t localX = x % binding.chunksZYX[2];
    if (binding.hasChannelDimension) {
        const auto& chunks = binding.array->metadata().chunks;
        chunkIndices = {
            binding.channelIndex / chunks[0],
            z / binding.chunksZYX[0],
            y / binding.chunksZYX[1],
            x / binding.chunksZYX[2],
        };
    } else {
        chunkIndices = {
            z / binding.chunksZYX[0],
            y / binding.chunksZYX[1],
            x / binding.chunksZYX[2],
        };
    }

    const ChunkKey key{binding.path, std::move(chunkIndices)};
    const std::shared_ptr<const std::vector<std::byte>> bytes = cache.get(*binding.array, key);
    if (bytes == nullptr) {
        return std::nullopt;
    }

    const size_t offset = localChunkOffset(binding, localZ, localY, localX);
    if (offset >= bytes->size()) {
        throw std::runtime_error("Lasagna zarr chunk is smaller than expected at chunk " +
                                 indicesToString(key.indices));
    }
    return static_cast<uint8_t>((*bytes)[offset]);
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
    const size_t x1 = std::min(x0 + 1, binding.shapeZYX[2] - 1);
    const size_t y1 = std::min(y0 + 1, binding.shapeZYX[1] - 1);
    const size_t z1 = std::min(z0 + 1, binding.shapeZYX[0] - 1);

    auto append = [&](size_t zz, size_t yy, size_t xx) {
        std::vector<size_t> chunkIndices;
        if (binding.hasChannelDimension) {
            const auto& chunks = binding.array->metadata().chunks;
            chunkIndices = {
                binding.channelIndex / chunks[0],
                zz / binding.chunksZYX[0],
                yy / binding.chunksZYX[1],
                xx / binding.chunksZYX[2],
            };
        } else {
            chunkIndices = {
                zz / binding.chunksZYX[0],
                yy / binding.chunksZYX[1],
                xx / binding.chunksZYX[2],
            };
        }
        keys.push_back(ChunkKey{binding.path, std::move(chunkIndices)});
    };

    append(z0, y0, x0);
    append(z0, y0, x1);
    append(z0, y1, x0);
    append(z0, y1, x1);
    append(z1, y0, x0);
    append(z1, y0, x1);
    append(z1, y1, x0);
    append(z1, y1, x1);
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
    const size_t x1 = std::min(x0 + 1, binding.shapeZYX[2] - 1);
    const size_t y1 = std::min(y0 + 1, binding.shapeZYX[1] - 1);
    const size_t z1 = std::min(z0 + 1, binding.shapeZYX[0] - 1);
    const double fx = x - static_cast<double>(x0);
    const double fy = y - static_cast<double>(y0);
    const double fz = z - static_cast<double>(z0);

    auto voxel = [&](size_t zz, size_t yy, size_t xx) -> std::optional<double> {
        if (auto value = readVoxel(binding, cache, zz, yy, xx)) {
            return static_cast<double>(*value);
        }
        return std::nullopt;
    };

    const auto c000 = voxel(z0, y0, x0);
    const auto c001 = voxel(z0, y0, x1);
    const auto c010 = voxel(z0, y1, x0);
    const auto c011 = voxel(z0, y1, x1);
    const auto c100 = voxel(z1, y0, x0);
    const auto c101 = voxel(z1, y0, x1);
    const auto c110 = voxel(z1, y1, x0);
    const auto c111 = voxel(z1, y1, x1);
    if (!c000 || !c001 || !c010 || !c011 || !c100 || !c101 || !c110 || !c111) {
        return std::nullopt;
    }

    const double c00 = *c000 * (1.0 - fx) + *c001 * fx;
    const double c01 = *c010 * (1.0 - fx) + *c011 * fx;
    const double c10 = *c100 * (1.0 - fx) + *c101 * fx;
    const double c11 = *c110 * (1.0 - fx) + *c111 * fx;
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
    const size_t x1 = std::min(x0 + 1, binding.shapeZYX[2] - 1);
    const size_t y1 = std::min(y0 + 1, binding.shapeZYX[1] - 1);
    const size_t z1 = std::min(z0 + 1, binding.shapeZYX[0] - 1);
    const double fx = x - static_cast<double>(x0);
    const double fy = y - static_cast<double>(y0);
    const double fz = z - static_cast<double>(z0);

    auto voxel = [&](size_t zz, size_t yy, size_t xx) -> std::optional<double> {
        if (auto value = readVoxel(binding, cache, zz, yy, xx)) {
            return static_cast<double>(*value);
        }
        return std::nullopt;
    };

    const auto c000 = voxel(z0, y0, x0);
    const auto c001 = voxel(z0, y0, x1);
    const auto c010 = voxel(z0, y1, x0);
    const auto c011 = voxel(z0, y1, x1);
    const auto c100 = voxel(z1, y0, x0);
    const auto c101 = voxel(z1, y0, x1);
    const auto c110 = voxel(z1, y1, x0);
    const auto c111 = voxel(z1, y1, x1);
    if (!c000 || !c001 || !c010 || !c011 || !c100 || !c101 || !c110 || !c111) {
        return std::nullopt;
    }

    const double c00 = *c000 * (1.0 - fx) + *c001 * fx;
    const double c01 = *c010 * (1.0 - fx) + *c011 * fx;
    const double c10 = *c100 * (1.0 - fx) + *c101 * fx;
    const double c11 = *c110 * (1.0 - fx) + *c111 * fx;
    const double c0 = c00 * (1.0 - fy) + c01 * fy;
    const double c1 = c10 * (1.0 - fy) + c11 * fy;

    const double dcDGridX =
        ((*c001 - *c000) * (1.0 - fy) + (*c011 - *c010) * fy) * (1.0 - fz) +
        ((*c101 - *c100) * (1.0 - fy) + (*c111 - *c110) * fy) * fz;
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
            cache_.prefetch(*gradMag_.array, gradMagKeys, workers);
        });
        auto nxPrefetch = std::async(std::launch::async, [&]() {
            cache_.prefetch(*nx_.array, nxKeys, workers);
        });
        auto nyPrefetch = std::async(std::launch::async, [&]() {
            cache_.prefetch(*ny_.array, nyKeys, workers);
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
