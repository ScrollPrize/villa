#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include "utils/zarr.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <list>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

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
    explicit ChunkCache(size_t capacity)
        : capacity_(std::max<size_t>(1, capacity))
    {
    }

    [[nodiscard]] const std::vector<std::byte>* get(
        const utils::ZarrArray& array,
        const ChunkKey& key) const
    {
        if (auto it = entries_.find(key); it != entries_.end()) {
            lru_.splice(lru_.begin(), lru_, it->second.lruIt);
            return it->second.bytes ? &*it->second.bytes : nullptr;
        }

        std::optional<std::vector<std::byte>> bytes = array.read_chunk(key.indices);
        lru_.push_front(key);
        entries_.emplace(key, Entry{std::move(bytes), lru_.begin()});
        trim();

        const auto inserted = entries_.find(key);
        if (inserted == entries_.end() || !inserted->second.bytes) {
            return nullptr;
        }
        return &*inserted->second.bytes;
    }

private:
    struct Entry {
        std::optional<std::vector<std::byte>> bytes;
        std::list<ChunkKey>::iterator lruIt;
    };

    void trim() const
    {
        while (entries_.size() > capacity_) {
            const ChunkKey evicted = lru_.back();
            lru_.pop_back();
            entries_.erase(evicted);
        }
    }

    size_t capacity_ = 128;
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
    const std::vector<std::byte>* bytes = cache.get(*binding.array, key);
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

} // namespace

class LasagnaNormalSampler::Impl {
public:
    Impl(const LasagnaDataset& dataset, LasagnaNormalSamplerOptions options)
        : nx_(bindChannel(dataset.manifest(), "nx"))
        , ny_(bindChannel(dataset.manifest(), "ny"))
        , gradMag_(bindChannel(dataset.manifest(), "grad_mag"))
        , options_(options)
        , cache_(options.maxCachedChunks)
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

} // namespace vc::lasagna
