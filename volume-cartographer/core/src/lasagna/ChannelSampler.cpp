#include "vc/lasagna/ChannelSampler.hpp"

#include "utils/thread_pool.hpp"
#include "utils/zarr.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <future>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>

#include <opencv2/core.hpp>

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

[[nodiscard]] utils::ThreadPool& lasagnaReadPool()
{
    // Keep workers alive so repeated short sampler calls can reuse HTTP/S3
    // connections through the underlying read-through store.
    static auto* pool = new utils::ThreadPool(lasagnaReadWorkersPerChannel() * 3);
    return *pool;
}

[[nodiscard]] uint32_t checkedChunkIndex(size_t value)
{
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Lasagna chunk index exceeds compact cache key range");
    }
    return static_cast<uint32_t>(value);
}

[[nodiscard]] std::vector<size_t> chunkPathIndicesForKey(
    const LasagnaChannelBinding& binding,
    const LasagnaChannelChunkKey& key)
{
    return {key.z, key.y, key.x};
}

[[nodiscard]] LasagnaChannelChunkKey chunkKeyForVoxel(
    const LasagnaChannelBinding& binding,
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

[[nodiscard]] std::string chunkKeyToString(const LasagnaChannelChunkKey& key)
{
    std::ostringstream out;
    out << "array=" << key.arrayId
        << " channel=" << key.channelIndex
        << " zyx=" << key.z << "," << key.y << "," << key.x;
    return out.str();
}

[[nodiscard]] size_t originalChunkOffset(
    const LasagnaChannelBinding& binding,
    size_t localZ,
    size_t localY,
    size_t localX)
{
    return (localZ * binding.chunksZYX[1] + localY) * binding.chunksZYX[2] + localX;
}

[[nodiscard]] std::shared_ptr<const LasagnaCachedChunk> readSourceChunk(
    const LasagnaChannelBinding& binding,
    const utils::ZarrArray& array,
    const LasagnaChannelChunkKey& key)
{
    const size_t originZ = static_cast<size_t>(key.z) * binding.chunksZYX[0];
    const size_t originY = static_cast<size_t>(key.y) * binding.chunksZYX[1];
    const size_t originX = static_cast<size_t>(key.x) * binding.chunksZYX[2];
    if (originZ >= binding.shapeZYX[0] ||
        originY >= binding.shapeZYX[1] ||
        originX >= binding.shapeZYX[2]) {
        return nullptr;
    }

    const auto source = array.read_chunk(chunkPathIndicesForKey(binding, key));
    auto cached = std::make_shared<LasagnaCachedChunk>();
    cached->dimsZYX = binding.chunksZYX;
    if (source.has_value()) {
        cached->values.resize(source->size());
        std::memcpy(cached->values.data(), source->data(), source->size());
    } else {
        const auto fillValue = array.metadata().fill_value;
        if (!fillValue.has_value()) {
            return nullptr;
        }
        if (!std::isfinite(*fillValue) ||
            *fillValue < 0.0 ||
            *fillValue > 255.0 ||
            std::trunc(*fillValue) != *fillValue) {
            throw std::runtime_error(
                "Lasagna uint8 Zarr has an invalid fill_value: " + binding.path.string());
        }
        cached->values.assign(
            array.metadata().chunk_byte_size(), static_cast<uint8_t>(*fillValue));
    }

    return cached;
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

[[nodiscard]] std::optional<double> readSourceVoxel(
    const LasagnaChannelBinding& binding,
    const LasagnaChannelChunkKey& key,
    const std::shared_ptr<const LasagnaCachedChunk>& chunk,
    size_t z,
    size_t y,
    size_t x)
{
    if (chunk == nullptr) {
        return std::nullopt;
    }
    const size_t offset = originalChunkOffset(
        binding,
        z % binding.chunksZYX[0],
        y % binding.chunksZYX[1],
        x % binding.chunksZYX[2]);
    if (offset >= chunk->values.size()) {
        throw std::runtime_error("Lasagna cached source chunk is smaller than expected at chunk " +
                                 chunkKeyToString(key));
    }
    return static_cast<double>(chunk->values[offset]);
}

[[nodiscard]] std::optional<LasagnaCubeValues> readInterpolationCube(
    const LasagnaChannelBinding& binding,
    const LasagnaCubeRequest& request)
{
    if (!request.valid) {
        return std::nullopt;
    }

    std::array<double, 8> values{};
    size_t cubeIndex = 0;
    for (size_t dz = 0; dz <= 1; ++dz) {
        const size_t z = std::min(request.z0 + dz, binding.shapeZYX[0] - 1);
        for (size_t dy = 0; dy <= 1; ++dy) {
            const size_t y = std::min(request.y0 + dy, binding.shapeZYX[1] - 1);
            for (size_t dx = 0; dx <= 1; ++dx) {
                const size_t x = std::min(request.x0 + dx, binding.shapeZYX[2] - 1);
                const size_t chunkIndex = request.singleChunk ? size_t{0} : cubeIndex;
                const auto value = readSourceVoxel(
                    binding,
                    request.keys[chunkIndex],
                    request.chunks[chunkIndex],
                    z,
                    y,
                    x);
                if (!value.has_value()) {
                    return std::nullopt;
                }
                values[cubeIndex++] = *value;
            }
        }
    }
    return LasagnaCubeValues{values[0], values[1], values[2], values[3],
                             values[4], values[5], values[6], values[7]};
}

[[nodiscard]] bool readInterpolationCubeValues(
    const LasagnaChannelBinding& binding,
    const LasagnaCubeRequest& request,
    std::array<double, 8>& values)
{
    if (!request.valid) {
        return false;
    }

    size_t cubeIndex = 0;
    for (size_t dz = 0; dz <= 1; ++dz) {
        const size_t z = std::min(request.z0 + dz, binding.shapeZYX[0] - 1);
        for (size_t dy = 0; dy <= 1; ++dy) {
            const size_t y = std::min(request.y0 + dy, binding.shapeZYX[1] - 1);
            for (size_t dx = 0; dx <= 1; ++dx) {
                const size_t x = std::min(request.x0 + dx, binding.shapeZYX[2] - 1);
                const size_t chunkIndex = request.singleChunk ? size_t{0} : cubeIndex;
                const auto& chunk = request.chunks[chunkIndex];
                if (chunk == nullptr) {
                    return false;
                }
                const size_t offset = originalChunkOffset(
                    binding,
                    z % binding.chunksZYX[0],
                    y % binding.chunksZYX[1],
                    x % binding.chunksZYX[2]);
                if (offset >= chunk->values.size()) {
                    throw std::runtime_error(
                        "Lasagna cached source chunk is smaller than expected at chunk " +
                        chunkKeyToString(request.keys[chunkIndex]));
                }
                values[cubeIndex++] = static_cast<double>(chunk->values[offset]);
            }
        }
    }
    return true;
}

[[nodiscard]] bool readInterpolationCubeBytes(
    const LasagnaChannelBinding& binding,
    const LasagnaCubeRequest& request,
    std::array<uint8_t, 8>& values)
{
    if (!request.valid) {
        return false;
    }

    size_t cubeIndex = 0;
    for (size_t dz = 0; dz <= 1; ++dz) {
        const size_t z = std::min(request.z0 + dz, binding.shapeZYX[0] - 1);
        for (size_t dy = 0; dy <= 1; ++dy) {
            const size_t y = std::min(request.y0 + dy, binding.shapeZYX[1] - 1);
            for (size_t dx = 0; dx <= 1; ++dx) {
                const size_t x = std::min(request.x0 + dx, binding.shapeZYX[2] - 1);
                const size_t chunkIndex = request.singleChunk ? size_t{0} : cubeIndex;
                const auto& chunk = request.chunks[chunkIndex];
                if (chunk == nullptr) {
                    return false;
                }
                const size_t offset = originalChunkOffset(
                    binding,
                    z % binding.chunksZYX[0],
                    y % binding.chunksZYX[1],
                    x % binding.chunksZYX[2]);
                if (offset >= chunk->values.size()) {
                    throw std::runtime_error(
                        "Lasagna cached source chunk is smaller than expected at chunk " +
                        chunkKeyToString(request.keys[chunkIndex]));
                }
                values[cubeIndex++] = chunk->values[offset];
            }
        }
    }
    return true;
}

[[nodiscard]] std::optional<double> sampleLasagnaChannelValues(
    const LasagnaChannelBinding& binding,
    const LasagnaCubeRequest& request)
{
    std::array<double, 8> values{};
    if (!readInterpolationCubeValues(binding, request, values)) {
        return std::nullopt;
    }

    const double c00 = values[0] * (1.0 - request.fx) + values[1] * request.fx;
    const double c01 = values[2] * (1.0 - request.fx) + values[3] * request.fx;
    const double c10 = values[4] * (1.0 - request.fx) + values[5] * request.fx;
    const double c11 = values[6] * (1.0 - request.fx) + values[7] * request.fx;
    const double c0 = c00 * (1.0 - request.fy) + c01 * request.fy;
    const double c1 = c10 * (1.0 - request.fy) + c11 * request.fy;
    return c0 * (1.0 - request.fz) + c1 * request.fz;
}

struct DecodedCompactNormal {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    bool valid = false;
};

[[nodiscard]] const std::array<DecodedCompactNormal, 256 * 256>&
decodedCompactNormalTable()
{
    static const auto table = [] {
        std::array<DecodedCompactNormal, 256 * 256> out{};
        for (size_t rawNx = 0; rawNx < 256; ++rawNx) {
            for (size_t rawNy = 0; rawNy < 256; ++rawNy) {
                double nx = decodeCompactNormalComponent(static_cast<double>(rawNx));
                double ny = decodeCompactNormalComponent(static_cast<double>(rawNy));
                const double nzSq = std::max(0.0, 1.0 - nx * nx - ny * ny);
                double nz = std::sqrt(nzSq);
                const double normalLength = std::sqrt(nx * nx + ny * ny + nz * nz);
                auto& entry = out[(rawNx << 8) | rawNy];
                if (!(normalLength > kEpsilon) || !std::isfinite(normalLength)) {
                    continue;
                }
                const double invLength = 1.0 / normalLength;
                entry.x = nx * invLength;
                entry.y = ny * invLength;
                entry.z = nz * invLength;
                entry.valid = true;
            }
        }
        return out;
    }();
    return table;
}

[[nodiscard]] cv::Vec3d fallbackPrincipalAxis(
    double a00,
    double a11,
    double a22)
{
    int axis = 0;
    double value = a00;
    if (a11 > value) {
        axis = 1;
        value = a11;
    }
    if (a22 > value) {
        axis = 2;
    }
    cv::Vec3d result{0.0, 0.0, 0.0};
    result[axis] = 1.0;
    return result;
}

[[nodiscard]] cv::Vec3d principalCompactTensorAxisFromComponents(
    double a00,
    double a01,
    double a02,
    double a11,
    double a12,
    double a22,
    const cv::Vec3d& hint)
{
    if (!std::isfinite(a00) || !std::isfinite(a01) || !std::isfinite(a02) ||
        !std::isfinite(a11) || !std::isfinite(a12) || !std::isfinite(a22)) {
        return {0.0, 0.0, 0.0};
    }

    cv::Vec3d axis{0.0, 0.0, 0.0};
    const double offDiagonal = a01 * a01 + a02 * a02 + a12 * a12;
    if (offDiagonal <= kEpsilon) {
        axis = fallbackPrincipalAxis(a00, a11, a22);
    } else {
        const double q = (a00 + a11 + a22) / 3.0;
        const double b00 = a00 - q;
        const double b11 = a11 - q;
        const double b22 = a22 - q;
        const double p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * offDiagonal;
        const double p = std::sqrt(std::max(0.0, p2 / 6.0));
        if (p <= kEpsilon) {
            axis = fallbackPrincipalAxis(a00, a11, a22);
        } else {
            const double invP = 1.0 / p;
            const double c00 = b00 * invP;
            const double c01 = a01 * invP;
            const double c02 = a02 * invP;
            const double c11 = b11 * invP;
            const double c12 = a12 * invP;
            const double c22 = b22 * invP;
            const double detC =
                c00 * (c11 * c22 - c12 * c12) -
                c01 * (c01 * c22 - c12 * c02) +
                c02 * (c01 * c12 - c11 * c02);
            const double r = std::clamp(0.5 * detC, -1.0, 1.0);
            const double phi = std::acos(r) / 3.0;
            const double lambda = q + 2.0 * p * std::cos(phi);
            const cv::Vec3d row0{a00 - lambda, a01, a02};
            const cv::Vec3d row1{a01, a11 - lambda, a12};
            const cv::Vec3d row2{a02, a12, a22 - lambda};
            const std::array<cv::Vec3d, 3> candidates = {
                row0.cross(row1),
                row0.cross(row2),
                row1.cross(row2),
            };
            double bestNorm2 = -1.0;
            for (const auto& candidate : candidates) {
                const double norm2 = candidate.dot(candidate);
                if (norm2 > bestNorm2) {
                    bestNorm2 = norm2;
                    axis = candidate;
                }
            }
            if (!(bestNorm2 > kEpsilon * kEpsilon)) {
                axis = fallbackPrincipalAxis(a00, a11, a22);
            }
        }
    }
    if (length(axis) <= kEpsilon) {
        axis = fallbackPrincipalAxis(a00, a11, a22);
    }
    axis = normalizedOrZero(axis);
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

[[nodiscard]] std::optional<LasagnaCubeValues> readInterpolationCube(
    const LasagnaChannelBinding& binding,
    const LasagnaChannelChunkCache& cache,
    size_t z0,
    size_t y0,
    size_t x0)
{
    LasagnaCubeRequest request;
    request.valid = true;
    request.z0 = z0;
    request.y0 = y0;
    request.x0 = x0;
    size_t cubeIndex = 0;
    for (size_t dz = 0; dz <= 1; ++dz) {
        const size_t z = std::min(z0 + dz, binding.shapeZYX[0] - 1);
        for (size_t dy = 0; dy <= 1; ++dy) {
            const size_t y = std::min(y0 + dy, binding.shapeZYX[1] - 1);
            for (size_t dx = 0; dx <= 1; ++dx) {
                const size_t x = std::min(x0 + dx, binding.shapeZYX[2] - 1);
                request.keys[cubeIndex] = chunkKeyForVoxel(binding, z, y, x);
                request.chunks[cubeIndex] = cache.get(
                    binding, *binding.array, request.keys[cubeIndex]);
                ++cubeIndex;
            }
        }
    }
    return readInterpolationCube(binding, request);
}

[[nodiscard]] double cubeValue(const LasagnaCubeValues& cube, int dz, int dy, int dx)
{
    if (dz == 0 && dy == 0 && dx == 0) return cube.c000;
    if (dz == 0 && dy == 0 && dx == 1) return cube.c001;
    if (dz == 0 && dy == 1 && dx == 0) return cube.c010;
    if (dz == 0 && dy == 1 && dx == 1) return cube.c011;
    if (dz == 1 && dy == 0 && dx == 0) return cube.c100;
    if (dz == 1 && dy == 0 && dx == 1) return cube.c101;
    if (dz == 1 && dy == 1 && dx == 0) return cube.c110;
    return cube.c111;
}

} // namespace

bool LasagnaChannelChunkKey::operator==(const LasagnaChannelChunkKey& other) const noexcept
{
    return arrayId == other.arrayId &&
           channelIndex == other.channelIndex &&
           z == other.z &&
           y == other.y &&
           x == other.x;
}

size_t LasagnaChannelChunkKeyHash::operator()(const LasagnaChannelChunkKey& key) const noexcept
{
    size_t hash = key.arrayId;
    hash ^= key.channelIndex + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
    hash ^= key.z + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
    hash ^= key.y + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
    hash ^= key.x + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
    return hash;
}

struct LasagnaChannelChunkCache::InFlightLoad {
    std::mutex mutex;
    std::condition_variable finished;
    bool done = false;
    std::shared_ptr<const LasagnaCachedChunk> bytes;
    std::exception_ptr error;
};

LasagnaChannelChunkCache::LasagnaChannelChunkCache(size_t capacityBytes)
    : capacityBytes_(std::max<size_t>(1, capacityBytes))
{
}

std::shared_ptr<const LasagnaCachedChunk> LasagnaChannelChunkCache::get(
    const LasagnaChannelBinding& binding,
    const utils::ZarrArray& array,
    const LasagnaChannelChunkKey& key) const
{
    return load(binding, array, key);
}

NormalPrefetchReport LasagnaChannelChunkCache::prefetchResolved(
    const LasagnaChannelBinding& binding,
    const utils::ZarrArray& array,
    const std::vector<LasagnaChannelChunkKey>& keys,
    size_t maxWorkers,
    ResolvedChunkMap& resolved) const
{
    NormalPrefetchReport report;
    resolved.clear();
    std::vector<LasagnaChannelChunkKey> missing;
    missing.reserve(keys.size());
    std::unordered_set<LasagnaChannelChunkKey, LasagnaChannelChunkKeyHash> seen;
    seen.reserve(keys.size());
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        for (const auto& key : keys) {
            if (!seen.insert(key).second) {
                continue;
            }
            ++report.requestedChunks;
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
            futures.push_back(lasagnaReadPool().submit(
                [this, &binding, &array, &missing, &next]() {
                    while (true) {
                        const size_t index = next.fetch_add(1);
                        if (index >= missing.size()) {
                            return;
                        }
                        const LasagnaChannelChunkKey key = missing[index];
                        (void)load(binding, array, key);
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

NormalPrefetchReport LasagnaChannelChunkCache::prefetchInterleaved(
    const std::vector<PrefetchRequest>& requests) const
{
    NormalPrefetchReport report;
    std::vector<PrefetchRequest> missing;
    missing.reserve(requests.size());
    std::unordered_set<LasagnaChannelChunkKey, LasagnaChannelChunkKeyHash> seen;
    seen.reserve(requests.size());
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        for (const auto& request : requests) {
            if (!seen.insert(request.second).second) {
                continue;
            }
            ++report.requestedChunks;
            if (entries_.find(request.second) == entries_.end()) {
                missing.push_back(request);
            }
        }
    }
    report.chunksRead = missing.size();

    const size_t maxWorkers = std::min(lasagnaReadPool().worker_count(), missing.size());
    std::vector<std::future<void>> futures;
    futures.reserve(maxWorkers);
    std::atomic<size_t> next{0};
    for (size_t worker = 0; worker < maxWorkers; ++worker) {
        futures.push_back(lasagnaReadPool().submit([this, &missing, &next]() {
            while (true) {
                const size_t index = next.fetch_add(1);
                if (index >= missing.size()) {
                    return;
                }
                const auto& request = missing[index];
                (void)load(*request.first, *request.first->array, request.second);
            }
        }));
    }
    for (auto& future : futures) {
        future.get();
    }
    return report;
}

std::shared_ptr<const LasagnaCachedChunk> LasagnaChannelChunkCache::load(
    const LasagnaChannelBinding& binding,
    const utils::ZarrArray& array,
    const LasagnaChannelChunkKey& key) const
{
    std::shared_ptr<InFlightLoad> request;
    bool ownsRequest = false;
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (auto it = entries_.find(key); it != entries_.end()) {
            lru_.splice(lru_.begin(), lru_, it->second.lruIt);
            return it->second.bytes;
        }
        if (auto it = inFlight_.find(key); it != inFlight_.end()) {
            request = it->second;
        } else {
            request = std::make_shared<InFlightLoad>();
            inFlight_.emplace(key, request);
            ownsRequest = true;
        }
    }

    if (!ownsRequest) {
        std::unique_lock<std::mutex> lock(request->mutex);
        request->finished.wait(lock, [&]() { return request->done; });
        if (request->error) {
            std::rethrow_exception(request->error);
        }
        return request->bytes;
    }

    std::shared_ptr<const LasagnaCachedChunk> bytes;
    std::exception_ptr error;
    try {
        bytes = readSourceChunk(binding, array, key);
        store(key, bytes);
    } catch (...) {
        error = std::current_exception();
    }
    {
        std::lock_guard<std::mutex> lock(request->mutex);
        request->bytes = bytes;
        request->error = error;
        request->done = true;
    }
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (auto it = inFlight_.find(key);
            it != inFlight_.end() && it->second == request) {
            inFlight_.erase(it);
        }
    }
    request->finished.notify_all();
    if (error) {
        std::rethrow_exception(error);
    }
    return bytes;
}

void LasagnaChannelChunkCache::store(
    LasagnaChannelChunkKey key,
    std::shared_ptr<const LasagnaCachedChunk> bytes) const
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

void LasagnaChannelChunkCache::trim() const
{
    while (cachedBytes_ > capacityBytes_ && !lru_.empty()) {
        const LasagnaChannelChunkKey evicted = lru_.back();
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

LasagnaLocalChunkResolver::LasagnaLocalChunkResolver(
    const LasagnaChannelBinding& binding,
    const LasagnaChannelChunkCache& cache)
    : binding_(&binding)
    , cache_(&cache)
{
}

std::shared_ptr<const LasagnaCachedChunk> LasagnaLocalChunkResolver::resolveKey(
    const LasagnaChannelChunkKey& key)
{
    if (hasLast_ && lastKey_ == key) {
        return lastChunk_;
    }
    for (size_t index = 0; index < size_; ++index) {
        if (keys_[index] == key) {
            lastKey_ = key;
            lastChunk_ = chunks_[index];
            hasLast_ = true;
            return chunks_[index];
        }
    }
    auto chunk = cache_->get(*binding_, *binding_->array, key);
    if (size_ < keys_.size()) {
        keys_[size_] = key;
        chunks_[size_] = chunk;
        ++size_;
    } else {
        keys_[next_] = key;
        chunks_[next_] = chunk;
        next_ = (next_ + 1) % keys_.size();
    }
    lastKey_ = key;
    lastChunk_ = chunk;
    hasLast_ = true;
    return chunk;
}

void LasagnaLocalChunkResolver::resolve(LasagnaCubeRequest& request)
{
    if (!request.valid)
        return;
    if (request.singleChunk) {
        request.chunks[0] = resolveKey(request.keys[0]);
        return;
    }
    for (size_t cubeIndex = 0; cubeIndex < request.keys.size(); ++cubeIndex) {
        bool reused = false;
        for (size_t previousIndex = 0; previousIndex < cubeIndex; ++previousIndex) {
            if (request.keys[cubeIndex] == request.keys[previousIndex]) {
                request.chunks[cubeIndex] = request.chunks[previousIndex];
                reused = true;
                break;
            }
        }
        if (!reused)
            request.chunks[cubeIndex] = resolveKey(request.keys[cubeIndex]);
    }
}

size_t lasagnaReadWorkersPerChannel()
{
    const unsigned hardwareThreads = std::thread::hardware_concurrency();
    return std::clamp<size_t>(
        hardwareThreads == 0 ? 4 : static_cast<size_t>(hardwareThreads), 1, 8);
}

std::shared_ptr<LasagnaChannelChunkCache>
sharedLasagnaChannelChunkCache(size_t capacityBytes)
{
    static std::mutex mutex;
    static std::shared_ptr<LasagnaChannelChunkCache> cache;
    std::lock_guard<std::mutex> lock(mutex);
    if (!cache) {
        cache = std::make_shared<LasagnaChannelChunkCache>(capacityBytes);
    }
    return cache;
}

double decodeCompactNormalComponent(double raw)
{
    return (raw - 128.0) / 127.0;
}

cv::Vec3d decodeCompactNormalFromRaw(double rawNx, double rawNy)
{
    const double nx = decodeCompactNormalComponent(rawNx);
    const double ny = decodeCompactNormalComponent(rawNy);
    const double nzSq = std::max(0.0, 1.0 - nx * nx - ny * ny);
    return normalizedOrZero({nx, ny, std::sqrt(nzSq)});
}

cv::Vec3d principalCompactTensorAxis(const cv::Matx33d& tensor, const cv::Vec3d& hint)
{
    auto fallbackTensorAxis = [](const cv::Matx33d& t) {
        int axis = 0;
        double value = t(0, 0);
        if (t(1, 1) > value) {
            axis = 1;
            value = t(1, 1);
        }
        if (t(2, 2) > value) {
            axis = 2;
        }
        cv::Vec3d result{0.0, 0.0, 0.0};
        result[axis] = 1.0;
        return result;
    };

    bool finiteTensor = true;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            finiteTensor = finiteTensor && std::isfinite(tensor(row, col));
        }
    }
    if (!finiteTensor) {
        return {0.0, 0.0, 0.0};
    }

    cv::Vec3d axis{0.0, 0.0, 0.0};

    const double a00 = tensor(0, 0);
    const double a01 = 0.5 * (tensor(0, 1) + tensor(1, 0));
    const double a02 = 0.5 * (tensor(0, 2) + tensor(2, 0));
    const double a11 = tensor(1, 1);
    const double a12 = 0.5 * (tensor(1, 2) + tensor(2, 1));
    const double a22 = tensor(2, 2);
    const double offDiagonal = a01 * a01 + a02 * a02 + a12 * a12;
    if (offDiagonal <= kEpsilon) {
        axis = fallbackTensorAxis(tensor);
    } else {
        const double q = (a00 + a11 + a22) / 3.0;
        const double b00 = a00 - q;
        const double b11 = a11 - q;
        const double b22 = a22 - q;
        const double p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * offDiagonal;
        const double p = std::sqrt(std::max(0.0, p2 / 6.0));
        if (p <= kEpsilon) {
            axis = fallbackTensorAxis(tensor);
        } else {
            const double invP = 1.0 / p;
            const double c00 = b00 * invP;
            const double c01 = a01 * invP;
            const double c02 = a02 * invP;
            const double c11 = b11 * invP;
            const double c12 = a12 * invP;
            const double c22 = b22 * invP;
            const double detC =
                c00 * (c11 * c22 - c12 * c12) -
                c01 * (c01 * c22 - c12 * c02) +
                c02 * (c01 * c12 - c11 * c02);
            const double r = std::clamp(0.5 * detC, -1.0, 1.0);
            const double phi = std::acos(r) / 3.0;
            const double lambda = q + 2.0 * p * std::cos(phi);
            const cv::Vec3d row0{a00 - lambda, a01, a02};
            const cv::Vec3d row1{a01, a11 - lambda, a12};
            const cv::Vec3d row2{a02, a12, a22 - lambda};
            const std::array<cv::Vec3d, 3> candidates = {
                row0.cross(row1),
                row0.cross(row2),
                row1.cross(row2),
            };
            double bestNorm2 = -1.0;
            for (const auto& candidate : candidates) {
                const double norm2 = candidate.dot(candidate);
                if (norm2 > bestNorm2) {
                    bestNorm2 = norm2;
                    axis = candidate;
                }
            }
            if (!(bestNorm2 > kEpsilon * kEpsilon)) {
                axis = fallbackTensorAxis(tensor);
            }
        }
    }
    if (length(axis) <= kEpsilon) {
        axis = fallbackTensorAxis(tensor);
    }
    axis = normalizedOrZero(axis);
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

LasagnaChannelBinding bindLasagnaChannel(
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
    LasagnaChannelBinding binding;
    binding.group = group;
    binding.channelIndex = *channelIndex;
    binding.path = group->zarrPath;
    binding.arrayId = arrayIdForPath(binding.path);
    binding.array = std::make_shared<utils::ZarrArray>(
        openLasagnaChannelArray(manifest, *group, 1));
    binding.spacing = static_cast<double>(group->scaleFactor()) *
                      manifest.sourceToBase / manifest.workingToBaseScale;

    const auto& meta = binding.array->metadata();
    if (meta.dtype != utils::ZarrDtype::uint8) {
        throw std::runtime_error("Lasagna channel '" + std::string(channel) + "' must be uint8");
    }
    if (meta.shape.size() != 3 || meta.chunks.size() != 3) {
        throw std::runtime_error("Lasagna channel '" + std::string(channel) +
                                 "' must reference a 3D (Z,Y,X) zarr");
    }
    binding.shapeZYX = {meta.shape[0], meta.shape[1], meta.shape[2]};
    binding.chunksZYX = {meta.chunks[0], meta.chunks[1], meta.chunks[2]};

    if (binding.spacing <= 0.0 || !std::isfinite(binding.spacing)) {
        throw std::runtime_error("Lasagna channel '" + std::string(channel) + "' has invalid spacing");
    }
    if (binding.shapeZYX[0] == 0 || binding.shapeZYX[1] == 0 || binding.shapeZYX[2] == 0 ||
        binding.chunksZYX[0] == 0 || binding.chunksZYX[1] == 0 || binding.chunksZYX[2] == 0) {
        throw std::runtime_error("Lasagna channel '" + std::string(channel) + "' has empty zarr shape/chunks");
    }
    return binding;
}

LasagnaCubeRequest prepareLasagnaCubeRequest(
    const LasagnaChannelBinding& binding,
    const cv::Vec3d& volumePoint)
{
    LasagnaCubeRequest request;
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
    const size_t x1 = std::min(request.x0 + size_t{1}, binding.shapeZYX[2] - 1);
    const size_t y1 = std::min(request.y0 + size_t{1}, binding.shapeZYX[1] - 1);
    const size_t z1 = std::min(request.z0 + size_t{1}, binding.shapeZYX[0] - 1);
    const bool singleChunk =
        request.x0 / binding.chunksZYX[2] == x1 / binding.chunksZYX[2] &&
        request.y0 / binding.chunksZYX[1] == y1 / binding.chunksZYX[1] &&
        request.z0 / binding.chunksZYX[0] == z1 / binding.chunksZYX[0];
    if (singleChunk) {
        request.keys[0] = chunkKeyForVoxel(binding, request.z0, request.y0, request.x0);
        request.singleChunk = true;
        request.valid = true;
        return request;
    }
    size_t cubeIndex = 0;
    for (size_t dz = 0; dz <= 1; ++dz) {
        const size_t gz = std::min(request.z0 + dz, binding.shapeZYX[0] - 1);
        for (size_t dy = 0; dy <= 1; ++dy) {
            const size_t gy = std::min(request.y0 + dy, binding.shapeZYX[1] - 1);
            for (size_t dx = 0; dx <= 1; ++dx) {
                const size_t gx = std::min(request.x0 + dx, binding.shapeZYX[2] - 1);
                request.keys[cubeIndex++] = chunkKeyForVoxel(binding, gz, gy, gx);
            }
        }
    }
    request.valid = true;
    return request;
}

bool sameLasagnaSamplingGrid(
    const LasagnaChannelBinding& a,
    const LasagnaChannelBinding& b)
{
    const double spacingTolerance =
        1.0e-12 * std::max({1.0, std::abs(a.spacing), std::abs(b.spacing)});
    return a.shapeZYX == b.shapeZYX &&
           a.chunksZYX == b.chunksZYX &&
           std::abs(a.spacing - b.spacing) <= spacingTolerance;
}

LasagnaCubeRequest cloneLasagnaCubeRequestForBinding(
    const LasagnaCubeRequest& source,
    const LasagnaChannelBinding& binding)
{
    LasagnaCubeRequest out = source;
    out.chunks = {};
    if (out.singleChunk) {
        out.keys[0].arrayId = binding.arrayId;
        out.keys[0].channelIndex = static_cast<uint32_t>(binding.channelIndex);
        return out;
    }
    for (auto& key : out.keys) {
        key.arrayId = binding.arrayId;
        key.channelIndex = static_cast<uint32_t>(binding.channelIndex);
    }
    return out;
}

void assignResolvedLasagnaCubeRequestChunks(
    LasagnaCubeRequest& request,
    const LasagnaChannelChunkCache::ResolvedChunkMap& resolved)
{
    if (!request.valid) {
        return;
    }
    if (request.singleChunk) {
        auto it = resolved.find(request.keys.front());
        if (it != resolved.end()) {
            request.chunks[0] = it->second;
        }
        return;
    }
    for (size_t cubeIndex = 0; cubeIndex < request.keys.size(); ++cubeIndex) {
        bool reused = false;
        for (size_t previousIndex = 0; previousIndex < cubeIndex; ++previousIndex) {
            if (request.keys[cubeIndex] == request.keys[previousIndex]) {
                request.chunks[cubeIndex] = request.chunks[previousIndex];
                reused = true;
                break;
            }
        }
        if (reused) {
            continue;
        }
        auto it = resolved.find(request.keys[cubeIndex]);
        if (it != resolved.end()) {
            request.chunks[cubeIndex] = it->second;
        }
    }
}

void appendUniqueLasagnaCubeRequestChunkKeys(
    const LasagnaCubeRequest& request,
    std::vector<LasagnaChannelChunkKey>& keys)
{
    auto appendIfNewTail = [&](const LasagnaChannelChunkKey& key) {
        if (keys.empty() || !(keys.back() == key)) {
            keys.push_back(key);
        }
    };
    if (!request.valid) {
        return;
    }
    if (request.singleChunk) {
        appendIfNewTail(request.keys.front());
        return;
    }
    for (size_t cubeIndex = 0; cubeIndex < request.keys.size(); ++cubeIndex) {
        bool duplicate = false;
        for (size_t previousIndex = 0; previousIndex < cubeIndex; ++previousIndex) {
            if (request.keys[cubeIndex] == request.keys[previousIndex]) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            appendIfNewTail(request.keys[cubeIndex]);
        }
    }
}

void deduplicateLasagnaChunkKeysInPlace(
    std::vector<LasagnaChannelChunkKey>& keys)
{
    std::unordered_set<LasagnaChannelChunkKey, LasagnaChannelChunkKeyHash> seen;
    seen.reserve(keys.size());
    size_t writeIndex = 0;
    for (const auto& key : keys) {
        if (!seen.insert(key).second) {
            continue;
        }
        keys[writeIndex++] = key;
    }
    keys.resize(writeIndex);
}

void appendLasagnaInterpolationChunkKeys(
    const LasagnaChannelBinding& binding,
    const cv::Vec3d& volumePoint,
    std::vector<LasagnaChannelChunkKey>& keys)
{
    const LasagnaCubeRequest request = prepareLasagnaCubeRequest(binding, volumePoint);
    if (!request.valid) {
        return;
    }
    keys.insert(keys.end(), request.keys.begin(), request.keys.end());
}

std::optional<double> sampleLasagnaChannel(
    const LasagnaChannelBinding& binding,
    const LasagnaChannelChunkCache& cache,
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

std::optional<double> sampleLasagnaChannel(
    const LasagnaChannelBinding& binding,
    const LasagnaCubeRequest& request)
{
    return sampleLasagnaChannelValues(binding, request);
}

std::optional<cv::Vec3d> sampleLasagnaCompactAxisTensor(
    const LasagnaChannelBinding& nxBinding,
    const LasagnaChannelBinding& nyBinding,
    const LasagnaChannelChunkCache& cache,
    const cv::Vec3d& volumePoint)
{
    const LasagnaCubeRequest nxRequest = prepareLasagnaCubeRequest(nxBinding, volumePoint);
    const LasagnaCubeRequest nyRequest = prepareLasagnaCubeRequest(nyBinding, volumePoint);
    if (!nxRequest.valid || !nyRequest.valid) {
        return std::nullopt;
    }
    LasagnaCubeRequest resolvedNx = nxRequest;
    LasagnaCubeRequest resolvedNy = nyRequest;
    for (size_t i = 0; i < resolvedNx.keys.size(); ++i) {
        resolvedNx.chunks[i] = cache.get(nxBinding, *nxBinding.array, resolvedNx.keys[i]);
        resolvedNy.chunks[i] = cache.get(nyBinding, *nyBinding.array, resolvedNy.keys[i]);
    }
    return sampleLasagnaCompactAxisTensor(nxBinding, nyBinding, resolvedNx, resolvedNy);
}

std::optional<cv::Vec3d> sampleLasagnaCompactAxisTensor(
    const LasagnaChannelBinding& nxBinding,
    const LasagnaChannelBinding& nyBinding,
    const LasagnaCubeRequest& nxRequest,
    const LasagnaCubeRequest& nyRequest)
{
    if (!nxRequest.valid || !nyRequest.valid) {
        return std::nullopt;
    }
    std::array<uint8_t, 8> nxValues{};
    std::array<uint8_t, 8> nyValues{};
    if (!readInterpolationCubeBytes(nxBinding, nxRequest, nxValues) ||
        !readInterpolationCubeBytes(nyBinding, nyRequest, nyValues)) {
        return std::nullopt;
    }

    const auto& normalTable = decodedCompactNormalTable();
    double t00 = 0.0;
    double t01 = 0.0;
    double t02 = 0.0;
    double t11 = 0.0;
    double t12 = 0.0;
    double t22 = 0.0;
    cv::Vec3d hint{0.0, 0.0, 0.0};
    double totalWeight = 0.0;
    size_t cubeIndex = 0;
    for (int dz = 0; dz <= 1; ++dz) {
        const double wz = dz == 0 ? (1.0 - nxRequest.fz) : nxRequest.fz;
        for (int dy = 0; dy <= 1; ++dy) {
            const double wy = dy == 0 ? (1.0 - nxRequest.fy) : nxRequest.fy;
            for (int dx = 0; dx <= 1; ++dx) {
                const double wx = dx == 0 ? (1.0 - nxRequest.fx) : nxRequest.fx;
                const double weight = wx * wy * wz;
                if (weight <= 0.0) {
                    ++cubeIndex;
                    continue;
                }
                const auto& decoded =
                    normalTable[(static_cast<size_t>(nxValues[cubeIndex]) << 8) |
                                static_cast<size_t>(nyValues[cubeIndex])];
                if (!decoded.valid) {
                    ++cubeIndex;
                    continue;
                }
                const double nx = decoded.x;
                const double ny = decoded.y;
                const double nz = decoded.z;
                t00 += weight * nx * nx;
                t01 += weight * nx * ny;
                t02 += weight * nx * nz;
                t11 += weight * ny * ny;
                t12 += weight * ny * nz;
                t22 += weight * nz * nz;
                const cv::Vec3d normal{nx, ny, nz};
                hint += normal * weight;
                totalWeight += weight;
                ++cubeIndex;
            }
        }
    }
    if (totalWeight <= kEpsilon) {
        return std::nullopt;
    }
    const cv::Vec3d normal = principalCompactTensorAxisFromComponents(
        t00, t01, t02, t11, t12, t22, hint);
    if (length(normal) <= kEpsilon) {
        return std::nullopt;
    }
    return normal;
}

} // namespace vc::lasagna
