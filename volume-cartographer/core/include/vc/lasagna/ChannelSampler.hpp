#pragma once

#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LineModel.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <opencv2/core/types.hpp>

namespace utils { class ZarrArray; }
namespace vc::render { class DecodedChunkCacheBudget; }

namespace vc::lasagna {

struct LasagnaChannelChunkKey {
    uint32_t arrayId = 0;
    uint32_t channelIndex = 0;
    uint32_t z = 0;
    uint32_t y = 0;
    uint32_t x = 0;

    [[nodiscard]] bool operator==(const LasagnaChannelChunkKey& other) const noexcept;
};

struct LasagnaChannelChunkKeyHash {
    [[nodiscard]] size_t operator()(const LasagnaChannelChunkKey& key) const noexcept;
};

struct LasagnaCachedChunk {
    std::vector<uint8_t> values;
    std::array<size_t, 3> dimsZYX{0, 0, 0};
};

struct LasagnaChannelBinding {
    const LasagnaChannelGroup* group = nullptr;
    uint32_t arrayId = 0;
    size_t channelIndex = 0;
    std::filesystem::path path;
    std::shared_ptr<utils::ZarrArray> array;
    std::array<size_t, 3> shapeZYX{0, 0, 0};
    std::array<size_t, 3> chunksZYX{0, 0, 0};
    double spacing = 1.0;
};

struct LasagnaCubeValues {
    double c000 = 0.0;
    double c001 = 0.0;
    double c010 = 0.0;
    double c011 = 0.0;
    double c100 = 0.0;
    double c101 = 0.0;
    double c110 = 0.0;
    double c111 = 0.0;
};

struct LasagnaCubeRequest {
    bool valid = false;
    bool singleChunk = false;
    size_t z0 = 0;
    size_t y0 = 0;
    size_t x0 = 0;
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    std::array<LasagnaChannelChunkKey, 8> keys{};
    std::array<std::shared_ptr<const LasagnaCachedChunk>, 8> chunks{};
};

class LasagnaChannelChunkCache {
public:
    using ResolvedChunkMap = std::unordered_map<
        LasagnaChannelChunkKey,
        std::shared_ptr<const LasagnaCachedChunk>,
        LasagnaChannelChunkKeyHash>;
    using PrefetchRequest = std::pair<const LasagnaChannelBinding*, LasagnaChannelChunkKey>;

    explicit LasagnaChannelChunkCache(size_t capacityBytes);

    [[nodiscard]] std::shared_ptr<const LasagnaCachedChunk> get(
        const LasagnaChannelBinding& binding,
        const utils::ZarrArray& array,
        const LasagnaChannelChunkKey& key) const;

    [[nodiscard]] NormalPrefetchReport prefetchResolved(
        const LasagnaChannelBinding& binding,
        const utils::ZarrArray& array,
        const std::vector<LasagnaChannelChunkKey>& keys,
        size_t maxWorkers,
        ResolvedChunkMap& resolved) const;

    [[nodiscard]] NormalPrefetchReport prefetchInterleaved(
        const std::vector<PrefetchRequest>& requests) const;

private:
    struct Entry {
        std::shared_ptr<const LasagnaCachedChunk> bytes;
        std::list<LasagnaChannelChunkKey>::iterator lruIt;
    };

    struct InFlightLoad;

    [[nodiscard]] std::shared_ptr<const LasagnaCachedChunk> load(
        const LasagnaChannelBinding& binding,
        const utils::ZarrArray& array,
        const LasagnaChannelChunkKey& key) const;

    void store(LasagnaChannelChunkKey key,
               std::shared_ptr<const LasagnaCachedChunk> bytes) const;
    void trim() const;

    size_t capacityBytes_ = 512ULL * 1024ULL * 1024ULL;
    mutable size_t cachedBytes_ = 0;
    mutable std::shared_mutex mutex_;
    mutable std::list<LasagnaChannelChunkKey> lru_;
    mutable std::unordered_map<LasagnaChannelChunkKey, Entry, LasagnaChannelChunkKeyHash> entries_;
    mutable std::unordered_map<
        LasagnaChannelChunkKey,
        std::shared_ptr<InFlightLoad>,
        LasagnaChannelChunkKeyHash> inFlight_;
};

class LasagnaLocalChunkResolver {
public:
    LasagnaLocalChunkResolver(
        const LasagnaChannelBinding& binding,
        const LasagnaChannelChunkCache& cache);

    void resolve(LasagnaCubeRequest& request);

private:
    [[nodiscard]] std::shared_ptr<const LasagnaCachedChunk> resolveKey(
        const LasagnaChannelChunkKey& key);

    const LasagnaChannelBinding* binding_ = nullptr;
    const LasagnaChannelChunkCache* cache_ = nullptr;
    LasagnaChannelChunkKey lastKey_{};
    std::shared_ptr<const LasagnaCachedChunk> lastChunk_;
    bool hasLast_ = false;
    std::array<LasagnaChannelChunkKey, 16> keys_{};
    std::array<std::shared_ptr<const LasagnaCachedChunk>, 16> chunks_{};
    size_t size_ = 0;
    size_t next_ = 0;
};

struct LasagnaPreparedCompactPoint {
    LasagnaCubeRequest nx;
    LasagnaCubeRequest ny;
};

struct LasagnaCornerSample {
    std::array<uint8_t, 8> values{};
    cv::Vec3f fractionXYZ{0.0f, 0.0f, 0.0f};
    bool valid = false;
};

struct LasagnaCornerBatch {
    std::vector<std::vector<std::array<uint8_t, 8>>> values;
    std::vector<cv::Vec3f> fractionsXYZ;
    std::vector<uint8_t> valid;
};

using LasagnaCornerPointVisitor = void (*)(
    void* context,
    size_t pointIndex,
    const cv::Vec3f& fractionXYZ,
    bool valid,
    std::span<const std::array<uint8_t, 8>> volumeCorners);

// Fetches the ordered eight nearest-neighbor voxel corners through VC3D's
// blocking requested-level reader for caller-side interpolation.
class LasagnaChannelCornerSampler {
public:
    explicit LasagnaChannelCornerSampler(
        const LasagnaChannelBinding& binding);
    ~LasagnaChannelCornerSampler();

    LasagnaChannelCornerSampler(const LasagnaChannelCornerSampler&) = delete;
    LasagnaChannelCornerSampler& operator=(const LasagnaChannelCornerSampler&) = delete;
    LasagnaChannelCornerSampler(LasagnaChannelCornerSampler&&) noexcept;
    LasagnaChannelCornerSampler& operator=(LasagnaChannelCornerSampler&&) noexcept;

    [[nodiscard]] NormalPrefetchReport sampleBatch(
        const std::vector<cv::Vec3f>& volumePoints,
        std::vector<LasagnaCornerSample>& samples) const;

private:
    friend NormalPrefetchReport sampleLasagnaChannelCornerBatch(
        const std::vector<const LasagnaChannelCornerSampler*>& samplers,
        const std::vector<cv::Vec3f>& volumePoints,
        std::vector<std::vector<LasagnaCornerSample>>& samples,
        int parallelThreads);
    friend NormalPrefetchReport sampleLasagnaChannelCornerBatch(
        const std::vector<const LasagnaChannelCornerSampler*>& samplers,
        const std::vector<cv::Vec3f>& volumePoints,
        LasagnaCornerBatch& samples,
        int parallelThreads);
    friend NormalPrefetchReport visitLasagnaChannelCorners(
        const std::vector<const LasagnaChannelCornerSampler*>& samplers,
        const std::vector<cv::Vec3f>& volumePoints,
        void* visitorContext,
        LasagnaCornerPointVisitor visitor,
        int parallelThreads,
        bool collectLocalityStats);

    class Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] NormalPrefetchReport sampleLasagnaChannelCornerBatch(
    const std::vector<const LasagnaChannelCornerSampler*>& samplers,
    const std::vector<cv::Vec3f>& volumePoints,
    std::vector<std::vector<LasagnaCornerSample>>& samples,
    int parallelThreads = 0);

[[nodiscard]] NormalPrefetchReport sampleLasagnaChannelCornerBatch(
    const std::vector<const LasagnaChannelCornerSampler*>& samplers,
    const std::vector<cv::Vec3f>& volumePoints,
    LasagnaCornerBatch& samples,
    int parallelThreads = 0);

[[nodiscard]] NormalPrefetchReport visitLasagnaChannelCorners(
    const std::vector<const LasagnaChannelCornerSampler*>& samplers,
    const std::vector<cv::Vec3f>& volumePoints,
    void* visitorContext,
    LasagnaCornerPointVisitor visitor,
    int parallelThreads = 0,
    bool collectLocalityStats = false);

[[nodiscard]] size_t lasagnaReadWorkersPerChannel();
[[nodiscard]] std::shared_ptr<LasagnaChannelChunkCache>
sharedLasagnaChannelChunkCache(size_t capacityBytes);

[[nodiscard]] double decodeCompactNormalComponent(double raw);
[[nodiscard]] cv::Vec3d decodeCompactNormalFromRaw(double rawNx, double rawNy);
[[nodiscard]] cv::Vec3d principalCompactTensorAxis(
    const cv::Matx33d& tensor,
    const cv::Vec3d& hint);

[[nodiscard]] float interpolateLasagnaCorners(
    const LasagnaCornerSample& sample);
[[nodiscard]] std::array<float, 8> lasagnaCornerWeights(
    const cv::Vec3f& fractionXYZ);
[[nodiscard]] float interpolateLasagnaCorners(
    const std::array<uint8_t, 8>& values,
    const std::array<float, 8>& weights);

[[nodiscard]] cv::Vec3f interpolateLasagnaCompactAxisCorners(
    const LasagnaCornerSample& nx,
    const LasagnaCornerSample& ny,
    const cv::Vec3f& hint = {0.0f, 0.0f, 0.0f});
[[nodiscard]] cv::Vec3f interpolateLasagnaCompactAxisCorners(
    const std::array<uint8_t, 8>& nx,
    const std::array<uint8_t, 8>& ny,
    const std::array<float, 8>& weights,
    const cv::Vec3f& hint = {0.0f, 0.0f, 0.0f});

[[nodiscard]] LasagnaChannelBinding bindLasagnaChannel(
    const LasagnaDatasetManifest& manifest,
    std::string_view channel);

[[nodiscard]] LasagnaCubeRequest prepareLasagnaCubeRequest(
    const LasagnaChannelBinding& binding,
    const cv::Vec3d& volumePoint);

[[nodiscard]] bool sameLasagnaSamplingGrid(
    const LasagnaChannelBinding& a,
    const LasagnaChannelBinding& b);

[[nodiscard]] LasagnaCubeRequest cloneLasagnaCubeRequestForBinding(
    const LasagnaCubeRequest& source,
    const LasagnaChannelBinding& binding);

void assignResolvedLasagnaCubeRequestChunks(
    LasagnaCubeRequest& request,
    const LasagnaChannelChunkCache::ResolvedChunkMap& resolved);

void appendUniqueLasagnaCubeRequestChunkKeys(
    const LasagnaCubeRequest& request,
    std::vector<LasagnaChannelChunkKey>& keys);

void deduplicateLasagnaChunkKeysInPlace(
    std::vector<LasagnaChannelChunkKey>& keys);

void appendLasagnaInterpolationChunkKeys(
    const LasagnaChannelBinding& binding,
    const cv::Vec3d& volumePoint,
    std::vector<LasagnaChannelChunkKey>& keys);

[[nodiscard]] std::optional<double> sampleLasagnaChannel(
    const LasagnaChannelBinding& binding,
    const LasagnaChannelChunkCache& cache,
    const cv::Vec3d& volumePoint);

[[nodiscard]] std::optional<double> sampleLasagnaChannel(
    const LasagnaChannelBinding& binding,
    const LasagnaCubeRequest& request);

[[nodiscard]] std::optional<cv::Vec3d> sampleLasagnaCompactAxisTensor(
    const LasagnaChannelBinding& nxBinding,
    const LasagnaChannelBinding& nyBinding,
    const LasagnaChannelChunkCache& cache,
    const cv::Vec3d& volumePoint);

[[nodiscard]] std::optional<cv::Vec3d> sampleLasagnaCompactAxisTensor(
    const LasagnaChannelBinding& nxBinding,
    const LasagnaChannelBinding& nyBinding,
    const LasagnaCubeRequest& nxRequest,
    const LasagnaCubeRequest& nyRequest);

} // namespace vc::lasagna
