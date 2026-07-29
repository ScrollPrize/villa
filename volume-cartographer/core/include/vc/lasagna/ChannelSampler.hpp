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
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <opencv2/core/types.hpp>

namespace utils { class ZarrArray; }

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
    bool hasChannelDimension = false;
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

struct LasagnaPreparedCompactPoint {
    LasagnaCubeRequest nx;
    LasagnaCubeRequest ny;
};

[[nodiscard]] size_t lasagnaReadWorkersPerChannel();
[[nodiscard]] std::shared_ptr<LasagnaChannelChunkCache>
sharedLasagnaChannelChunkCache(size_t capacityBytes);

[[nodiscard]] double decodeCompactNormalComponent(double raw);
[[nodiscard]] cv::Vec3d decodeCompactNormalFromRaw(double rawNx, double rawNy);
[[nodiscard]] cv::Vec3d principalCompactTensorAxis(
    const cv::Matx33d& tensor,
    const cv::Vec3d& hint);

[[nodiscard]] LasagnaChannelBinding bindLasagnaChannel(
    const LasagnaDatasetManifest& manifest,
    std::string_view channel);

[[nodiscard]] LasagnaCubeRequest prepareLasagnaCubeRequest(
    const LasagnaChannelBinding& binding,
    const cv::Vec3d& volumePoint);

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
