#pragma once

#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LineModel.hpp"

#include <vc_reader/zarr.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core/types.hpp>

namespace vc::lasagna {

// The channel-group arrays of one dataset, each opened once through vc_reader
// (openLasagnaGroupArray) and sharing one memory cache. Channels of one packed
// group share its array.
struct LasagnaGroupArrays {
    explicit LasagnaGroupArrays(std::size_t memoryCacheBytes)
        : cache(std::make_shared<vcr::MemoryCache>(memoryCacheBytes)) {}
    explicit LasagnaGroupArrays(std::shared_ptr<vcr::MemoryCache> sharedCache)
        : cache(std::move(sharedCache)) {}
    struct Opened {
        std::shared_ptr<vcr::Array> array;
        vcr::zarr::ArrayMeta meta;
    };
    std::shared_ptr<vcr::MemoryCache> cache;
    std::map<const LasagnaChannelGroup*, Opened> arrays;
};

// One channel of a dataset in working coordinates: working point p (xyz) lies
// at channel voxel p / spacing.
struct LasagnaChannelBinding {
    const LasagnaChannelGroup* group = nullptr;  // in the manifest it was bound from
    size_t channelIndex = 0;
    // The group array: (Z,Y,X), or (C,Z,Y,X) holding every channel of the group.
    std::shared_ptr<vcr::Array> array;
    bool packed = false;
    bool float32 = false;  // stored as float32; otherwise uint8
    std::array<size_t, 3> shapeZYX{0, 0, 0};
    std::array<size_t, 3> chunksZYX{0, 0, 0};
    double spacing = 1.0;
};

[[nodiscard]] LasagnaChannelBinding bindLasagnaChannel(
    const LasagnaDatasetManifest& manifest,
    std::string_view channel,
    LasagnaGroupArrays& arrays,
    bool allowFloat32 = false);

[[nodiscard]] std::vector<LasagnaChannelBinding> bindLasagnaEmbeddingChannels(
    const LasagnaDatasetManifest& manifest, const LasagnaChannelBinding& lattice,
    LasagnaGroupArrays& arrays);

[[nodiscard]] bool sameLasagnaSamplingGrid(
    const LasagnaChannelBinding& a,
    const LasagnaChannelBinding& b);

// Trilinear cubes. A point's cube is the voxel below it (floor of p / spacing)
// and the seven above; corner k adds (k & 1, k >> 1 & 1, k >> 2) to (x, y, z),
// clamped to the last voxel. Points that are not finite or lie outside
// [0, n - 1] on an axis have none. Each binding's corners are one
// Array::read_points call; chunks that are not stored read as the fill value.

// Byte channels (uint8, or float32 holding integers in [0, 255]) in float
// geometry (p / float(spacing)), for the tracer's hot paths. All bindings must
// share one sampling grid.
struct LasagnaCornerBatch {
    std::vector<std::vector<std::array<uint8_t, 8>>> values;  // [binding][point]
    std::vector<cv::Vec3f> fractionsXYZ;
    std::vector<uint8_t> valid;
};

using LasagnaCornerPointVisitor = void (*)(
    void* context,
    size_t pointIndex,
    const cv::Vec3f& fractionXYZ,
    bool valid,
    std::span<const std::array<uint8_t, 8>> volumeCorners);

void sampleLasagnaChannelCornerBatch(
    const std::vector<const LasagnaChannelBinding*>& bindings,
    const std::vector<cv::Vec3f>& volumePoints,
    LasagnaCornerBatch& samples);

// Calls `visitor` for every point (concurrently on up to parallelThreads threads).
void visitLasagnaChannelCorners(
    const std::vector<const LasagnaChannelBinding*>& bindings,
    const std::vector<cv::Vec3f>& volumePoints,
    void* visitorContext,
    LasagnaCornerPointVisitor visitor,
    int parallelThreads,
    LasagnaCornerBatch& scratch);

// Any channel in double geometry (p / spacing).
struct LasagnaCubes {
    std::vector<std::array<double, 8>> values;
    std::vector<cv::Vec3d> fractionsXYZ;
    std::vector<uint8_t> valid;
};

void readLasagnaCubes(
    const LasagnaChannelBinding& binding,
    std::span<const cv::Vec3d> volumePoints,
    LasagnaCubes& cubes);

[[nodiscard]] double interpolateLasagnaCube(
    const std::array<double, 8>& values, const cv::Vec3d& fractionXYZ);

// Compact-normal (nx, ny byte) cubes to their weighted principal axis; corners
// whose `validity` value is zero are left out. nullopt when no corner counts.
[[nodiscard]] std::optional<cv::Vec3d> lasagnaCompactAxisCube(
    const std::array<double, 8>& nx,
    const std::array<double, 8>& ny,
    const cv::Vec3d& fractionXYZ,
    const std::array<double, 8>* validity = nullptr);

[[nodiscard]] std::optional<double> sampleLasagnaChannel(
    const LasagnaChannelBinding& binding,
    const cv::Vec3d& volumePoint);

[[nodiscard]] std::optional<cv::Vec3d> sampleLasagnaCompactAxisTensor(
    const LasagnaChannelBinding& nxBinding,
    const LasagnaChannelBinding& nyBinding,
    const cv::Vec3d& volumePoint, const LasagnaChannelBinding* validity = nullptr);

// Interpolate each component before optional L2 normalization; empty denotes
// invalid (or normalized zero). With normalize=false retain physical
// scalar/vector units, including valid zero. With `validity`, corners whose
// validity value is zero are left out of each component's interpolation.
void sampleLasagnaVectorBatch(const std::vector<LasagnaChannelBinding>& channels,
    const std::vector<cv::Vec3f>& points, std::vector<std::vector<float>>& vectors,
    bool normalize = true, const LasagnaChannelBinding* validity = nullptr);

[[nodiscard]] double decodeCompactNormalComponent(double raw);
[[nodiscard]] cv::Vec3d decodeCompactNormalFromRaw(double rawNx, double rawNy);
[[nodiscard]] std::optional<std::array<uint8_t, 2>>
encodeCompactNormalToRaw(const cv::Vec3d& normal);
[[nodiscard]] cv::Vec3d principalCompactTensorAxis(
    const cv::Matx33d& tensor,
    const cv::Vec3d& hint);

[[nodiscard]] std::array<float, 8> lasagnaCornerWeights(
    const cv::Vec3f& fractionXYZ);
[[nodiscard]] std::array<float, 8> validLasagnaCornerWeights(
    const std::array<float, 8>& weights, const std::array<uint8_t, 8>& validity);
[[nodiscard]] float interpolateLasagnaCorners(
    const std::array<uint8_t, 8>& values,
    const std::array<float, 8>& weights);

[[nodiscard]] cv::Vec3f interpolateLasagnaCompactAxisCorners(
    const std::array<uint8_t, 8>& nx,
    const std::array<uint8_t, 8>& ny,
    const std::array<float, 8>& weights,
    const cv::Vec3f& hint = {0.0f, 0.0f, 0.0f}, bool vectorized = false);

} // namespace vc::lasagna
