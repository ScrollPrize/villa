#pragma once

#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/fiber_tracer/FiberGraph.hpp"
#include "vc/fiber_tracer/FiberPaths.hpp"
#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vc::fiber_tracer
{

class FiberReplayTubeContainmentQuery {
public:
    FiberReplayTubeContainmentQuery(const FiberReplayTubeContainmentQuery&) noexcept = default;
    FiberReplayTubeContainmentQuery(FiberReplayTubeContainmentQuery&&) noexcept = default;
    FiberReplayTubeContainmentQuery& operator=(const FiberReplayTubeContainmentQuery&) noexcept = default;
    FiberReplayTubeContainmentQuery& operator=(FiberReplayTubeContainmentQuery&&) noexcept = default;

    [[nodiscard]] bool containsPredictionPoint(
        const cv::Vec3d& pointPredictionXYZ) const;

private:
    struct Impl;

    explicit FiberReplayTubeContainmentQuery(std::shared_ptr<const Impl> impl) noexcept;

    std::shared_ptr<const Impl> impl_;

    friend struct FiberReplayTube;
};

struct FiberReplayTube {
    PolylineArcGeometry reference;
    double beginArcBase = 0.0;
    double endArcBase = 0.0;
    double radiusBaseVoxels = 0.0;
    std::vector<cv::Vec3d> referenceIntervalBase;
    std::vector<std::array<size_t, 3>> cellsZYX;
    std::array<size_t, 6> volumeCropBaseXYZWHD{0, 0, 0, 0, 0, 0};

    [[nodiscard]] bool containsBasePoint(const cv::Vec3d& point) const;
    [[nodiscard]] double distanceToBasePoint(const cv::Vec3d& point) const;
    [[nodiscard]] bool containsPredictionPoint(const cv::Vec3d& pointPredictionXYZ, double predictionToBaseScale) const;
    [[nodiscard]] FiberReplayTubeContainmentQuery makePredictionContainmentQuery(
        double predictionToBaseScale) const;
};

[[nodiscard]] FiberReplayTube makeFiberReplayTube(
    const std::vector<cv::Vec3d>& referenceLineBase,
    double centerArcBase,
    double alongBaseVoxels,
    double radiusBaseVoxels,
    const FiberPredictionGridInfo& grid,
    int anchorCellSizePredictionVoxels);

enum class FiberReplayTracer {
    Greedy,
    Fiberlet,
};

[[nodiscard]] const char* fiberReplayTracerName(FiberReplayTracer tracer) noexcept;

struct FiberReplayVisualizationInput {
    FiberReplayTracer tracer = FiberReplayTracer::Greedy;
    size_t tracerFailureIndex = 0;
    FiberReplayTube tube;
    FiberAnchorExtractionReport anchors;
    FiberAnchorArtifactInfo anchorArtifact;
    FiberletPathReport paths;
    FiberletArtifactInfo pathArtifact;
};

struct FiberReplayBundleInput {
    FiberReplayTraceRequest request;
    FiberReplayTraceResult greedyReplay;
    FiberletGraphReplayResult fiberletReplay;
    FiberletGraphReplayConfig fiberletReplayConfig;
    std::optional<double> requestedLengthBaseVoxels;
    std::vector<cv::Vec3d> referenceGeometryBase;
    nlohmann::json sources;
    nlohmann::json traceBinding;
    nlohmann::json predictionBinding;
    nlohmann::json requestedTraceConfig;
    nlohmann::json effectiveTraceConfig;
    std::vector<FiberReplayVisualizationInput> visualizations;
};

[[nodiscard]] nlohmann::json writeFiberReplayBundle(const std::filesystem::path& outputDirectory, const FiberReplayBundleInput& input);

}  // namespace vc::fiber_tracer
