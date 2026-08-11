#pragma once

#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/fiber_tracer/FiberPaths.hpp"
#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vc::fiber_tracer
{

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
    [[nodiscard]] bool containsPredictionPoint(
        const cv::Vec3d& pointPredictionXYZ,
        double predictionToBaseScale) const;
};

[[nodiscard]] FiberReplayTube makeFiberReplayTube(
    const std::vector<cv::Vec3d>& referenceLineBase,
    double centerArcBase,
    double alongBaseVoxels,
    double radiusBaseVoxels,
    const FiberPredictionGridInfo& grid,
    int anchorCellSizePredictionVoxels);

struct FiberReplayBundleInput {
    FiberReplayTraceRequest request;
    FiberReplayTraceResult replay;
    std::vector<cv::Vec3d> referenceGeometryBase;
    std::optional<FiberReplayTube> tube;
    nlohmann::json sources;
    nlohmann::json traceBinding;
    nlohmann::json predictionBinding;
    nlohmann::json requestedTraceConfig;
    nlohmann::json effectiveTraceConfig;
    std::optional<FiberAnchorExtractionReport> anchors;
    std::optional<FiberAnchorArtifactInfo> anchorArtifact;
    std::optional<FiberletPathReport> paths;
    std::optional<FiberletArtifactInfo> pathArtifact;
};

[[nodiscard]] nlohmann::json writeFiberReplayBundle(
    const std::filesystem::path& outputDirectory,
    const FiberReplayBundleInput& input);

} // namespace vc::fiber_tracer
