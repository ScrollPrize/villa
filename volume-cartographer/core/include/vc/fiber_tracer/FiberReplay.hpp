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
#include <opencv2/core/mat.hpp>

class Volume;
class QuadSurface;

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
    [[nodiscard]] bool containsPredictionPoint(const cv::Vec3d& pointPredictionXYZ, double predictionToBaseScale) const;
};

struct FiberReplayStripComponent {
    size_t sourceSegmentIndex = 0;
    std::vector<cv::Vec3d> centerlineBaseXYZ;
    std::shared_ptr<QuadSurface> lineSurface;
    cv::Mat_<uint8_t> texture;
};

struct FiberReplayStripTextureSource {
    std::string locator;
    int renderScale = 1;
    std::array<int, 3> shapeZYX{};
    std::array<double, 3> scaleFromBaseXYZ{};
    std::array<double, 3> offsetFromBaseXYZ{};
};

struct FiberReplayStripMeshes {
    std::vector<FiberReplayStripComponent> reference;
    std::vector<FiberReplayStripComponent> greedy;
    std::vector<FiberReplayStripComponent> fiberlet;
    std::optional<FiberReplayStripTextureSource> textureSource;
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

[[nodiscard]] FiberReplayStripMeshes makeFiberReplayStripSurfaces(
    const FiberReplayTube& tube,
    const FiberReplayTraceResult& greedyReplay,
    const FiberletGraphReplayResult& fiberletReplay,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    int parallelThreads);

void renderFiberReplayStripTextures(
    FiberReplayStripMeshes& meshes,
    ::Volume& volume,
    const std::string& sourceLocator,
    int renderScale);

[[nodiscard]] FiberReplayStripTextureSource validateFiberReplayStripCtVolume(
    ::Volume& volume,
    const std::string& sourceLocator,
    int renderScale);

struct FiberReplayVisualizationInput {
    FiberReplayTracer tracer = FiberReplayTracer::Greedy;
    size_t tracerFailureIndex = 0;
    FiberReplayTube tube;
    FiberAnchorExtractionReport anchors;
    FiberAnchorArtifactInfo anchorArtifact;
    FiberletPathReport paths;
    FiberletArtifactInfo pathArtifact;
    std::optional<FiberReplayStripMeshes> strips;
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
