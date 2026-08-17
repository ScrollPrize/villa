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

struct FiberReplayOverviewPanel {
    double progressFractionBegin = 0.0;
    double progressFractionEnd = 0.0;
    std::array<int, 2> referenceTopColumns{};
    std::array<int, 2> referenceSideColumns{};
    std::array<int, 2> fiberletTopColumns{};
    std::array<int, 2> fiberletSideColumns{};
    std::array<int, 2> referenceTopRows{};
    std::array<int, 2> referenceSideRows{};
    std::array<int, 2> fiberletTopRows{};
    std::array<int, 2> fiberletSideRows{};
};

struct FiberReplayOverviewPage {
    cv::Mat_<cv::Vec3b> image;
    std::vector<FiberReplayOverviewPanel> panels;
};

struct FiberReplayOverviewFiberletComponent {
    size_t sourceSegmentIndex = 0;
    double referenceArcBeginBase = 0.0;
    double referenceArcEndBase = 0.0;
    std::array<int, 2> topColumns{};
    std::array<int, 2> sideColumns{};
    std::array<int, 2> topRows{};
    std::array<int, 2> sideRows{};
};

struct FiberReplayOverview {
    FiberReplayStripTextureSource textureSource;
    std::array<int, 2> referenceTopShapeYX{};
    std::array<int, 2> referenceSideShapeYX{};
    std::array<int, 2> fiberletTopShapeYX{};
    std::array<int, 2> fiberletSideShapeYX{};
    int renderScale = 1;
    int markerWidthPixels = 0;
    int fiberletComponentGapColumns = 0;
    std::vector<FiberReplayOverviewFiberletComponent> fiberletComponents;
    std::vector<FiberReplayOverviewPage> pages;
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
    const std::string& sourceLocator);

[[nodiscard]] FiberReplayOverview renderFiberReplayOverview(
    const std::vector<cv::Vec3d>& referenceGeometryBase,
    const FiberReplayTraceResult& greedyReplay,
    const FiberletGraphReplayResult& fiberletReplay,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    int parallelThreads,
    ::Volume& volume,
    const std::string& sourceLocator);

[[nodiscard]] FiberReplayStripTextureSource validateFiberReplayStripCtVolume(
    ::Volume& volume,
    const std::string& sourceLocator);

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
    std::optional<FiberReplayOverview> overview;
    std::vector<FiberReplayVisualizationInput> visualizations;
};

[[nodiscard]] nlohmann::json writeFiberReplayBundle(const std::filesystem::path& outputDirectory, const FiberReplayBundleInput& input);

#ifdef VC_TESTING
namespace testing
{
[[nodiscard]] FiberReplayOverview composeFiberReplayOverviewForTesting(
    const cv::Mat_<cv::Vec3b>& referenceTop,
    const cv::Mat_<cv::Vec3b>& referenceSide,
    const cv::Mat_<cv::Vec3b>& fiberletTop,
    const cv::Mat_<cv::Vec3b>& fiberletSide,
    int maximumPageRows = 65000);
}  // namespace testing
#endif

}  // namespace vc::fiber_tracer
