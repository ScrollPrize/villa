#pragma once
#include "vc/lasagna/LineViewBuilder.hpp"

#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/fiber_tracer/FiberGraph.hpp"
#include "vc/fiber_tracer/FiberPaths.hpp"
#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

#include <array>
#include <filesystem>
#include <functional>
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

class FiberReplayTubeContainmentQuery {
public:
    FiberReplayTubeContainmentQuery(const FiberReplayTubeContainmentQuery&) noexcept = default;
    FiberReplayTubeContainmentQuery(FiberReplayTubeContainmentQuery&&) noexcept = default;
    FiberReplayTubeContainmentQuery& operator=(const FiberReplayTubeContainmentQuery&) noexcept = default;
    FiberReplayTubeContainmentQuery& operator=(FiberReplayTubeContainmentQuery&&) noexcept = default;

    [[nodiscard]] bool containsPredictionPoint(
        const cv::Vec3d& pointPredictionXYZ) const;
    [[nodiscard]] bool intersectsPredictionCell(
        const std::array<size_t, 3>& cellZYX,
        const FiberPredictionGridInfo& grid,
        int anchorCellSizePredictionVoxels) const;
    [[nodiscard]] FiberAnchorRetainEvaluation evaluatePredictionAnchor(
        const FiberAnchor& anchor) const;

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

struct FiberReplayStripComponent {
    size_t sourceSegmentIndex = 0;
    std::vector<cv::Vec3d> centerlineBaseXYZ;
    std::shared_ptr<QuadSurface> lineSurface;
    cv::Mat_<uint8_t> texture;
    vc::lasagna::LineStripPositionMap stripPositionMap;
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
    // Original traced line + inverse display map; never bridge restart gaps.
    std::vector<cv::Vec3d> pointsBase;
    std::vector<double> referenceArcsBase;
    vc::lasagna::LineStripPositionMap positionMap;
    int topSourceColumns = 0;
};

struct FiberReplayStripPoint {
    cv::Vec3d pointBase, tangentBase;
    double referenceArcBase = 0, originalPosition = 0;
};
[[nodiscard]] FiberReplayStripPoint sampleFiberReplayStripColumn(
    const FiberReplayOverviewFiberletComponent& component, double assembledColumn);

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
    int anchorCellSizePredictionVoxels,
    bool collectAnchorCells = true);

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

// Standalone native replay inspection, without graph/fiberlet artifacts.
[[nodiscard]] nlohmann::json fiberReplayTraceJson(
    const FiberReplayTraceResult& replay, double normalThresholdBaseVoxels);
[[nodiscard]] FiberReplayTraceResult fiberReplayTraceFromJson(const nlohmann::json& document);
void writeFiberReplayTraceArtifacts(
    const std::filesystem::path& outputDirectory,
    const std::vector<cv::Vec3d>& referenceGeometryBase,
    const FiberReplayTraceResult& replay,
    double normalThresholdBaseVoxels,
    const nlohmann::json& provenance);
[[nodiscard]] FiberReplayOverview renderFiberReplayTraceOverview(
    const std::vector<cv::Vec3d>& referenceGeometryBase,
    const FiberReplayTraceResult& replay,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale, int parallelThreads,
    ::Volume& volume, const std::string& sourceLocator);

// Coordinates supplied to the texture callback are in the declared source
// lattice, resampled at native spacing. The callback applies renderScale.
using FiberReplayTextureRenderer = std::function<cv::Mat_<uint8_t>(
    const cv::Mat_<cv::Vec3f>&, int renderScale)>;
[[nodiscard]] FiberReplayOverview renderFiberReplayTraceOverview(
    const std::vector<cv::Vec3d>& referenceGeometryBase,
    const FiberReplayTraceResult& replay,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale, int parallelThreads,
    const FiberReplayStripTextureSource& source,
    const FiberReplayTextureRenderer& renderTexture,
    int renderScale = 8, bool labelRestarts = false, double baseSamplesPerPixel = 0);

struct FiberReplayInspection {
    std::vector<cv::Mat_<cv::Vec3b>> pages;
    nlohmann::json metadata;
};

struct FiberReplaySectionFrame {
    cv::Vec3d horizontal, vertical, tangent;
    double horizontalRadius, verticalRadius;
};
[[nodiscard]] FiberReplaySectionFrame fiberReplaySectionFrame(
    const cv::Vec3d& center, const cv::Vec3d& tangent,
    const vc::lasagna::NormalSampler& normals, double workingToBaseScale, double thresholdBase,
    double tangentialFactor = kFiberReplayTangentialThresholdFactor);

// Aligned CT/presence strips and transverse acceptance cross-sections.
[[nodiscard]] FiberReplayInspection renderFiberReplayInspection(
    const std::vector<cv::Vec3d>& referenceGeometryBase,
    const FiberReplayTraceResult& replay,
    const vc::lasagna::NormalSampler& normals, double workingToBaseScale,
    const FiberReplayStripTextureSource& source,
    const FiberReplayTextureRenderer& ct, const FiberReplayTextureRenderer& presence,
    double thresholdBase, bool reverse, double voxelSizeUm,
    const std::string& ctSamplingDescription = "CT: trilinear interpolation",
    // When positive, each presence strip is followed by the same strip
    // thresholded at this stored presence level, which is what the skeleton
    // backend actually builds its topology from. Zero omits those rows.
    int presenceThresholdU8 = 0);

// Uses the existing prediction source (including its readiness gate). Multiple
// prediction options are visualized as maximum presence, not direction validity.
// This uses complete presence/nx/ny products, not standalone presence arrays.
[[nodiscard]] cv::Mat_<uint8_t> renderFiberReplayPresenceTexture(
    const cv::Mat_<cv::Vec3f>& sourceCoordinates, int renderScale,
    const FiberReplayStripTextureSource& source, double workingToBaseScale,
    const FiberPredictionSource& predictions);

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
    nlohmann::json fiberletEvaluationProfile;
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
