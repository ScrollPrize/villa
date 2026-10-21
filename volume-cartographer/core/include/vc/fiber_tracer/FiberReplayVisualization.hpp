#pragma once
#include "vc/lasagna/LineViewBuilder.hpp"
#include "vc/fiber_tracer/FiberGraph.hpp"
#include "vc/fiber_tracer/FiberTrace.hpp"
#include <functional>
#include <filesystem>
#include <opencv2/core/mat.hpp>
class Volume;
namespace vc::fiber_tracer {
struct FiberReplayStripTextureSource {
    std::string locator;
    std::array<int, 3> shapeZYX{};
    std::array<double, 3> scaleFromBaseXYZ{};
    std::array<double, 3> offsetFromBaseXYZ{};
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
[[nodiscard]] FiberReplayOverview renderFiberReplayOverview(
    const std::vector<cv::Vec3d>& referenceGeometryBase,
    const FiberReplayTraceResult& greedyReplay,
    const FiberletGraphReplayResult& fiberletReplay,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    int parallelThreads,
    ::Volume& volume,
    const std::string& sourceLocator);
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
