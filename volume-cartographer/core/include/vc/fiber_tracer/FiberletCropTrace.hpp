#pragma once

#include "vc/fiber_tracer/FiberGraph.hpp"
#include "vc/fiber_tracer/FiberTrace.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

struct FiberletCropTraceConfig {
    cv::Vec3d minimumBaseXYZ{0.0, 0.0, 0.0};
    cv::Vec3d maximumBaseXYZ{0.0, 0.0, 0.0};
    std::size_t beamWidth = 16;
    double lookaheadDistanceBaseVoxels = 384.0;
    std::size_t maximumGeneratedStatesPerStep = 1'000'000;
    std::size_t maximumFiberletsPerSide = 100'000;
    double coverageNormalRadiusBaseVoxels = 20.0;
    std::optional<double> coverageTangentialRadiusBaseVoxels;
    bool reverseConsistency = false;
    double reverseNormalRadiusBaseVoxels = 20.0;
    double reverseTangentialRadiusBaseVoxels = 40.0;
    double reverseBackoffBaseVoxels = 40.0;
    // Zero preserves endpoint/backoff filtering. Positive enables full-prefix
    // checks at this spacing and every terminal endpoint, without GT reseeding.
    double reversePrefixIntervalBaseVoxels = 0.0;
    double coverageDirectionDegrees = 25.0;
    bool stopAtCoveredAnchors = false;
    bool stopAtCoveredTraces = false; // Dense only; indexed accepted segments.
    std::optional<double> coveredTraceNormalRadiusBaseVoxels;
    std::optional<double> coveredTraceTangentialRadiusBaseVoxels;
    std::optional<double> ambiguityRelativeCostMargin;
    double ambiguityNormalRadiusBaseVoxels = 20.0;
    std::optional<double> maximumAcceptedCostDensity;
    std::size_t parallelThreads = 0;
    std::size_t maximumAttempts = 0;
    std::size_t maximumFibers = 0;
    bool captureStepCosts = false; // Optional segment costs persisted by trace artifacts.
};

struct FiberletCropTraceSearchBox {
    cv::Vec3d minimumBaseXYZ{0.0, 0.0, 0.0};
    cv::Vec3d maximumBaseXYZ{0.0, 0.0, 0.0};
};

[[nodiscard]] double fiberletCropCoverageTangentialRadius(const FiberletCropTraceConfig& config);

[[nodiscard]] FiberletCropTraceSearchBox fiberletCropTraceSearchBox(
    const FiberletCropTraceConfig& config);

struct FiberletCropTraceLine {
    FiberletStorageKey seed;
    cv::Vec3d seedBaseXYZ{0.0, 0.0, 0.0};
    std::optional<cv::Vec3d> originalSeedBaseXYZ;
    float seedPresence = 0.0F;
  double totalMetricCost = 0.0;
  double pathLengthPredictionVoxels = 0.0;
    std::vector<cv::Vec3d> pointsBaseXYZ;
    std::vector<double> segmentMetricCosts;
    std::vector<double> segmentOriginalStepCosts;
    std::string negativeTermination;
    std::string positiveTermination;
    std::size_t negativeFiberlets = 0;
    std::size_t positiveFiberlets = 0;
};

struct FiberletCropTraceResult {
    std::vector<FiberletCropTraceLine> lines;
    std::size_t candidateAnchors = 0;
    std::size_t computedCandidates = 0;
    std::size_t discardedCandidates = 0;
    std::size_t attemptedAnchors = 0;
    std::size_t coveredAnchors = 0;
    std::size_t noEdgeAnchors = 0;
    std::size_t oneSidedLines = 0;
    std::size_t bidirectionalLines = 0;
    std::size_t coveredAnchorStops = 0;
    std::size_t coveredTraceStops = 0;
    std::size_t coverageCommitRechecks = 0;
    std::size_t coverageQueries = 0;
    std::size_t coverageSegmentTests = 0;
    std::size_t ambiguityDecisions = 0;
    std::size_t ambiguityRouteComparisons = 0;
    std::size_t acceptedAmbiguityStops = 0;
    std::optional<double> minimumAmbiguityRelativeCostGap;
    double maximumAmbiguityThresholdRatio = 0.0;
    std::size_t qualityRejectedAnchors = 0;
    std::size_t reverseAttempts = 0;
    std::size_t reverseFailures = 0;
    double reverseSkippedLengthBaseVoxels = 0.0;
    double candidateBatchSeconds = 0.0;
    double candidateBatchCpuSeconds = 0.0;
    double candidateTaskSeconds = 0.0;
    double maximumCandidateTaskSeconds = 0.0;
    double integrationSeconds = 0.0;
    std::size_t maximumLookaheadRouteNodes = 0;
    std::size_t maximumLookaheadRouteBytes = 0;
};

enum class FiberDirectionGroup {
    Direction1,
    Direction2,
    Mixed,
};

inline constexpr double kFiberDirectionDominanceFraction = 0.75;

struct FiberDirectionLineClassification {
    FiberDirectionGroup group = FiberDirectionGroup::Mixed;
    double direction1SupportBaseVoxels = 0.0;
    double direction2SupportBaseVoxels = 0.0;
    double totalLengthBaseVoxels = 0.0;
};

struct FiberDirectionClassification {
    cv::Vec3d direction1BaseXYZ{1.0, 0.0, 0.0};
    cv::Vec3d direction2BaseXYZ{0.0, 1.0, 0.0};
    double dominanceFraction = kFiberDirectionDominanceFraction;
    std::vector<FiberDirectionLineClassification> lines;
    std::array<std::size_t, 3> groupCounts{0, 0, 0};
    std::size_t analyzedSteps = 0;
    double analyzedLengthBaseVoxels = 0.0;
};

struct FiberDirectionAblationCandidate {
    std::size_t lineIndex = 0;
    double confidence = 0.0;
};

[[nodiscard]] std::vector<FiberDirectionAblationCandidate>
rankMixedFiberDirections(const FiberDirectionClassification& classification);

struct FiberDirectionObjPaths {
    std::filesystem::path all;
    std::filesystem::path direction1;
    std::filesystem::path direction2;
    std::filesystem::path mixed;
    std::filesystem::path allAnchors;
    std::filesystem::path direction1Anchors;
    std::filesystem::path direction2Anchors;
    std::filesystem::path mixedAnchors;
};

struct FiberQualityHistogramBin {
  std::vector<std::size_t> lineIndices;
  double minimumTotalMetricCost = 0.0;
  double meanTotalMetricCost = 0.0;
  double maximumTotalMetricCost = 0.0;
  double minimumCostDensity = 0.0;
  double meanCostDensity = 0.0;
  double maximumCostDensity = 0.0;
};

struct FiberQualityHistogram {
  std::array<FiberQualityHistogramBin, 10> bins;
};

struct FiberQualitySelection {
  std::vector<std::size_t> lineIndices;
  std::size_t inputLines = 0;
  double requestedFraction = 1.0;
  std::optional<double> requestedMaximumCostDensity;
  double effectiveFraction = 1.0;
  std::optional<double> maximumRetainedCostDensity;
};

struct FiberQualityObjPaths {
  std::array<std::filesystem::path, 10> deciles;
  std::filesystem::path histogramCsv;
};

struct FiberValueBand {
    std::vector<std::size_t> lineIndices;
    double minimumValue = 0.0;
    double meanValue = 0.0;
    double maximumValue = 0.0;
};

struct FiberValueBands {
    std::array<FiberValueBand, 10> bands;
};

enum class FiberTernaryState : unsigned char {
    Vertical,
    Mixed,
    Horizontal,
    Tie,
};

struct FiberTernaryStateObjPaths {
    std::filesystem::path vertical;
    std::filesystem::path mixed;
    std::filesystem::path horizontal;
    std::filesystem::path tie;
};

struct FiberValueBandObjPaths {
    std::array<std::filesystem::path, 10> bands;
};

using FiberletCropTraceProgress = std::function<void(const FiberletCropTraceResult& result, std::size_t remainingAnchors)>;

// Cost is unavailable (serialized zero sentinel), not a quality score. Preserve
// source order and separate contiguous runs; never connect a crop-external gap.
[[nodiscard]] std::vector<FiberletCropTraceLine> importFiberCropLines(
    std::span<const std::vector<cv::Vec3d>> sourceLines,
    const FiberletCropTraceConfig& config);

// Canonical anchors, deterministic crop scheduling and coverage, dense production beam.
[[nodiscard]] FiberletCropTraceResult traceDenseBeamCrop(
    const FiberPredictionSource& predictions,
    std::vector<FiberletStoredAnchor> anchors,
    double predictionToBaseScale,
    const vc::lasagna::NormalSampler& normalSampler,
    const FiberTraceConfig& denseConfig,
    const FiberletCropTraceConfig& config,
    const FiberletCropTraceProgress& progress = {});

[[nodiscard]] FiberletCropTraceResult traceFiberletCrop(
    const FiberletReplayGraphSource& graph,
    std::vector<FiberletStoredAnchor> anchors,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    const FiberletCropTraceConfig& config,
    const FiberletCropTraceProgress& progress = {});

[[nodiscard]] FiberDirectionClassification classifyFiberletCropDirections(
    const std::vector<FiberletCropTraceLine>& lines,
    double dominanceFraction = kFiberDirectionDominanceFraction);

[[nodiscard]] FiberDirectionObjPaths fiberDirectionObjPaths(
    const std::filesystem::path& allOutputPath);

void writeFiberletCropDirectionObjs(
    const std::vector<FiberletCropTraceLine>& lines,
    const FiberDirectionClassification& classification,
    const std::filesystem::path& allOutputPath);

[[nodiscard]] FiberQualityHistogram
classifyFiberletCropQuality(const std::vector<FiberletCropTraceLine> &lines);

[[nodiscard]] FiberQualitySelection
selectFiberletCropQuality(
    const std::vector<FiberletCropTraceLine>& lines,
    double fraction);

[[nodiscard]] FiberQualitySelection
selectFiberletCropQualityThreshold(
    const std::vector<FiberletCropTraceLine>& lines,
    double maximumCostDensity);

[[nodiscard]] FiberQualityObjPaths
fiberQualityObjPaths(const std::filesystem::path &allOutputPath);

void writeFiberletCropQualityArtifacts(
    const std::vector<FiberletCropTraceLine> &lines,
    const FiberQualityHistogram &histogram,
    const std::filesystem::path &allOutputPath);

[[nodiscard]] FiberValueBands classifyFiberValues(
    std::span<const double> values);

[[nodiscard]] FiberValueBandObjPaths fiberValueBandObjPaths(
    const std::filesystem::path& outputBase);

[[nodiscard]] FiberValueBandObjPaths writeFiberletCropValueBandObjs(
    const std::vector<FiberletCropTraceLine>& lines,
    const FiberValueBands& bands,
    const std::filesystem::path& outputBase);

[[nodiscard]] FiberTernaryStateObjPaths writeFiberletCropTernaryStateObjs(
    const std::vector<FiberletCropTraceLine>& lines,
    std::span<const FiberTernaryState> states,
    const std::filesystem::path& outputBase);

struct FiberPieceOverlapConfig {
    double normalRadiusBase = 10, tangentialRadiusBase = 20;
    double directionDegrees = 45, fraction = 0.8, sampleStepBase = 4;
    int threads = 1;
};
struct FiberPieceOverlapSelection {
    std::vector<std::size_t> retained;
    // Original input ordinals; -1 means retained. Every removed piece has a retained suppressor.
    std::vector<std::int64_t> suppressedBy;
    std::vector<double> overlapFraction, lengthBase;
};
// All strictly positive overlaps, measured from the shorter piece. Independent
// of ranking and suppression threshold, reusable across a parameter sweep.
struct FiberPieceOverlapGraph {
    std::vector<double> lengthBase;
    std::vector<std::vector<std::pair<std::size_t,double>>> shorterPairs;
};
[[nodiscard]] FiberPieceOverlapGraph measureFiberPieceOverlap(
    const std::vector<FiberletCropTraceLine>& lines,
    const vc::lasagna::NormalSampler& normals, double normalScale,
    const FiberPieceOverlapConfig& config,
    std::span<const std::size_t> groups = {}, bool bothDirections = false);
[[nodiscard]] FiberPieceOverlapSelection selectFiberPieceOverlap(
    const FiberPieceOverlapGraph& graph, const std::vector<double>& quality, double fraction);
// Higher quality wins, length breaks ties, then original ordinal. Every input
// line is independent, including siblings produced by reverse consistency.
[[nodiscard]] FiberPieceOverlapSelection selectFiberPieceOverlap(
    const std::vector<FiberletCropTraceLine>& lines, const std::vector<double>& quality,
    const vc::lasagna::NormalSampler& normals, double normalScale,
    const FiberPieceOverlapConfig& config);

}  // namespace vc::fiber_tracer
