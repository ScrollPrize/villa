#pragma once

#include "vc/fiber_tracer/FiberGraph.hpp"
#include "vc/fiber_tracer/FiberTrace.hpp"

#include <array>
#include <cstddef>
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
    bool reverseConsistency = false; // Dense backend only; graph rejects it.
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

} // namespace vc::fiber_tracer
