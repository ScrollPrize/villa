#pragma once

#include "vc/fiber_tracer/FiberGraph.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

struct FiberReferenceReplayCase {
    std::size_t sourceIndex = 0;
    std::size_t runIndex = 0;
    bool reverse = false;
    std::vector<cv::Vec3d> pointsBaseXYZ;
    double referenceLengthBaseVoxels = 0.0;
};

struct FiberReferenceReplayFailure {
    std::size_t index = 0;
    std::size_t segmentIndex = 0;
    std::string reason;
    double directionalReferenceArcBaseVoxels = 0.0;
    double directionalReferenceArcFraction = 0.0;
    double sourceReferenceArcBaseVoxels = 0.0;
    double sourceReferenceArcFraction = 0.0;
    double tracedSpanBaseVoxels = 0.0;
    cv::Vec3d referencePointBaseXYZ{0.0, 0.0, 0.0};
    std::optional<cv::Vec3d> evaluatorPointBaseXYZ;
    std::optional<FiberReplayThresholdMeasurement> thresholdMeasurement;
};

struct FiberReferenceReplayOutcome {
    std::size_t sourceIndex = 0;
    std::size_t runIndex = 0;
    bool reverse = false;
    double referenceLengthBaseVoxels = 0.0;
    double tracedLengthBaseVoxels = 0.0;
    double evaluatedThroughBaseVoxels = 0.0;
    std::size_t seededSpans = 0;
    bool evaluationComplete = false;
    bool failureFree = false;
    std::vector<FiberReferenceReplayFailure> failures;
};

struct FiberReferenceReplaySummary {
    std::size_t selectedSources = 0;
    std::size_t sourcesWithoutRuns = 0;
    std::size_t inCropRuns = 0;
    std::size_t directedCases = 0;
    std::size_t evaluatedCases = 0;
    std::size_t incompleteCases = 0;
    std::size_t failureFreeCases = 0;
    std::size_t casesWithFailures = 0;
    std::size_t totalFailures = 0;
    std::size_t seededSpans = 0;
    double undirectedReferenceLengthBaseVoxels = 0.0;
    double directedReferenceLengthBaseVoxels = 0.0;
    double tracedLengthBaseVoxels = 0.0;
    double meanCreditedLengthBaseVoxels = 0.0;
    double meanSeededSpanLengthBaseVoxels = 0.0;
    double meanFailedSpanLengthBaseVoxels = 0.0;
    double baseVoxelSizeUm = 0.0;
    double tracedLengthMillimeters = 0.0;
    double meanCreditedLengthMillimeters = 0.0;
    double meanSeededSpanLengthMillimeters = 0.0;
    double meanFailedSpanLengthMillimeters = 0.0;
    double lengthWeightedSuccessPercent = 0.0;
    double failureFreeCasesPercent = 0.0;
    double failuresPerDirectedMillimeter = 0.0;
    double meanDistancePerFailureBaseVoxels = 0.0;
    double meanDistancePerFailureMillimeters = 0.0;
    double meanDistancePerFailurePercent = 0.0;
    std::vector<std::pair<std::string, std::size_t>> failureReasons;
};

[[nodiscard]] std::vector<FiberReferenceReplayCase> makeFiberReferenceReplayCases(
    std::span<const std::vector<cv::Vec3d>> sourceFibersBaseXYZ, const cv::Vec3d& minimumBaseXYZ, const cv::Vec3d& maximumBaseXYZ);

[[nodiscard]] FiberReferenceReplayOutcome measureFiberReferenceReplayOutcome(const FiberReferenceReplayCase& replayCase, const FiberletGraphReplayResult& replay);

[[nodiscard]] FiberReferenceReplaySummary summarizeFiberReferenceReplay(
    std::size_t selectedSources, std::span<const FiberReferenceReplayOutcome> outcomes, double baseVoxelSizeUm);

[[nodiscard]] nlohmann::json fiberReferenceReplayBenchmarkJson(
    const FiberReferenceReplaySummary& summary,
    std::span<const FiberReferenceReplayOutcome> outcomes,
    const FiberletGraphReplayConfig& replayConfig,
    double seedWindowBaseVoxels,
    const cv::Vec3d& minimumBaseXYZ,
    const cv::Vec3d& maximumBaseXYZ);

}  // namespace vc::fiber_tracer
