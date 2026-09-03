#pragma once

#include "vc/fiber_tracer/FiberGraph.hpp"

#include <cstddef>
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

struct FiberReferenceReplayOutcome {
    std::size_t sourceIndex = 0;
    std::size_t runIndex = 0;
    bool reverse = false;
    double referenceLengthBaseVoxels = 0.0;
    double tracedLengthBaseVoxels = 0.0;
    bool completed = false;
    std::string failureReason;
};

struct FiberReferenceReplaySummary {
    std::size_t selectedSources = 0;
    std::size_t sourcesWithoutRuns = 0;
    std::size_t inCropRuns = 0;
    std::size_t directedCases = 0;
    std::size_t completedCases = 0;
    std::size_t failedCases = 0;
    double undirectedReferenceLengthBaseVoxels = 0.0;
    double directedReferenceLengthBaseVoxels = 0.0;
    double tracedLengthBaseVoxels = 0.0;
    double meanCreditedLengthBaseVoxels = 0.0;
    double meanFailureLengthBaseVoxels = 0.0;
    double baseVoxelSizeUm = 0.0;
    double tracedLengthMillimeters = 0.0;
    double meanCreditedLengthMillimeters = 0.0;
    double meanFailureLengthMillimeters = 0.0;
    double lengthWeightedSuccessPercent = 0.0;
    double completedCasesPercent = 0.0;
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
    const cv::Vec3d& minimumBaseXYZ,
    const cv::Vec3d& maximumBaseXYZ);

}  // namespace vc::fiber_tracer
