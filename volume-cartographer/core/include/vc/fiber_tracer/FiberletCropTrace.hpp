#pragma once

#include "vc/fiber_tracer/FiberGraph.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <functional>
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
    double coverageDirectionDegrees = 25.0;
    std::size_t parallelThreads = 0;
    std::size_t maximumAttempts = 0;
    std::size_t maximumFibers = 0;
};

struct FiberletCropTraceLine {
    FiberletStorageKey seed;
    cv::Vec3d seedBaseXYZ{0.0, 0.0, 0.0};
    float seedPresence = 0.0F;
    std::vector<cv::Vec3d> pointsBaseXYZ;
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
    double direction1LengthBaseVoxels = 0.0;
    double direction2LengthBaseVoxels = 0.0;
};

struct FiberDirectionClassification {
    cv::Vec3d direction1BaseXYZ{1.0, 0.0, 0.0};
    cv::Vec3d direction2BaseXYZ{0.0, 1.0, 0.0};
    std::vector<FiberDirectionLineClassification> lines;
    std::array<std::size_t, 3> groupCounts{0, 0, 0};
    std::size_t analyzedSteps = 0;
    double analyzedLengthBaseVoxels = 0.0;
};

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

using FiberletCropTraceProgress = std::function<void(const FiberletCropTraceResult& result, std::size_t remainingAnchors)>;

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

}  // namespace vc::fiber_tracer
