#pragma once

#include "vc/fiber_tracer/FiberGraph.hpp"

#include <cstddef>
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
    std::size_t maximumAttempts = 0;
    std::size_t maximumFibers = 0;
};

struct FiberletCropTraceLine {
    FiberletStorageKey seed;
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
    std::size_t attemptedAnchors = 0;
    std::size_t coveredAnchors = 0;
    std::size_t noEdgeAnchors = 0;
    std::size_t oneSidedLines = 0;
    std::size_t bidirectionalLines = 0;
};

using FiberletCropTraceProgress = std::function<void(const FiberletCropTraceResult& result, std::size_t remainingAnchors)>;

[[nodiscard]] FiberletCropTraceResult traceFiberletCrop(
    const FiberletReplayGraphSource& graph,
    std::vector<FiberletStoredAnchor> anchors,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    const FiberletCropTraceConfig& config,
    const FiberletCropTraceProgress& progress = {});

}  // namespace vc::fiber_tracer
