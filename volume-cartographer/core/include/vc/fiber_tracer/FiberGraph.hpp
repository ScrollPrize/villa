#pragma once

#include "vc/fiber_tracer/FiberPaths.hpp"
#include "vc/fiber_tracer/FiberTrace.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

struct FiberletGraphNode {
    FiberletAnchorId anchor;
    cv::Vec3f positionBaseXYZ{0.0F, 0.0F, 0.0F};
    FiberletPredictionSample prediction;
    cv::Vec3f normalXYZ{0.0F, 0.0F, 0.0F};
    bool normalValid = false;
    std::vector<size_t> outgoingArcs;
};

struct FiberletGraphEdge {
    size_t candidateIndex = 0;
    size_t startNode = 0;
    size_t targetNode = 0;
    std::vector<cv::Vec3f> pointsBaseXYZ;
    float pathLengthPredictionVoxels = 0.0F;
    FiberletPathCost cost;
};

struct FiberletGraphTransition {
    size_t incomingArc = 0;
    size_t outgoingArc = 0;
    float angleDegrees = 0.0F;
    float incomingLengthPredictionVoxels = 0.0F;
    float outgoingLengthPredictionVoxels = 0.0F;
    FiberletPathCost cost;
};

struct FiberletGraph {
    float predictionToBaseScale = 1.0F;
    int anchorCellSizePredictionVoxels = 0;
    float maximumJoinAngleDegrees = 45.0F;
    std::vector<FiberletGraphNode> nodes;
    std::vector<FiberletGraphEdge> edges;
    std::vector<FiberletGraphTransition> transitions;
};

struct FiberletGraphReplayConfig {
    size_t beamWidth = 16;
    size_t lookaheadEdges = 3;
    double errorThresholdBaseVoxels = 20.0;
    double matchRefineSteps = 1.0;
    double minimumResetAdvanceBaseVoxels = 1.0;
    double referenceBeginArcBase = 0.0;
    std::optional<double> referenceEndArcBase;
};

struct FiberletGraphReplayMatch {
    size_t routePointIndex = 0;
    double predictedReferenceArcBase = 0.0;
    double matchedReferenceArcBase = 0.0;
    cv::Vec3d matchedReferencePointBaseXYZ{0.0, 0.0, 0.0};
    double searchBeginArcBase = 0.0;
    double searchEndArcBase = 0.0;
    double errorBaseVoxels = 0.0;
};

struct FiberletGraphReplayCost {
    double invalidPrediction = 0.0;
    double alignment = 0.0;
    double isotropicSmoothness = 0.0;
    double tangentSmoothness = 0.0;
    double normalSmoothness = 0.0;

    [[nodiscard]] double total() const noexcept
    {
        return invalidPrediction + alignment + isotropicSmoothness +
            tangentSmoothness + normalSmoothness;
    }

    FiberletGraphReplayCost& operator+=(const FiberletPathCost& other) noexcept
    {
        invalidPrediction += static_cast<double>(other.invalidPrediction);
        alignment += static_cast<double>(other.alignment);
        isotropicSmoothness += static_cast<double>(other.isotropicSmoothness);
        tangentSmoothness += static_cast<double>(other.tangentSmoothness);
        normalSmoothness += static_cast<double>(other.normalSmoothness);
        return *this;
    }
};

struct FiberletGraphReplaySegment {
    double startReferenceArcBase = 0.0;
    double endReferenceArcBase = 0.0;
    std::string terminationReason;
    std::vector<cv::Vec3d> routePointsBaseXYZ;
    std::vector<size_t> candidateIndices;
    std::vector<size_t> arcIndices;
    std::vector<size_t> transitionIndices;
    std::optional<size_t> stopNodeIndex;
    bool terminalPartialEdge = false;
    std::vector<FiberletGraphReplayMatch> matches;
    double totalLoss = 0.0;
    FiberletGraphReplayCost edgeCost;
    FiberletGraphReplayCost transitionCost;
    double pathLengthPredictionVoxels = 0.0;
};

struct FiberletGraphReplayResult {
    double referenceBeginArcBase = 0.0;
    double referenceEndArcBase = 0.0;
    double completedReferenceArcBase = 0.0;
    std::vector<FiberletGraphReplaySegment> segments;
    std::vector<FiberReplayFailure> failures;
};

[[nodiscard]] FiberletGraph buildFiberletGraph(const FiberletPathReport& paths, float maximumJoinAngleDegrees = 45.0F);

[[nodiscard]] FiberletGraphReplayResult traceFiberletGraphReplay(
    const FiberletGraph& graph,
    const std::vector<cv::Vec3d>& referencePointsBaseXYZ,
    const FiberletGraphReplayConfig& config,
    const FiberReplayFailureCallback& failure = {});

[[nodiscard]] nlohmann::json fiberletGraphJson(const FiberletGraph& graph);

[[nodiscard]] std::string fiberletGraphReplayObj(const FiberletGraphReplayResult& replay);

[[nodiscard]] nlohmann::json fiberletGraphReplayJson(const FiberletGraphReplayResult& replay, const FiberletGraphReplayConfig& config);

}  // namespace vc::fiber_tracer
