#pragma once

#include "vc/fiber_tracer/FiberPaths.hpp"

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
    cv::Vec3d positionBaseXYZ{0.0, 0.0, 0.0};
    FiberStoredPredictionSample prediction;
    cv::Vec3d normalXYZ{0.0, 0.0, 0.0};
    bool normalValid = false;
    std::vector<size_t> outgoingArcs;
};

struct FiberletGraphEdge {
    size_t candidateIndex = 0;
    size_t startNode = 0;
    size_t targetNode = 0;
    std::vector<cv::Vec3d> pointsBaseXYZ;
    double pathLengthPredictionVoxels = 0.0;
    FiberletPathCost cost;
};

struct FiberletGraphTransition {
    size_t incomingArc = 0;
    size_t outgoingArc = 0;
    double angleDegrees = 0.0;
    double incomingLengthPredictionVoxels = 0.0;
    double outgoingLengthPredictionVoxels = 0.0;
    FiberletPathCost cost;
};

struct FiberletGraph {
    double predictionToBaseScale = 1.0;
    int anchorCellSizePredictionVoxels = 0;
    double maximumJoinAngleDegrees = 45.0;
    std::vector<FiberletGraphNode> nodes;
    std::vector<FiberletGraphEdge> edges;
    std::vector<FiberletGraphTransition> transitions;
};

enum class FiberletGraphReplayStatus {
    FailureWithPostroll,
    FailureTruncated,
    ReferenceEnd,
    GraphExhausted,
    NoUsableStart,
};

struct FiberletGraphReplayConfig {
    size_t beamWidth = 16;
    size_t lookaheadEdges = 3;
    double errorThresholdBaseVoxels = 20.0;
    double matchRefineSteps = 1.0;
    double postrollDistanceBaseVoxels = 0.0;
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

struct FiberletGraphReplayResult {
    FiberletGraphReplayStatus status = FiberletGraphReplayStatus::NoUsableStart;
    std::string reason;
    std::vector<cv::Vec3d> routePointsBaseXYZ;
    std::vector<size_t> candidateIndices;
    std::vector<size_t> arcIndices;
    std::vector<size_t> transitionIndices;
    std::optional<size_t> failureCandidateIndex;
    std::optional<size_t> failureCandidatePathPointIndex;
    std::optional<size_t> failureArcIndex;
    std::optional<size_t> stopNodeIndex;
    std::vector<FiberletGraphReplayMatch> matches;
    std::optional<size_t> failureRoutePointIndex;
    std::optional<double> failureReferenceArcBase;
    double requestedPostrollDistanceBaseVoxels = 0.0;
    double completedPostrollDistanceBaseVoxels = 0.0;
    double totalLoss = 0.0;
    FiberletPathCost edgeCost;
    FiberletPathCost transitionCost;
    double pathLengthPredictionVoxels = 0.0;
};

[[nodiscard]] FiberletGraph buildFiberletGraph(const FiberletPathReport& paths, double maximumJoinAngleDegrees = 45.0);

[[nodiscard]] FiberletGraphReplayResult traceFiberletGraphReplay(
    const FiberletGraph& graph, const std::vector<cv::Vec3d>& referencePointsBaseXYZ, const FiberletGraphReplayConfig& config);

[[nodiscard]] const char* fiberletGraphReplayStatusName(FiberletGraphReplayStatus status) noexcept;

[[nodiscard]] nlohmann::json fiberletGraphJson(const FiberletGraph& graph);

[[nodiscard]] std::string fiberletGraphReplayObj(const FiberletGraphReplayResult& replay);

[[nodiscard]] nlohmann::json fiberletGraphReplayJson(const FiberletGraphReplayResult& replay, const FiberletGraphReplayConfig& config);

}  // namespace vc::fiber_tracer
