#pragma once

#include "vc/fiber_tracer/FiberPaths.hpp"
#include "vc/fiber_tracer/FiberletStorage.hpp"
#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
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
    std::vector<float> segmentLengthsPredictionVoxels;
    std::vector<float> segmentCostDensities;
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

[[nodiscard]] FiberletGraph buildFiberletGraph(const FiberletPathReport& paths, float maximumJoinAngleDegrees = 45.0F);
[[nodiscard]] nlohmann::json fiberletGraphJson(const FiberletGraph& graph);
} // namespace vc::fiber_tracer
