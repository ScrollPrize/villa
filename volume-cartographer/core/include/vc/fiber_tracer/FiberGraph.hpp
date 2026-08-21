#pragma once

#include "vc/fiber_tracer/FiberPaths.hpp"
#include "vc/fiber_tracer/FiberletStorage.hpp"
#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

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
    size_t expansionThreads = 1;
    double beamStepDistanceBaseVoxels = 48.0;
    double lookaheadDistanceBaseVoxels = 192.0;
    size_t searchWidth = 128;
    double pruneDistanceBaseVoxels = 48.0;
    size_t maximumGeneratedStatesPerIteration = 1'000'000;
    double errorThresholdBaseVoxels = 20.0;
    double matchRefineSteps = 1.0;
    double minimumResetAdvanceBaseVoxels = 1.0;
    double referenceBeginArcBase = 0.0;
    std::optional<double> referenceEndArcBase;
    std::optional<FiberletStorageKey> initialSeedKey;
    bool recordDecisionDiagnostics = false;
};

struct FiberletGraphReplayMatch {
    size_t routePointIndex = 0;
    double predictedReferenceArcBase = 0.0;
    double matchedReferenceArcBase = 0.0;
    cv::Vec3d matchedReferencePointBaseXYZ{0.0, 0.0, 0.0};
    double searchBeginArcBase = 0.0;
    double searchEndArcBase = 0.0;
    FiberReplayThresholdMeasurement thresholdMeasurement;
};

struct FiberletGraphReplayCost {
    double invalidPrediction = 0.0;
    double alignment = 0.0;
    double isotropicSmoothness = 0.0;
    double tangentSmoothness = 0.0;
    double normalSmoothness = 0.0;

    [[nodiscard]] double total() const noexcept
    {
        return invalidPrediction + alignment + isotropicSmoothness + tangentSmoothness + normalSmoothness;
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

struct FiberletGraphReplayDecisionRoute {
    std::vector<DirectedFiberletStorageId> prefixLogicalArcs;
    std::vector<DirectedFiberletStorageId> logicalArcs;
    std::vector<cv::Vec3d> routePointsBaseXYZ;
    FiberletGraphReplayCost edgeCost;
    FiberletGraphReplayCost transitionCost;
    FiberletGraphReplayCost committedEdgeCost;
    FiberletGraphReplayCost committedTransitionCost;
    double committedPathLengthPredictionVoxels = 0.0;
    double pathLengthPredictionVoxels = 0.0;
    double completePathLengthPredictionVoxels = 0.0;
    double totalLoss = 0.0;
    double lossPerPredictionVoxel = 0.0;
};

struct FiberletGraphReplayPruneFront {
    double horizonPathLengthPredictionVoxels = 0.0;
    size_t inputRouteCount = 0;
    size_t localCandidateLimit = 0;
    size_t generatedStateCount = 0;
    size_t expandedStateCount = 0;
    size_t rejectedStateCount = 0;
    size_t dominatedStateCount = 0;
    size_t costPrunedStateCount = 0;
    size_t completedCandidateCount = 0;
    size_t distinctPrefixCount = 0;
    size_t diversityProtectedCount = 0;
    size_t globalFillCount = 0;
    size_t retainedRouteCount = 0;
    size_t prunedCandidateCount = 0;
    size_t cumulativeGeneratedStateCount = 0;
    bool searchWidthBound = false;
};

struct FiberletGraphReplayDecision {
    size_t routePointIndex = 0;
    double referenceArcBase = 0.0;
    double checkpointPathLengthPredictionVoxels = 0.0;
    double nextCheckpointPathLengthPredictionVoxels = 0.0;
    double scoringHorizonPathLengthPredictionVoxels = 0.0;
    size_t generatedStateCount = 0;
    size_t expandedStateCount = 0;
    size_t evaluatedCandidateCount = 0;
    size_t costPrunedStateCount = 0;
    size_t rejectedStateCount = 0;
    size_t dominatedStateCount = 0;
    size_t retainedBeamCount = 0;
    std::string searchMode;
    size_t searchWidth = 0;
    double pruneDistancePredictionVoxels = 0.0;
    std::vector<FiberletGraphReplayPruneFront> pruneFronts;
    std::vector<DirectedFiberletStorageId> selectedPrefixLogicalArcs;
    FiberletStorageKey sourceKey;
    std::optional<DirectedFiberletStorageId> incomingLogicalArc;
    std::optional<size_t> selectedRouteIndex;
    std::vector<FiberletGraphReplayDecisionRoute> routes;
};

struct FiberletGraphReplayCommittedStep {
    double referenceBeginArcBase = 0.0;
    double referenceEndArcBase = 0.0;
    FiberletGraphReplayCost edgeCost;
    FiberletGraphReplayCost transitionCost;
    double pathLengthPredictionVoxels = 0.0;
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
    std::optional<FiberletStorageKey> seedKey;
    std::vector<FiberletGraphReplayDecision> decisions;
    std::vector<FiberletGraphReplayCommittedStep> committedSteps;
    bool terminalPartialEdge = false;
    std::vector<FiberletGraphReplayMatch> matches;
    double totalLoss = 0.0;
    FiberletGraphReplayCost edgeCost;
    FiberletGraphReplayCost transitionCost;
    double pathLengthPredictionVoxels = 0.0;
};

struct FiberletGraphReplayResult {
    double predictionToBaseScale = 1.0;
    double referenceBeginArcBase = 0.0;
    double referenceEndArcBase = 0.0;
    double completedReferenceArcBase = 0.0;
    std::vector<FiberletGraphReplaySegment> segments;
    std::vector<FiberReplayFailure> failures;
};

struct FiberletGraphReplayFailureWindow {
    size_t failureIndex = 0;
    size_t segmentIndex = 0;
    std::string reason;
    double failureReferenceArcBase = 0.0;
    double replayBeginArcBase = 0.0;
    double replayEndArcBase = 0.0;
    std::optional<FiberletStorageKey> seedKey;
};

struct FiberletGraphReplayProgress {
    size_t segmentIndex = 0;
    double referenceArcBase = 0.0;
    double referenceArcFraction = 0.0;
    std::string state;
};

using FiberletGraphReplayProgressCallback = std::function<void(const FiberletGraphReplayProgress&)>;

struct FiberletReplaySourceAnchor {
    FiberletStorageKey id;
    cv::Vec3f positionBaseXYZ{0.0F, 0.0F, 0.0F};
};

struct FiberletReplaySourceArc {
    DirectedFiberletStorageId id;
    FiberletStorageKey source;
    FiberletStorageKey target;
    cv::Vec3d sourcePositionBaseXYZ{0.0, 0.0, 0.0};
    cv::Vec3d targetPositionBaseXYZ{0.0, 0.0, 0.0};
    cv::Vec3f startStepBaseXYZ{0.0F, 0.0F, 0.0F};
    cv::Vec3f endStepBaseXYZ{0.0F, 0.0F, 0.0F};
    float pathLengthPredictionVoxels = 0.0F;
    FiberletPathCost cost;
    std::optional<size_t> diagnosticCandidateIndex;
    std::optional<size_t> diagnosticArcIndex;
};

struct FiberletReplaySourceTransition {
    DirectedFiberletStorageId incoming;
    DirectedFiberletStorageId outgoing;
    FiberletPathCost cost;
    std::optional<size_t> diagnosticTransitionIndex;
};

class FiberletReplayGraphSource
{
public:
    virtual ~FiberletReplayGraphSource() = default;
    [[nodiscard]] virtual float predictionToBaseScale() const noexcept = 0;
    [[nodiscard]] virtual int anchorCellSizePredictionVoxels() const noexcept = 0;
    [[nodiscard]] virtual float maximumJoinAngleDegrees() const noexcept = 0;
    [[nodiscard]] virtual FiberletStorageKey logicalAnchorId(const FiberletStorageKey& physical) const { return physical; }
    [[nodiscard]] virtual DirectedFiberletStorageId logicalArcId(const DirectedFiberletStorageId& physical) const { return physical; }
    [[nodiscard]] virtual std::vector<FiberletReplaySourceAnchor> anchorsNearReference(
        const PolylineArcGeometry& reference, double beginArcBase, double endArcBase, double broadPhaseRadiusBaseVoxels) const = 0;
    [[nodiscard]] virtual std::vector<DirectedFiberletStorageId> outgoing(const FiberletStorageKey& anchor) const = 0;
    [[nodiscard]] virtual FiberletReplaySourceArc arc(const DirectedFiberletStorageId& id) const = 0;
    [[nodiscard]] virtual std::vector<cv::Vec3d> routePoints(const DirectedFiberletStorageId& id) const = 0;
    [[nodiscard]] virtual std::optional<FiberletReplaySourceTransition> transition(
        const FiberletReplaySourceArc& incoming, const FiberletReplaySourceArc& outgoing) const = 0;
};

[[nodiscard]] FiberletGraph buildFiberletGraph(const FiberletPathReport& paths, float maximumJoinAngleDegrees = 45.0F);

[[nodiscard]] FiberletGraphReplayResult traceFiberletGraphReplay(
    const FiberletGraph& graph,
    const std::vector<cv::Vec3d>& referencePointsBaseXYZ,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    const FiberletGraphReplayConfig& config,
    const FiberReplayFailureCallback& failure = {},
    const FiberletGraphReplayProgressCallback& progress = {});

[[nodiscard]] FiberletGraphReplayResult traceFiberletGraphReplay(
    const FiberletReplayGraphSource& graph,
    const std::vector<cv::Vec3d>& referencePointsBaseXYZ,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    const FiberletGraphReplayConfig& config,
    const FiberReplayFailureCallback& failure = {},
    const FiberletGraphReplayProgressCallback& progress = {});

[[nodiscard]] nlohmann::json fiberletGraphJson(const FiberletGraph& graph);

[[nodiscard]] std::string fiberletGraphReplayObj(const FiberletGraphReplayResult& replay);

[[nodiscard]] nlohmann::json fiberletGraphReplayJson(const FiberletGraphReplayResult& replay, const FiberletGraphReplayConfig& config);

[[nodiscard]] std::vector<FiberletGraphReplayFailureWindow> fiberletGraphReplayFailureWindows(const FiberletGraphReplayResult& replay);

}  // namespace vc::fiber_tracer
