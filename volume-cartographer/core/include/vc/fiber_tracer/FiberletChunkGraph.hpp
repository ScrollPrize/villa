#pragma once

#include "vc/fiber_tracer/FiberletDataset.hpp"
#include "vc/fiber_tracer/FiberGraph.hpp"
#include "vc/fiber_tracer/FiberletOnDemand.hpp"
#include "vc/fiber_tracer/FiberPaths.hpp"

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace vc::fiber_tracer
{

enum class FiberletGraphQueryStatus {
    Ready,
    Pending,
    Error,
};

template <typename T>
struct FiberletGraphQuery {
    FiberletGraphQueryStatus status = FiberletGraphQueryStatus::Pending;
    T value;
    std::string error;
};

struct FiberletIncidentEdge {
    DirectedFiberletStorageId id;
    FiberletStoredPrefix prefix;
};

struct FiberletIncidentLease {
    std::vector<std::shared_ptr<const FiberletPrefixChunkPayload>> payloadLeases;
    std::vector<FiberletIncidentEdge> edges;
};

struct FiberletAnchorLease {
    std::shared_ptr<const FiberletAnchorChunkPayload> payloadLease;
    FiberletStoredAnchor anchor;
};

struct FiberletAnchorChunkLease {
    std::shared_ptr<const FiberletAnchorChunkPayload> payloadLease;
    std::shared_ptr<const std::vector<FiberletStoredAnchor>> anchors;
};

struct FiberletPrefixChunkLease {
    std::shared_ptr<const FiberletPrefixChunkPayload> payloadLease;
};

struct FiberletEdgeLease {
    std::shared_ptr<const FiberletPrefixChunkPayload> prefixPayloadLease;
    std::array<std::shared_ptr<const FiberletAnchorChunkPayload>, 2> anchorPayloadLeases;
    FiberletStoredPrefix prefix;
    FiberletStoredAnchor firstAnchor;
    FiberletStoredAnchor secondAnchor;
};

struct FiberletRouteLease {
    std::shared_ptr<const FiberletPrefixChunkPayload> prefixPayloadLease;
    std::shared_ptr<const FiberletRouteChunkPayload> routePayloadLease;
    std::array<std::shared_ptr<const FiberletAnchorChunkPayload>, 2> anchorPayloadLeases;
    FiberletStoredPrefix prefix;
    FiberletStoredRoute route;
    std::vector<cv::Vec3f> pointsPredictionXYZ;
};

struct FiberletStoredRouteLease {
    std::shared_ptr<const FiberletPrefixChunkPayload> prefixPayloadLease;
    std::shared_ptr<const FiberletRouteChunkPayload> routePayloadLease;
    FiberletStoredPrefix prefix;
    FiberletStoredRoute route;
};

struct FiberletLogicalProjectionStats {
    std::size_t projectedAnchors = 0;
    std::size_t coincidentPositionGroups = 0;
    std::size_t maximumVariants = 0;
    std::size_t compactCostChunks = 0;
};

using FiberletAnchorView =
    std::function<std::shared_ptr<const std::vector<FiberletStoredAnchor>>(const vc::render::ChunkKey& key, std::shared_ptr<const FiberletAnchorChunkPayload> canonicalChunk)>;

class FiberletChunkGraphSource
{
public:
    FiberletChunkGraphSource(
        std::shared_ptr<FiberletChunkDataset> anchorDataset,
        std::shared_ptr<vc::render::ChunkCache> anchorCache,
        std::shared_ptr<FiberletChunkDataset> fiberletDataset,
        std::shared_ptr<vc::render::ChunkCache> fiberletCache,
        FiberletPathConfig pathConfig = {},
        FiberletAnchorView anchorView = {});

    [[nodiscard]] FiberletGraphQuery<FiberletIncidentLease> incidentEdges(const FiberletStorageKey& anchor, bool blocking = false) const;
    [[nodiscard]] FiberletGraphQuery<FiberletAnchorLease> anchor(const FiberletStorageKey& anchor, bool blocking = false) const;
    [[nodiscard]] FiberletGraphQuery<FiberletAnchorChunkLease> anchorsInChunk(const vc::render::ChunkKey& chunk, bool blocking = false) const;
    [[nodiscard]] FiberletGraphQuery<FiberletPrefixChunkLease> prefixesInChunk(const vc::render::ChunkKey& chunk, bool blocking = false) const;
    [[nodiscard]] FiberletGraphQuery<FiberletEdgeLease> edge(const FiberletStorageId& fiberlet, bool blocking = false) const;
    [[nodiscard]] FiberletGraphQuery<FiberletStoredRouteLease> storedRoute(
        const FiberletStorageId& fiberlet, bool blocking = false) const;
    [[nodiscard]] FiberletGraphQuery<FiberletRouteLease> route(const FiberletStorageId& fiberlet, bool blocking = false) const;
    [[nodiscard]] FiberletGraphQuery<FiberletReplaySourceArc> directedEdge(
        const DirectedFiberletStorageId& fiberlet, bool blocking = false) const;
    [[nodiscard]] FiberletGraphQuery<std::optional<FiberletReplaySourceTransition>> transition(
        const FiberletReplaySourceArc& incoming,
        const FiberletReplaySourceArc& outgoing,
        float maximumJoinAngleDegrees,
        bool blocking = false) const;
    [[nodiscard]] const FiberletDatasetMetadata& metadata() const noexcept;

private:
    [[nodiscard]] vc::render::ChunkKey ownerChunk(const FiberletStorageKey& anchor, int level) const;
    [[nodiscard]] std::vector<vc::render::ChunkKey> incidentOwnerChunks(const FiberletStorageKey& anchor) const;

    std::shared_ptr<FiberletChunkDataset> anchorDataset_;
    std::shared_ptr<vc::render::ChunkCache> anchorCache_;
    std::shared_ptr<FiberletChunkDataset> fiberletDataset_;
    std::shared_ptr<vc::render::ChunkCache> fiberletCache_;
    FiberletPathConfig pathConfig_;
    FiberletAnchorView anchorView_;
};

enum class FiberletChunkRouteEdgeCostView {
    Stored,
    SqrtUint16Max256,
};

struct FiberletChunkRouteDistribution {
    std::size_t count = 0;
    std::optional<double> minimum;
    std::optional<double> mean;
    std::optional<double> median;
    std::optional<double> maximum;
};

struct FiberletChunkRouteAnalysisConfig {
    cv::Vec3d minimumBaseXYZ{0.0, 0.0, 0.0};
    cv::Vec3d maximumBaseXYZ{0.0, 0.0, 0.0};
    float maximumJoinAngleDegrees = 45.0F;
    FiberletChunkRouteEdgeCostView edgeCostView =
        FiberletChunkRouteEdgeCostView::Stored;
    std::size_t parallelThreads = 1;
    std::size_t maximumGeneratedStatesPerEntry = 1'000'000;
};

struct FiberletChunkRouteAnalysisReport {
    cv::Vec3d minimumBaseXYZ{0.0, 0.0, 0.0};
    cv::Vec3d maximumBaseXYZ{0.0, 0.0, 0.0};
    std::vector<vc::render::ChunkKey> seedAnchorStorageChunks;
    std::size_t insideAnchors = 0;
    std::size_t physicalFiberlets = 0;
    std::size_t internalFiberlets = 0;
    std::size_t crossingFiberlets = 0;
    std::vector<FiberletStorageId> physicalFiberletIds;
    std::vector<FiberletStorageId> internalFiberletIds;
    std::size_t directedEntries = 0;
    std::size_t directedExits = 0;
    std::size_t reachableEntries = 0;
    std::size_t unreachableEntries = 0;
    std::size_t tiedOptimalEntries = 0;
    std::size_t optimalRoutes = 0;
    std::size_t usedInsideAnchors = 0;
    std::size_t unusedInsideAnchors = 0;
    std::size_t usedPhysicalFiberlets = 0;
    std::size_t unusedPhysicalFiberlets = 0;
    std::size_t usedInternalFiberlets = 0;
    std::size_t unusedInternalFiberlets = 0;
    std::vector<FiberletStorageId> retainedPhysicalFiberlets;
    std::size_t directedStates = 0;
    std::size_t admissibleTransitions = 0;
    std::size_t generatedSearchStates = 0;
    std::size_t expandedSearchStates = 0;
    std::size_t rejectedVisitedTargets = 0;
    FiberletChunkRouteDistribution routeFiberletCounts;
    FiberletChunkRouteDistribution routeLengthsPredictionVoxels;
    FiberletChunkRouteDistribution routeLosses;
    FiberletChunkRouteDistribution routeLossesPerPredictionVoxel;
    double elapsedSeconds = 0.0;
    double cpuSeconds = 0.0;
};

// Conservative prefix-owner set needed by any anchor in the analysis box.
// Callers may feed these keys to the ordinary on-demand prefetch scheduler;
// analysis remains correct without prefetching.
[[nodiscard]] std::vector<vc::render::ChunkKey>
fiberletChunkRoutePrefetchChunks(
    const FiberletDatasetMetadata& metadata,
    const FiberletChunkRouteAnalysisConfig& config);

struct FiberletChunkRoutePopulation {
    std::size_t insideAnchors = 0;
    std::vector<FiberletStorageId> physicalFiberletIds;
    std::vector<FiberletStorageId> internalFiberletIds;
};

struct FiberletChunkRouteDirectedMacroId {
    std::size_t macro = 0;
    bool reverse = false;

    auto operator<=>(const FiberletChunkRouteDirectedMacroId&) const = default;
};

struct FiberletChunkRouteMacroDirection {
    bool live = false;
    std::vector<DirectedFiberletStorageId> physicalFiberlets;
    // Source followed by every physical target, including hidden anchors.
    std::vector<FiberletStorageKey> anchors;
    std::vector<double> edgeLosses;
    std::vector<double> internalJoinLosses;
    std::vector<double> edgeLengthsPredictionVoxels;
    double diagnosticLoss = 0.0;
    double diagnosticLengthPredictionVoxels = 0.0;
};

struct FiberletChunkRouteMacro {
    std::size_t index = 0;
    std::array<FiberletChunkRouteMacroDirection, 2> directions;
    bool firstBoundaryPortal = false;
    bool secondBoundaryPortal = false;
};

struct FiberletChunkRouteMacroTransition {
    FiberletChunkRouteDirectedMacroId incoming;
    FiberletChunkRouteDirectedMacroId outgoing;
    double joinLoss = 0.0;
};

struct FiberletChunkRouteDeterministicRollout {
    FiberletChunkRouteDirectedMacroId start;
    std::vector<FiberletChunkRouteDirectedMacroId> macros;
    std::vector<FiberletStorageKey> anchors;
    std::vector<double> transitionJoinLosses;
    double diagnosticLoss = 0.0;
    double diagnosticLengthPredictionVoxels = 0.0;
};

struct FiberletChunkRouteSimplificationReport {
    cv::Vec3d minimumBaseXYZ{0.0, 0.0, 0.0};
    cv::Vec3d maximumBaseXYZ{0.0, 0.0, 0.0};
    std::size_t inputAnchors = 0;
    std::size_t retainedAnchors = 0;
    std::size_t unusedAnchorsRemoved = 0;
    std::size_t inputInsideAnchors = 0;
    std::size_t retainedInsideAnchors = 0;
    std::size_t unusedInsideAnchorsRemoved = 0;
    std::size_t boundaryPortals = 0;
    std::size_t inputPhysicalFiberlets = 0;
    std::size_t livePhysicalFiberlets = 0;
    std::size_t deadPhysicalFiberletsRemoved = 0;
    std::size_t inputDirectedStates = 0;
    std::size_t liveDirectedStates = 0;
    std::size_t deadDirectedStatesRemoved = 0;
    std::size_t contractibleInsideAnchors = 0;
    std::size_t physicalMacros = 0;
    std::size_t physicalFiberletsMerged = 0;
    std::size_t liveDirectedMacros = 0;
    std::size_t macroTransitions = 0;
    std::size_t zeroContinuationStates = 0;
    std::size_t forcedContinuationStates = 0;
    std::size_t branchingStates = 0;
    std::size_t directedChainMacros = 0;
    std::size_t directedMacrosMerged = 0;
    std::size_t deterministicRollouts = 0;
    std::size_t structuralDuplicateFiberlets = 0;
    FiberletChunkRouteDistribution physicalFiberletsPerMacro;
    FiberletChunkRouteDistribution macrosPerDeterministicRollout;
    std::vector<FiberletStorageKey> retainedInsideAnchorIds;
    std::vector<FiberletStorageKey> boundaryPortalIds;
    std::vector<FiberletStorageId> livePhysicalFiberletIds;
    std::vector<std::array<bool, 2>> livePhysicalDirections;
    std::vector<FiberletChunkRouteMacro> macros;
    std::vector<FiberletChunkRouteMacroTransition> transitions;
    std::vector<FiberletChunkRouteDeterministicRollout> rollouts;
};

// Appends the macro using the same left-associated edge/join sequence as the
// physical route search. diagnosticLoss is never authoritative for ranking.
[[nodiscard]] double appendFiberletChunkRouteMacroLoss(
    double prefixLoss,
    double incomingJoinLoss,
    const FiberletChunkRouteMacroDirection& direction);

// The shared source anchor is anchors.front() and is already present in route
// history. Every hidden/target anchor is checked atomically before append.
[[nodiscard]] bool canAppendFiberletChunkRouteMacro(
    const FiberletChunkRouteMacroDirection& direction,
    std::span<const FiberletStorageKey> visitedAnchors);

[[nodiscard]] FiberletChunkRouteSimplificationReport
simplifyFiberletChunkRoutes(
    const FiberletChunkGraphSource& graph,
    const FiberletChunkRouteAnalysisConfig& config,
    std::span<const FiberletStorageId> retainedPhysicalFiberlets);

[[nodiscard]] FiberletChunkRoutePopulation collectFiberletChunkRoutePopulation(
    const FiberletChunkGraphSource& graph,
    const FiberletChunkRouteAnalysisConfig& config);

[[nodiscard]] FiberletChunkRouteAnalysisReport analyzeFiberletChunkRoutes(
    const FiberletChunkGraphSource& graph,
    const FiberletChunkRouteAnalysisConfig& config);

[[nodiscard]] vc::render::ChunkKey fiberletStorageOwnerChunk(
    const FiberletDatasetMetadata& metadata,
    const FiberletStorageKey& anchor,
    int level = 0);

struct FiberletReductionWriteReport {
    vc::render::ChunkKey owner;
    std::size_t inputFiberlets = 0;
    std::size_t retainedFiberlets = 0;
    bool reused = false;
};

struct FiberletReductionOverlayBoxWriteReport {
    std::size_t touchedAnchorChunks = 0;
    std::size_t touchedFiberletChunks = 0;
    std::size_t inputAnchors = 0;
    std::size_t retainedAnchors = 0;
    std::size_t inputFiberlets = 0;
    std::size_t retainedFiberlets = 0;
};

// Rewrite every storage chunk intersecting config into a temporary sparse
// overlay. Only records whose stored base-space owner position is inside the
// half-open box may be removed; records and routes are otherwise exact copies.
[[nodiscard]] FiberletReductionOverlayBoxWriteReport
writeFiberletReductionOverlayBox(
    const FiberletChunkGraphSource& source,
    const std::shared_ptr<FiberletChunkDataset>& outputAnchors,
    const std::shared_ptr<FiberletChunkDataset>& outputFiberlets,
    const FiberletChunkRouteAnalysisConfig& config,
    std::span<const FiberletStorageId> inputPhysicalFiberlets,
    std::span<const FiberletStorageId> retainedPhysicalFiberlets);

[[nodiscard]] FiberletReductionWriteReport writeReducedFiberletChunk(
    const FiberletChunkGraphSource& source,
    const std::shared_ptr<FiberletChunkDataset>& outputDataset,
    const vc::render::ChunkKey& owner,
    std::span<const FiberletStorageId> inputFiberlets,
    std::span<const FiberletStorageId> retainedFiberlets);

class FiberletCachedReplayGraphSource final : public FiberletReplayGraphSource
{
public:
    FiberletCachedReplayGraphSource(
        std::shared_ptr<FiberletOnDemandPreprocessor> preprocessor,
        FiberletPathConfig pathConfig,
        FiberletEvaluationQuantization evaluationQuantization =
            defaultFiberletReplayQuantization(),
        float maximumJoinAngleDegrees = 45.0F);

    [[nodiscard]] float predictionToBaseScale() const noexcept override;
    [[nodiscard]] int anchorCellSizePredictionVoxels() const noexcept override;
    [[nodiscard]] float maximumJoinAngleDegrees() const noexcept override;
    [[nodiscard]] FiberletStorageKey logicalAnchorId(const FiberletStorageKey& physical) const override;
    [[nodiscard]] DirectedFiberletStorageId logicalArcId(const DirectedFiberletStorageId& physical) const override;
    [[nodiscard]] std::vector<FiberletReplaySourceAnchor> anchorsNearReference(
        const PolylineArcGeometry& reference, double beginArcBase, double endArcBase, double broadPhaseRadiusBaseVoxels) const override;
    [[nodiscard]] std::vector<DirectedFiberletStorageId> outgoing(const FiberletStorageKey& anchor) const override;
    [[nodiscard]] FiberletReplaySourceArc arc(const DirectedFiberletStorageId& id) const override;
    [[nodiscard]] FiberletReplaySourceCostProfile costProfile(const DirectedFiberletStorageId& id) const override;
    [[nodiscard]] std::vector<cv::Vec3d> routePoints(const DirectedFiberletStorageId& id) const override;
    [[nodiscard]] std::optional<FiberletReplaySourceTransition> transition(const FiberletReplaySourceArc& incoming, const FiberletReplaySourceArc& outgoing) const override;
    [[nodiscard]] FiberletLogicalProjectionStats logicalProjectionStats() const;

private:
    [[nodiscard]] bool anchorCellInCorridor(const FiberletStorageKey& anchor) const;
    [[nodiscard]] FiberletStorageKey logicalAnchorKey(const FiberletStorageKey& physical) const;
    [[nodiscard]] DirectedFiberletStorageId logicalFiberletId(const DirectedFiberletStorageId& physical) const;
    [[nodiscard]] std::array<int, 3> compactCostOwner(const FiberletStorageId& physical) const;
    [[nodiscard]] float quantizedCost(
        const FiberletStorageId& physical,
        float cost,
        float pathLengthPredictionVoxels) const;

    std::shared_ptr<FiberletOnDemandPreprocessor> preprocessor_;
    FiberletChunkGraphSource chunks_;
    FiberletPathConfig pathConfig_;
    FiberletEvaluationQuantization evaluationQuantization_;
    float maximumJoinAngleDegrees_ = 45.0F;
    struct QuantizationState;
    std::shared_ptr<QuantizationState> quantizationState_;
};

}  // namespace vc::fiber_tracer
