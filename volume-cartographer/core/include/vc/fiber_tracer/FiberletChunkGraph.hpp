#pragma once

#include "vc/fiber_tracer/FiberletDataset.hpp"
#include "vc/fiber_tracer/FiberGraph.hpp"
#include "vc/fiber_tracer/FiberletOnDemand.hpp"
#include "vc/fiber_tracer/FiberPaths.hpp"

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
    [[nodiscard]] FiberletGraphQuery<FiberletRouteLease> route(const FiberletStorageId& fiberlet, bool blocking = false) const;

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
