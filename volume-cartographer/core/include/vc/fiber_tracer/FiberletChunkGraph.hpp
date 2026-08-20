#pragma once

#include "vc/fiber_tracer/FiberletDataset.hpp"
#include "vc/fiber_tracer/FiberGraph.hpp"
#include "vc/fiber_tracer/FiberletOnDemand.hpp"
#include "vc/fiber_tracer/FiberPaths.hpp"

#include <memory>
#include <optional>
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
};

struct FiberletEdgeLease {
    std::shared_ptr<const FiberletPrefixChunkPayload> prefixPayloadLease;
    std::array<std::shared_ptr<const FiberletAnchorChunkPayload>, 2> anchorPayloadLeases;
    FiberletStoredPrefix prefix;
    FiberletStoredAnchor firstAnchor;
    FiberletStoredAnchor secondAnchor;
    FiberletRouteEndpointSteps endpointSteps;
};

struct FiberletRouteLease {
    std::shared_ptr<const FiberletPrefixChunkPayload> prefixPayloadLease;
    std::shared_ptr<const FiberletRouteChunkPayload> routePayloadLease;
    std::array<std::shared_ptr<const FiberletAnchorChunkPayload>, 2> anchorPayloadLeases;
    FiberletStoredPrefix prefix;
    FiberletStoredRoute route;
    std::vector<cv::Vec3f> pointsPredictionXYZ;
};

class FiberletChunkGraphSource
{
public:
    FiberletChunkGraphSource(
        std::shared_ptr<FiberletChunkDataset> anchorDataset,
        std::shared_ptr<vc::render::ChunkCache> anchorCache,
        std::shared_ptr<FiberletChunkDataset> fiberletDataset,
        std::shared_ptr<vc::render::ChunkCache> fiberletCache,
        FiberletPathConfig pathConfig = {});

    [[nodiscard]] FiberletGraphQuery<FiberletIncidentLease> incidentEdges(const FiberletStorageKey& anchor, bool blocking = false) const;
    [[nodiscard]] FiberletGraphQuery<FiberletAnchorLease> anchor(const FiberletStorageKey& anchor, bool blocking = false) const;
    [[nodiscard]] FiberletGraphQuery<FiberletAnchorChunkLease> anchorsInChunk(const vc::render::ChunkKey& chunk, bool blocking = false) const;
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
};

class FiberletCachedReplayGraphSource final : public FiberletReplayGraphSource
{
public:
    FiberletCachedReplayGraphSource(
        std::shared_ptr<FiberletOnDemandPreprocessor> preprocessor,
        const FiberPredictionSource& predictionSource,
        std::shared_ptr<const vc::lasagna::NormalSampler> normalSampler,
        FiberletPathConfig pathConfig,
        float maximumJoinAngleDegrees = 45.0F);

    [[nodiscard]] float predictionToBaseScale() const noexcept override;
    [[nodiscard]] int anchorCellSizePredictionVoxels() const noexcept override;
    [[nodiscard]] float maximumJoinAngleDegrees() const noexcept override;
    [[nodiscard]] std::vector<FiberletReplaySourceAnchor> anchorsNearReference(
        const PolylineArcGeometry& reference, double beginArcBase, double endArcBase, double broadPhaseRadiusBaseVoxels) const override;
    [[nodiscard]] std::vector<DirectedFiberletStorageId> outgoing(const FiberletStorageKey& anchor) const override;
    [[nodiscard]] FiberletReplaySourceArc arc(const DirectedFiberletStorageId& id) const override;
    [[nodiscard]] std::vector<cv::Vec3d> routePoints(const DirectedFiberletStorageId& id) const override;
    [[nodiscard]] std::optional<FiberletReplaySourceTransition> transition(const FiberletReplaySourceArc& incoming, const FiberletReplaySourceArc& outgoing) const override;

private:
    [[nodiscard]] bool anchorCellInCorridor(const FiberletStorageKey& anchor) const;

    std::shared_ptr<FiberletOnDemandPreprocessor> preprocessor_;
    FiberletChunkGraphSource chunks_;
    const FiberPredictionSource* predictionSource_ = nullptr;
    std::shared_ptr<const vc::lasagna::NormalSampler> normalSampler_;
    FiberletPathConfig pathConfig_;
    float maximumJoinAngleDegrees_ = 45.0F;
};

}  // namespace vc::fiber_tracer
