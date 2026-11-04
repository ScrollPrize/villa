#pragma once

#include "vc/fiber_tracer/FiberletDataset.hpp"
#include "vc/fiber_tracer/FiberGraph.hpp"
#include "vc/fiber_tracer/FiberletOnDemand.hpp"
#include "vc/fiber_tracer/FiberPaths.hpp"

#include <array>
#include <cstdint>
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
    Error,
};

template <typename T>
struct FiberletGraphQuery {
    FiberletGraphQueryStatus status = FiberletGraphQueryStatus::Error;
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

struct FiberletRouteChunkLease {
    std::shared_ptr<const FiberletRouteChunkPayload> payloadLease;
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

using FiberletAnchorView =
    std::function<std::shared_ptr<const std::vector<FiberletStoredAnchor>>(const FiberletChunkKey& key, std::shared_ptr<const FiberletAnchorChunkPayload> canonicalChunk)>;

class FiberletChunkGraphSource
{
public:
    FiberletChunkGraphSource(
        std::shared_ptr<FiberletChunkDataset> anchorDataset,
        FiberletChunkRead anchorRead,
        std::shared_ptr<FiberletChunkDataset> fiberletDataset,
        FiberletChunkRead fiberletRead,
        FiberletPathConfig pathConfig = {},
        FiberletAnchorView anchorView = {});

    [[nodiscard]] FiberletGraphQuery<FiberletIncidentLease> incidentEdges(const FiberletStorageKey& anchor) const;
    [[nodiscard]] FiberletGraphQuery<FiberletAnchorLease> anchor(const FiberletStorageKey& anchor) const;
    [[nodiscard]] FiberletGraphQuery<FiberletAnchorChunkLease> anchorsInChunk(const FiberletChunkKey& chunk) const;
    [[nodiscard]] FiberletGraphQuery<FiberletPrefixChunkLease> prefixesInChunk(const FiberletChunkKey& chunk) const;
    [[nodiscard]] FiberletGraphQuery<FiberletRouteChunkLease> routesInChunk(const FiberletChunkKey& chunk) const;
    [[nodiscard]] std::vector<FiberletChunkKey> incidentPrefixChunks(
        std::span<const FiberletStorageKey> anchors) const;
    [[nodiscard]] FiberletGraphQuery<FiberletEdgeLease> edge(const FiberletStorageId& fiberlet) const;
    [[nodiscard]] FiberletGraphQuery<FiberletStoredRouteLease> storedRoute(
        const FiberletStorageId& fiberlet) const;
    [[nodiscard]] FiberletGraphQuery<FiberletRouteLease> route(const FiberletStorageId& fiberlet) const;
    [[nodiscard]] FiberletGraphQuery<FiberletReplaySourceArc> directedEdge(
        const DirectedFiberletStorageId& fiberlet) const;
    [[nodiscard]] FiberletGraphQuery<std::optional<FiberletReplaySourceTransition>> transition(
        const FiberletReplaySourceArc& incoming,
        const FiberletReplaySourceArc& outgoing,
        float maximumJoinAngleDegrees) const;
    [[nodiscard]] FiberletGraphQuery<std::optional<FiberletReplaySourceTransition>> transitionAtAnchor(
        const FiberletReplaySourceArc& incoming,
        const FiberletReplaySourceArc& outgoing,
        const FiberletStoredAnchor& sharedAnchor,
        float maximumJoinAngleDegrees) const;
    [[nodiscard]] FiberletGraphQuery<std::optional<FiberletPathCost>>
    transitionCostAtAnchor(
        const FiberletReplaySourceArc& incoming,
        const FiberletReplaySourceArc& outgoing,
        const FiberletStoredAnchor& sharedAnchor,
        float maximumJoinAngleDegrees) const;
    [[nodiscard]] const FiberletDatasetMetadata& metadata() const noexcept;
    [[nodiscard]] const FiberletPathConfig& pathConfig() const noexcept;

private:
    [[nodiscard]] FiberletGraphQuery<FiberletAnchorLease> endpoint(
        const FiberletStorageKey&, const FiberletPrefixChunkPayload&) const;
    [[nodiscard]] FiberletChunkKey ownerChunk(const FiberletStorageKey& anchor, int level) const;
    [[nodiscard]] std::vector<FiberletChunkKey> incidentOwnerChunks(const FiberletStorageKey& anchor) const;

    std::shared_ptr<FiberletChunkDataset> anchorDataset_;
    FiberletChunkRead anchorRead_;
    std::shared_ptr<FiberletChunkDataset> fiberletDataset_;
    FiberletChunkRead fiberletRead_;
    FiberletPathConfig pathConfig_;
    FiberletAnchorView anchorView_;
};

// Recover the path metric needed to reconstruct routes and join costs from a
// stored dataset's authoritative processing metadata.
[[nodiscard]] FiberletPathConfig fiberletPathConfigFromDatasetMetadata(
    const FiberletDatasetMetadata& metadata,
    int parallelThreads = 1);

// Blocking stored-graph adapter. Missing sparse chunks are empty; all edge,
// route, and transition interpretation remains owned by FiberletChunkGraphSource.
struct FiberletMaterializedReplayGraph {
    std::shared_ptr<FiberletImmutableReplayGraphSource> graph;
    std::vector<FiberletStoredAnchor> insideAnchors;
};

class FiberletStoredReplayGraphSource final : public FiberletReplayGraphSource
{
public:
    FiberletStoredReplayGraphSource(
        std::shared_ptr<FiberletChunkDataset> dataset,
        std::shared_ptr<vcr::MemoryCache> memoryCache = {},
        float maximumJoinAngleDegrees = 45.0F);
    FiberletStoredReplayGraphSource(
        std::shared_ptr<FiberletChunkDataset> anchorDataset,
        FiberletChunkRead anchorRead,
        std::shared_ptr<FiberletChunkDataset> fiberletDataset,
        FiberletChunkRead fiberletRead,
        FiberletPathConfig pathConfig,
        float maximumJoinAngleDegrees = 45.0F,
        FiberletAnchorView anchorView = {});

    [[nodiscard]] bool supportsConcurrentQueries() const noexcept override { return false; }
    [[nodiscard]] float predictionToBaseScale() const noexcept override;
    [[nodiscard]] int anchorCellSizePredictionVoxels() const noexcept override;
    [[nodiscard]] float maximumJoinAngleDegrees() const noexcept override;
    [[nodiscard]] std::vector<FiberletReplaySourceAnchor> anchorsNearReference(
        const PolylineArcGeometry& reference,
        double beginArcBase,
        double endArcBase,
        double broadPhaseRadiusBaseVoxels) const override;
    [[nodiscard]] std::vector<DirectedFiberletStorageId> outgoing(
        const FiberletStorageKey& anchor) const override;
    [[nodiscard]] FiberletReplaySourceArc arc(
        const DirectedFiberletStorageId& id) const override;
    [[nodiscard]] FiberletReplaySourceCostProfile costProfile(
        const DirectedFiberletStorageId& id) const override;
    [[nodiscard]] std::vector<cv::Vec3d> routePoints(
        const DirectedFiberletStorageId& id) const override;
    [[nodiscard]] std::optional<FiberletReplaySourceTransition> transition(
        const FiberletReplaySourceArc& incoming,
        const FiberletReplaySourceArc& outgoing) const override;

    [[nodiscard]] std::vector<FiberletStoredAnchor> anchorsInBaseBox(
        const cv::Vec3d& minimumBaseXYZ,
        const cv::Vec3d& maximumBaseXYZ) const;
    [[nodiscard]] const FiberletDatasetMetadata& metadata() const noexcept;

    // Include all edges incident to anchors in the half-open graph box and
    // their outside endpoints, but no joins at outside endpoints. The seed
    // box is separately bounded so lookahead may extend beyond the output crop.
    // Each chunk is decoded once for the pass; the returned graph owns its
    // geometry and supports concurrent, cache-independent queries.
    [[nodiscard]] FiberletMaterializedReplayGraph materializeBaseBox(
        const cv::Vec3d& minimumBaseXYZ, const cv::Vec3d& maximumBaseXYZ,
        std::size_t parallelThreads = 1) const;
    [[nodiscard]] FiberletMaterializedReplayGraph materializeBaseBoxForSeeds(
        const cv::Vec3d& minimumBaseXYZ, const cv::Vec3d& maximumBaseXYZ,
        const cv::Vec3d& seedMinimumBaseXYZ, const cv::Vec3d& seedMaximumBaseXYZ,
        std::size_t parallelThreads = 1) const;

private:
    std::shared_ptr<FiberletChunkDataset> anchorDataset_;
    std::shared_ptr<FiberletChunkDataset> dataset_;
    FiberletPathConfig pathConfig_;
    FiberletChunkRead anchorRead_;
    FiberletChunkRead pathRead_;
    FiberletAnchorView anchorView_;
    FiberletChunkGraphSource chunks_;
    float maximumJoinAngleDegrees_ = 45.0F;
};


} // namespace vc::fiber_tracer
