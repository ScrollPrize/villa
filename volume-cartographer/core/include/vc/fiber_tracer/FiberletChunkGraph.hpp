#pragma once

#include "vc/fiber_tracer/FiberletDataset.hpp"
#include "vc/fiber_tracer/FiberGraph.hpp"
#include "vc/fiber_tracer/FiberletOnDemand.hpp"
#include "vc/fiber_tracer/FiberPaths.hpp"

#include <array>
#include <atomic>
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
    std::function<std::shared_ptr<const std::vector<FiberletStoredAnchor>>(const vc::render::ChunkKey& key, std::shared_ptr<const FiberletAnchorChunkPayload> canonicalChunk)>;

class FiberletChunkGraphSource
{
public:
    using WaitExecutor = std::function<void(const std::function<void()>&)>;
    void setWaitExecutor(WaitExecutor wait) { wait_ = std::move(wait); }
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
    [[nodiscard]] FiberletGraphQuery<FiberletRouteChunkLease> routesInChunk(const vc::render::ChunkKey& chunk, bool blocking = false) const;
    void prefetchAnchorChunks(
        std::span<const vc::render::ChunkKey> chunks,
        bool blocking = false) const;
    void prefetchFiberletChunks(
        std::span<const vc::render::ChunkKey> chunks,
        bool blocking = false) const;
    [[nodiscard]] std::vector<vc::render::ChunkKey> incidentPrefixChunks(
        std::span<const FiberletStorageKey> anchors) const;
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
        const FiberletStorageKey&, const FiberletPrefixChunkPayload&, bool blocking) const;
    [[nodiscard]] vc::render::ChunkResult fetchChunk(
        const std::shared_ptr<vc::render::ChunkCache>& cache,
        const vc::render::ChunkKey& key, bool blocking) const;
    WaitExecutor wait_;
    [[nodiscard]] vc::render::ChunkKey ownerChunk(const FiberletStorageKey& anchor, int level) const;
    [[nodiscard]] std::vector<vc::render::ChunkKey> incidentOwnerChunks(const FiberletStorageKey& anchor) const;

    std::shared_ptr<FiberletChunkDataset> anchorDataset_;
    std::shared_ptr<vc::render::ChunkCache> anchorCache_;
    std::shared_ptr<FiberletChunkDataset> fiberletDataset_;
    std::shared_ptr<vc::render::ChunkCache> fiberletCache_;
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
        FiberletChunkCacheOptions cacheOptions = {},
        float maximumJoinAngleDegrees = 45.0F);
    FiberletStoredReplayGraphSource(
        std::shared_ptr<FiberletChunkDataset> anchorDataset,
        std::shared_ptr<vc::render::ChunkCache> anchorCache,
        std::shared_ptr<FiberletChunkDataset> fiberletDataset,
        std::shared_ptr<vc::render::ChunkCache> fiberletCache,
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
        const cv::Vec3d& maximumBaseXYZ,
        std::size_t prefetchWindow = 0) const;
    [[nodiscard]] const FiberletDatasetMetadata& metadata() const noexcept;

    // Include all edges incident to anchors in the half-open graph box and
    // their outside endpoints, but no joins at outside endpoints. The seed
    // box is separately bounded so lookahead may extend beyond the output crop.
    // Loading uses the source's existing cache; the returned graph owns its
    // geometry and supports concurrent, cache-independent queries.
    [[nodiscard]] FiberletMaterializedReplayGraph materializeBaseBox(
        const cv::Vec3d& minimumBaseXYZ, const cv::Vec3d& maximumBaseXYZ,
        std::size_t parallelThreads = 1) const;
    [[nodiscard]] FiberletMaterializedReplayGraph materializeBaseBoxForSeeds(
        const cv::Vec3d& minimumBaseXYZ, const cv::Vec3d& maximumBaseXYZ,
        const cv::Vec3d& seedMinimumBaseXYZ, const cv::Vec3d& seedMaximumBaseXYZ,
        std::size_t parallelThreads = 1) const;

private:
    std::shared_ptr<FiberletChunkDataset> dataset_;
    std::shared_ptr<vc::render::ChunkCache> anchorCache_;
    std::shared_ptr<vc::render::ChunkCache> pathCache_;
    FiberletPathConfig pathConfig_;
    FiberletChunkGraphSource chunks_;
    float maximumJoinAngleDegrees_ = 45.0F;
};


} // namespace vc::fiber_tracer
