#include "vc/fiber_tracer/FiberletChunkGraph.hpp"

#include "vc/fiber_tracer/FiberLocalScoring.hpp"
#include "utils/thread_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <exception>
#include <filesystem>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace vc::fiber_tracer
{
namespace
{

template <typename Function>
void indexedParallelFor(
    std::size_t count, std::size_t requestedThreads, Function&& function)
{
    if (count == 0)
        return;
    const std::size_t workerCount = std::min(
        count, std::max<std::size_t>(1, requestedThreads));
    if (workerCount == 1) {
        for (std::size_t index = 0; index < count; ++index)
            function(index);
        return;
    }

    const auto poolFor = [](std::size_t threads) -> utils::ThreadPool& {
        static std::mutex poolsMutex;
        static std::unordered_map<
            std::size_t, std::unique_ptr<utils::ThreadPool>> pools;
        std::lock_guard lock(poolsMutex);
        auto& pool = pools[threads];
        if (!pool)
            pool = std::make_unique<utils::ThreadPool>(threads);
        return *pool;
    };

    std::vector<std::exception_ptr> failures(count);
    poolFor(std::max<std::size_t>(1, requestedThreads))
        .run_indexed_batch(workerCount, [&](std::size_t worker) {
            // Each worker owns a fixed strided partition. The hot loop has no
            // shared scheduler or synchronization.
            for (std::size_t index = worker; index < count;
                 index += workerCount) {
                try {
                    function(index);
                } catch (...) {
                    failures[index] = std::current_exception();
                }
            }
        });
    for (const auto& failure : failures) {
        if (failure)
            std::rethrow_exception(failure);
    }
}

std::int64_t floorDiv(std::int64_t numerator, std::int64_t denominator)
{
    if (denominator <= 0)
        throw std::invalid_argument("fiberlet graph chunk divisor must be positive");
    auto quotient = numerator / denominator;
    if (numerator % denominator < 0)
        --quotient;
    return quotient;
}

vc::render::ChunkResult fetchChunk(const std::shared_ptr<vc::render::ChunkCache>& cache, const vc::render::ChunkKey& key, bool blocking)
{
    return blocking ? cache->getChunkBlocking(key.level, key.iz, key.iy, key.ix) : cache->tryGetChunk(key.level, key.iz, key.iy, key.ix);
}

template <typename T>
FiberletGraphQuery<T> chunkFailure(const vc::render::ChunkResult& chunk)
{
    FiberletGraphQuery<T> result;
    if (chunk.status == vc::render::ChunkStatus::MissQueued) {
        result.status = FiberletGraphQueryStatus::Pending;
    } else {
        result.status = FiberletGraphQueryStatus::Error;
        result.error = chunk.status == vc::render::ChunkStatus::Error ? chunk.error : "required generated fiberlet chunk is not data";
    }
    return result;
}

template <typename Payload, typename Query>
std::shared_ptr<const Payload> chunkPayload(const vc::render::ChunkResult& chunk, Query& result)
{
    if (chunk.status != vc::render::ChunkStatus::Data || !chunk.payload) {
        result.status = chunk.status == vc::render::ChunkStatus::MissQueued ? FiberletGraphQueryStatus::Pending : FiberletGraphQueryStatus::Error;
        result.error = chunk.status == vc::render::ChunkStatus::Error ? chunk.error : "required generated fiberlet chunk has no decoded payload";
        return {};
    }
    auto payload = std::dynamic_pointer_cast<const Payload>(chunk.payload);
    if (!payload) {
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "generated fiberlet chunk has the wrong decoded payload type";
    }
    return payload;
}

float vectorLength(const cv::Vec3f& value)
{
    return std::sqrt(value.dot(value));
}

cv::Vec3f normalized(const cv::Vec3f& value)
{
    const float magnitude = vectorLength(value);
    if (!(magnitude > 1.0e-6F) || !std::isfinite(magnitude))
        return {0.0F, 0.0F, 0.0F};
    return value / magnitude;
}

FiberletPathCost storedPathCost(const FiberletStoredPathCost& stored)
{
    FiberletPathCost result;
    result.invalidPrediction = stored.invalidPrediction;
    result.alignment = stored.alignment;
    result.isotropicSmoothness = stored.isotropicSmoothness;
    result.tangentSmoothness = stored.tangentSmoothness;
    result.normalSmoothness = stored.normalSmoothness;
    return result;
}

FiberletPathCost pathCost(const FiberLocalMetricCost& local)
{
    FiberletPathCost result;
    result.invalidPrediction = local.invalidPrediction;
    result.alignment = local.alignment;
    result.isotropicSmoothness = local.isotropicSmoothness;
    result.tangentSmoothness = local.tangentSmoothness;
    result.normalSmoothness = local.normalSmoothness;
    return result;
}

const FiberletStorageKey& directedSource(const DirectedFiberletStorageId& id)
{
    return id.reverse ? id.fiberlet.second : id.fiberlet.first;
}

const FiberletStorageKey& directedTarget(const DirectedFiberletStorageId& id)
{
    return id.reverse ? id.fiberlet.first : id.fiberlet.second;
}

using ChunkCoordinate = std::array<int, 3>;

FiberletReplaySourceArc directedStoredArc(
    const FiberletEdgeLease& loaded,
    const DirectedFiberletStorageId& id,
    float predictionToBaseScale)
{
    FiberletReplaySourceArc result;
    result.id = id;
    result.source = directedSource(id);
    result.target = directedTarget(id);
    result.pathLengthPredictionVoxels =
        loaded.prefix.pathLengthPredictionVoxels;
    result.cost = storedPathCost(loaded.prefix.cost);
    const cv::Vec3d firstPosition(
        loaded.firstAnchor.positionPredictionXYZ * predictionToBaseScale);
    const cv::Vec3d secondPosition(
        loaded.secondAnchor.positionPredictionXYZ * predictionToBaseScale);
    if (!id.reverse) {
        result.sourcePositionBaseXYZ = firstPosition;
        result.targetPositionBaseXYZ = secondPosition;
        result.startStepBaseXYZ = loaded.prefix.firstStepBaseXYZ;
        result.endStepBaseXYZ = loaded.prefix.lastStepBaseXYZ;
    } else {
        result.sourcePositionBaseXYZ = secondPosition;
        result.targetPositionBaseXYZ = firstPosition;
        result.startStepBaseXYZ = -loaded.prefix.lastStepBaseXYZ;
        result.endStepBaseXYZ = -loaded.prefix.firstStepBaseXYZ;
    }
    return result;
}

std::optional<FiberletReplaySourceTransition> storedTransition(
    const FiberletReplaySourceArc& incomingArc,
    const FiberletReplaySourceArc& outgoingArc,
    const FiberletStoredAnchor& sharedAnchor,
    float predictionToBaseScale,
    const FiberletPathConfig& pathConfig,
    float maximumJoinAngleDegrees)
{
    const auto& incoming = incomingArc.id;
    const auto& outgoing = outgoingArc.id;
    if (incoming.fiberlet == outgoing.fiberlet ||
        directedTarget(incoming) != directedSource(outgoing)) {
        return std::nullopt;
    }
    const cv::Vec3f incomingStepBase = incomingArc.endStepBaseXYZ;
    const cv::Vec3f outgoingStepBase = outgoingArc.startStepBaseXYZ;
    const cv::Vec3f incomingDirection = normalized(incomingStepBase);
    const cv::Vec3f outgoingDirection = normalized(outgoingStepBase);
    const float minimumJoinDot = std::cos(
        maximumJoinAngleDegrees *
        static_cast<float>(3.14159265358979323846 / 180.0));
    if (!(incomingDirection.dot(outgoingDirection) > minimumJoinDot) ||
        !sharedAnchor.predictionValid) {
        return std::nullopt;
    }
    const FiberLocalMetricSample prediction{
        sharedAnchor.predictionAxisXYZ,
        sharedAnchor.predictionPresence,
        true};
    const auto local = fiberLocalMetricCost(
        &prediction,
        prediction,
        incomingDirection,
        vectorLength(incomingStepBase) / predictionToBaseScale,
        outgoingDirection,
        vectorLength(outgoingStepBase) / predictionToBaseScale,
        sharedAnchor.normalXYZ,
        sharedAnchor.normalValid,
        FiberLocalMetricConfig{
            pathConfig.invalidPredictionCostPerVoxel,
            FiberLocalSmoothnessConfig{
                pathConfig.smoothnessWeight,
                pathConfig.smoothnessNormalWeight,
                pathConfig.smoothnessTangentWeight,
                pathConfig.smoothnessFreeAngleDegrees *
                    static_cast<float>(3.14159265358979323846 / 180.0)}});
    return FiberletReplaySourceTransition{
        incoming, outgoing, pathCost(local)};
}

}  // namespace

struct FiberletCachedReplayGraphSource::QuantizationState {
    mutable std::mutex mutex;
    std::size_t projectedAnchors = 0;
    struct Contribution {
        float minimum = std::numeric_limits<float>::infinity();
        float maximum = -std::numeric_limits<float>::infinity();
        bool populated = false;
    };
    std::map<ChunkCoordinate, std::map<ChunkCoordinate, Contribution>> physicalCostContributions;
    std::map<ChunkCoordinate, Contribution> compactCostRanges;
    std::size_t coincidentPositionGroups = 0;
    std::size_t maximumVariants = 0;
};

FiberletChunkGraphSource::FiberletChunkGraphSource(
    std::shared_ptr<FiberletChunkDataset> anchorDataset,
    std::shared_ptr<vc::render::ChunkCache> anchorCache,
    std::shared_ptr<FiberletChunkDataset> fiberletDataset,
    std::shared_ptr<vc::render::ChunkCache> fiberletCache,
    FiberletPathConfig pathConfig,
    FiberletAnchorView anchorView)
    : anchorDataset_(std::move(anchorDataset))
    , anchorCache_(std::move(anchorCache))
    , fiberletDataset_(std::move(fiberletDataset))
    , fiberletCache_(std::move(fiberletCache))
    , pathConfig_(std::move(pathConfig))
    , anchorView_(std::move(anchorView))
{
    if (!anchorDataset_ || !anchorCache_ || !fiberletDataset_ || !fiberletCache_)
        throw std::invalid_argument("fiberlet chunk graph requires both datasets and caches");
    if ((anchorDataset_->metadata().kind != FiberletDatasetKind::Anchors && anchorDataset_->metadata().kind != FiberletDatasetKind::Combined) ||
        (fiberletDataset_->metadata().kind != FiberletDatasetKind::Fiberlets && fiberletDataset_->metadata().kind != FiberletDatasetKind::Combined))
        throw std::invalid_argument("fiberlet chunk graph dataset kinds are invalid");
    if (anchorDataset_->metadata().profile != fiberletDataset_->metadata().profile ||
        anchorDataset_->metadata().chunkGridShapeZYX != fiberletDataset_->metadata().chunkGridShapeZYX ||
        anchorDataset_->metadata().coordinateOriginZYX != fiberletDataset_->metadata().coordinateOriginZYX ||
        anchorDataset_->metadata().coordinateUnitsPerChunkZYX != fiberletDataset_->metadata().coordinateUnitsPerChunkZYX ||
        anchorDataset_->metadata().maximumEndpointReachCoordinateUnitsZYX != fiberletDataset_->metadata().maximumEndpointReachCoordinateUnitsZYX ||
        anchorDataset_->metadata().spatialChunkSideBaseVoxels != fiberletDataset_->metadata().spatialChunkSideBaseVoxels ||
        anchorDataset_->metadata().predictionToBaseScale != fiberletDataset_->metadata().predictionToBaseScale ||
        anchorDataset_->metadata().sources != fiberletDataset_->metadata().sources)
        throw std::invalid_argument("fiberlet chunk graph datasets are incompatible");
    if (!anchorView_) {
        anchorView_ = [](const vc::render::ChunkKey&, std::shared_ptr<const FiberletAnchorChunkPayload> payload) {
            const auto* anchors = &payload->anchors;
            return std::shared_ptr<const std::vector<FiberletStoredAnchor>>(std::move(payload), anchors);
        };
    }
    validateFiberletPathConfig(pathConfig_);
}

vc::render::ChunkKey FiberletChunkGraphSource::ownerChunk(const FiberletStorageKey& anchor, int level) const
{
    return fiberletStorageOwnerChunk(fiberletDataset_->metadata(), anchor, level);
}

vc::render::ChunkKey fiberletStorageOwnerChunk(
    const FiberletDatasetMetadata& metadata,
    const FiberletStorageKey& anchor,
    int level)
{
    std::array<std::int64_t, 3> coordinate{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        coordinate[axis] = floorDiv(anchor.coordinateZYX[axis] - metadata.coordinateOriginZYX[axis], metadata.coordinateUnitsPerChunkZYX[axis]);
        if (coordinate[axis] < 0 || coordinate[axis] >= metadata.chunkGridShapeZYX[axis])
            throw std::out_of_range("fiberlet anchor lies outside the dataset grid");
    }
    return {level, static_cast<int>(coordinate[0]), static_cast<int>(coordinate[1]), static_cast<int>(coordinate[2])};
}

std::vector<vc::render::ChunkKey> FiberletChunkGraphSource::incidentOwnerChunks(const FiberletStorageKey& anchor) const
{
    const auto& metadata = fiberletDataset_->metadata();
    std::array<std::int64_t, 3> begin{};
    std::array<std::int64_t, 3> end{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto relative = anchor.coordinateZYX[axis] - metadata.coordinateOriginZYX[axis];
        begin[axis] = floorDiv(relative - metadata.maximumEndpointReachCoordinateUnitsZYX[axis], metadata.coordinateUnitsPerChunkZYX[axis]);
        // Canonical endpoint ordering is lexicographic, not component-wise.
        // A first endpoint may therefore be greater on a lower-priority axis.
        // Load the complete bounded reach cube and filter exact endpoint IDs.
        end[axis] = floorDiv(relative + metadata.maximumEndpointReachCoordinateUnitsZYX[axis], metadata.coordinateUnitsPerChunkZYX[axis]);
    }
    std::vector<vc::render::ChunkKey> result;
    for (auto z = begin[0]; z <= end[0]; ++z) {
        for (auto y = begin[1]; y <= end[1]; ++y) {
            for (auto x = begin[2]; x <= end[2]; ++x) {
                if (z >= 0 && y >= 0 && x >= 0 && z < metadata.chunkGridShapeZYX[0] && y < metadata.chunkGridShapeZYX[1] &&
                    x < metadata.chunkGridShapeZYX[2])
                    result.push_back({0, static_cast<int>(z), static_cast<int>(y), static_cast<int>(x)});
            }
        }
    }
    return result;
}

FiberletGraphQuery<FiberletIncidentLease> FiberletChunkGraphSource::incidentEdges(const FiberletStorageKey& anchorKey, bool blocking) const
{
    FiberletGraphQuery<FiberletIncidentLease> result;
    const auto keys = incidentOwnerChunks(anchorKey);
    fiberletCache_->prefetchChunks(keys, blocking);
    std::vector<vc::render::ChunkResult> chunks;
    chunks.reserve(keys.size());
    bool pending = false;
    for (const auto& key : keys) {
        auto chunk = fetchChunk(fiberletCache_, key, blocking);
        if (chunk.status == vc::render::ChunkStatus::MissQueued) {
            pending = true;
        } else if (chunk.status != vc::render::ChunkStatus::Data || !chunk.payload) {
            return chunkFailure<FiberletIncidentLease>(chunk);
        }
        chunks.push_back(std::move(chunk));
    }
    if (pending) {
        result.status = FiberletGraphQueryStatus::Pending;
        return result;
    }
    for (const auto& chunk : chunks) {
        auto payload = chunkPayload<FiberletPrefixChunkPayload>(chunk, result);
        if (!payload)
            return result;
        for (auto&& incident : payload->incident(anchorKey))
            result.value.edges.push_back({incident.id, std::move(incident.prefix)});
        result.value.payloadLeases.push_back(std::move(payload));
    }
    std::sort(result.value.edges.begin(), result.value.edges.end(), [](const auto& left, const auto& right) { return left.id < right.id; });
    const auto duplicate = std::adjacent_find(result.value.edges.begin(), result.value.edges.end(), [](const auto& left, const auto& right) {
        return left.id == right.id;
    });
    if (duplicate != result.value.edges.end()) {
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet graph contains a duplicate incident edge";
        result.value = {};
        return result;
    }
    result.status = FiberletGraphQueryStatus::Ready;
    return result;
}

FiberletGraphQuery<FiberletAnchorLease> FiberletChunkGraphSource::anchor(const FiberletStorageKey& anchorKey, bool blocking) const
{
    const auto key = ownerChunk(anchorKey, 0);
    auto chunk = fetchChunk(anchorCache_, key, blocking);
    FiberletGraphQuery<FiberletAnchorLease> result;
    auto payload = chunkPayload<FiberletAnchorChunkPayload>(chunk, result);
    if (!payload)
        return result;
    const auto transformed = anchorView_(key, payload);
    if (!transformed || transformed->size() != payload->anchors.size()) {
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet graph anchor view changed the chunk count";
        return result;
    }
    const auto found =
        std::lower_bound(transformed->begin(), transformed->end(), anchorKey, [](const FiberletStoredAnchor& anchor, const FiberletStorageKey& key) {
            return anchor.key < key;
        });
    if (found == transformed->end() || found->key != anchorKey) {
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet graph endpoint anchor is absent from its owner chunk";
        return result;
    }
    result.status = FiberletGraphQueryStatus::Ready;
    result.value.payloadLease = std::move(payload);
    result.value.anchor = *found;
    return result;
}

FiberletGraphQuery<FiberletAnchorChunkLease> FiberletChunkGraphSource::anchorsInChunk(const vc::render::ChunkKey& key, bool blocking) const
{
    if (key.level != 0) {
        FiberletGraphQuery<FiberletAnchorChunkLease> result;
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet anchor chunk query requires level zero";
        return result;
    }
    auto chunk = fetchChunk(anchorCache_, key, blocking);
    FiberletGraphQuery<FiberletAnchorChunkLease> result;
    auto payload = chunkPayload<FiberletAnchorChunkPayload>(chunk, result);
    if (!payload)
        return result;
    result.status = FiberletGraphQueryStatus::Ready;
    result.value.anchors = anchorView_(key, payload);
    if (!result.value.anchors || result.value.anchors->size() != payload->anchors.size()) {
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet graph anchor view changed the chunk count";
        return result;
    }
    result.value.payloadLease = std::move(payload);
    return result;
}

FiberletGraphQuery<FiberletPrefixChunkLease> FiberletChunkGraphSource::prefixesInChunk(const vc::render::ChunkKey& key, bool blocking) const
{
    if (key.level != 0) {
        FiberletGraphQuery<FiberletPrefixChunkLease> result;
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet prefix chunk query requires level zero";
        return result;
    }
    auto chunk = fetchChunk(fiberletCache_, key, blocking);
    FiberletGraphQuery<FiberletPrefixChunkLease> result;
    auto payload = chunkPayload<FiberletPrefixChunkPayload>(chunk, result);
    if (!payload)
        return result;
    result.status = FiberletGraphQueryStatus::Ready;
    result.value.payloadLease = std::move(payload);
    return result;
}

FiberletGraphQuery<FiberletRouteChunkLease>
FiberletChunkGraphSource::routesInChunk(
    const vc::render::ChunkKey& requested, bool blocking) const
{
    auto key = requested;
    if (key.level == 0)
        key.level = 1;
    if (key.level != 1) {
        FiberletGraphQuery<FiberletRouteChunkLease> result;
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet route chunk query requires level one";
        return result;
    }
    auto chunk = fetchChunk(fiberletCache_, key, blocking);
    FiberletGraphQuery<FiberletRouteChunkLease> result;
    auto payload = chunkPayload<FiberletRouteChunkPayload>(chunk, result);
    if (!payload)
        return result;
    result.status = FiberletGraphQueryStatus::Ready;
    result.value.payloadLease = std::move(payload);
    return result;
}

std::vector<vc::render::ChunkKey>
FiberletChunkGraphSource::incidentPrefixChunks(
    std::span<const FiberletStorageKey> anchors) const
{
    std::set<std::tuple<int, int, int>> coordinates;
    for (const auto& anchor : anchors) {
        for (const auto& chunk : incidentOwnerChunks(anchor))
            coordinates.emplace(chunk.iz, chunk.iy, chunk.ix);
    }
    std::vector<vc::render::ChunkKey> result;
    result.reserve(coordinates.size());
    for (const auto& [z, y, x] : coordinates)
        result.push_back({0, z, y, x});
    return result;
}

FiberletGraphQuery<FiberletEdgeLease> FiberletChunkGraphSource::edge(const FiberletStorageId& fiberlet, bool blocking) const
{
    const auto prefixKey = ownerChunk(fiberlet.first, 0);
    const auto firstAnchorKey = ownerChunk(fiberlet.first, 0);
    const auto secondAnchorKey = ownerChunk(fiberlet.second, 0);
    fiberletCache_->prefetchChunks({prefixKey}, blocking);
    anchorCache_->prefetchChunks({firstAnchorKey, secondAnchorKey}, blocking);

    FiberletGraphQuery<FiberletEdgeLease> result;
    const auto prefixChunk = fetchChunk(fiberletCache_, prefixKey, blocking);
    auto prefixPayload = chunkPayload<FiberletPrefixChunkPayload>(prefixChunk, result);
    if (!prefixPayload)
        return result;
    const auto* prefix = prefixPayload->find(fiberlet);
    if (!prefix) {
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet edge is absent from its owner chunk";
        return result;
    }
    const auto firstAnchor = anchor(fiberlet.first, blocking);
    if (firstAnchor.status != FiberletGraphQueryStatus::Ready) {
        result.status = firstAnchor.status;
        result.error = firstAnchor.error;
        return result;
    }
    const auto secondAnchor = anchor(fiberlet.second, blocking);
    if (secondAnchor.status != FiberletGraphQueryStatus::Ready) {
        result.status = secondAnchor.status;
        result.error = secondAnchor.error;
        return result;
    }
    result.status = FiberletGraphQueryStatus::Ready;
    result.value.prefixPayloadLease = std::move(prefixPayload);
    result.value.anchorPayloadLeases = {firstAnchor.value.payloadLease, secondAnchor.value.payloadLease};
    result.value.prefix = *prefix;
    result.value.firstAnchor = firstAnchor.value.anchor;
    result.value.secondAnchor = secondAnchor.value.anchor;
    return result;
}

FiberletGraphQuery<FiberletStoredRouteLease>
FiberletChunkGraphSource::storedRoute(
    const FiberletStorageId& fiberlet, bool blocking) const
{
    const auto prefixKey = ownerChunk(fiberlet.first, 0);
    auto routeKey = prefixKey;
    routeKey.level = 1;
    fiberletCache_->prefetchChunks({prefixKey, routeKey}, blocking);
    FiberletGraphQuery<FiberletStoredRouteLease> result;
    const auto prefixChunk = fetchChunk(fiberletCache_, prefixKey, blocking);
    auto prefixPayload = chunkPayload<FiberletPrefixChunkPayload>(prefixChunk, result);
    if (!prefixPayload)
        return result;
    const auto routeChunk = fetchChunk(fiberletCache_, routeKey, blocking);
    auto routePayload = chunkPayload<FiberletRouteChunkPayload>(routeChunk, result);
    if (!routePayload)
        return result;
    if (prefixPayload->prefixes.size() != routePayload->routes.size()) {
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet prefix and route chunk record counts differ";
        return result;
    }
    const auto* found = prefixPayload->find(fiberlet);
    if (!found) {
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet route is absent from its owner chunk";
        return result;
    }
    const auto index = static_cast<std::size_t>(found - prefixPayload->prefixes.data());
    result.status = FiberletGraphQueryStatus::Ready;
    result.value.prefixPayloadLease = std::move(prefixPayload);
    result.value.routePayloadLease = std::move(routePayload);
    result.value.prefix = *found;
    result.value.route = result.value.routePayloadLease->routes[index];
    return result;
}

FiberletGraphQuery<FiberletRouteLease> FiberletChunkGraphSource::route(
    const FiberletStorageId& fiberlet, bool blocking) const
{
    FiberletGraphQuery<FiberletRouteLease> result;
    auto stored = storedRoute(fiberlet, blocking);
    if (stored.status != FiberletGraphQueryStatus::Ready) {
        result.status = stored.status;
        result.error = stored.error;
        return result;
    }
    result.status = FiberletGraphQueryStatus::Ready;
    result.value.prefixPayloadLease =
        std::move(stored.value.prefixPayloadLease);
    result.value.routePayloadLease =
        std::move(stored.value.routePayloadLease);
    result.value.prefix = stored.value.prefix;
    result.value.route = stored.value.route;
    const auto firstAnchor = anchor(fiberlet.first, blocking);
    if (firstAnchor.status != FiberletGraphQueryStatus::Ready) {
        result.status = firstAnchor.status;
        result.error = firstAnchor.error;
        result.value = {};
        return result;
    }
    const auto secondAnchor = anchor(fiberlet.second, blocking);
    if (secondAnchor.status != FiberletGraphQueryStatus::Ready) {
        result.status = secondAnchor.status;
        result.error = secondAnchor.error;
        result.value = {};
        return result;
    }
    std::vector<std::array<std::int16_t, 2>> lattice;
    if (result.value.prefix.interiorPointCount == 1) {
        lattice.push_back(result.value.prefix.entryUV);
    } else if (result.value.prefix.interiorPointCount >= 2) {
        if (result.value.route.middleUV.size() + 2 != result.value.prefix.interiorPointCount) {
            result.status = FiberletGraphQueryStatus::Error;
            result.error = "fiberlet route interior count differs from its prefix";
            return result;
        }
        lattice.push_back(result.value.prefix.entryUV);
        lattice.insert(lattice.end(), result.value.route.middleUV.begin(), result.value.route.middleUV.end());
        lattice.push_back(result.value.prefix.exitUV);
    } else if (!result.value.route.middleUV.empty()) {
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "direct fiberlet route unexpectedly has interior geometry";
        return result;
    }
    const size_t expectedSegmentCosts =
        static_cast<size_t>(result.value.prefix.interiorPointCount) + 1;
    if (result.value.route.segmentCostDensities.size() != expectedSegmentCosts) {
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet route cost-density count differs from its geometry";
        return result;
    }
    result.value.anchorPayloadLeases = {firstAnchor.value.payloadLease, secondAnchor.value.payloadLease};
    result.value.pointsPredictionXYZ = reconstructFiberletRoutePoints(
        firstAnchor.value.anchor.positionPredictionXYZ,
        firstAnchor.value.anchor.fittedAxisXYZ,
        secondAnchor.value.anchor.positionPredictionXYZ,
        secondAnchor.value.anchor.fittedAxisXYZ,
        lattice,
        pathConfig_);
    return result;
}

FiberletGraphQuery<FiberletReplaySourceArc>
FiberletChunkGraphSource::directedEdge(
    const DirectedFiberletStorageId& fiberlet, bool blocking) const
{
    FiberletGraphQuery<FiberletReplaySourceArc> result;
    const auto loaded = edge(fiberlet.fiberlet, blocking);
    if (loaded.status != FiberletGraphQueryStatus::Ready) {
        result.status = loaded.status;
        result.error = loaded.error;
        return result;
    }
    result.status = FiberletGraphQueryStatus::Ready;
    result.value = directedStoredArc(
        loaded.value, fiberlet,
        static_cast<float>(fiberletDataset_->metadata().predictionToBaseScale));
    return result;
}

FiberletGraphQuery<std::optional<FiberletReplaySourceTransition>>
FiberletChunkGraphSource::transition(
    const FiberletReplaySourceArc& incoming,
    const FiberletReplaySourceArc& outgoing,
    float maximumJoinAngleDegrees,
    bool blocking) const
{
    FiberletGraphQuery<std::optional<FiberletReplaySourceTransition>> result;
    if (!(maximumJoinAngleDegrees >= 0.0F) ||
        !(maximumJoinAngleDegrees <= 180.0F) ||
        !std::isfinite(maximumJoinAngleDegrees)) {
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet transition maximum join angle is invalid";
        return result;
    }
    if (incoming.id.fiberlet == outgoing.id.fiberlet ||
        incoming.target != outgoing.source) {
        result.status = FiberletGraphQueryStatus::Ready;
        return result;
    }
    const auto shared = anchor(incoming.target, blocking);
    if (shared.status != FiberletGraphQueryStatus::Ready) {
        result.status = shared.status;
        result.error = shared.error;
        return result;
    }
    return transitionAtAnchor(
        incoming, outgoing, shared.value.anchor, maximumJoinAngleDegrees);
}

FiberletGraphQuery<std::optional<FiberletReplaySourceTransition>>
FiberletChunkGraphSource::transitionAtAnchor(
    const FiberletReplaySourceArc& incoming,
    const FiberletReplaySourceArc& outgoing,
    const FiberletStoredAnchor& sharedAnchor,
    float maximumJoinAngleDegrees) const
{
    FiberletGraphQuery<std::optional<FiberletReplaySourceTransition>> result;
    if (!(maximumJoinAngleDegrees >= 0.0F) ||
        !(maximumJoinAngleDegrees <= 180.0F) ||
        !std::isfinite(maximumJoinAngleDegrees)) {
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet transition maximum join angle is invalid";
        return result;
    }
    result.status = FiberletGraphQueryStatus::Ready;
    result.value = storedTransition(
        incoming, outgoing, sharedAnchor,
        static_cast<float>(fiberletDataset_->metadata().predictionToBaseScale),
        pathConfig_, maximumJoinAngleDegrees);
    return result;
}

const FiberletDatasetMetadata& FiberletChunkGraphSource::metadata() const noexcept
{
    return fiberletDataset_->metadata();
}

FiberletPathConfig fiberletPathConfigFromDatasetMetadata(
    const FiberletDatasetMetadata& metadata,
    int parallelThreads)
{
    if (parallelThreads < 1)
        throw std::invalid_argument(
            "fiberlet stored path parallel thread count must be positive");
    const auto& paths = metadata.processing.at("paths");
    if (!paths.is_object())
        throw std::invalid_argument(
            "fiberlet dataset processing.paths must be an object");
    FiberletPathConfig result;
    result.cellRadius = paths.at("cell_radius").get<int>();
    result.neighborhoodMarginCells =
        paths.at("neighborhood_margin_cells").get<float>();
    result.longitudinalStepPredictionVoxels =
        paths.at("longitudinal_step_prediction").get<float>();
    result.transverseStepPredictionVoxels =
        paths.at("transverse_step_prediction").get<float>();
    result.maximumEndpointAngleDegrees =
        paths.at("maximum_endpoint_angle_degrees").get<float>();
    result.maximumPredictionDeviationDegrees =
        paths.at("maximum_prediction_deviation_degrees").get<float>();
    result.corridorRadiusPredictionVoxels =
        paths.at("corridor_radius_prediction").get<float>();
    result.invalidPredictionCostPerVoxel =
        paths.at("invalid_prediction_cost_per_voxel").get<float>();
    result.smoothnessWeight = paths.at("smoothness_weight").get<float>();
    result.smoothnessNormalWeight =
        paths.at("smoothness_normal_weight").get<float>();
    result.smoothnessTangentWeight =
        paths.at("smoothness_tangent_weight").get<float>();
    result.smoothnessFreeAngleDegrees =
        paths.at("smoothness_free_angle_degrees").get<float>();
    result.parallelThreads = parallelThreads;
    validateFiberletPathConfig(result);
    return result;
}

FiberletStoredReplayGraphSource::FiberletStoredReplayGraphSource(
    std::shared_ptr<FiberletChunkDataset> dataset,
    FiberletChunkCacheOptions cacheOptions,
    float maximumJoinAngleDegrees)
    : dataset_(std::move(dataset))
    , anchorCache_(createStoredFiberletAnchorChunkCache(dataset_, cacheOptions))
    , pathCache_(createStoredFiberletPathChunkCache(dataset_, cacheOptions))
    , pathConfig_(fiberletPathConfigFromDatasetMetadata(
          dataset_->metadata(),
          static_cast<int>(std::max<std::size_t>(
              1, cacheOptions.service.fetchConcurrency.workerCapacity))))
    , chunks_(dataset_, anchorCache_, dataset_, pathCache_, pathConfig_)
    , maximumJoinAngleDegrees_(maximumJoinAngleDegrees)
{
    if (dataset_->metadata().kind != FiberletDatasetKind::Combined)
        throw std::invalid_argument(
            "stored replay graph requires a combined Fiberlet dataset");
    if (!(maximumJoinAngleDegrees_ >= 0.0F) ||
        !(maximumJoinAngleDegrees_ <= 180.0F) ||
        !std::isfinite(maximumJoinAngleDegrees_)) {
        throw std::invalid_argument(
            "stored replay graph maximum join angle is invalid");
    }
}

float FiberletStoredReplayGraphSource::predictionToBaseScale() const noexcept
{
    return static_cast<float>(dataset_->metadata().predictionToBaseScale);
}

int FiberletStoredReplayGraphSource::anchorCellSizePredictionVoxels() const noexcept
{
    const auto& metadata = dataset_->metadata();
    const double cellBase =
        static_cast<double>(metadata.spatialChunkSideBaseVoxels) /
        static_cast<double>(metadata.coordinateUnitsPerChunkZYX[0]);
    return static_cast<int>(
        std::llround(cellBase / metadata.predictionToBaseScale));
}

float FiberletStoredReplayGraphSource::maximumJoinAngleDegrees() const noexcept
{
    return maximumJoinAngleDegrees_;
}

std::vector<FiberletStoredAnchor>
FiberletStoredReplayGraphSource::anchorsInBaseBox(
    const cv::Vec3d& minimumBaseXYZ,
    const cv::Vec3d& maximumBaseXYZ) const
{
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(minimumBaseXYZ[axis]) ||
            !std::isfinite(maximumBaseXYZ[axis]) ||
            !(maximumBaseXYZ[axis] > minimumBaseXYZ[axis])) {
            throw std::invalid_argument(
                "fiberlet anchor query box must be finite and nonempty");
        }
    }
    const auto& metadata = dataset_->metadata();
    std::array<int, 3> begin{};
    std::array<int, 3> end{};
    for (std::size_t zyx = 0; zyx < 3; ++zyx) {
        const std::size_t xyz = 2 - zyx;
        const double chunkSide =
            static_cast<double>(metadata.spatialChunkSideBaseVoxels);
        const double cellSide = chunkSide /
            static_cast<double>(metadata.coordinateUnitsPerChunkZYX[zyx]);
        const double origin =
            static_cast<double>(metadata.coordinateOriginZYX[zyx]) * cellSide;
        begin[zyx] = static_cast<int>(std::floor(
            (minimumBaseXYZ[static_cast<int>(xyz)] - origin) / chunkSide));
        end[zyx] = static_cast<int>(std::floor(
            (std::nextafter(
                 maximumBaseXYZ[static_cast<int>(xyz)],
                 -std::numeric_limits<double>::infinity()) - origin) /
            chunkSide));
        begin[zyx] = std::max(begin[zyx], 0);
        end[zyx] = std::min(
            end[zyx], metadata.chunkGridShapeZYX[zyx] - 1);
    }

    std::vector<FiberletStoredAnchor> result;
    if (begin[0] > end[0] || begin[1] > end[1] || begin[2] > end[2])
        return result;
    const float scale = predictionToBaseScale();
    for (int z = begin[0]; z <= end[0]; ++z) {
        for (int y = begin[1]; y <= end[1]; ++y) {
            for (int x = begin[2]; x <= end[2]; ++x) {
                const auto loaded = chunks_.anchorsInChunk({0, z, y, x}, true);
                if (loaded.status != FiberletGraphQueryStatus::Ready) {
                    throw std::runtime_error(
                        "stored Fiberlet anchor chunk failed: " + loaded.error);
                }
                for (const auto& anchor : *loaded.value.anchors) {
                    const cv::Vec3d point(anchor.positionPredictionXYZ * scale);
                    bool inside = true;
                    for (int axis = 0; axis < 3; ++axis) {
                        inside = inside &&
                            point[axis] >= minimumBaseXYZ[axis] &&
                            point[axis] < maximumBaseXYZ[axis];
                    }
                    if (inside)
                        result.push_back(anchor);
                }
            }
        }
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
        if (left.predictionPresence != right.predictionPresence)
            return left.predictionPresence > right.predictionPresence;
        return left.key < right.key;
    });
    return result;
}

std::vector<FiberletReplaySourceAnchor>
FiberletStoredReplayGraphSource::anchorsNearReference(
    const PolylineArcGeometry& reference,
    double beginArcBase,
    double endArcBase,
    double broadPhaseRadiusBaseVoxels) const
{
    if (reference.points.empty())
        return {};
    cv::Vec3d minimum = reference.points.front();
    cv::Vec3d maximum = reference.points.front();
    for (const auto& point : reference.points) {
        for (int axis = 0; axis < 3; ++axis) {
            minimum[axis] = std::min(minimum[axis], point[axis]);
            maximum[axis] = std::max(maximum[axis], point[axis]);
        }
    }
    minimum -= cv::Vec3d::all(broadPhaseRadiusBaseVoxels);
    maximum += cv::Vec3d::all(broadPhaseRadiusBaseVoxels);
    for (int axis = 0; axis < 3; ++axis)
        maximum[axis] = std::nextafter(
            maximum[axis], std::numeric_limits<double>::infinity());
    std::vector<FiberletReplaySourceAnchor> result;
    for (const auto& anchor : anchorsInBaseBox(minimum, maximum)) {
        const cv::Vec3d point(anchor.positionPredictionXYZ * predictionToBaseScale());
        const auto projection = projectPointToPolylineArc(
            reference, point, beginArcBase, endArcBase);
        if (projection.distance <= broadPhaseRadiusBaseVoxels)
            result.push_back({anchor.key, point});
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
        return left.id < right.id;
    });
    return result;
}

std::vector<DirectedFiberletStorageId>
FiberletStoredReplayGraphSource::outgoing(
    const FiberletStorageKey& anchor) const
{
    const auto loaded = chunks_.incidentEdges(anchor, true);
    if (loaded.status != FiberletGraphQueryStatus::Ready)
        throw std::runtime_error(
            "stored Fiberlet adjacency failed: " + loaded.error);
    std::vector<DirectedFiberletStorageId> result;
    result.reserve(loaded.value.edges.size());
    for (const auto& edge : loaded.value.edges)
        result.push_back(edge.id);
    std::sort(result.begin(), result.end());
    return result;
}

FiberletReplaySourceArc FiberletStoredReplayGraphSource::arc(
    const DirectedFiberletStorageId& id) const
{
    const auto loaded = chunks_.directedEdge(id, true);
    if (loaded.status != FiberletGraphQueryStatus::Ready)
        throw std::runtime_error(
            "stored Fiberlet edge failed: " + loaded.error);
    return loaded.value;
}

FiberletReplaySourceCostProfile
FiberletStoredReplayGraphSource::costProfile(
    const DirectedFiberletStorageId& id) const
{
    const auto loaded = chunks_.route(id.fiberlet, true);
    if (loaded.status != FiberletGraphQueryStatus::Ready)
        throw std::runtime_error(
            "stored Fiberlet cost profile failed: " + loaded.error);
    FiberletReplaySourceCostProfile result;
    result.segmentLengthsPredictionVoxels.reserve(
        loaded.value.pointsPredictionXYZ.size() - 1);
    for (std::size_t index = 1;
         index < loaded.value.pointsPredictionXYZ.size(); ++index) {
        const float length = vectorLength(
            loaded.value.pointsPredictionXYZ[index] -
            loaded.value.pointsPredictionXYZ[index - 1]);
        if (!(length > 0.0F) || !std::isfinite(length))
            throw std::runtime_error(
                "stored Fiberlet route segment length is invalid");
        result.segmentLengthsPredictionVoxels.push_back(length);
    }
    result.segmentCostDensities = loaded.value.route.segmentCostDensities;
    if (id.reverse) {
        std::reverse(
            result.segmentLengthsPredictionVoxels.begin(),
            result.segmentLengthsPredictionVoxels.end());
        std::reverse(
            result.segmentCostDensities.begin(),
            result.segmentCostDensities.end());
    }
    return result;
}

std::vector<cv::Vec3d> FiberletStoredReplayGraphSource::routePoints(
    const DirectedFiberletStorageId& id) const
{
    const auto loaded = chunks_.route(id.fiberlet, true);
    if (loaded.status != FiberletGraphQueryStatus::Ready)
        throw std::runtime_error(
            "stored Fiberlet route failed: " + loaded.error);
    std::vector<cv::Vec3d> result;
    result.reserve(loaded.value.pointsPredictionXYZ.size());
    for (const auto& point : loaded.value.pointsPredictionXYZ)
        result.emplace_back(point * predictionToBaseScale());
    if (id.reverse)
        std::reverse(result.begin(), result.end());
    return result;
}

std::optional<FiberletReplaySourceTransition>
FiberletStoredReplayGraphSource::transition(
    const FiberletReplaySourceArc& incoming,
    const FiberletReplaySourceArc& outgoing) const
{
    const auto loaded = chunks_.transition(
        incoming, outgoing, maximumJoinAngleDegrees_, true);
    if (loaded.status != FiberletGraphQueryStatus::Ready)
        throw std::runtime_error(
            "stored Fiberlet transition failed: " + loaded.error);
    return loaded.value;
}

const FiberletDatasetMetadata&
FiberletStoredReplayGraphSource::metadata() const noexcept
{
    return dataset_->metadata();
}

namespace
{

struct ChunkRouteAnchor {
    FiberletStorageKey key;
    FiberletStoredAnchor stored;
    cv::Vec3d positionBaseXYZ{0.0, 0.0, 0.0};
    bool inside = false;
};

struct ChunkRouteArc {
    FiberletReplaySourceArc source;
    std::size_t sourceAnchor = 0;
    std::size_t targetAnchor = 0;
    double loss = 0.0;
    double lengthPredictionVoxels = 0.0;
};

struct ChunkRouteSuccessor {
    std::size_t arc = 0;
    double joinLoss = 0.0;
};

struct ChunkRouteLocalGraph {
    std::vector<ChunkRouteAnchor> anchors;
    std::vector<FiberletStorageId> physicalFiberlets;
    std::vector<ChunkRouteArc> arcs;
    std::vector<std::vector<std::size_t>> outgoing;
    std::vector<std::vector<ChunkRouteSuccessor>> successors;
    std::vector<std::size_t> entries;
    std::size_t exits = 0;
    std::size_t admissibleTransitions = 0;
};

bool pointInsideChunk(
    const cv::Vec3d& point,
    const FiberletChunkRouteAnalysisConfig& config)
{
    for (int axis = 0; axis < 3; ++axis) {
        if (!(point[axis] >= config.minimumBaseXYZ[axis] &&
              point[axis] < config.maximumBaseXYZ[axis])) {
            return false;
        }
    }
    return true;
}

FiberletPathCost chunkRouteEdgeCost(
    FiberletPathCost cost,
    float lengthPredictionVoxels,
    FiberletChunkRouteEdgeCostView view)
{
    if (view == FiberletChunkRouteEdgeCostView::Stored)
        return cost;
    const float total = quantizeFiberletCostForEvaluation(
        cost.total(), lengthPredictionVoxels, 0.0F, 1.0F, 16,
        FiberletCostQuantizationDomain::SqrtPerPredictionVoxel,
        kFiberletStoredCostDensityMaximum);
    return {total, 0.0F, 0.0F, 0.0F, 0.0F};
}

FiberletChunkRouteDistribution chunkRouteDistribution(
    std::vector<double> values)
{
    FiberletChunkRouteDistribution result;
    result.count = values.size();
    if (values.empty())
        return result;
    std::sort(values.begin(), values.end());
    result.minimum = values.front();
    result.maximum = values.back();
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());
    const std::size_t middle = values.size() / 2;
    result.median = values.size() % 2 == 0
        ? 0.5 * (values[middle - 1] + values[middle])
        : values[middle];
    return result;
}

std::vector<vc::render::ChunkKey> analysisAnchorChunks(
    const FiberletDatasetMetadata& metadata,
    const FiberletChunkRouteAnalysisConfig& config)
{
    std::array<int, 3> begin{};
    std::array<int, 3> end{};
    for (std::size_t zyx = 0; zyx < 3; ++zyx) {
        const std::size_t xyz = 2 - zyx;
        const double units = static_cast<double>(
            metadata.coordinateUnitsPerChunkZYX[zyx]);
        const double side = static_cast<double>(
            metadata.spatialChunkSideBaseVoxels);
        if (!(units > 0.0) || !(side > 0.0))
            throw std::invalid_argument(
                "fiberlet chunk-route storage geometry is invalid");
        const double cellSide = side / units;
        const double origin =
            static_cast<double>(metadata.coordinateOriginZYX[zyx]) * cellSide;
        const double maximumInside = std::nextafter(
            config.maximumBaseXYZ[static_cast<int>(xyz)],
            -std::numeric_limits<double>::infinity());
        begin[zyx] = static_cast<int>(std::floor(
            (config.minimumBaseXYZ[static_cast<int>(xyz)] - origin) / side));
        end[zyx] = static_cast<int>(std::floor(
            (maximumInside - origin) / side));
        begin[zyx] = std::max(begin[zyx], 0);
        end[zyx] = std::min(
            end[zyx], metadata.chunkGridShapeZYX[zyx] - 1);
    }
    std::vector<vc::render::ChunkKey> result;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (begin[axis] > end[axis])
            return result;
    }
    for (int z = begin[0]; z <= end[0]; ++z) {
        for (int y = begin[1]; y <= end[1]; ++y) {
            for (int x = begin[2]; x <= end[2]; ++x)
                result.push_back({0, z, y, x});
        }
    }
    return result;
}

std::vector<vc::render::ChunkKey> analysisPrefixOwnerChunks(
    const FiberletDatasetMetadata& metadata,
    const FiberletChunkRouteAnalysisConfig& config)
{
    const auto seeds = analysisAnchorChunks(metadata, config);
    std::array<int, 3> halo{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto units = metadata.coordinateUnitsPerChunkZYX[axis];
        const auto reach = metadata.maximumEndpointReachCoordinateUnitsZYX[axis];
        if (units <= 0 || reach < 0)
            throw std::invalid_argument(
                "fiberlet chunk-route storage reach is invalid");
        halo[axis] = static_cast<int>((reach + units - 1) / units);
    }
    std::set<std::array<int, 3>> owners;
    for (const auto& seed : seeds) {
        for (int z = seed.iz - halo[0]; z <= seed.iz + halo[0]; ++z) {
            for (int y = seed.iy - halo[1]; y <= seed.iy + halo[1]; ++y) {
                for (int x = seed.ix - halo[2]; x <= seed.ix + halo[2]; ++x) {
                    if (z >= 0 && y >= 0 && x >= 0 &&
                        z < metadata.chunkGridShapeZYX[0] &&
                        y < metadata.chunkGridShapeZYX[1] &&
                        x < metadata.chunkGridShapeZYX[2]) {
                        owners.insert({z, y, x});
                    }
                }
            }
        }
    }
    std::vector<vc::render::ChunkKey> result;
    result.reserve(owners.size());
    for (const auto& owner : owners)
        result.push_back({0, owner[0], owner[1], owner[2]});
    return result;
}

ChunkRouteLocalGraph materializeChunkRouteGraph(
    const FiberletChunkGraphSource& graph,
    const FiberletChunkRouteAnalysisConfig& config,
    std::vector<vc::render::ChunkKey>& seedChunks,
    bool buildTransitions)
{
    const float predictionToBaseScale =
        static_cast<float>(graph.metadata().predictionToBaseScale);
    std::map<FiberletStorageKey, FiberletStoredAnchor> insideAnchors;
    seedChunks = analysisAnchorChunks(graph.metadata(), config);
    for (const auto& chunk : seedChunks) {
        const auto loaded = graph.anchorsInChunk(chunk, true);
        if (loaded.status != FiberletGraphQueryStatus::Ready) {
            throw std::runtime_error(
                "fiberlet chunk-route anchor chunk failed: " + loaded.error);
        }
        for (const auto& anchor : *loaded.value.anchors) {
            const cv::Vec3d position(
                anchor.positionPredictionXYZ * predictionToBaseScale);
            if (pointInsideChunk(position, config))
                insideAnchors.emplace(anchor.key, anchor);
        }
    }

    std::vector<FiberletStorageKey> insideKeys;
    insideKeys.reserve(insideAnchors.size());
    for (const auto& [key, anchor] : insideAnchors) {
        (void)anchor;
        insideKeys.push_back(key);
    }

    std::map<FiberletStorageId, FiberletStoredPrefix> physicalPrefixes;
    for (const auto& chunk : graph.incidentPrefixChunks(insideKeys)) {
        const auto loaded = graph.prefixesInChunk(chunk, true);
        if (loaded.status != FiberletGraphQueryStatus::Ready) {
            throw std::runtime_error(
                "fiberlet chunk-route prefix halo failed: " + loaded.error);
        }
        for (const auto& prefix : loaded.value.payloadLease->prefixes) {
            if (!insideAnchors.contains(prefix.id.first) &&
                !insideAnchors.contains(prefix.id.second)) {
                continue;
            }
            if (!physicalPrefixes.emplace(prefix.id, prefix).second) {
                throw std::runtime_error(
                    "fiberlet graph contains a duplicate incident edge");
            }
        }
    }

    std::set<FiberletStorageKey> requiredAnchorKeys;
    for (const auto& [id, prefix] : physicalPrefixes) {
        (void)prefix;
        requiredAnchorKeys.insert(id.first);
        requiredAnchorKeys.insert(id.second);
    }
    std::set<std::tuple<int, int, int>> requiredAnchorOwners;
    for (const auto& key : requiredAnchorKeys) {
        const auto owner = fiberletStorageOwnerChunk(
            graph.metadata(), key, 0);
        requiredAnchorOwners.emplace(owner.iz, owner.iy, owner.ix);
    }
    std::map<FiberletStorageKey, FiberletStoredAnchor> endpointAnchors;
    for (const auto& [z, y, x] : requiredAnchorOwners) {
        const auto loaded = graph.anchorsInChunk({0, z, y, x}, true);
        if (loaded.status != FiberletGraphQueryStatus::Ready) {
            throw std::runtime_error(
                "fiberlet chunk-route endpoint anchor chunk failed: " +
                loaded.error);
        }
        for (const auto& anchor : *loaded.value.anchors) {
            if (!requiredAnchorKeys.contains(anchor.key))
                continue;
            if (!endpointAnchors.emplace(anchor.key, anchor).second) {
                throw std::runtime_error(
                    "fiberlet graph contains a duplicate endpoint anchor");
            }
        }
    }
    if (endpointAnchors.size() != requiredAnchorKeys.size()) {
        throw std::runtime_error(
            "fiberlet graph endpoint anchor is absent from its owner chunk");
    }

    ChunkRouteLocalGraph result;
    std::map<FiberletStorageKey, std::size_t> anchorIndices;
    auto ensureAnchor = [&](const FiberletStoredAnchor& anchor) {
        const cv::Vec3d position(
            anchor.positionPredictionXYZ * predictionToBaseScale);
        const auto [found, inserted] =
            anchorIndices.emplace(anchor.key, result.anchors.size());
        if (inserted) {
            result.anchors.push_back(
                {anchor.key, anchor, position,
                 pointInsideChunk(position, config)});
        } else if (cv::norm(result.anchors[found->second].positionBaseXYZ -
                            position) > 1.0e-5) {
            throw std::runtime_error(
                "fiberlet chunk-route anchor position is inconsistent");
        }
        return found->second;
    };

    result.physicalFiberlets.reserve(physicalPrefixes.size());
    for (const auto& [id, prefix] : physicalPrefixes) {
        (void)prefix;
        result.physicalFiberlets.push_back(id);
    }
    result.arcs.reserve(result.physicalFiberlets.size() * 2);
    for (const auto& id : result.physicalFiberlets) {
        const auto prefix = physicalPrefixes.find(id);
        const auto firstAnchor = endpointAnchors.find(id.first);
        const auto secondAnchor = endpointAnchors.find(id.second);
        if (prefix == physicalPrefixes.end() ||
            firstAnchor == endpointAnchors.end() ||
            secondAnchor == endpointAnchors.end()) {
            throw std::logic_error(
                "fiberlet chunk-route bulk materialization lost a record");
        }
        FiberletEdgeLease loaded;
        loaded.prefix = prefix->second;
        loaded.firstAnchor = firstAnchor->second;
        loaded.secondAnchor = secondAnchor->second;
        std::array<FiberletReplaySourceArc, 2> directed;
        for (std::size_t reverse = 0; reverse < 2; ++reverse) {
            directed[reverse] = directedStoredArc(
                loaded, {id, reverse != 0}, predictionToBaseScale);
        }
        const std::size_t first = ensureAnchor(firstAnchor->second);
        const std::size_t second = ensureAnchor(secondAnchor->second);
        for (std::size_t reverse = 0; reverse < 2; ++reverse) {
            auto arc = directed[reverse];
            arc.cost = chunkRouteEdgeCost(
                arc.cost, arc.pathLengthPredictionVoxels,
                config.edgeCostView);
            const double loss = static_cast<double>(arc.cost.total());
            const double length =
                static_cast<double>(arc.pathLengthPredictionVoxels);
            if (!(loss >= 0.0) || !std::isfinite(loss) ||
                !(length > 0.0) || !std::isfinite(length)) {
                throw std::runtime_error(
                    "fiberlet chunk-route edge objective is invalid");
            }
            result.arcs.push_back({
                std::move(arc), reverse == 0 ? first : second,
                reverse == 0 ? second : first, loss, length});
        }
    }

    result.outgoing.resize(result.anchors.size());
    for (std::size_t arc = 0; arc < result.arcs.size(); ++arc)
        result.outgoing[result.arcs[arc].sourceAnchor].push_back(arc);
    for (auto& outgoing : result.outgoing)
        std::sort(outgoing.begin(), outgoing.end());

    if (!buildTransitions)
        return result;
    result.successors.resize(result.arcs.size());
    indexedParallelFor(
        result.arcs.size(), config.parallelThreads,
        [&](std::size_t incoming) {
            const auto& incomingArc = result.arcs[incoming];
            if (!result.anchors[incomingArc.targetAnchor].inside)
                return;
            auto& successors = result.successors[incoming];
            for (const std::size_t outgoing :
                 result.outgoing[incomingArc.targetAnchor]) {
                const auto transition = graph.transitionAtAnchor(
                    incomingArc.source, result.arcs[outgoing].source,
                    result.anchors[incomingArc.targetAnchor].stored,
                    config.maximumJoinAngleDegrees);
                if (transition.status != FiberletGraphQueryStatus::Ready) {
                    throw std::runtime_error(
                        "fiberlet chunk-route transition failed: " +
                        transition.error);
                }
                if (!transition.value.has_value())
                    continue;
                const double joinLoss =
                    static_cast<double>(transition.value->cost.total());
                if (!(joinLoss >= 0.0) || !std::isfinite(joinLoss)) {
                    throw std::runtime_error(
                        "fiberlet chunk-route join objective is invalid");
                }
                successors.push_back({outgoing, joinLoss});
            }
        });
    for (const auto& successors : result.successors)
        result.admissibleTransitions += successors.size();

    for (std::size_t arc = 0; arc < result.arcs.size(); ++arc) {
        const bool sourceInside =
            result.anchors[result.arcs[arc].sourceAnchor].inside;
        const bool targetInside =
            result.anchors[result.arcs[arc].targetAnchor].inside;
        if (!sourceInside && targetInside)
            result.entries.push_back(arc);
        if (sourceInside && !targetInside)
            ++result.exits;
    }
    return result;
}

std::vector<FiberletStorageId> internalChunkRouteFiberlets(
    const ChunkRouteLocalGraph& graph)
{
    std::vector<FiberletStorageId> result;
    for (std::size_t edge = 0; edge < graph.physicalFiberlets.size(); ++edge) {
        const auto& arc = graph.arcs[edge * 2];
        if (graph.anchors[arc.sourceAnchor].inside &&
            graph.anchors[arc.targetAnchor].inside) {
            result.push_back(graph.physicalFiberlets[edge]);
        }
    }
    return result;
}

struct ChunkRouteSearchHistory {
    std::uint32_t arc = 0;
    std::uint32_t parent = std::numeric_limits<std::uint32_t>::max();
};

struct ChunkRouteTerminal {
    std::uint32_t parent = 0;
    std::uint32_t exitArc = 0;
    double loss = 0.0;
    double lengthPredictionVoxels = 0.0;
    std::uint32_t fiberletCount = 0;
};

struct ChunkRouteEntryResult {
    std::vector<std::vector<std::size_t>> optimalArcRoutes;
    std::vector<double> lengths;
    std::vector<double> losses;
    std::vector<double> fiberletCounts;
    std::size_t generatedStates = 0;
    std::size_t expandedStates = 0;
    std::size_t rejectedVisitedTargets = 0;
};

struct ChunkRouteQueueItem {
    double loss = 0.0;
    std::uint32_t node = 0;
};

struct ChunkRouteSearchScratch {
    std::vector<ChunkRouteQueueItem> queue;
    std::vector<ChunkRouteSearchHistory> history;
    std::vector<double> lengths;
    std::vector<std::uint32_t> fiberletCounts;
    std::vector<ChunkRouteTerminal> terminals;
};

bool chunkRouteContainsAnchor(
    const ChunkRouteLocalGraph& graph,
    const std::vector<ChunkRouteSearchHistory>& history,
    std::uint32_t node,
    std::size_t rootSourceAnchor,
    std::size_t candidate)
{
    if (candidate == rootSourceAnchor)
        return true;
    while (node != std::numeric_limits<std::uint32_t>::max()) {
        if (graph.arcs[history[node].arc].targetAnchor == candidate)
            return true;
        node = history[node].parent;
    }
    return false;
}

ChunkRouteEntryResult searchChunkRouteEntry(
    const ChunkRouteLocalGraph& graph,
    std::size_t entryArc,
    std::size_t maximumGeneratedStates)
{
    const auto greater = [](const ChunkRouteQueueItem& left,
                            const ChunkRouteQueueItem& right) {
        if (left.loss != right.loss)
            return left.loss > right.loss;
        return left.node > right.node;
    };
    thread_local ChunkRouteSearchScratch scratch;
    auto& queue = scratch.queue;
    auto& history = scratch.history;
    auto& lengths = scratch.lengths;
    auto& fiberletCounts = scratch.fiberletCounts;
    auto& terminals = scratch.terminals;
    queue.clear();
    history.clear();
    lengths.clear();
    fiberletCounts.clear();
    terminals.clear();
    if (entryArc > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("fiberlet chunk-route arc index is too large");
    const auto& entry = graph.arcs[entryArc];
    history.push_back({
        static_cast<std::uint32_t>(entryArc),
        std::numeric_limits<std::uint32_t>::max()});
    lengths.push_back(entry.lengthPredictionVoxels);
    fiberletCounts.push_back(1);
    queue.push_back({entry.loss, 0});
    ChunkRouteEntryResult result;
    result.generatedStates = 1;
    std::optional<double> bestLoss;
    while (!queue.empty()) {
        const ChunkRouteQueueItem item = queue.front();
        if (bestLoss.has_value() && item.loss > *bestLoss)
            break;
        std::pop_heap(queue.begin(), queue.end(), greater);
        queue.pop_back();
        ++result.expandedStates;
        const auto state = history[item.node];
        for (const auto& successor : graph.successors[state.arc]) {
            const auto& outgoing = graph.arcs[successor.arc];
            if (chunkRouteContainsAnchor(
                    graph, history, item.node, entry.sourceAnchor,
                    outgoing.targetAnchor)) {
                ++result.rejectedVisitedTargets;
                continue;
            }
            if (++result.generatedStates > maximumGeneratedStates) {
                throw std::runtime_error(
                    "fiberlet chunk-route entry exceeded the exact state limit");
            }
            const double loss =
                item.loss + successor.joinLoss + outgoing.loss;
            const double length = lengths[item.node] +
                outgoing.lengthPredictionVoxels;
            const std::uint32_t count = fiberletCounts[item.node] + 1;
            if (!graph.anchors[outgoing.targetAnchor].inside) {
                if (!bestLoss.has_value() || loss < *bestLoss) {
                    bestLoss = loss;
                    terminals.clear();
                }
                if (loss == *bestLoss)
                    terminals.push_back({
                        item.node, static_cast<std::uint32_t>(successor.arc),
                        loss, length, count});
                continue;
            }
            if (history.size() >=
                    static_cast<std::size_t>(
                        std::numeric_limits<std::uint32_t>::max()) ||
                successor.arc > std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error(
                    "fiberlet chunk-route search history is too large");
            }
            const auto next = static_cast<std::uint32_t>(history.size());
            history.push_back({
                static_cast<std::uint32_t>(successor.arc), item.node});
            lengths.push_back(length);
            fiberletCounts.push_back(count);
            queue.push_back({loss, next});
            std::push_heap(queue.begin(), queue.end(), greater);
        }
    }

    for (const auto& terminal : terminals) {
        std::vector<std::size_t> route{terminal.exitArc};
        std::uint32_t node = terminal.parent;
        while (node != std::numeric_limits<std::uint32_t>::max()) {
            route.push_back(history[node].arc);
            node = history[node].parent;
        }
        std::reverse(route.begin(), route.end());
        result.optimalArcRoutes.push_back(std::move(route));
        result.lengths.push_back(terminal.lengthPredictionVoxels);
        result.losses.push_back(terminal.loss);
        result.fiberletCounts.push_back(
            static_cast<double>(terminal.fiberletCount));
    }
    return result;
}

}  // namespace

std::vector<vc::render::ChunkKey> fiberletChunkRoutePrefetchChunks(
    const FiberletDatasetMetadata& metadata,
    const FiberletChunkRouteAnalysisConfig& config)
{
    return analysisPrefixOwnerChunks(metadata, config);
}

FiberletChunkRoutePopulation collectFiberletChunkRoutePopulation(
    const FiberletChunkGraphSource& graph,
    const FiberletChunkRouteAnalysisConfig& config)
{
    if (config.minimumBaseXYZ[0] >= config.maximumBaseXYZ[0] ||
        config.minimumBaseXYZ[1] >= config.maximumBaseXYZ[1] ||
        config.minimumBaseXYZ[2] >= config.maximumBaseXYZ[2]) {
        throw std::invalid_argument(
            "fiberlet chunk-route bounds must have positive extent");
    }
    std::vector<vc::render::ChunkKey> seeds;
    const auto local = materializeChunkRouteGraph(
        graph, config, seeds, false);
    FiberletChunkRoutePopulation result;
    result.insideAnchors = static_cast<std::size_t>(std::count_if(
        local.anchors.begin(), local.anchors.end(),
        [](const auto& anchor) { return anchor.inside; }));
    result.physicalFiberletIds = local.physicalFiberlets;
    result.internalFiberletIds = internalChunkRouteFiberlets(local);
    return result;
}

double appendFiberletChunkRouteMacroLoss(
    double prefixLoss,
    double incomingJoinLoss,
    const FiberletChunkRouteMacroDirection& direction)
{
    if (!direction.live || direction.edgeLosses.empty() ||
        direction.internalJoinLosses.size() + 1 !=
            direction.edgeLosses.size()) {
        throw std::invalid_argument(
            "fiberlet chunk-route macro loss sequence is invalid");
    }
    double result = prefixLoss + incomingJoinLoss +
        direction.edgeLosses.front();
    for (std::size_t index = 1; index < direction.edgeLosses.size(); ++index) {
        result = result + direction.internalJoinLosses[index - 1] +
            direction.edgeLosses[index];
    }
    return result;
}

bool canAppendFiberletChunkRouteMacro(
    const FiberletChunkRouteMacroDirection& direction,
    std::span<const FiberletStorageKey> visitedAnchors)
{
    if (!direction.live || direction.anchors.size() < 2)
        return false;
    for (std::size_t index = 1; index < direction.anchors.size(); ++index) {
        if (std::find(
                visitedAnchors.begin(), visitedAnchors.end(),
                direction.anchors[index]) != visitedAnchors.end()) {
            return false;
        }
    }
    return true;
}

static FiberletChunkRouteSimplificationReport simplifyMaterializedChunkRoutes(
    const ChunkRouteLocalGraph& local,
    const FiberletChunkRouteAnalysisConfig& config,
    std::span<const FiberletStorageId> retainedPhysicalFiberlets)
{
    FiberletChunkRouteSimplificationReport report;
    report.minimumBaseXYZ = config.minimumBaseXYZ;
    report.maximumBaseXYZ = config.maximumBaseXYZ;
    std::vector<FiberletStorageId> selected(
        retainedPhysicalFiberlets.begin(), retainedPhysicalFiberlets.end());
    std::sort(selected.begin(), selected.end());
    if (std::adjacent_find(selected.begin(), selected.end()) != selected.end()) {
        throw std::invalid_argument(
            "fiberlet simplification input contains duplicate physical IDs");
    }

    report.inputAnchors = local.anchors.size();
    report.inputInsideAnchors = static_cast<std::size_t>(std::count_if(
        local.anchors.begin(), local.anchors.end(),
        [](const auto& anchor) { return anchor.inside; }));
    report.inputPhysicalFiberlets = selected.size();
    report.inputDirectedStates = selected.size() * 2;

    std::vector<bool> selectedEdge(local.physicalFiberlets.size(), false);
    for (const auto& id : selected) {
        const auto found = std::lower_bound(
            local.physicalFiberlets.begin(), local.physicalFiberlets.end(), id);
        if (found == local.physicalFiberlets.end() || *found != id) {
            throw std::invalid_argument(
                "fiberlet simplification input is outside the analysis graph");
        }
        selectedEdge[static_cast<std::size_t>(
            found - local.physicalFiberlets.begin())] = true;
    }

    std::vector<bool> active(local.arcs.size(), false);
    for (std::size_t edge = 0; edge < selectedEdge.size(); ++edge) {
        if (selectedEdge[edge]) {
            active[edge * 2] = true;
            active[edge * 2 + 1] = true;
        }
    }
    std::vector<std::vector<std::size_t>> predecessors(local.arcs.size());
    for (std::size_t incoming = 0; incoming < local.successors.size();
         ++incoming) {
        if (!active[incoming])
            continue;
        for (const auto& successor : local.successors[incoming]) {
            if (active[successor.arc])
                predecessors[successor.arc].push_back(incoming);
        }
    }
    std::vector<bool> forward(local.arcs.size(), false);
    std::vector<std::size_t> pending;
    for (const auto entry : local.entries) {
        if (active[entry] && !forward[entry]) {
            forward[entry] = true;
            pending.push_back(entry);
        }
    }
    for (std::size_t index = 0; index < pending.size(); ++index) {
        for (const auto& successor : local.successors[pending[index]]) {
            if (active[successor.arc] && !forward[successor.arc]) {
                forward[successor.arc] = true;
                pending.push_back(successor.arc);
            }
        }
    }
    std::vector<bool> backward(local.arcs.size(), false);
    pending.clear();
    for (std::size_t arc = 0; arc < local.arcs.size(); ++arc) {
        if (active[arc] &&
            local.anchors[local.arcs[arc].sourceAnchor].inside &&
            !local.anchors[local.arcs[arc].targetAnchor].inside) {
            backward[arc] = true;
            pending.push_back(arc);
        }
    }
    for (std::size_t index = 0; index < pending.size(); ++index) {
        for (const auto predecessor : predecessors[pending[index]]) {
            if (!backward[predecessor]) {
                backward[predecessor] = true;
                pending.push_back(predecessor);
            }
        }
    }
    std::vector<bool> live(local.arcs.size(), false);
    for (std::size_t arc = 0; arc < local.arcs.size(); ++arc)
        live[arc] = active[arc] && forward[arc] && backward[arc];
    report.liveDirectedStates = static_cast<std::size_t>(
        std::count(live.begin(), live.end(), true));
    report.deadDirectedStatesRemoved =
        report.inputDirectedStates - report.liveDirectedStates;

    std::vector<bool> liveEdge(local.physicalFiberlets.size(), false);
    for (std::size_t edge = 0; edge < local.physicalFiberlets.size(); ++edge) {
        if (!selectedEdge[edge])
            continue;
        const std::array<bool, 2> directions{
            live[edge * 2], live[edge * 2 + 1]};
        if (directions[0] || directions[1]) {
            liveEdge[edge] = true;
            report.livePhysicalFiberletIds.push_back(
                local.physicalFiberlets[edge]);
            report.livePhysicalDirections.push_back(directions);
        }
    }
    report.livePhysicalFiberlets = report.livePhysicalFiberletIds.size();
    report.deadPhysicalFiberletsRemoved =
        report.inputPhysicalFiberlets - report.livePhysicalFiberlets;

    std::vector<std::vector<std::size_t>> incident(local.anchors.size());
    std::set<FiberletStorageKey> retainedInside;
    std::set<FiberletStorageKey> portals;
    for (std::size_t edge = 0; edge < liveEdge.size(); ++edge) {
        if (!liveEdge[edge])
            continue;
        const auto& physical = local.arcs[edge * 2];
        incident[physical.sourceAnchor].push_back(edge);
        incident[physical.targetAnchor].push_back(edge);
        for (const auto anchorIndex :
             {physical.sourceAnchor, physical.targetAnchor}) {
            const auto& anchor = local.anchors[anchorIndex];
            if (anchor.inside)
                retainedInside.insert(anchor.key);
            else
                portals.insert(anchor.key);
        }
    }
    report.retainedInsideAnchorIds.assign(
        retainedInside.begin(), retainedInside.end());
    report.boundaryPortalIds.assign(portals.begin(), portals.end());
    report.retainedInsideAnchors = report.retainedInsideAnchorIds.size();
    report.unusedInsideAnchorsRemoved =
        report.inputInsideAnchors - report.retainedInsideAnchors;
    report.boundaryPortals = report.boundaryPortalIds.size();
    report.retainedAnchors =
        report.retainedInsideAnchors + report.boundaryPortals;
    report.unusedAnchorsRemoved =
        report.inputAnchors - report.retainedAnchors;
    for (auto& edges : incident)
        std::sort(edges.begin(), edges.end());

    auto arcFrom = [&](std::size_t edge, std::size_t sourceAnchor) {
        const std::size_t forwardArc = edge * 2;
        if (local.arcs[forwardArc].sourceAnchor == sourceAnchor)
            return forwardArc;
        if (local.arcs[forwardArc + 1].sourceAnchor == sourceAnchor)
            return forwardArc + 1;
        throw std::logic_error(
            "fiberlet simplification edge does not touch its anchor");
    };
    auto arcTo = [&](std::size_t edge, std::size_t targetAnchor) {
        return arcFrom(edge, targetAnchor) ^ 1U;
    };
    auto successorLoss = [&](std::size_t incoming,
                             std::size_t outgoing) -> std::optional<double> {
        const auto& successors = local.successors[incoming];
        const auto found = std::find_if(
            successors.begin(), successors.end(), [&](const auto& value) {
                return value.arc == outgoing;
            });
        if (found == successors.end())
            return std::nullopt;
        return found->joinLoss;
    };

    std::vector<bool> contractible(local.anchors.size(), false);
    for (std::size_t anchor = 0; anchor < local.anchors.size(); ++anchor) {
        if (!local.anchors[anchor].inside || incident[anchor].size() != 2)
            continue;
        const std::size_t first = incident[anchor][0];
        const std::size_t second = incident[anchor][1];
        const std::size_t firstIn = arcTo(first, anchor);
        const std::size_t firstOut = arcFrom(first, anchor);
        const std::size_t secondIn = arcTo(second, anchor);
        const std::size_t secondOut = arcFrom(second, anchor);
        if (live[firstIn] && live[firstOut] && live[secondIn] &&
            live[secondOut] && successorLoss(firstIn, secondOut) &&
            successorLoss(secondIn, firstOut)) {
            contractible[anchor] = true;
            ++report.contractibleInsideAnchors;
        }
    }

    auto reverseArc = [](std::size_t arc) { return arc ^ 1U; };
    auto makeDirection = [&](std::span<const std::size_t> arcs) {
        FiberletChunkRouteMacroDirection direction;
        if (arcs.empty())
            return direction;
        direction.live = std::all_of(
            arcs.begin(), arcs.end(), [&](std::size_t arc) {
                return live[arc];
            });
        direction.anchors.push_back(
            local.anchors[local.arcs[arcs.front()].sourceAnchor].key);
        double loss = 0.0;
        bool first = true;
        for (std::size_t index = 0; index < arcs.size(); ++index) {
            const auto arc = arcs[index];
            direction.physicalFiberlets.push_back(local.arcs[arc].source.id);
            direction.anchors.push_back(
                local.anchors[local.arcs[arc].targetAnchor].key);
            direction.edgeLosses.push_back(local.arcs[arc].loss);
            direction.edgeLengthsPredictionVoxels.push_back(
                local.arcs[arc].lengthPredictionVoxels);
            direction.diagnosticLengthPredictionVoxels +=
                local.arcs[arc].lengthPredictionVoxels;
            if (first) {
                loss = local.arcs[arc].loss;
                first = false;
            } else {
                const auto join = successorLoss(arcs[index - 1], arc);
                if (!join) {
                    direction.live = false;
                    direction.internalJoinLosses.push_back(0.0);
                } else {
                    direction.internalJoinLosses.push_back(*join);
                    loss = loss + *join + local.arcs[arc].loss;
                }
            }
        }
        direction.diagnosticLoss = loss;
        return direction;
    };
    auto appendMacro = [&](const std::vector<std::size_t>& forwardArcs) {
        FiberletChunkRouteMacro macro;
        macro.index = report.macros.size();
        macro.directions[0] = makeDirection(forwardArcs);
        std::vector<std::size_t> reverseArcs;
        reverseArcs.reserve(forwardArcs.size());
        for (auto iterator = forwardArcs.rbegin();
             iterator != forwardArcs.rend(); ++iterator) {
            reverseArcs.push_back(reverseArc(*iterator));
        }
        macro.directions[1] = makeDirection(reverseArcs);
        const auto& anchors = macro.directions[0].anchors;
        if (anchors.size() >= 2) {
            const auto first = std::find_if(
                local.anchors.begin(), local.anchors.end(),
                [&](const auto& value) { return value.key == anchors.front(); });
            const auto second = std::find_if(
                local.anchors.begin(), local.anchors.end(),
                [&](const auto& value) { return value.key == anchors.back(); });
            macro.firstBoundaryPortal =
                first != local.anchors.end() && !first->inside;
            macro.secondBoundaryPortal =
                second != local.anchors.end() && !second->inside;
        }
        report.macros.push_back(std::move(macro));
    };

    std::vector<bool> visitedEdge(local.physicalFiberlets.size(), false);
    for (std::size_t start = 0; start < local.anchors.size(); ++start) {
        if (contractible[start])
            continue;
        for (const std::size_t firstEdge : incident[start]) {
            if (visitedEdge[firstEdge])
                continue;
            std::vector<std::size_t> arcs;
            std::vector<std::size_t> edges;
            std::size_t currentAnchor = start;
            std::size_t edge = firstEdge;
            bool cycle = false;
            while (true) {
                if (std::find(edges.begin(), edges.end(), edge) != edges.end() ||
                    visitedEdge[edge]) {
                    cycle = true;
                    break;
                }
                edges.push_back(edge);
                const auto arc = arcFrom(edge, currentAnchor);
                arcs.push_back(arc);
                const auto target = local.arcs[arc].targetAnchor;
                if (!contractible[target]) {
                    if (target == start && edges.size() > 1)
                        cycle = true;
                    break;
                }
                const auto& next = incident[target];
                edge = next[0] == edge ? next[1] : next[0];
                currentAnchor = target;
            }
            if (cycle) {
                for (const auto physical : edges) {
                    if (!visitedEdge[physical]) {
                        visitedEdge[physical] = true;
                        appendMacro({arcFrom(
                            physical,
                            local.arcs[physical * 2].sourceAnchor)});
                    }
                }
            } else {
                for (const auto physical : edges)
                    visitedEdge[physical] = true;
                appendMacro(arcs);
            }
        }
    }
    for (std::size_t edge = 0; edge < liveEdge.size(); ++edge) {
        if (liveEdge[edge] && !visitedEdge[edge]) {
            visitedEdge[edge] = true;
            appendMacro({edge * 2});
        }
    }
    report.physicalMacros = report.macros.size();
    report.physicalFiberletsMerged =
        report.livePhysicalFiberlets - report.physicalMacros;
    std::vector<double> macroSizes;
    macroSizes.reserve(report.macros.size());
    for (const auto& macro : report.macros) {
        macroSizes.push_back(static_cast<double>(
            macro.directions[0].physicalFiberlets.size()));
    }
    report.physicalFiberletsPerMacro =
        chunkRouteDistribution(std::move(macroSizes));

    std::map<DirectedFiberletStorageId, std::size_t> arcById;
    for (std::size_t arc = 0; arc < local.arcs.size(); ++arc)
        arcById.emplace(local.arcs[arc].source.id, arc);
    const auto directedIndex = [](FiberletChunkRouteDirectedMacroId id) {
        return id.macro * 2 + static_cast<std::size_t>(id.reverse);
    };
    std::vector<std::optional<FiberletChunkRouteDirectedMacroId>>
        macroByFirstArc(local.arcs.size());
    std::vector<std::size_t> lastArc(report.macros.size() * 2, 0);
    for (const auto& macro : report.macros) {
        for (std::size_t reverse = 0; reverse < 2; ++reverse) {
            const auto& direction = macro.directions[reverse];
            if (!direction.live)
                continue;
            ++report.liveDirectedMacros;
            const auto first = arcById.at(direction.physicalFiberlets.front());
            const auto last = arcById.at(direction.physicalFiberlets.back());
            const FiberletChunkRouteDirectedMacroId id{
                macro.index, reverse != 0};
            if (macroByFirstArc[first]) {
                throw std::logic_error(
                    "fiberlet simplification has duplicate macro starts");
            }
            macroByFirstArc[first] = id;
            lastArc[directedIndex(id)] = last;
        }
    }
    std::vector<std::vector<FiberletChunkRouteMacroTransition>> adjacency(
        report.macros.size() * 2);
    for (const auto& macro : report.macros) {
        for (std::size_t reverse = 0; reverse < 2; ++reverse) {
            const FiberletChunkRouteDirectedMacroId incoming{
                macro.index, reverse != 0};
            if (!macro.directions[reverse].live)
                continue;
            for (const auto& successor :
                 local.successors[lastArc[directedIndex(incoming)]]) {
                if (!live[successor.arc] ||
                    !macroByFirstArc[successor.arc]) {
                    continue;
                }
                FiberletChunkRouteMacroTransition transition{
                    incoming, *macroByFirstArc[successor.arc],
                    successor.joinLoss};
                adjacency[directedIndex(incoming)].push_back(transition);
                report.transitions.push_back(transition);
            }
            auto& values = adjacency[directedIndex(incoming)];
            std::sort(values.begin(), values.end(), [](const auto& left,
                                                       const auto& right) {
                return left.outgoing < right.outgoing;
            });
        }
    }
    report.macroTransitions = report.transitions.size();
    for (const auto& macro : report.macros) {
        for (std::size_t reverse = 0; reverse < 2; ++reverse) {
            if (!macro.directions[reverse].live)
                continue;
            const auto count = adjacency[macro.index * 2 + reverse].size();
            if (count == 0)
                ++report.zeroContinuationStates;
            else if (count == 1)
                ++report.forcedContinuationStates;
            else
                ++report.branchingStates;
        }
    }

    const std::size_t directedCount = report.macros.size() * 2;
    std::vector<std::size_t> incomingCounts(directedCount, 0);
    for (const auto& transition : report.transitions)
        ++incomingCounts[directedIndex(transition.outgoing)];
    std::vector<std::optional<FiberletChunkRouteMacroTransition>>
        forcedNext(directedCount);
    std::vector<std::size_t> forcedPredecessors(directedCount, 0);
    for (std::size_t state = 0; state < directedCount; ++state) {
        if (adjacency[state].size() != 1)
            continue;
        const auto transition = adjacency[state].front();
        if (incomingCounts[directedIndex(transition.outgoing)] != 1)
            continue;
        const auto current = FiberletChunkRouteDirectedMacroId{
            state / 2, state % 2 != 0};
        const auto& currentDirection = report.macros[current.macro]
            .directions[static_cast<std::size_t>(current.reverse)];
        const auto& nextDirection = report.macros[transition.outgoing.macro]
            .directions[static_cast<std::size_t>(
                transition.outgoing.reverse)];
        if (!canAppendFiberletChunkRouteMacro(
                nextDirection, currentDirection.anchors)) {
            continue;
        }
        forcedNext[state] = transition;
        ++forcedPredecessors[directedIndex(transition.outgoing)];
    }

    // Functional forced-successor cycles cannot be replaced by a linear
    // macro without changing their topology. Leave every state in such a
    // cycle explicit.
    std::vector<bool> forcedCycle(directedCount, false);
    std::vector<std::uint8_t> finished(directedCount, 0);
    std::vector<std::ptrdiff_t> pathPosition(directedCount, -1);
    for (std::size_t start = 0; start < directedCount; ++start) {
        if (finished[start] || !forcedNext[start])
            continue;
        std::vector<std::size_t> path;
        std::size_t current = start;
        while (!finished[current] && pathPosition[current] < 0 &&
               forcedNext[current]) {
            pathPosition[current] = static_cast<std::ptrdiff_t>(path.size());
            path.push_back(current);
            current = directedIndex(forcedNext[current]->outgoing);
        }
        if (pathPosition[current] >= 0) {
            for (std::size_t index = static_cast<std::size_t>(
                     pathPosition[current]);
                 index < path.size(); ++index) {
                forcedCycle[path[index]] = true;
            }
        }
        for (const auto state : path) {
            finished[state] = 1;
            pathPosition[state] = -1;
        }
    }

    std::vector<bool> consumed(directedCount, false);
    auto buildRollout = [&](std::size_t start) {
        const FiberletChunkRouteDirectedMacroId startId{
            start / 2, start % 2 != 0};
        const auto& initial = report.macros[startId.macro]
            .directions[static_cast<std::size_t>(startId.reverse)];
        FiberletChunkRouteDeterministicRollout rollout;
        rollout.start = startId;
        rollout.macros.push_back(startId);
        rollout.anchors = initial.anchors;
        rollout.diagnosticLoss = initial.diagnosticLoss;
        rollout.diagnosticLengthPredictionVoxels =
            initial.diagnosticLengthPredictionVoxels;
        auto current = start;
        while (forcedNext[current]) {
            const auto transition = *forcedNext[current];
            const auto nextIndex = directedIndex(transition.outgoing);
            if (forcedCycle[nextIndex] || consumed[nextIndex])
                break;
            const auto& next = report.macros[transition.outgoing.macro]
                .directions[static_cast<std::size_t>(
                    transition.outgoing.reverse)];
            if (!canAppendFiberletChunkRouteMacro(next, rollout.anchors))
                break;
            rollout.diagnosticLoss = appendFiberletChunkRouteMacroLoss(
                rollout.diagnosticLoss, transition.joinLoss, next);
            rollout.diagnosticLengthPredictionVoxels +=
                next.diagnosticLengthPredictionVoxels;
            rollout.transitionJoinLosses.push_back(transition.joinLoss);
            rollout.macros.push_back(transition.outgoing);
            rollout.anchors.insert(
                rollout.anchors.end(), next.anchors.begin() + 1,
                next.anchors.end());
            current = nextIndex;
        }
        if (rollout.macros.size() <= 1)
            return;
        for (const auto id : rollout.macros)
            consumed[directedIndex(id)] = true;
        report.directedMacrosMerged += rollout.macros.size() - 1;
    };
    for (std::size_t state = 0; state < directedCount; ++state) {
        if (!consumed[state] && !forcedCycle[state] && forcedNext[state] &&
            forcedPredecessors[state] == 0) {
            buildRollout(state);
        }
    }
    for (std::size_t state = 0; state < directedCount; ++state) {
        if (!consumed[state] && !forcedCycle[state] && forcedNext[state])
            buildRollout(state);
    }
    report.directedChainMacros =
        report.liveDirectedMacros - report.directedMacrosMerged;

    // A convergence prevents disjoint graph contraction, but once replay has
    // reached a directed state with one successor there is still no choice to
    // make. Precompute the maximal continuation from every such state so an
    // arbitrary replay prefix can apply it atomically.
    std::vector<double> rolloutSizes;
    for (const auto& macro : report.macros) {
        for (std::size_t reverse = 0; reverse < 2; ++reverse) {
            const FiberletChunkRouteDirectedMacroId start{
                macro.index, reverse != 0};
            const auto& initial = macro.directions[reverse];
            if (!initial.live || adjacency[directedIndex(start)].size() != 1)
                continue;
            FiberletChunkRouteDeterministicRollout rollout;
            rollout.start = start;
            rollout.macros.push_back(start);
            rollout.anchors = initial.anchors;
            rollout.diagnosticLoss = initial.diagnosticLoss;
            rollout.diagnosticLengthPredictionVoxels =
                initial.diagnosticLengthPredictionVoxels;
            std::set<FiberletChunkRouteDirectedMacroId> seen{start};
            auto current = start;
            while (adjacency[directedIndex(current)].size() == 1) {
                const auto transition =
                    adjacency[directedIndex(current)].front();
                if (seen.contains(transition.outgoing))
                    break;
                const auto& next = report.macros[transition.outgoing.macro]
                    .directions[static_cast<std::size_t>(
                        transition.outgoing.reverse)];
                if (!canAppendFiberletChunkRouteMacro(next, rollout.anchors))
                    break;
                rollout.diagnosticLoss = appendFiberletChunkRouteMacroLoss(
                    rollout.diagnosticLoss, transition.joinLoss, next);
                rollout.diagnosticLengthPredictionVoxels +=
                    next.diagnosticLengthPredictionVoxels;
                rollout.transitionJoinLosses.push_back(transition.joinLoss);
                rollout.macros.push_back(transition.outgoing);
                rollout.anchors.insert(
                    rollout.anchors.end(), next.anchors.begin() + 1,
                    next.anchors.end());
                seen.insert(transition.outgoing);
                current = transition.outgoing;
            }
            if (rollout.macros.size() > 1) {
                rolloutSizes.push_back(
                    static_cast<double>(rollout.macros.size()));
                report.rollouts.push_back(std::move(rollout));
            }
        }
    }
    report.deterministicRollouts = report.rollouts.size();
    report.macrosPerDeterministicRollout =
        chunkRouteDistribution(std::move(rolloutSizes));
    return report;
}

static FiberletChunkRouteAnalysisReport analyzeMaterializedChunkRoutes(
    const ChunkRouteLocalGraph& local,
    const FiberletChunkRouteAnalysisConfig& config,
    std::span<const vc::render::ChunkKey> seedChunks)
{
    const auto started = std::chrono::steady_clock::now();
    const std::clock_t cpuStarted = std::clock();
    FiberletChunkRouteAnalysisReport report;
    report.minimumBaseXYZ = config.minimumBaseXYZ;
    report.maximumBaseXYZ = config.maximumBaseXYZ;
    report.seedAnchorStorageChunks.assign(seedChunks.begin(), seedChunks.end());
    report.insideAnchors = static_cast<std::size_t>(std::count_if(
        local.anchors.begin(), local.anchors.end(),
        [](const auto& anchor) { return anchor.inside; }));
    report.physicalFiberlets = local.physicalFiberlets.size();
    report.physicalFiberletIds = local.physicalFiberlets;
    report.directedEntries = local.entries.size();
    report.directedExits = local.exits;
    report.admissibleTransitions = local.admissibleTransitions;
    report.internalFiberletIds = internalChunkRouteFiberlets(local);
    report.internalFiberlets = report.internalFiberletIds.size();
    for (std::size_t edge = 0; edge < local.physicalFiberlets.size(); ++edge) {
        const auto& arc = local.arcs[edge * 2];
        const bool firstInside = local.anchors[arc.sourceAnchor].inside;
        const bool secondInside = local.anchors[arc.targetAnchor].inside;
        if ((firstInside || secondInside) && !(firstInside && secondInside))
            ++report.crossingFiberlets;
    }
    report.directedStates = static_cast<std::size_t>(std::count_if(
        local.arcs.begin(), local.arcs.end(), [&](const auto& arc) {
            return local.anchors[arc.targetAnchor].inside;
        }));

    std::vector<ChunkRouteEntryResult> entries(local.entries.size());
    indexedParallelFor(
        local.entries.size(), config.parallelThreads,
        [&](std::size_t index) {
            entries[index] = searchChunkRouteEntry(
                local, local.entries[index],
                config.maximumGeneratedStatesPerEntry);
        });

    std::set<std::size_t> usedAnchors;
    std::set<std::size_t> usedPhysicalFiberlets;
    std::vector<double> counts;
    std::vector<double> lengths;
    std::vector<double> losses;
    std::vector<double> densities;
    for (const auto& entry : entries) {
        report.generatedSearchStates += entry.generatedStates;
        report.expandedSearchStates += entry.expandedStates;
        report.rejectedVisitedTargets += entry.rejectedVisitedTargets;
        if (entry.optimalArcRoutes.empty()) {
            ++report.unreachableEntries;
            continue;
        }
        ++report.reachableEntries;
        if (entry.optimalArcRoutes.size() > 1)
            ++report.tiedOptimalEntries;
        report.optimalRoutes += entry.optimalArcRoutes.size();
        for (std::size_t routeIndex = 0;
             routeIndex < entry.optimalArcRoutes.size(); ++routeIndex) {
            for (const std::size_t arcIndex :
                 entry.optimalArcRoutes[routeIndex]) {
                const auto& arc = local.arcs[arcIndex];
                usedPhysicalFiberlets.insert(arcIndex / 2);
                if (local.anchors[arc.sourceAnchor].inside)
                    usedAnchors.insert(arc.sourceAnchor);
                if (local.anchors[arc.targetAnchor].inside)
                    usedAnchors.insert(arc.targetAnchor);
            }
            counts.push_back(entry.fiberletCounts[routeIndex]);
            lengths.push_back(entry.lengths[routeIndex]);
            losses.push_back(entry.losses[routeIndex]);
            densities.push_back(
                entry.losses[routeIndex] / entry.lengths[routeIndex]);
        }
    }
    report.usedInsideAnchors = usedAnchors.size();
    report.unusedInsideAnchors = report.insideAnchors - usedAnchors.size();
    report.usedPhysicalFiberlets = usedPhysicalFiberlets.size();
    report.unusedPhysicalFiberlets =
        report.physicalFiberlets - usedPhysicalFiberlets.size();
    for (const std::size_t edge : usedPhysicalFiberlets) {
        const auto& arc = local.arcs[edge * 2];
        if (local.anchors[arc.sourceAnchor].inside &&
            local.anchors[arc.targetAnchor].inside) {
            ++report.usedInternalFiberlets;
        }
    }
    report.unusedInternalFiberlets =
        report.internalFiberlets - report.usedInternalFiberlets;
    report.retainedPhysicalFiberlets.reserve(usedPhysicalFiberlets.size());
    for (const std::size_t edge : usedPhysicalFiberlets)
        report.retainedPhysicalFiberlets.push_back(
            local.physicalFiberlets[edge]);
    report.routeFiberletCounts = chunkRouteDistribution(std::move(counts));
    report.routeLengthsPredictionVoxels =
        chunkRouteDistribution(std::move(lengths));
    report.routeLosses = chunkRouteDistribution(std::move(losses));
    report.routeLossesPerPredictionVoxel =
        chunkRouteDistribution(std::move(densities));
    report.elapsedSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    if (cpuStarted != static_cast<std::clock_t>(-1)) {
        const std::clock_t cpuEnded = std::clock();
        if (cpuEnded != static_cast<std::clock_t>(-1)) {
            report.cpuSeconds = static_cast<double>(cpuEnded - cpuStarted) /
                static_cast<double>(CLOCKS_PER_SEC);
        }
    }
    return report;
}

static void validateChunkRouteAnalysisConfig(
    const FiberletChunkRouteAnalysisConfig& config)
{
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(config.minimumBaseXYZ[axis]) ||
            !std::isfinite(config.maximumBaseXYZ[axis]) ||
            !(config.maximumBaseXYZ[axis] > config.minimumBaseXYZ[axis])) {
            throw std::invalid_argument(
                "fiberlet chunk-route bounds are invalid");
        }
    }
    if (!(config.maximumJoinAngleDegrees >= 0.0F) ||
        !(config.maximumJoinAngleDegrees <= 180.0F) ||
        !std::isfinite(config.maximumJoinAngleDegrees) ||
        config.parallelThreads == 0 ||
        config.maximumGeneratedStatesPerEntry == 0) {
        throw std::invalid_argument(
            "fiberlet chunk-route configuration is invalid");
    }
}

static double clockSecondsSince(std::clock_t started)
{
    if (started == static_cast<std::clock_t>(-1))
        return 0.0;
    const auto ended = std::clock();
    if (ended == static_cast<std::clock_t>(-1))
        return 0.0;
    return static_cast<double>(ended - started) /
        static_cast<double>(CLOCKS_PER_SEC);
}

FiberletChunkRouteSimplificationReport simplifyFiberletChunkRoutes(
    const FiberletChunkGraphSource& graph,
    const FiberletChunkRouteAnalysisConfig& config,
    std::span<const FiberletStorageId> retainedPhysicalFiberlets)
{
    validateChunkRouteAnalysisConfig(config);
    std::vector<vc::render::ChunkKey> seedChunks;
    const auto local = materializeChunkRouteGraph(
        graph, config, seedChunks, true);
    return simplifyMaterializedChunkRoutes(
        local, config, retainedPhysicalFiberlets);
}

FiberletChunkRouteAnalysisReport analyzeFiberletChunkRoutes(
    const FiberletChunkGraphSource& graph,
    const FiberletChunkRouteAnalysisConfig& config)
{
    validateChunkRouteAnalysisConfig(config);
    const auto started = std::chrono::steady_clock::now();
    const auto cpuStarted = std::clock();
    std::vector<vc::render::ChunkKey> seedChunks;
    const auto local = materializeChunkRouteGraph(
        graph, config, seedChunks, true);
    auto report = analyzeMaterializedChunkRoutes(
        local, config, seedChunks);
    report.elapsedSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    report.cpuSeconds = clockSecondsSince(cpuStarted);
    return report;
}

FiberletChunkRouteReductionReport analyzeAndSimplifyFiberletChunkRoutes(
    const FiberletChunkGraphSource& graph,
    const FiberletChunkRouteAnalysisConfig& config)
{
    validateChunkRouteAnalysisConfig(config);
    FiberletChunkRouteReductionReport result;
    std::vector<vc::render::ChunkKey> seedChunks;

    auto wallStarted = std::chrono::steady_clock::now();
    auto cpuStarted = std::clock();
    const auto local = materializeChunkRouteGraph(
        graph, config, seedChunks, true);
    result.materializationSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wallStarted).count();
    result.materializationCpuSeconds = clockSecondsSince(cpuStarted);

    wallStarted = std::chrono::steady_clock::now();
    cpuStarted = std::clock();
    result.analysis = analyzeMaterializedChunkRoutes(
        local, config, seedChunks);
    result.analysisSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wallStarted).count();
    result.analysisCpuSeconds = clockSecondsSince(cpuStarted);

    wallStarted = std::chrono::steady_clock::now();
    cpuStarted = std::clock();
    result.simplification = simplifyMaterializedChunkRoutes(
        local, config, result.analysis.retainedPhysicalFiberlets);
    result.simplificationSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wallStarted).count();
    result.simplificationCpuSeconds = clockSecondsSince(cpuStarted);
    return result;
}

FiberletReductionOverlayBoxWriteReport
writeFiberletReductionOverlayBox(
    const FiberletChunkGraphSource& source,
    const std::shared_ptr<FiberletChunkDataset>& outputAnchors,
    const std::shared_ptr<FiberletChunkDataset>& outputFiberlets,
    const FiberletChunkRouteAnalysisConfig& config,
    std::span<const FiberletStorageId> inputPhysicalFiberlets,
    std::span<const FiberletStorageId> retainedPhysicalFiberlets)
{
    if (!outputAnchors || !outputFiberlets ||
        outputAnchors->metadata().kind != FiberletDatasetKind::Anchors ||
        outputFiberlets->metadata().kind != FiberletDatasetKind::Fiberlets) {
        throw std::invalid_argument(
            "fiberlet reduction overlay requires anchor and path datasets");
    }
    const auto& sourceMetadata = source.metadata();
    const auto& outputMetadata = outputFiberlets->metadata();
    if (outputAnchors->metadata().chunkGridShapeZYX !=
            outputMetadata.chunkGridShapeZYX ||
        outputAnchors->metadata().coordinateOriginZYX !=
            outputMetadata.coordinateOriginZYX ||
        outputAnchors->metadata().coordinateUnitsPerChunkZYX !=
            outputMetadata.coordinateUnitsPerChunkZYX ||
        outputMetadata.chunkGridShapeZYX !=
            sourceMetadata.chunkGridShapeZYX ||
        outputMetadata.coordinateOriginZYX !=
            sourceMetadata.coordinateOriginZYX ||
        outputMetadata.coordinateUnitsPerChunkZYX !=
            sourceMetadata.coordinateUnitsPerChunkZYX ||
        outputMetadata.spatialChunkSideBaseVoxels !=
            sourceMetadata.spatialChunkSideBaseVoxels ||
        outputMetadata.predictionToBaseScale !=
            sourceMetadata.predictionToBaseScale) {
        throw std::invalid_argument(
            "fiberlet reduction overlay layouts are incompatible");
    }

    std::vector<FiberletStorageId> input(
        inputPhysicalFiberlets.begin(), inputPhysicalFiberlets.end());
    std::vector<FiberletStorageId> retained(
        retainedPhysicalFiberlets.begin(), retainedPhysicalFiberlets.end());
    std::sort(input.begin(), input.end());
    std::sort(retained.begin(), retained.end());
    if (std::adjacent_find(input.begin(), input.end()) != input.end() ||
        std::adjacent_find(retained.begin(), retained.end()) !=
            retained.end()) {
        throw std::invalid_argument(
            "fiberlet reduction overlay IDs contain duplicates");
    }
    if (!std::includes(
            input.begin(), input.end(), retained.begin(), retained.end())) {
        throw std::invalid_argument(
            "fiberlet reduction overlay restores a physical Fiberlet");
    }

    const double predictionToBase = sourceMetadata.predictionToBaseScale;
    std::set<FiberletStorageKey> inputAnchorKeys;
    std::set<std::tuple<int, int, int>> inputAnchorOwners;
    for (const auto& id : input) {
        inputAnchorKeys.insert(id.first);
        inputAnchorKeys.insert(id.second);
    }
    for (const auto& key : inputAnchorKeys) {
        const auto owner = fiberletStorageOwnerChunk(
            sourceMetadata, key, 0);
        inputAnchorOwners.emplace(owner.iz, owner.iy, owner.ix);
    }
    std::map<FiberletStorageKey, bool> anchorInside;
    for (const auto& [z, y, x] : inputAnchorOwners) {
        const auto loaded = source.anchorsInChunk({0, z, y, x}, true);
        if (loaded.status != FiberletGraphQueryStatus::Ready) {
            throw std::runtime_error(
                "fiberlet reduction overlay anchor chunk failed: " +
                loaded.error);
        }
        for (const auto& anchor : *loaded.value.anchors) {
            if (!inputAnchorKeys.contains(anchor.key))
                continue;
            const bool inside = pointInsideChunk(
                cv::Vec3d(anchor.positionPredictionXYZ) * predictionToBase,
                config);
            if (!anchorInside.emplace(anchor.key, inside).second) {
                throw std::runtime_error(
                    "fiberlet graph contains a duplicate endpoint anchor");
            }
        }
    }
    if (anchorInside.size() != inputAnchorKeys.size()) {
        throw std::runtime_error(
            "fiberlet graph endpoint anchor is absent from its owner chunk");
    }
    auto isAnchorInside = [&](const FiberletStorageKey& key) {
        const auto found = anchorInside.find(key);
        if (found == anchorInside.end())
            throw std::logic_error(
                "fiberlet reduction overlay lost an endpoint anchor");
        return found->second;
    };
    auto survives = [&](const FiberletStorageId& id) {
        return !isAnchorInside(id.first) ||
            std::binary_search(retained.begin(), retained.end(), id);
    };

    FiberletReductionOverlayBoxWriteReport result;
    result.inputFiberlets = input.size();
    result.retainedFiberlets = static_cast<std::size_t>(std::count_if(
        input.begin(), input.end(), survives));
    std::set<FiberletStorageKey> referencedAnchors;
    for (const auto& id : input) {
        if (survives(id)) {
            referencedAnchors.insert(id.first);
            referencedAnchors.insert(id.second);
        }
    }
    const auto anchorOwners = analysisAnchorChunks(outputMetadata, config);
    std::set<std::tuple<int, int, int>> fiberletOwnerCoordinates;
    std::vector<vc::render::ChunkKey> fiberletOwners = anchorOwners;
    for (const auto& owner : fiberletOwners) {
        fiberletOwnerCoordinates.emplace(owner.iz, owner.iy, owner.ix);
    }
    for (const auto& id : input) {
        if (!isAnchorInside(id.first))
            continue;
        const auto owner = fiberletStorageOwnerChunk(
            outputMetadata, id.first, 0);
        if (fiberletOwnerCoordinates.emplace(
                owner.iz, owner.iy, owner.ix).second) {
            fiberletOwners.push_back(owner);
        }
    }
    std::sort(
        fiberletOwners.begin(), fiberletOwners.end(),
        [](const auto& left, const auto& right) {
            return std::tie(left.iz, left.iy, left.ix) <
                std::tie(right.iz, right.iy, right.ix);
        });
    result.touchedAnchorChunks = anchorOwners.size();
    result.touchedFiberletChunks = fiberletOwners.size();

    struct PreparedFiberletPair {
        vc::render::ChunkKey owner;
        vc::render::ChunkKey routeOwner;
        FiberletChunkDataset::MaterializedChunk currentPrefix;
        FiberletChunkDataset::MaterializedChunk replacementPrefix;
        FiberletChunkDataset::MaterializedChunk currentRoute;
        FiberletChunkDataset::MaterializedChunk replacementRoute;
    };
    struct CurrentFiberletPair {
        vc::render::ChunkKey owner;
        std::shared_ptr<const FiberletPrefixChunkPayload> prefixes;
        std::shared_ptr<const FiberletRouteChunkPayload> routes;
    };
    std::vector<CurrentFiberletPair> currentFiberlets;
    currentFiberlets.reserve(fiberletOwners.size());
    for (const auto& owner : fiberletOwners) {
        const auto current = source.prefixesInChunk(owner, true);
        if (current.status != FiberletGraphQueryStatus::Ready ||
            !current.value.payloadLease) {
            throw std::runtime_error(
                "fiberlet reduction overlay prefix chunk failed: " +
                current.error);
        }
        const auto routes = source.routesInChunk(owner, true);
        if (routes.status != FiberletGraphQueryStatus::Ready ||
            !routes.value.payloadLease) {
            throw std::runtime_error(
                "fiberlet reduction overlay route chunk failed: " +
                routes.error);
        }
        if (current.value.payloadLease->prefixes.size() !=
            routes.value.payloadLease->routes.size()) {
            throw std::runtime_error(
                "fiberlet prefix and route chunk record counts differ");
        }
        currentFiberlets.push_back({
            owner, current.value.payloadLease, routes.value.payloadLease});
    }
    std::set<FiberletStorageKey> additionalAnchorKeys;
    for (const auto& current : currentFiberlets) {
        for (const auto& prefix : current.prefixes->prefixes) {
            if (!anchorInside.contains(prefix.id.first))
                additionalAnchorKeys.insert(prefix.id.first);
            if (!anchorInside.contains(prefix.id.second))
                additionalAnchorKeys.insert(prefix.id.second);
        }
    }
    std::set<std::tuple<int, int, int>> additionalAnchorOwners;
    for (const auto& key : additionalAnchorKeys) {
        const auto owner = fiberletStorageOwnerChunk(
            sourceMetadata, key, 0);
        additionalAnchorOwners.emplace(owner.iz, owner.iy, owner.ix);
    }
    for (const auto& [z, y, x] : additionalAnchorOwners) {
        const auto loaded = source.anchorsInChunk({0, z, y, x}, true);
        if (loaded.status != FiberletGraphQueryStatus::Ready) {
            throw std::runtime_error(
                "fiberlet reduction overlay anchor chunk failed: " +
                loaded.error);
        }
        for (const auto& anchor : *loaded.value.anchors) {
            if (!additionalAnchorKeys.contains(anchor.key))
                continue;
            anchorInside.emplace(
                anchor.key,
                pointInsideChunk(
                    cv::Vec3d(anchor.positionPredictionXYZ) * predictionToBase,
                    config));
        }
    }
    if (std::any_of(
            additionalAnchorKeys.begin(), additionalAnchorKeys.end(),
            [&](const auto& key) { return !anchorInside.contains(key); })) {
        throw std::runtime_error(
            "fiberlet graph endpoint anchor is absent from its owner chunk");
    }
    std::vector<PreparedFiberletPair> preparedFiberlets(
        currentFiberlets.size());
    indexedParallelFor(
        currentFiberlets.size(), config.parallelThreads,
        [&](std::size_t index) {
        const auto& current = currentFiberlets[index];
        const auto& owner = current.owner;
        const auto& currentPrefixes = current.prefixes->prefixes;
        const auto& currentRoutes = current.routes->routes;
        std::vector<FiberletStoredPrefix> replacementPrefixes;
        std::vector<FiberletStoredRoute> replacementRoutes;
        replacementPrefixes.reserve(currentPrefixes.size());
        replacementRoutes.reserve(currentPrefixes.size());
        for (std::size_t route = 0; route < currentPrefixes.size(); ++route) {
            const auto& prefix = currentPrefixes[route];
            if (survives(prefix.id)) {
                replacementPrefixes.push_back(prefix);
                replacementRoutes.push_back(currentRoutes[route]);
            }
        }
        auto routeOwner = owner;
        routeOwner.level = 1;
        const auto prefixCodec = outputFiberlets->codecConfig(
            FiberletStorageChunkKind::FiberletPrefix, owner);
        const auto routeCodec = outputFiberlets->codecConfig(
            FiberletStorageChunkKind::FiberletRoutes, routeOwner);
        const auto makePrefixChunk = [&](const auto& prefixes) {
            auto bytes = serializeFiberletPrefixes(prefixCodec, prefixes);
            return FiberletChunkDataset::MaterializedChunk{
                bytes,
                std::make_shared<const FiberletPrefixChunkPayload>(
                    FiberletDecodedPrefixes{prefixCodec, prefixes}),
                false};
        };
        const auto makeRouteChunk = [&](const auto& routes) {
            auto bytes = serializeFiberletRoutes(routeCodec, routes);
            return FiberletChunkDataset::MaterializedChunk{
                bytes,
                std::make_shared<const FiberletRouteChunkPayload>(
                    FiberletDecodedRoutes{routeCodec, routes}),
                false};
        };
        preparedFiberlets[index] = {
            owner, routeOwner, makePrefixChunk(currentPrefixes),
            makePrefixChunk(replacementPrefixes), makeRouteChunk(currentRoutes),
            makeRouteChunk(replacementRoutes)};
        });

    struct PreparedAnchorChunk {
        vc::render::ChunkKey owner;
        FiberletChunkDataset::MaterializedChunk current;
        FiberletChunkDataset::MaterializedChunk replacement;
        std::size_t inputAnchors = 0;
        std::size_t retainedAnchors = 0;
    };
    struct CurrentAnchorChunk {
        vc::render::ChunkKey owner;
        std::shared_ptr<const std::vector<FiberletStoredAnchor>> anchors;
    };
    std::vector<CurrentAnchorChunk> currentAnchors;
    currentAnchors.reserve(anchorOwners.size());
    for (const auto& owner : anchorOwners) {
        const auto current = source.anchorsInChunk(owner, true);
        if (current.status != FiberletGraphQueryStatus::Ready ||
            !current.value.anchors) {
            throw std::runtime_error(
                "fiberlet reduction overlay anchor chunk failed: " +
                current.error);
        }
        currentAnchors.push_back({owner, current.value.anchors});
    }
    std::vector<PreparedAnchorChunk> preparedAnchors(currentAnchors.size());
    indexedParallelFor(
        currentAnchors.size(), config.parallelThreads,
        [&](std::size_t index) {
        const auto& current = currentAnchors[index];
        const auto& owner = current.owner;
        std::vector<FiberletStoredAnchor> replacement;
        replacement.reserve(current.anchors->size());
        std::size_t inputAnchors = 0;
        std::size_t retainedInsideAnchors = 0;
        for (const auto& anchor : *current.anchors) {
            const bool inside = pointInsideChunk(
                cv::Vec3d(anchor.positionPredictionXYZ) * predictionToBase,
                config);
            if (inside)
                ++inputAnchors;
            const bool referenced =
                !inside || referencedAnchors.contains(anchor.key);
            if (referenced) {
                replacement.push_back(anchor);
                if (inside)
                    ++retainedInsideAnchors;
            }
        }
        const auto codec = outputAnchors->codecConfig(
            FiberletStorageChunkKind::Anchors, owner);
        const auto makeAnchorChunk = [&](const auto& anchors) {
            auto bytes = serializeFiberletAnchors(codec, anchors);
            return FiberletChunkDataset::MaterializedChunk{
                bytes,
                std::make_shared<const FiberletAnchorChunkPayload>(
                    FiberletDecodedAnchors{codec, anchors}),
                false};
        };
        preparedAnchors[index] = {
            owner, makeAnchorChunk(*current.anchors),
            makeAnchorChunk(replacement), inputAnchors,
            retainedInsideAnchors};
        });
    for (const auto& prepared : preparedAnchors) {
        result.inputAnchors += prepared.inputAnchors;
        result.retainedAnchors += prepared.retainedAnchors;
    }

    indexedParallelFor(
        preparedFiberlets.size(), config.parallelThreads,
        [&](std::size_t index) {
            const auto& prepared = preparedFiberlets[index];
            replaceFiberletOverlayChunkPair(
                outputFiberlets, prepared.owner, prepared.currentPrefix,
                prepared.replacementPrefix, prepared.routeOwner,
                prepared.currentRoute, prepared.replacementRoute);
        });
    indexedParallelFor(
        preparedAnchors.size(), config.parallelThreads,
        [&](std::size_t index) {
            const auto& prepared = preparedAnchors[index];
            replaceFiberletOverlayChunk(
                outputAnchors, FiberletStorageChunkKind::Anchors,
                prepared.owner, prepared.current, prepared.replacement);
        });
    return result;
}

FiberletReductionWriteReport writeReducedFiberletChunk(
    const FiberletChunkGraphSource& source,
    const std::shared_ptr<FiberletChunkDataset>& outputDataset,
    const vc::render::ChunkKey& requestedOwner,
    std::span<const FiberletStorageId> inputFiberlets,
    std::span<const FiberletStorageId> retainedFiberlets)
{
    if (!outputDataset ||
        outputDataset->metadata().kind != FiberletDatasetKind::Fiberlets) {
        throw std::invalid_argument(
            "reduced fiberlet output must be a Fiberlets dataset");
    }
    auto owner = requestedOwner;
    if (owner.level != 0)
        throw std::invalid_argument(
            "reduced fiberlet owner must use level zero");
    (void)outputDataset->codecConfig(
        FiberletStorageChunkKind::FiberletPrefix, owner);
    const auto& inputMetadata = source.metadata();
    const auto& outputMetadata = outputDataset->metadata();
    if (outputMetadata.coordinateOriginZYX !=
            inputMetadata.coordinateOriginZYX ||
        outputMetadata.maximumEndpointReachCoordinateUnitsZYX !=
            inputMetadata.maximumEndpointReachCoordinateUnitsZYX ||
        outputMetadata.predictionToBaseScale !=
            inputMetadata.predictionToBaseScale) {
        throw std::invalid_argument(
            "reduced fiberlet output geometry differs from its source");
    }

    std::vector<FiberletStorageId> retained(
        retainedFiberlets.begin(), retainedFiberlets.end());
    std::sort(retained.begin(), retained.end());
    retained.erase(std::unique(retained.begin(), retained.end()), retained.end());

    FiberletReductionWriteReport report;
    report.owner = owner;
    std::vector<FiberletStoredPrefix> prefixes;
    std::vector<FiberletStoredRoute> routes;
    for (const auto& id : inputFiberlets) {
        if (fiberletStorageOwnerChunk(outputMetadata, id.first, 0) != owner)
            continue;
        ++report.inputFiberlets;
        if (!std::binary_search(retained.begin(), retained.end(), id))
            continue;
        const auto route = source.storedRoute(id, true);
        if (route.status != FiberletGraphQueryStatus::Ready) {
            throw std::runtime_error(
                "reduced fiberlet source route failed: " + route.error);
        }
        prefixes.push_back(route.value.prefix);
        routes.push_back(route.value.route);
    }
    report.retainedFiberlets = prefixes.size();
    auto routeOwner = owner;
    routeOwner.level = 1;
    const auto prefixCodec = outputDataset->codecConfig(
        FiberletStorageChunkKind::FiberletPrefix, owner);
    const auto routeCodec = outputDataset->codecConfig(
        FiberletStorageChunkKind::FiberletRoutes, routeOwner);
    auto prefixBytes = serializeFiberletPrefixes(prefixCodec, prefixes);
    auto routeBytes = serializeFiberletRoutes(routeCodec, routes);
    FiberletChunkDataset::MaterializedChunk prefixChunk{
        prefixBytes,
        std::make_shared<const FiberletPrefixChunkPayload>(
            FiberletDecodedPrefixes{prefixCodec, prefixes}),
        false};
    FiberletChunkDataset::MaterializedChunk routeChunk{
        routeBytes,
        std::make_shared<const FiberletRouteChunkPayload>(
            FiberletDecodedRoutes{routeCodec, routes}),
        false};
    const bool prefixExists = std::filesystem::exists(outputDataset->chunkPath(
        FiberletStorageChunkKind::FiberletPrefix, owner));
    const bool routeExists = std::filesystem::exists(outputDataset->chunkPath(
        FiberletStorageChunkKind::FiberletRoutes, routeOwner));
    if (prefixExists && routeExists) {
        const auto existingPrefix = outputDataset->readMaterializedChunk(
            FiberletStorageChunkKind::FiberletPrefix, owner);
        const auto existingRoute = outputDataset->readMaterializedChunk(
            FiberletStorageChunkKind::FiberletRoutes, routeOwner);
        if (!existingPrefix || !existingRoute ||
            existingPrefix->bytes != prefixBytes ||
            existingRoute->bytes != routeBytes) {
            throw std::runtime_error(
                "existing reduced fiberlet chunk differs from the "
                "deterministic reduction");
        }
        report.reused = true;
        return report;
    }
    if (prefixExists != routeExists) {
        std::error_code error;
        const auto partialPath = outputDataset->chunkPath(
            prefixExists
                ? FiberletStorageChunkKind::FiberletPrefix
                : FiberletStorageChunkKind::FiberletRoutes,
            prefixExists ? owner : routeOwner);
        if (!std::filesystem::remove(partialPath, error) || error) {
            throw std::runtime_error(
                "cannot remove incomplete reduced fiberlet chunk pair: " +
                error.message());
        }
    }
    outputDataset->publishFiberletChunkPair(
        owner, prefixChunk, routeOwner, routeChunk);
    return report;
}

FiberletCachedReplayGraphSource::FiberletCachedReplayGraphSource(
    std::shared_ptr<FiberletOnDemandPreprocessor> preprocessor, FiberletPathConfig pathConfig, FiberletEvaluationQuantization evaluationQuantization, float maximumJoinAngleDegrees)
    : preprocessor_(std::move(preprocessor))
    , chunks_(
          preprocessor_ ? preprocessor_->anchorDataset() : nullptr,
          preprocessor_ ? preprocessor_->anchorCache() : nullptr,
          preprocessor_ ? preprocessor_->fiberletDataset() : nullptr,
          preprocessor_ ? preprocessor_->fiberletCache() : nullptr,
          pathConfig,
          [preprocessor = preprocessor_](const vc::render::ChunkKey& key, std::shared_ptr<const FiberletAnchorChunkPayload> anchors) {
              return preprocessor->evaluationAnchorChunk(key, std::move(anchors));
          })
    , pathConfig_(std::move(pathConfig))
    , evaluationQuantization_(std::move(evaluationQuantization))
    , maximumJoinAngleDegrees_(maximumJoinAngleDegrees)
    , quantizationState_(std::make_shared<QuantizationState>())
{
    if (!preprocessor_ || !(maximumJoinAngleDegrees_ >= 0.0F) || !(maximumJoinAngleDegrees_ <= 180.0F) || !std::isfinite(maximumJoinAngleDegrees_)) {
        throw std::invalid_argument("cached fiberlet replay graph configuration is invalid");
    }
    if ((evaluationQuantization_.costBits != 0 && evaluationQuantization_.costBits != 8 && evaluationQuantization_.costBits != 16) ||
        (evaluationQuantization_.costDomain != FiberletCostQuantizationDomain::RawTotal &&
         evaluationQuantization_.costDomain != FiberletCostQuantizationDomain::SqrtPerPredictionVoxel) ||
        (evaluationQuantization_.costDomain == FiberletCostQuantizationDomain::SqrtPerPredictionVoxel &&
         (!(evaluationQuantization_.costDensityMaximum > 0.0F) ||
          !std::isfinite(evaluationQuantization_.costDensityMaximum))) ||
        evaluationQuantization_.storageChunkSideBaseVoxels <= 0) {
        throw std::invalid_argument("cached fiberlet logical quantization is invalid");
    }
    (void)fiberletPositionBinCountForEvaluation(
        evaluationQuantization_.storageChunkSideBaseVoxels,
        evaluationQuantization_.positionQuantumBaseVoxels);
}

float FiberletCachedReplayGraphSource::predictionToBaseScale() const noexcept
{
    return static_cast<float>(preprocessor_->anchorDataset()->metadata().predictionToBaseScale);
}

int FiberletCachedReplayGraphSource::anchorCellSizePredictionVoxels() const noexcept
{
    const auto& metadata = preprocessor_->anchorDataset()->metadata();
    const double cellBase = static_cast<double>(metadata.spatialChunkSideBaseVoxels) / static_cast<double>(metadata.coordinateUnitsPerChunkZYX[0]);
    return static_cast<int>(std::llround(cellBase / metadata.predictionToBaseScale));
}

float FiberletCachedReplayGraphSource::maximumJoinAngleDegrees() const noexcept
{
    return maximumJoinAngleDegrees_;
}

FiberletStorageKey FiberletCachedReplayGraphSource::logicalAnchorId(const FiberletStorageKey& physical) const
{
    return logicalAnchorKey(physical);
}

DirectedFiberletStorageId FiberletCachedReplayGraphSource::logicalArcId(const DirectedFiberletStorageId& physical) const
{
    return logicalFiberletId(physical);
}

bool FiberletCachedReplayGraphSource::anchorCellInCorridor(const FiberletStorageKey& anchorKey) const
{
    return preprocessor_->isSelectedAnchorCell(anchorKey);
}

FiberletStorageKey FiberletCachedReplayGraphSource::logicalAnchorKey(const FiberletStorageKey& physical) const
{
    if (!evaluationQuantization_.enabled())
        return physical;
    if (physical.variant > 1)
        throw std::invalid_argument("quantized anchor source cell has more than two variants");
    std::lock_guard lock(quantizationState_->mutex);
    quantizationState_->maximumVariants = std::max(quantizationState_->maximumVariants, static_cast<std::size_t>(physical.variant) + 1);
    return physical;
}

DirectedFiberletStorageId FiberletCachedReplayGraphSource::logicalFiberletId(const DirectedFiberletStorageId& physical) const
{
    const auto source = logicalAnchorKey(directedSource(physical));
    const auto target = logicalAnchorKey(directedTarget(physical));
    if (source == target)
        throw std::invalid_argument("quantized fiberlet endpoints collapse to one anchor key");
    FiberletStorageId logical{std::min(source, target), std::max(source, target)};
    for (size_t axis = 0; axis < 3; ++axis) {
        const auto delta = logical.second.coordinateZYX[axis] - logical.first.coordinateZYX[axis];
        if (delta < std::numeric_limits<std::int16_t>::min() || delta > std::numeric_limits<std::int16_t>::max()) {
            throw std::invalid_argument("quantized fiberlet endpoint delta exceeds int16");
        }
    }
    return {logical, source != logical.first};
}

std::array<int, 3> FiberletCachedReplayGraphSource::compactCostOwner(const FiberletStorageId& physical) const
{
    const auto logical = logicalFiberletId({physical, false}).fiberlet;
    const auto firstLogical = logicalAnchorKey(physical.first);
    const FiberletStorageKey& firstPhysical = firstLogical == logical.first ? physical.first : physical.second;
    const auto anchor = chunks_.anchor(firstPhysical, true);
    if (anchor.status != FiberletGraphQueryStatus::Ready) {
        throw std::runtime_error("cached fiberlet compact cost owner failed: " + anchor.error);
    }
    const cv::Vec3f base = anchor.value.anchor.positionPredictionXYZ * predictionToBaseScale();
    ChunkCoordinate result{};
    for (size_t zyx = 0; zyx < 3; ++zyx) {
        const int xyz = static_cast<int>(2 - zyx);
        result[zyx] =
            static_cast<int>(std::floor(static_cast<double>(base[xyz]) / static_cast<double>(evaluationQuantization_.storageChunkSideBaseVoxels)));
    }
    return result;
}

float FiberletCachedReplayGraphSource::quantizedCost(
    const FiberletStorageId& physical,
    float cost,
    float pathLengthPredictionVoxels) const
{
    if (evaluationQuantization_.costBits != 8 && evaluationQuantization_.costBits != 16) {
        return cost;
    }
    if (evaluationQuantization_.costDomain ==
        FiberletCostQuantizationDomain::SqrtPerPredictionVoxel) {
        return quantizeFiberletCostForEvaluation(
            cost, pathLengthPredictionVoxels, 0.0F, 1.0F,
            evaluationQuantization_.costBits,
            evaluationQuantization_.costDomain,
            evaluationQuantization_.costDensityMaximum);
    }
    const ChunkCoordinate compactOwner = compactCostOwner(physical);
    {
        std::lock_guard lock(quantizationState_->mutex);
        if (const auto found = quantizationState_->compactCostRanges.find(compactOwner); found != quantizationState_->compactCostRanges.end()) {
            const auto& range = found->second;
            return quantizeFiberletCostForEvaluation(
                cost,
                pathLengthPredictionVoxels,
                range.minimum,
                range.maximum,
                evaluationQuantization_.costBits,
                evaluationQuantization_.costDomain,
                evaluationQuantization_.costDensityMaximum);
        }
    }

    const auto& metadata = preprocessor_->fiberletDataset()->metadata();
    const int64_t unitsPerPhysicalChunk = metadata.coordinateUnitsPerChunkZYX[0];
    const int64_t maximumReach =
        *std::max_element(metadata.maximumEndpointReachCoordinateUnitsZYX.begin(), metadata.maximumEndpointReachCoordinateUnitsZYX.end());
    const int halo = static_cast<int>((maximumReach + unitsPerPhysicalChunk) / unitsPerPhysicalChunk);
    QuantizationState::Contribution combined;
    for (int z = compactOwner[0] - halo; z <= compactOwner[0] + halo; ++z) {
        for (int y = compactOwner[1] - halo; y <= compactOwner[1] + halo; ++y) {
            for (int x = compactOwner[2] - halo; x <= compactOwner[2] + halo; ++x) {
                const ChunkCoordinate physicalChunk{z, y, x};
                if (z < 0 || y < 0 || x < 0 || z >= metadata.chunkGridShapeZYX[0] || y >= metadata.chunkGridShapeZYX[1] ||
                    x >= metadata.chunkGridShapeZYX[2]) {
                    continue;
                }
                bool scanned = false;
                {
                    std::lock_guard lock(quantizationState_->mutex);
                    scanned = quantizationState_->physicalCostContributions.contains(physicalChunk);
                }
                if (!scanned) {
                    const auto loaded = chunks_.prefixesInChunk({0, z, y, x}, true);
                    if (loaded.status != FiberletGraphQueryStatus::Ready) {
                        throw std::runtime_error("cached fiberlet compact cost chunk failed: " + loaded.error);
                    }
                    std::map<ChunkCoordinate, QuantizationState::Contribution> contributions;
                    for (const auto& prefix : loaded.value.payloadLease->prefixes) {
                        const ChunkCoordinate group = compactCostOwner(prefix.id);
                        auto& contribution = contributions[group];
                        const float value = fiberletCostQuantizationValueForEvaluation(
                            prefix.cost.total(),
                            prefix.pathLengthPredictionVoxels,
                            evaluationQuantization_.costDomain,
                            evaluationQuantization_.costDensityMaximum);
                        contribution.minimum = std::min(contribution.minimum, value);
                        contribution.maximum = std::max(contribution.maximum, value);
                        contribution.populated = true;
                    }
                    std::lock_guard lock(quantizationState_->mutex);
                    quantizationState_->physicalCostContributions.emplace(physicalChunk, std::move(contributions));
                }
                std::lock_guard lock(quantizationState_->mutex);
                const auto& contributions = quantizationState_->physicalCostContributions.at(physicalChunk);
                if (const auto found = contributions.find(compactOwner); found != contributions.end() && found->second.populated) {
                    combined.minimum = std::min(combined.minimum, found->second.minimum);
                    combined.maximum = std::max(combined.maximum, found->second.maximum);
                    combined.populated = true;
                }
            }
        }
    }
    if (!combined.populated)
        throw std::logic_error("fiberlet compact cost group is empty");
    {
        std::lock_guard lock(quantizationState_->mutex);
        quantizationState_->compactCostRanges.emplace(compactOwner, combined);
    }
    return quantizeFiberletCostForEvaluation(
        cost,
        pathLengthPredictionVoxels,
        combined.minimum,
        combined.maximum,
        evaluationQuantization_.costBits,
        evaluationQuantization_.costDomain,
        evaluationQuantization_.costDensityMaximum);
}

std::vector<FiberletReplaySourceAnchor> FiberletCachedReplayGraphSource::anchorsNearReference(
    const PolylineArcGeometry& reference, double beginArcBase, double endArcBase, double broadPhaseRadiusBaseVoxels) const
{
    const double anchorCellDiagonalBase = std::sqrt(3.0) * static_cast<double>(anchorCellSizePredictionVoxels()) * predictionToBaseScale();
    const auto schedule =
        preprocessor_->referenceChunkSchedule(reference, beginArcBase, endArcBase, broadPhaseRadiusBaseVoxels + anchorCellDiagonalBase);
    std::map<FiberletStorageKey, FiberletReplaySourceAnchor> unique;
    for (const auto& scheduled : schedule) {
        const auto loaded = chunks_.anchorsInChunk(scheduled.key, true);
        if (loaded.status != FiberletGraphQueryStatus::Ready)
            throw std::runtime_error("cached fiberlet seed chunk failed: " + loaded.error);
        for (const auto& anchor : *loaded.value.anchors) {
            if (!anchorCellInCorridor(anchor.key))
                continue;
            const cv::Vec3f positionBase = anchor.positionPredictionXYZ * predictionToBaseScale();
            const auto projection = projectPointToPolylineArc(reference, cv::Vec3d(positionBase), beginArcBase, endArcBase);
            if (projection.arc + 1.0e-12 < beginArcBase || projection.arc > endArcBase + 1.0e-12 || projection.distance > broadPhaseRadiusBaseVoxels)
                continue;
            const auto [found, inserted] = unique.emplace(anchor.key, FiberletReplaySourceAnchor{anchor.key, positionBase});
            if (!inserted && vectorLength(found->second.positionBaseXYZ - positionBase) > 1.0e-5F) {
                throw std::runtime_error("cached fiberlet anchor has inconsistent positions");
            }
        }
    }
    std::vector<FiberletReplaySourceAnchor> result;
    result.reserve(unique.size());
    for (auto& [id, anchor] : unique) {
        (void)id;
        result.push_back(std::move(anchor));
    }
    if (evaluationQuantization_.enabled()) {
        std::lock_guard lock(quantizationState_->mutex);
        quantizationState_->projectedAnchors = std::max(quantizationState_->projectedAnchors, result.size());
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) { return left.id < right.id; });
    return result;
}

std::vector<DirectedFiberletStorageId> FiberletCachedReplayGraphSource::outgoing(const FiberletStorageKey& anchorKey) const
{
    const auto loaded = chunks_.incidentEdges(anchorKey, true);
    if (loaded.status != FiberletGraphQueryStatus::Ready)
        throw std::runtime_error("cached fiberlet adjacency failed: " + loaded.error);
    std::vector<DirectedFiberletStorageId> result;
    result.reserve(loaded.value.edges.size());
    for (const auto& edge : loaded.value.edges) {
        if (!anchorCellInCorridor(directedTarget(edge.id)))
            continue;
        result.push_back(edge.id);
    }
    std::sort(result.begin(), result.end(), [&](const auto& left, const auto& right) {
        return std::tuple{logicalFiberletId(left), left} < std::tuple{logicalFiberletId(right), right};
    });
    return result;
}

FiberletReplaySourceArc FiberletCachedReplayGraphSource::arc(const DirectedFiberletStorageId& id) const
{
    const auto loaded = chunks_.directedEdge(id, true);
    if (loaded.status != FiberletGraphQueryStatus::Ready)
        throw std::runtime_error("cached fiberlet edge failed: " + loaded.error);
    FiberletReplaySourceArc result = loaded.value;
    if (evaluationQuantization_.costBits == 8 || evaluationQuantization_.costBits == 16) {
        result.cost = {
            quantizedCost(
                id.fiberlet,
                result.cost.total(),
                result.pathLengthPredictionVoxels),
            0.0F,
            0.0F,
            0.0F,
            0.0F};
    }
    return result;
}

FiberletReplaySourceCostProfile FiberletCachedReplayGraphSource::costProfile(
    const DirectedFiberletStorageId& id) const
{
    const auto loaded = chunks_.route(id.fiberlet, true);
    if (loaded.status != FiberletGraphQueryStatus::Ready)
        throw std::runtime_error("cached fiberlet cost profile failed: " + loaded.error);
    FiberletReplaySourceCostProfile result;
    result.segmentLengthsPredictionVoxels.reserve(
        loaded.value.pointsPredictionXYZ.size() - 1);
    for (size_t segment = 1; segment < loaded.value.pointsPredictionXYZ.size(); ++segment) {
        const float segmentLength = vectorLength(
            loaded.value.pointsPredictionXYZ[segment] -
            loaded.value.pointsPredictionXYZ[segment - 1]);
        if (!(segmentLength > 0.0F) || !std::isfinite(segmentLength))
            throw std::runtime_error("cached fiberlet route segment length is invalid");
        result.segmentLengthsPredictionVoxels.push_back(segmentLength);
    }
    result.segmentCostDensities = loaded.value.route.segmentCostDensities;
    if (id.reverse) {
        std::reverse(
            result.segmentLengthsPredictionVoxels.begin(),
            result.segmentLengthsPredictionVoxels.end());
        std::reverse(
            result.segmentCostDensities.begin(),
            result.segmentCostDensities.end());
    }
    return result;
}

std::vector<cv::Vec3d> FiberletCachedReplayGraphSource::routePoints(const DirectedFiberletStorageId& id) const
{
    const auto loaded = chunks_.route(id.fiberlet, true);
    if (loaded.status != FiberletGraphQueryStatus::Ready)
        throw std::runtime_error("cached fiberlet route failed: " + loaded.error);
    std::vector<cv::Vec3d> result;
    result.reserve(loaded.value.pointsPredictionXYZ.size());
    for (const auto& point : loaded.value.pointsPredictionXYZ)
        result.emplace_back(point * predictionToBaseScale());
    if (id.reverse)
        std::reverse(result.begin(), result.end());
    return result;
}

std::optional<FiberletReplaySourceTransition> FiberletCachedReplayGraphSource::transition(
    const FiberletReplaySourceArc& incomingArc, const FiberletReplaySourceArc& outgoingArc) const
{
    const auto loaded = chunks_.transition(
        incomingArc, outgoingArc, maximumJoinAngleDegrees_, true);
    if (loaded.status != FiberletGraphQueryStatus::Ready)
        throw std::runtime_error(
            "cached fiberlet transition failed: " + loaded.error);
    return loaded.value;
}

FiberletLogicalProjectionStats FiberletCachedReplayGraphSource::logicalProjectionStats() const
{
    std::lock_guard lock(quantizationState_->mutex);
    return {
        quantizationState_->projectedAnchors,
        quantizationState_->coincidentPositionGroups,
        quantizationState_->maximumVariants,
        quantizationState_->compactCostRanges.size(),
    };
}

}  // namespace vc::fiber_tracer
