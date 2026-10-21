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
#include <type_traits>
#include <unordered_map>

namespace vc::fiber_tracer
{
namespace
{

std::int64_t floorDiv(std::int64_t numerator, std::int64_t denominator)
{
    if (denominator <= 0)
        throw std::invalid_argument("fiberlet graph chunk divisor must be positive");
    auto quotient = numerator / denominator;
    if (numerator % denominator < 0)
        --quotient;
    return quotient;
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

std::optional<FiberletPathCost> storedTransitionCost(
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
    return pathCost(local);
}

std::vector<cv::Vec3f> reconstructedStoredRoutePoints(
    const FiberletStoredPrefix& prefix,
    const FiberletStoredRoute& route,
    const FiberletStoredAnchor& firstAnchor,
    const FiberletStoredAnchor& secondAnchor,
    const FiberletPathConfig& pathConfig)
{
    std::vector<std::array<std::int16_t, 2>> lattice;
    if (prefix.interiorPointCount == 1) {
        lattice.push_back(prefix.entryUV);
    } else if (prefix.interiorPointCount >= 2) {
        if (route.middleUV.size() + 2 != prefix.interiorPointCount) {
            throw std::runtime_error(
                "fiberlet route interior count differs from its prefix");
        }
        lattice.push_back(prefix.entryUV);
        lattice.insert(
            lattice.end(), route.middleUV.begin(), route.middleUV.end());
        lattice.push_back(prefix.exitUV);
    } else if (!route.middleUV.empty()) {
        throw std::runtime_error(
            "direct fiberlet route unexpectedly has interior geometry");
    }
    const auto expectedCosts =
        static_cast<std::size_t>(prefix.interiorPointCount) + 1;
    if (route.segmentCostDensities.size() != expectedCosts) {
        throw std::runtime_error(
            "fiberlet route cost-density count differs from its geometry");
    }
    return reconstructFiberletRoutePoints(
        firstAnchor.positionPredictionXYZ,
        firstAnchor.fittedAxisXYZ,
        secondAnchor.positionPredictionXYZ,
        secondAnchor.fittedAxisXYZ,
        lattice,
        pathConfig);
}

}  // namespace

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
    if ((anchorDataset_->metadata().kind != FiberletDatasetKind::Anchors && anchorDataset_->metadata().kind != FiberletDatasetKind::Combined && anchorDataset_->metadata().kind != FiberletDatasetKind::Published) ||
        (fiberletDataset_->metadata().kind != FiberletDatasetKind::Fiberlets && fiberletDataset_->metadata().kind != FiberletDatasetKind::Combined && fiberletDataset_->metadata().kind != FiberletDatasetKind::Published))
        throw std::invalid_argument("fiberlet chunk graph dataset kinds are invalid");
    // Production generation retains float32 intermediate anchors while final
    // paths use compact directions/fixed costs. Both retain float positions
    // and identical stable IDs; the supplied anchor view applies direction
    // quantization before graph scoring.
    const bool productionIntermediate = anchorView_ &&
        anchorDataset_->metadata().profile == FiberletStorageProfile::Float32Cache &&
        fiberletDataset_->metadata().profile == FiberletStorageProfile::CompactDirectionsFixedCost;
    if ((!productionIntermediate && anchorDataset_->metadata().profile != fiberletDataset_->metadata().profile) ||
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
    return fiberletDataset_->chunkOwningAnchor(anchor, level);
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

vc::render::ChunkResult FiberletChunkGraphSource::fetchChunk(
    const std::shared_ptr<vc::render::ChunkCache>& cache,
    const vc::render::ChunkKey& key, bool blocking) const
{
    if (!blocking) return cache->tryGetChunk(key.level, key.iz, key.iy, key.ix);
    auto result = cache->getChunkIfCached(key.level, key.iz, key.iy, key.ix);
    if (result.status == vc::render::ChunkStatus::Data) return result;
    const auto wait = [&] { result = cache->getChunkBlocking(key.level, key.iz, key.iy, key.ix); };
    if (wait_) wait_(wait); else wait();
    return result;
}

FiberletGraphQuery<FiberletIncidentLease> FiberletChunkGraphSource::incidentEdges(const FiberletStorageKey& anchorKey, bool blocking) const
{
    FiberletGraphQuery<FiberletIncidentLease> result;
    const auto keys = incidentOwnerChunks(anchorKey);
    fiberletCache_->prefetchChunks(keys, false);
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
    std::shared_ptr<const std::vector<FiberletStoredAnchor>> transformed;
    const auto view = [&] { transformed = anchorView_(key, payload); };
    if (wait_) wait_(view); else view();
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
    const auto view = [&] { result.value.anchors = anchorView_(key, payload); };
    if (wait_) wait_(view); else view();
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

void FiberletChunkGraphSource::prefetchAnchorChunks(
    std::span<const vc::render::ChunkKey> chunks,
    bool blocking) const
{
    anchorCache_->prefetchChunks(
        std::vector<vc::render::ChunkKey>(chunks.begin(), chunks.end()),
        blocking);
}

void FiberletChunkGraphSource::prefetchFiberletChunks(
    std::span<const vc::render::ChunkKey> chunks,
    bool blocking) const
{
    fiberletCache_->prefetchChunks(
        std::vector<vc::render::ChunkKey>(chunks.begin(), chunks.end()),
        blocking);
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

FiberletGraphQuery<FiberletAnchorLease> FiberletChunkGraphSource::endpoint(
    const FiberletStorageKey& key, const FiberletPrefixChunkPayload& prefix, bool blocking) const
{
    if (!prefix.endpoints)
        return anchor(key, blocking);
    const auto* found = prefix.endpoints->find(key);
    if (!found)
        return {FiberletGraphQueryStatus::Error, {}, "published endpoint missing from owner payload"};
    return {FiberletGraphQueryStatus::Ready, {prefix.endpoints, *found}, {}};
}

FiberletGraphQuery<FiberletEdgeLease> FiberletChunkGraphSource::edge(const FiberletStorageId& fiberlet, bool blocking) const
{
    const auto prefixKey = ownerChunk(fiberlet.first, 0);
    const auto firstAnchorKey = ownerChunk(fiberlet.first, 0);
    const auto secondAnchorKey = ownerChunk(fiberlet.second, 0);
    fiberletCache_->prefetchChunks({prefixKey}, false);
    if (fiberletDataset_->metadata().kind != FiberletDatasetKind::Published)
        anchorCache_->prefetchChunks({firstAnchorKey, secondAnchorKey}, false);

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
    const auto firstAnchor = endpoint(fiberlet.first, *prefixPayload, blocking);
    if (firstAnchor.status != FiberletGraphQueryStatus::Ready) {
        result.status = firstAnchor.status;
        result.error = firstAnchor.error;
        return result;
    }
    const auto secondAnchor = endpoint(fiberlet.second, *prefixPayload, blocking);
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
    fiberletCache_->prefetchChunks({prefixKey, routeKey}, false);
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
    const auto firstAnchor = endpoint(fiberlet.first, *result.value.prefixPayloadLease, blocking);
    if (firstAnchor.status != FiberletGraphQueryStatus::Ready) {
        result.status = firstAnchor.status;
        result.error = firstAnchor.error;
        result.value = {};
        return result;
    }
    const auto secondAnchor = endpoint(fiberlet.second, *result.value.prefixPayloadLease, blocking);
    if (secondAnchor.status != FiberletGraphQueryStatus::Ready) {
        result.status = secondAnchor.status;
        result.error = secondAnchor.error;
        result.value = {};
        return result;
    }
    result.value.anchorPayloadLeases = {firstAnchor.value.payloadLease, secondAnchor.value.payloadLease};
    try {
        result.value.pointsPredictionXYZ = reconstructedStoredRoutePoints(
            result.value.prefix, result.value.route,
            firstAnchor.value.anchor, secondAnchor.value.anchor,
            pathConfig_);
    } catch (const std::exception& error) {
        result.status = FiberletGraphQueryStatus::Error;
        const auto& first = fiberlet.first;
        const auto& second = fiberlet.second;
        result.error = std::string(error.what()) +
            " fiberlet=" +
            std::to_string(first.coordinateZYX[0]) + ',' +
            std::to_string(first.coordinateZYX[1]) + ',' +
            std::to_string(first.coordinateZYX[2]) + ',' +
            std::to_string(first.variant) + "->" +
            std::to_string(second.coordinateZYX[0]) + ',' +
            std::to_string(second.coordinateZYX[1]) + ',' +
            std::to_string(second.coordinateZYX[2]) + ',' +
            std::to_string(second.variant);
    }
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
    const auto shared = [&]() -> FiberletGraphQuery<FiberletAnchorLease> {
        if (fiberletDataset_->metadata().kind != FiberletDatasetKind::Published)
            return anchor(incoming.target, blocking);
        const auto prefix = prefixesInChunk(ownerChunk(outgoing.id.fiberlet.first, 0), blocking);
        if (prefix.status != FiberletGraphQueryStatus::Ready)
            return {prefix.status, {}, prefix.error};
        return endpoint(incoming.target, *prefix.value.payloadLease, blocking);
    }();
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
    const auto cost = transitionCostAtAnchor(
        incoming, outgoing, sharedAnchor, maximumJoinAngleDegrees);
    result.status = cost.status;
    result.error = cost.error;
    if (cost.status == FiberletGraphQueryStatus::Ready && cost.value)
        result.value = FiberletReplaySourceTransition{
            incoming.id, outgoing.id, *cost.value};
    return result;
}

FiberletGraphQuery<std::optional<FiberletPathCost>>
FiberletChunkGraphSource::transitionCostAtAnchor(
    const FiberletReplaySourceArc& incoming,
    const FiberletReplaySourceArc& outgoing,
    const FiberletStoredAnchor& sharedAnchor,
    float maximumJoinAngleDegrees) const
{
    FiberletGraphQuery<std::optional<FiberletPathCost>> result;
    if (!(maximumJoinAngleDegrees >= 0.0F) ||
        !(maximumJoinAngleDegrees <= 180.0F) ||
        !std::isfinite(maximumJoinAngleDegrees)) {
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet transition maximum join angle is invalid";
        return result;
    }
    result.status = FiberletGraphQueryStatus::Ready;
    result.value = storedTransitionCost(
        incoming, outgoing, sharedAnchor,
        static_cast<float>(fiberletDataset_->metadata().predictionToBaseScale),
        pathConfig_, maximumJoinAngleDegrees);
    return result;
}

const FiberletDatasetMetadata& FiberletChunkGraphSource::metadata() const noexcept
{
    return fiberletDataset_->metadata();
}

const FiberletPathConfig& FiberletChunkGraphSource::pathConfig() const noexcept
{
    return pathConfig_;
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
    if (dataset_->metadata().kind != FiberletDatasetKind::Combined &&
        dataset_->metadata().kind != FiberletDatasetKind::Published)
        throw std::invalid_argument(
            "stored replay graph requires a combined Fiberlet dataset");
    if (!(maximumJoinAngleDegrees_ >= 0.0F) ||
        !(maximumJoinAngleDegrees_ <= 180.0F) ||
        !std::isfinite(maximumJoinAngleDegrees_)) {
        throw std::invalid_argument(
            "stored replay graph maximum join angle is invalid");
    }
}

FiberletStoredReplayGraphSource::FiberletStoredReplayGraphSource(
    std::shared_ptr<FiberletChunkDataset> anchorDataset,
    std::shared_ptr<vc::render::ChunkCache> anchorCache,
    std::shared_ptr<FiberletChunkDataset> fiberletDataset,
    std::shared_ptr<vc::render::ChunkCache> fiberletCache,
    FiberletPathConfig pathConfig,
    float maximumJoinAngleDegrees,
    FiberletAnchorView anchorView)
    : dataset_(std::move(fiberletDataset))
    , anchorCache_(std::move(anchorCache))
    , pathCache_(std::move(fiberletCache))
    , pathConfig_(std::move(pathConfig))
    , chunks_(
          std::move(anchorDataset), anchorCache_, dataset_, pathCache_,
          pathConfig_, std::move(anchorView))
    , maximumJoinAngleDegrees_(maximumJoinAngleDegrees)
{
    if (!dataset_ ||
        dataset_->metadata().kind != FiberletDatasetKind::Fiberlets) {
        throw std::invalid_argument(
            "layered stored replay graph requires a Fiberlet path dataset");
    }
    validateFiberletPathConfig(pathConfig_);
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
    const cv::Vec3d& maximumBaseXYZ,
    std::size_t prefetchWindow) const
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
        begin[zyx] = static_cast<int>(std::clamp(std::floor(
            (minimumBaseXYZ[static_cast<int>(xyz)] - origin) / chunkSide),
            0.0, static_cast<double>(metadata.chunkGridShapeZYX[zyx])));
        end[zyx] = static_cast<int>(std::clamp(std::floor(
            (std::nextafter(
                 maximumBaseXYZ[static_cast<int>(xyz)],
                 -std::numeric_limits<double>::infinity()) - origin) /
            chunkSide), -1.0,
            static_cast<double>(metadata.chunkGridShapeZYX[zyx] - 1)));
    }

    std::vector<FiberletStoredAnchor> result;
    if (begin[0] > end[0] || begin[1] > end[1] || begin[2] > end[2])
        return result;
    const float scale = predictionToBaseScale();
    std::vector<vc::render::ChunkKey> ownerKeys;
    if(prefetchWindow){
        for(int z=begin[0];z<=end[0];++z)for(int y=begin[1];y<=end[1];++y)for(int x=begin[2];x<=end[2];++x)
            ownerKeys.push_back({0,z,y,x});
        chunks_.prefetchAnchorChunks(std::span(ownerKeys).first(std::min(prefetchWindow,ownerKeys.size())),false);
    }
    std::size_t ownerIndex=0;
    for (int z = begin[0]; z <= end[0]; ++z) {
        for (int y = begin[1]; y <= end[1]; ++y) {
            for (int x = begin[2]; x <= end[2]; ++x) {
                const auto loaded = chunks_.anchorsInChunk({0, z, y, x}, true);
                if (loaded.status != FiberletGraphQueryStatus::Ready) {
                    throw std::runtime_error(
                        "stored Fiberlet anchor chunk failed: " + loaded.error);
                }
                if(prefetchWindow && ownerIndex+prefetchWindow<ownerKeys.size())
                    chunks_.prefetchAnchorChunks(std::span(ownerKeys).subspan(ownerIndex+prefetchWindow,1),false);
                ++ownerIndex;
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

} // namespace vc::fiber_tracer
