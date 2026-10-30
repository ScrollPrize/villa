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

float vectorLength(const cv::Vec3f& value);

FiberletReplaySourceCostProfile storedRouteCostProfile(const FiberletRouteLease& route)
{
    FiberletReplaySourceCostProfile result;
    result.segmentLengthsPredictionVoxels.reserve(
        route.pointsPredictionXYZ.size() - 1);
    for (std::size_t index = 1;
         index < route.pointsPredictionXYZ.size(); ++index) {
        const float length = vectorLength(
            route.pointsPredictionXYZ[index] -
            route.pointsPredictionXYZ[index - 1]);
        if (!(length > 0.0F) || !std::isfinite(length))
            throw std::runtime_error(
                "stored Fiberlet route segment length is invalid");
        result.segmentLengthsPredictionVoxels.push_back(length);
    }
    result.segmentCostDensities = route.route.segmentCostDensities;
    return result;
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

template <typename Payload, typename Query>
std::shared_ptr<const Payload> chunkPayload(const FiberletChunkRead& read, const FiberletChunkKey& key, Query& result)
{
    std::shared_ptr<const FiberletChunkPayload> chunk;
    try {
        chunk = read(key);
    } catch (const std::exception& error) {
        result.status = FiberletGraphQueryStatus::Error;
        result.error = error.what();
        return {};
    }
    auto payload = std::dynamic_pointer_cast<const Payload>(chunk);
    if (!payload) {
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet chunk has the wrong decoded payload type";
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
    FiberletChunkRead anchorRead,
    std::shared_ptr<FiberletChunkDataset> fiberletDataset,
    FiberletChunkRead fiberletRead,
    FiberletPathConfig pathConfig,
    FiberletAnchorView anchorView)
    : anchorDataset_(std::move(anchorDataset))
    , anchorRead_(std::move(anchorRead))
    , fiberletDataset_(std::move(fiberletDataset))
    , fiberletRead_(std::move(fiberletRead))
    , pathConfig_(std::move(pathConfig))
    , anchorView_(std::move(anchorView))
{
    if (!anchorDataset_ || !anchorRead_ || !fiberletDataset_ || !fiberletRead_)
        throw std::invalid_argument("fiberlet chunk graph requires both datasets and reads");
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
        anchorView_ = [](const FiberletChunkKey&, std::shared_ptr<const FiberletAnchorChunkPayload> payload) {
            const auto* anchors = &payload->anchors;
            return std::shared_ptr<const std::vector<FiberletStoredAnchor>>(std::move(payload), anchors);
        };
    }
    validateFiberletPathConfig(pathConfig_);
}

FiberletChunkKey FiberletChunkGraphSource::ownerChunk(const FiberletStorageKey& anchor, int level) const
{
    return fiberletDataset_->chunkOwningAnchor(anchor, level);
}

std::vector<FiberletChunkKey> FiberletChunkGraphSource::incidentOwnerChunks(const FiberletStorageKey& anchor) const
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
    std::vector<FiberletChunkKey> result;
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

FiberletGraphQuery<FiberletIncidentLease> FiberletChunkGraphSource::incidentEdges(const FiberletStorageKey& anchorKey) const
{
    FiberletGraphQuery<FiberletIncidentLease> result;
    const auto keys = incidentOwnerChunks(anchorKey);
    for (const auto& key : keys) {
        auto payload = chunkPayload<FiberletPrefixChunkPayload>(fiberletRead_, key, result);
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

FiberletGraphQuery<FiberletAnchorLease> FiberletChunkGraphSource::anchor(const FiberletStorageKey& anchorKey) const
{
    const auto key = ownerChunk(anchorKey, 0);
    FiberletGraphQuery<FiberletAnchorLease> result;
    auto payload = chunkPayload<FiberletAnchorChunkPayload>(anchorRead_, key, result);
    if (!payload)
        return result;
    std::shared_ptr<const std::vector<FiberletStoredAnchor>> transformed;
    transformed = anchorView_(key, payload);
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

FiberletGraphQuery<FiberletAnchorChunkLease> FiberletChunkGraphSource::anchorsInChunk(const FiberletChunkKey& key) const
{
    if (key.level != 0) {
        FiberletGraphQuery<FiberletAnchorChunkLease> result;
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet anchor chunk query requires level zero";
        return result;
    }
    FiberletGraphQuery<FiberletAnchorChunkLease> result;
    auto payload = chunkPayload<FiberletAnchorChunkPayload>(anchorRead_, key, result);
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

FiberletGraphQuery<FiberletPrefixChunkLease> FiberletChunkGraphSource::prefixesInChunk(const FiberletChunkKey& key) const
{
    if (key.level != 0) {
        FiberletGraphQuery<FiberletPrefixChunkLease> result;
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet prefix chunk query requires level zero";
        return result;
    }
    FiberletGraphQuery<FiberletPrefixChunkLease> result;
    auto payload = chunkPayload<FiberletPrefixChunkPayload>(fiberletRead_, key, result);
    if (!payload)
        return result;
    result.status = FiberletGraphQueryStatus::Ready;
    result.value.payloadLease = std::move(payload);
    return result;
}

FiberletGraphQuery<FiberletRouteChunkLease>
FiberletChunkGraphSource::routesInChunk(
    const FiberletChunkKey& requested) const
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
    FiberletGraphQuery<FiberletRouteChunkLease> result;
    auto payload = chunkPayload<FiberletRouteChunkPayload>(fiberletRead_, key, result);
    if (!payload)
        return result;
    result.status = FiberletGraphQueryStatus::Ready;
    result.value.payloadLease = std::move(payload);
    return result;
}

std::vector<FiberletChunkKey>
FiberletChunkGraphSource::incidentPrefixChunks(
    std::span<const FiberletStorageKey> anchors) const
{
    std::set<std::tuple<int, int, int>> coordinates;
    for (const auto& anchor : anchors) {
        for (const auto& chunk : incidentOwnerChunks(anchor))
            coordinates.emplace(chunk.iz, chunk.iy, chunk.ix);
    }
    std::vector<FiberletChunkKey> result;
    result.reserve(coordinates.size());
    for (const auto& [z, y, x] : coordinates)
        result.push_back({0, z, y, x});
    return result;
}

FiberletGraphQuery<FiberletAnchorLease> FiberletChunkGraphSource::endpoint(
    const FiberletStorageKey& key, const FiberletPrefixChunkPayload& prefix) const
{
    if (!prefix.endpoints)
        return anchor(key);
    const auto* found = prefix.endpoints->find(key);
    if (!found)
        return {FiberletGraphQueryStatus::Error, {}, "published endpoint missing from owner payload"};
    return {FiberletGraphQueryStatus::Ready, {prefix.endpoints, *found}, {}};
}

FiberletGraphQuery<FiberletEdgeLease> FiberletChunkGraphSource::edge(const FiberletStorageId& fiberlet) const
{
    const auto prefixKey = ownerChunk(fiberlet.first, 0);
    FiberletGraphQuery<FiberletEdgeLease> result;
    auto prefixPayload = chunkPayload<FiberletPrefixChunkPayload>(fiberletRead_, prefixKey, result);
    if (!prefixPayload)
        return result;
    const auto* prefix = prefixPayload->find(fiberlet);
    if (!prefix) {
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet edge is absent from its owner chunk";
        return result;
    }
    const auto firstAnchor = endpoint(fiberlet.first, *prefixPayload);
    if (firstAnchor.status != FiberletGraphQueryStatus::Ready) {
        result.status = firstAnchor.status;
        result.error = firstAnchor.error;
        return result;
    }
    const auto secondAnchor = endpoint(fiberlet.second, *prefixPayload);
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
    const FiberletStorageId& fiberlet) const
{
    const auto prefixKey = ownerChunk(fiberlet.first, 0);
    auto routeKey = prefixKey;
    routeKey.level = 1;
    FiberletGraphQuery<FiberletStoredRouteLease> result;
    auto prefixPayload = chunkPayload<FiberletPrefixChunkPayload>(fiberletRead_, prefixKey, result);
    if (!prefixPayload)
        return result;
    auto routePayload = chunkPayload<FiberletRouteChunkPayload>(fiberletRead_, routeKey, result);
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
    const FiberletStorageId& fiberlet) const
{
    FiberletGraphQuery<FiberletRouteLease> result;
    auto stored = storedRoute(fiberlet);
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
    const auto firstAnchor = endpoint(fiberlet.first, *result.value.prefixPayloadLease);
    if (firstAnchor.status != FiberletGraphQueryStatus::Ready) {
        result.status = firstAnchor.status;
        result.error = firstAnchor.error;
        result.value = {};
        return result;
    }
    const auto secondAnchor = endpoint(fiberlet.second, *result.value.prefixPayloadLease);
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
    const DirectedFiberletStorageId& fiberlet) const
{
    FiberletGraphQuery<FiberletReplaySourceArc> result;
    const auto loaded = edge(fiberlet.fiberlet);
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
    if (incoming.id.fiberlet == outgoing.id.fiberlet ||
        incoming.target != outgoing.source) {
        result.status = FiberletGraphQueryStatus::Ready;
        return result;
    }
    const auto shared = [&]() -> FiberletGraphQuery<FiberletAnchorLease> {
        if (fiberletDataset_->metadata().kind != FiberletDatasetKind::Published)
            return anchor(incoming.target);
        const auto prefix = prefixesInChunk(ownerChunk(outgoing.id.fiberlet.first, 0));
        if (prefix.status != FiberletGraphQueryStatus::Ready)
            return {prefix.status, {}, prefix.error};
        return endpoint(incoming.target, *prefix.value.payloadLease);
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
    std::shared_ptr<vcr::MemoryCache> memoryCache,
    float maximumJoinAngleDegrees)
    : anchorDataset_(dataset)
    , dataset_(std::move(dataset))
    , pathConfig_(fiberletPathConfigFromDatasetMetadata(dataset_->metadata()))
    , anchorRead_(fiberletStoredAnchorRead(dataset_, memoryCache))
    , pathRead_(fiberletStoredPathRead(dataset_, memoryCache))
    , chunks_(dataset_, anchorRead_, dataset_, pathRead_, pathConfig_)
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
    FiberletChunkRead anchorRead,
    std::shared_ptr<FiberletChunkDataset> fiberletDataset,
    FiberletChunkRead fiberletRead,
    FiberletPathConfig pathConfig,
    float maximumJoinAngleDegrees,
    FiberletAnchorView anchorView)
    : anchorDataset_(anchorDataset)
    , dataset_(std::move(fiberletDataset))
    , pathConfig_(std::move(pathConfig))
    , anchorRead_(std::move(anchorRead))
    , pathRead_(std::move(fiberletRead))
    , anchorView_(anchorView)
    , chunks_(
          std::move(anchorDataset), anchorRead_, dataset_, pathRead_,
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
    for (int z = begin[0]; z <= end[0]; ++z) {
        for (int y = begin[1]; y <= end[1]; ++y) {
            for (int x = begin[2]; x <= end[2]; ++x) {
                const auto loaded = chunks_.anchorsInChunk({0, z, y, x});
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
    const auto loaded = chunks_.incidentEdges(anchor);
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
    const auto loaded = chunks_.directedEdge(id);
    if (loaded.status != FiberletGraphQueryStatus::Ready)
        throw std::runtime_error(
            "stored Fiberlet edge failed: " + loaded.error);
    return loaded.value;
}

FiberletReplaySourceCostProfile
FiberletStoredReplayGraphSource::costProfile(
    const DirectedFiberletStorageId& id) const
{
    const auto loaded = chunks_.route(id.fiberlet);
    if (loaded.status != FiberletGraphQueryStatus::Ready)
        throw std::runtime_error(
            "stored Fiberlet cost profile failed: " + loaded.error);
    auto result = storedRouteCostProfile(loaded.value);
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
    const auto loaded = chunks_.route(id.fiberlet);
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
        incoming, outgoing, maximumJoinAngleDegrees_);
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


FiberletMaterializedReplayGraph
FiberletStoredReplayGraphSource::materializeBaseBox(
    const cv::Vec3d& minimum, const cv::Vec3d& maximum,
    std::size_t parallelThreads) const
{
    return materializeBaseBoxForSeeds(
        minimum, maximum, minimum, maximum, parallelThreads);
}

FiberletMaterializedReplayGraph
FiberletStoredReplayGraphSource::materializeBaseBoxForSeeds(
    const cv::Vec3d& minimum, const cv::Vec3d& maximum,
    const cv::Vec3d& seedMinimum, const cv::Vec3d& seedMaximum,
    std::size_t parallelThreads) const
{
    if (parallelThreads == 0)
        throw std::invalid_argument("fiberlet materialization needs positive worker count");
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(seedMinimum[axis]) ||
            !std::isfinite(seedMaximum[axis]) ||
            !(seedMaximum[axis] > seedMinimum[axis]) ||
            !(seedMinimum[axis] >= minimum[axis]) ||
            !(seedMaximum[axis] <= maximum[axis]))
            throw std::invalid_argument("fiberlet seed box must be inside the graph box");
    }
    const auto inside = [](const cv::Vec3d& p, const cv::Vec3d& lo, const cv::Vec3d& hi) {
        return p[0] >= lo[0] && p[0] < hi[0] &&
               p[1] >= lo[1] && p[1] < hi[1] &&
               p[2] >= lo[2] && p[2] < hi[2];
    };
    const float scale = predictionToBaseScale();
    const auto seeds = anchorsInBaseBox(minimum, maximum);
    // One pass reads a chunk for every edge it touches; decode each one once.
    const auto memoized = [](const FiberletChunkRead& read) {
        using Decoded = std::shared_ptr<const FiberletChunkPayload>;
        return [read, decoded = std::make_shared<std::map<std::array<int, 4>, Decoded>>()](
                   const FiberletChunkKey& key) {
            auto& slot = (*decoded)[{key.level, key.iz, key.iy, key.ix}];
            if (!slot)
                slot = read(key);
            return slot;
        };
    };
    const FiberletChunkGraphSource chunks(
        anchorDataset_, memoized(anchorRead_), dataset_, memoized(pathRead_),
        pathConfig_, anchorView_);
    FiberletMaterializedReplayGraph result;
    std::map<FiberletStorageKey, FiberletStoredAnchor> anchors;
    std::set<FiberletStorageKey> insideKeys;
    std::vector<FiberletStorageKey> keys;
    for (const auto& seed : seeds) {
        anchors.emplace(seed.key, seed);
        insideKeys.insert(seed.key);
        keys.push_back(seed.key);
        if (inside(cv::Vec3d(seed.positionPredictionXYZ * scale), seedMinimum, seedMaximum))
            result.insideAnchors.push_back(seed);
    }

    // Read each prefix owner once. Filter before resolving geometry so unrelated
    // sparse halo tuples cannot force unavailable endpoint/route dependencies.
    std::set<FiberletStorageId> ids;
    for (const auto& owner : chunks.incidentPrefixChunks(keys)) {
        const auto prefixes = chunks.prefixesInChunk(owner);
        if (prefixes.status != FiberletGraphQueryStatus::Ready)
            throw std::runtime_error("fiberlet materialization prefixes: " + prefixes.error);
        for (const auto& prefix : prefixes.value.payloadLease->prefixes) {
            if (insideKeys.contains(prefix.id.first) || insideKeys.contains(prefix.id.second)) {
                if (!ids.insert(prefix.id).second)
                    throw std::runtime_error("fiberlet materialization duplicate edge");
            }
        }
    }
    if (ids.size() > std::numeric_limits<std::uint32_t>::max() / 2)
        throw std::overflow_error("too many directed fiberlet arcs to materialize");

    std::vector<FiberletImmutableReplayEdge> edges;
    std::vector<FiberletReplaySourceArc> arcs;
    std::map<FiberletStorageKey, std::vector<std::size_t>> outgoing;
    edges.reserve(ids.size());
    arcs.reserve(ids.size() * 2);
    for (const auto& id : ids) {
        // These shared accessors handle both generated anchor views and format 3
        // embedded endpoints. Do not bypass them with direct owner-chunk loads.
        const auto edge = chunks.edge(id);
        if (edge.status != FiberletGraphQueryStatus::Ready)
            throw std::runtime_error("fiberlet materialization edge: " + edge.error);
        const auto route = chunks.route(id);
        if (route.status != FiberletGraphQueryStatus::Ready)
            throw std::runtime_error("fiberlet materialization route: " + route.error);
        for (const auto& anchor : {edge.value.firstAnchor, edge.value.secondAnchor}) {
            const auto [found, inserted] = anchors.emplace(anchor.key, anchor);
            if (!inserted && cv::norm(found->second.positionPredictionXYZ -
                                     anchor.positionPredictionXYZ) * scale > 1.e-5)
                throw std::runtime_error("fiberlet materialization inconsistent endpoint");
        }
        auto forward = directedStoredArc(edge.value, {id, false}, scale);
        auto reverse = directedStoredArc(edge.value, {id, true}, scale);
        outgoing[forward.source].push_back(arcs.size());
        arcs.push_back(forward);
        outgoing[reverse.source].push_back(arcs.size());
        arcs.push_back(reverse);
        std::vector<cv::Vec3d> points;
        points.reserve(route.value.pointsPredictionXYZ.size());
        for (const auto& point : route.value.pointsPredictionXYZ)
            points.emplace_back(point * scale);
        edges.push_back({forward, storedRouteCostProfile(route.value), std::move(points)});
    }

    // Once materialized, joins are CPU-only calculations over fixed anchors.
    // Per-input slots preserve canonical order regardless of worker completion.
    std::vector<std::vector<FiberletImmutableReplayTransition>> transitions(arcs.size());
    std::vector<std::exception_ptr> failures(arcs.size());
    const auto build = [&](std::size_t incoming) {
        try {
            const auto& arc = arcs[incoming];
            if (!insideKeys.contains(arc.target))
                return;
            const auto found = outgoing.find(arc.target);
            if (found == outgoing.end())
                return;
            for (const auto next : found->second) {
                const auto transition = chunks.transitionCostAtAnchor(
                    arc, arcs[next], anchors.at(arc.target), maximumJoinAngleDegrees_);
                if (transition.status != FiberletGraphQueryStatus::Ready)
                    throw std::runtime_error("fiberlet materialization join: " + transition.error);
                if (transition.value)
                    transitions[incoming].push_back({
                        static_cast<std::uint32_t>(next), *transition.value});
            }
        } catch (...) {
            failures[incoming] = std::current_exception();
        }
    };
    if (parallelThreads > 1 && !arcs.empty()) {
        utils::ThreadPool pool(std::min(parallelThreads, arcs.size()));
        pool.run_indexed_batch(arcs.size(), build);
    } else {
        for (std::size_t i = 0; i < arcs.size(); ++i)
            build(i);
    }
    FiberletImmutableReplayTransitionCsr csr;
    csr.offsets.push_back(0);
    for (std::size_t i = 0; i < arcs.size(); ++i) {
        if (failures[i])
            std::rethrow_exception(failures[i]);
        csr.entries.insert(csr.entries.end(), transitions[i].begin(), transitions[i].end());
        csr.offsets.push_back(csr.entries.size());
    }
    std::vector<FiberletReplaySourceAnchor> positions;
    positions.reserve(anchors.size());
    for (const auto& [key, anchor] : anchors)
        positions.push_back({key, cv::Vec3d(anchor.positionPredictionXYZ * scale)});
    result.graph = std::make_shared<FiberletImmutableReplayGraphSource>(
        scale, anchorCellSizePredictionVoxels(), maximumJoinAngleDegrees_,
        std::move(positions), std::move(edges), std::move(csr));
    return result;
}
} // namespace vc::fiber_tracer
