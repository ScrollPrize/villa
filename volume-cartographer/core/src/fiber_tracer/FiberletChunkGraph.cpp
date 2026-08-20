#include "vc/fiber_tracer/FiberletChunkGraph.hpp"

#include "vc/fiber_tracer/FiberLocalScoring.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

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

FiberletPathCost storedTotalCost(float total)
{
    FiberletPathCost result;
    result.alignment = total;
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

FiberLocalMetricSample bestAlignedSample(const FiberPredictionSample& sampled, const cv::Vec3f& reference)
{
    const cv::Vec3f unitReference = normalized(reference);
    FiberLocalMetricSample best;
    float bestScore = -1.0F;
    for (const auto& option : sampled.options) {
        if (!option.valid)
            continue;
        cv::Vec3f direction = normalized(option.direction);
        if (direction.dot(unitReference) < 0.0F)
            direction = -direction;
        const float score = std::max(0.0F, direction.dot(unitReference)) * std::clamp(option.presence, 0.0F, 1.0F);
        if (score > bestScore) {
            bestScore = score;
            best = {direction, option.presence, true};
        }
    }
    return best;
}

const FiberletStorageKey& directedSource(const DirectedFiberletStorageId& id)
{
    return id.reverse ? id.fiberlet.second : id.fiberlet.first;
}

const FiberletStorageKey& directedTarget(const DirectedFiberletStorageId& id)
{
    return id.reverse ? id.fiberlet.first : id.fiberlet.second;
}

}  // namespace

FiberletChunkGraphSource::FiberletChunkGraphSource(
    std::shared_ptr<FiberletChunkDataset> anchorDataset,
    std::shared_ptr<vc::render::ChunkCache> anchorCache,
    std::shared_ptr<FiberletChunkDataset> fiberletDataset,
    std::shared_ptr<vc::render::ChunkCache> fiberletCache,
    FiberletPathConfig pathConfig)
    : anchorDataset_(std::move(anchorDataset))
    , anchorCache_(std::move(anchorCache))
    , fiberletDataset_(std::move(fiberletDataset))
    , fiberletCache_(std::move(fiberletCache))
    , pathConfig_(std::move(pathConfig))
{
    if (!anchorDataset_ || !anchorCache_ || !fiberletDataset_ || !fiberletCache_)
        throw std::invalid_argument("fiberlet chunk graph requires both datasets and caches");
    if (anchorDataset_->metadata().kind != FiberletDatasetKind::Anchors || fiberletDataset_->metadata().kind != FiberletDatasetKind::Fiberlets)
        throw std::invalid_argument("fiberlet chunk graph dataset kinds are invalid");
    if (anchorDataset_->metadata().profile != fiberletDataset_->metadata().profile ||
        anchorDataset_->metadata().chunkGridShapeZYX != fiberletDataset_->metadata().chunkGridShapeZYX ||
        anchorDataset_->metadata().coordinateOriginZYX != fiberletDataset_->metadata().coordinateOriginZYX ||
        anchorDataset_->metadata().coordinateUnitsPerChunkZYX != fiberletDataset_->metadata().coordinateUnitsPerChunkZYX ||
        anchorDataset_->metadata().datasetFingerprint != fiberletDataset_->metadata().datasetFingerprint)
        throw std::invalid_argument("fiberlet chunk graph datasets are incompatible");
    validateFiberletPathConfig(pathConfig_);
}

vc::render::ChunkKey FiberletChunkGraphSource::ownerChunk(const FiberletStorageKey& anchor, int level) const
{
    const auto& metadata = fiberletDataset_->metadata();
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
    std::vector<vc::render::ChunkResult> chunks;
    chunks.reserve(keys.size());
    bool pending = false;
    for (const auto& key : keys) {
        auto chunk = fetchChunk(fiberletCache_, key, blocking);
        if (chunk.status == vc::render::ChunkStatus::MissQueued) {
            pending = true;
        } else if (chunk.status != vc::render::ChunkStatus::Data || !chunk.bytes) {
            return chunkFailure<FiberletIncidentLease>(chunk);
        }
        chunks.push_back(std::move(chunk));
    }
    if (pending) {
        result.status = FiberletGraphQueryStatus::Pending;
        return result;
    }
    for (auto& chunk : chunks) {
        auto decoded = deserializeFiberletPrefixes(*chunk.bytes);
        for (const auto& prefix : decoded.prefixes) {
            if (prefix.id.first == anchorKey)
                result.value.edges.push_back({{prefix.id, false}, prefix});
            else if (prefix.id.second == anchorKey)
                result.value.edges.push_back({{prefix.id, true}, prefix});
        }
        result.value.payloadLeases.push_back(std::move(chunk.bytes));
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
    if (chunk.status != vc::render::ChunkStatus::Data || !chunk.bytes)
        return chunkFailure<FiberletAnchorLease>(chunk);
    auto decoded = deserializeFiberletAnchors(*chunk.bytes);
    const auto found = std::lower_bound(decoded.anchors.begin(), decoded.anchors.end(), anchorKey, [](const auto& candidate, const auto& value) {
        return candidate.key < value;
    });
    if (found == decoded.anchors.end() || found->key != anchorKey) {
        FiberletGraphQuery<FiberletAnchorLease> result;
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet graph endpoint anchor is absent from its owner chunk";
        return result;
    }
    FiberletGraphQuery<FiberletAnchorLease> result;
    result.status = FiberletGraphQueryStatus::Ready;
    result.value.payloadLease = std::move(chunk.bytes);
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
    if (chunk.status != vc::render::ChunkStatus::Data || !chunk.bytes)
        return chunkFailure<FiberletAnchorChunkLease>(chunk);
    auto decoded = deserializeFiberletAnchors(*chunk.bytes);
    FiberletGraphQuery<FiberletAnchorChunkLease> result;
    result.status = FiberletGraphQueryStatus::Ready;
    result.value.payloadLease = std::move(chunk.bytes);
    result.value.anchors = std::move(decoded.anchors);
    return result;
}

FiberletGraphQuery<FiberletRouteLease> FiberletChunkGraphSource::route(const FiberletStorageId& fiberlet, bool blocking) const
{
    const auto prefixKey = ownerChunk(fiberlet.first, 0);
    auto routeKey = prefixKey;
    routeKey.level = 1;
    auto prefixChunk = fetchChunk(fiberletCache_, prefixKey, blocking);
    if (prefixChunk.status != vc::render::ChunkStatus::Data || !prefixChunk.bytes)
        return chunkFailure<FiberletRouteLease>(prefixChunk);
    auto routeChunk = fetchChunk(fiberletCache_, routeKey, blocking);
    if (routeChunk.status != vc::render::ChunkStatus::Data || !routeChunk.bytes)
        return chunkFailure<FiberletRouteLease>(routeChunk);
    auto prefixes = deserializeFiberletPrefixes(*prefixChunk.bytes);
    auto routes = deserializeFiberletRoutes(*routeChunk.bytes);
    if (prefixes.prefixes.size() != routes.routes.size()) {
        FiberletGraphQuery<FiberletRouteLease> result;
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet prefix and route chunk record counts differ";
        return result;
    }
    const auto found = std::lower_bound(prefixes.prefixes.begin(), prefixes.prefixes.end(), fiberlet, [](const auto& candidate, const auto& value) {
        return candidate.id < value;
    });
    if (found == prefixes.prefixes.end() || found->id != fiberlet) {
        FiberletGraphQuery<FiberletRouteLease> result;
        result.status = FiberletGraphQueryStatus::Error;
        result.error = "fiberlet route is absent from its owner chunk";
        return result;
    }
    const auto index = static_cast<std::size_t>(found - prefixes.prefixes.begin());
    FiberletGraphQuery<FiberletRouteLease> result;
    result.status = FiberletGraphQueryStatus::Ready;
    result.value.prefixPayloadLease = std::move(prefixChunk.bytes);
    result.value.routePayloadLease = std::move(routeChunk.bytes);
    result.value.prefix = *found;
    result.value.route = std::move(routes.routes[index]);
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

FiberletCachedReplayGraphSource::FiberletCachedReplayGraphSource(
    std::shared_ptr<FiberletOnDemandPreprocessor> preprocessor,
    const FiberPredictionSource& predictionSource,
    std::shared_ptr<const vc::lasagna::NormalSampler> normalSampler,
    FiberletPathConfig pathConfig,
    std::vector<cv::Vec3d> corridorReferenceBaseXYZ,
    double corridorRadiusBaseVoxels,
    float maximumJoinAngleDegrees)
    : preprocessor_(std::move(preprocessor))
    , chunks_(
          preprocessor_ ? preprocessor_->anchorDataset() : nullptr,
          preprocessor_ ? preprocessor_->anchorCache() : nullptr,
          preprocessor_ ? preprocessor_->fiberletDataset() : nullptr,
          preprocessor_ ? preprocessor_->fiberletCache() : nullptr,
          pathConfig)
    , predictionSource_(&predictionSource)
    , normalSampler_(std::move(normalSampler))
    , pathConfig_(std::move(pathConfig))
    , corridorReferenceBaseXYZ_(std::move(corridorReferenceBaseXYZ))
    , corridorRadiusBaseVoxels_(corridorRadiusBaseVoxels)
    , maximumJoinAngleDegrees_(maximumJoinAngleDegrees)
{
    if (!preprocessor_ || !normalSampler_ || corridorReferenceBaseXYZ_.size() < 2 || !(corridorRadiusBaseVoxels_ > 0.0) ||
        !std::isfinite(corridorRadiusBaseVoxels_) || !(maximumJoinAngleDegrees_ >= 0.0F) || !(maximumJoinAngleDegrees_ <= 180.0F) ||
        !std::isfinite(maximumJoinAngleDegrees_)) {
        throw std::invalid_argument("cached fiberlet replay graph configuration is invalid");
    }
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

bool FiberletCachedReplayGraphSource::anchorCellInCorridor(const FiberletStorageKey& anchorKey) const
{
    std::array<std::size_t, 3> cell{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (anchorKey.coordinateZYX[axis] < 0 || static_cast<std::uint64_t>(anchorKey.coordinateZYX[axis]) > std::numeric_limits<std::size_t>::max()) {
            return false;
        }
        cell[axis] = static_cast<std::size_t>(anchorKey.coordinateZYX[axis]);
    }
    return fiberAnchorCellIntersectsPolylineTube(
        cell, corridorReferenceBaseXYZ_, corridorRadiusBaseVoxels_, preprocessor_->grid(), preprocessor_->anchorConfig().cellSizePredictionVoxels);
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
        for (const auto& anchor : loaded.value.anchors) {
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
    return result;
}

FiberletReplaySourceArc FiberletCachedReplayGraphSource::arc(const DirectedFiberletStorageId& id) const
{
    const auto loaded = chunks_.route(id.fiberlet, true);
    if (loaded.status != FiberletGraphQueryStatus::Ready)
        throw std::runtime_error("cached fiberlet route failed: " + loaded.error);
    FiberletReplaySourceArc result;
    result.id = id;
    result.source = directedSource(id);
    result.target = directedTarget(id);
    result.pathLengthPredictionVoxels = loaded.value.prefix.pathLengthPredictionVoxels;
    result.cost = storedTotalCost(loaded.value.prefix.totalCost);
    result.pointsBaseXYZ.reserve(loaded.value.pointsPredictionXYZ.size());
    for (const auto& point : loaded.value.pointsPredictionXYZ)
        result.pointsBaseXYZ.emplace_back(point * predictionToBaseScale());
    if (id.reverse)
        std::reverse(result.pointsBaseXYZ.begin(), result.pointsBaseXYZ.end());
    return result;
}

std::optional<FiberletReplaySourceTransition> FiberletCachedReplayGraphSource::transition(
    const FiberletReplaySourceArc& incomingArc, const FiberletReplaySourceArc& outgoingArc) const
{
    const auto& incoming = incomingArc.id;
    const auto& outgoingId = outgoingArc.id;
    if (incoming.fiberlet == outgoingId.fiberlet || directedTarget(incoming) != directedSource(outgoingId))
        return std::nullopt;
    if (incomingArc.pointsBaseXYZ.size() < 2 || outgoingArc.pointsBaseXYZ.size() < 2)
        throw std::runtime_error("cached fiberlet transition has a short route");
    const cv::Vec3f incomingStepBase(incomingArc.pointsBaseXYZ.back() - incomingArc.pointsBaseXYZ[incomingArc.pointsBaseXYZ.size() - 2]);
    const cv::Vec3f outgoingStepBase(outgoingArc.pointsBaseXYZ[1] - outgoingArc.pointsBaseXYZ[0]);
    const cv::Vec3f incomingDirection = normalized(incomingStepBase);
    const cv::Vec3f outgoingDirection = normalized(outgoingStepBase);
    const float joinDot = incomingDirection.dot(outgoingDirection);
    const float minimumJoinDot = std::cos(maximumJoinAngleDegrees_ * static_cast<float>(3.14159265358979323846 / 180.0));
    if (!(joinDot > minimumJoinDot))
        return std::nullopt;

    const cv::Vec3d sharedPrediction = incomingArc.pointsBaseXYZ.back() / static_cast<double>(predictionToBaseScale());
    const auto sampled = predictionSource_->sample(sharedPrediction, cv::Vec3d(outgoingDirection));
    const auto prediction = bestAlignedSample(sampled, outgoingDirection);
    if (!prediction.valid)
        return std::nullopt;
    const auto normal = normalSampler_->sampleNormal(sharedPrediction);
    const auto local = fiberLocalMetricCost(
        &prediction,
        prediction,
        incomingDirection,
        vectorLength(incomingStepBase) / predictionToBaseScale(),
        outgoingDirection,
        vectorLength(outgoingStepBase) / predictionToBaseScale(),
        cv::Vec3f(normal.normal),
        normal.valid,
        FiberLocalMetricConfig{
            pathConfig_.invalidPredictionCostPerVoxel,
            FiberLocalSmoothnessConfig{
                pathConfig_.smoothnessWeight,
                pathConfig_.smoothnessNormalWeight,
                pathConfig_.smoothnessTangentWeight,
                pathConfig_.smoothnessFreeAngleDegrees * static_cast<float>(3.14159265358979323846 / 180.0)}});
    return FiberletReplaySourceTransition{incoming, outgoingId, pathCost(local)};
}

}  // namespace vc::fiber_tracer
