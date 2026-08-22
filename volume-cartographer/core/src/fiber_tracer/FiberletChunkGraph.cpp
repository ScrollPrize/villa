#include "vc/fiber_tracer/FiberletChunkGraph.hpp"

#include "vc/fiber_tracer/FiberLocalScoring.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
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
    if (anchorDataset_->metadata().kind != FiberletDatasetKind::Anchors || fiberletDataset_->metadata().kind != FiberletDatasetKind::Fiberlets)
        throw std::invalid_argument("fiberlet chunk graph dataset kinds are invalid");
    if (anchorDataset_->metadata().profile != fiberletDataset_->metadata().profile ||
        anchorDataset_->metadata().chunkGridShapeZYX != fiberletDataset_->metadata().chunkGridShapeZYX ||
        anchorDataset_->metadata().coordinateOriginZYX != fiberletDataset_->metadata().coordinateOriginZYX ||
        anchorDataset_->metadata().coordinateUnitsPerChunkZYX != fiberletDataset_->metadata().coordinateUnitsPerChunkZYX ||
        anchorDataset_->metadata().maximumEndpointReachCoordinateUnitsZYX != fiberletDataset_->metadata().maximumEndpointReachCoordinateUnitsZYX ||
        anchorDataset_->metadata().spatialChunkSideBaseVoxels != fiberletDataset_->metadata().spatialChunkSideBaseVoxels ||
        anchorDataset_->metadata().predictionToBaseScale != fiberletDataset_->metadata().predictionToBaseScale ||
        anchorDataset_->metadata().fiberManifest != fiberletDataset_->metadata().fiberManifest ||
        anchorDataset_->metadata().fiberManifestHash != fiberletDataset_->metadata().fiberManifestHash ||
        anchorDataset_->metadata().normalManifest != fiberletDataset_->metadata().normalManifest ||
        anchorDataset_->metadata().normalManifestHash != fiberletDataset_->metadata().normalManifestHash)
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

FiberletGraphQuery<FiberletRouteLease> FiberletChunkGraphSource::route(const FiberletStorageId& fiberlet, bool blocking) const
{
    const auto prefixKey = ownerChunk(fiberlet.first, 0);
    auto routeKey = prefixKey;
    routeKey.level = 1;
    fiberletCache_->prefetchChunks({prefixKey, routeKey}, blocking);
    const auto firstAnchorKey = ownerChunk(fiberlet.first, 0);
    const auto secondAnchorKey = ownerChunk(fiberlet.second, 0);
    anchorCache_->prefetchChunks({firstAnchorKey, secondAnchorKey}, blocking);
    FiberletGraphQuery<FiberletRouteLease> result;
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
    const auto loaded = chunks_.edge(id.fiberlet, true);
    if (loaded.status != FiberletGraphQueryStatus::Ready)
        throw std::runtime_error("cached fiberlet edge failed: " + loaded.error);
    FiberletReplaySourceArc result;
    result.id = id;
    result.source = directedSource(id);
    result.target = directedTarget(id);
    result.pathLengthPredictionVoxels = loaded.value.prefix.pathLengthPredictionVoxels;
    result.cost = storedPathCost(loaded.value.prefix.cost);
    if (evaluationQuantization_.costBits == 8 || evaluationQuantization_.costBits == 16) {
        result.cost = {
            quantizedCost(
                id.fiberlet,
                result.cost.total(),
                loaded.value.prefix.pathLengthPredictionVoxels),
            0.0F,
            0.0F,
            0.0F,
            0.0F};
    }
    const float scale = predictionToBaseScale();
    const cv::Vec3d firstPosition(loaded.value.firstAnchor.positionPredictionXYZ * scale);
    const cv::Vec3d secondPosition(loaded.value.secondAnchor.positionPredictionXYZ * scale);
    const cv::Vec3f firstStep = loaded.value.prefix.firstStepBaseXYZ;
    const cv::Vec3f lastStep = loaded.value.prefix.lastStepBaseXYZ;
    if (!id.reverse) {
        result.sourcePositionBaseXYZ = firstPosition;
        result.targetPositionBaseXYZ = secondPosition;
        result.startStepBaseXYZ = firstStep;
        result.endStepBaseXYZ = lastStep;
    } else {
        result.sourcePositionBaseXYZ = secondPosition;
        result.targetPositionBaseXYZ = firstPosition;
        result.startStepBaseXYZ = -lastStep;
        result.endStepBaseXYZ = -firstStep;
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
    const auto& incoming = incomingArc.id;
    const auto& outgoingId = outgoingArc.id;
    if (incoming.fiberlet == outgoingId.fiberlet || directedTarget(incoming) != directedSource(outgoingId))
        return std::nullopt;
    const cv::Vec3f incomingStepBase = incomingArc.endStepBaseXYZ;
    const cv::Vec3f outgoingStepBase = outgoingArc.startStepBaseXYZ;
    const cv::Vec3f incomingDirection = normalized(incomingStepBase);
    const cv::Vec3f outgoingDirection = normalized(outgoingStepBase);
    const float joinDot = incomingDirection.dot(outgoingDirection);
    const float minimumJoinDot = std::cos(maximumJoinAngleDegrees_ * static_cast<float>(3.14159265358979323846 / 180.0));
    if (!(joinDot > minimumJoinDot))
        return std::nullopt;

    const auto shared = chunks_.anchor(incomingArc.target, true);
    if (shared.status != FiberletGraphQueryStatus::Ready)
        throw std::runtime_error("cached fiberlet transition anchor failed: " + shared.error);
    if (!shared.value.anchor.predictionValid)
        return std::nullopt;
    cv::Vec3f predictionAxis = shared.value.anchor.predictionAxisXYZ;
    float predictionPresence = shared.value.anchor.predictionPresence;
    cv::Vec3f normal = shared.value.anchor.normalXYZ;
    const FiberLocalMetricSample prediction{predictionAxis, predictionPresence, true};
    const auto local = fiberLocalMetricCost(
        &prediction,
        prediction,
        incomingDirection,
        vectorLength(incomingStepBase) / predictionToBaseScale(),
        outgoingDirection,
        vectorLength(outgoingStepBase) / predictionToBaseScale(),
        normal,
        shared.value.anchor.normalValid,
        FiberLocalMetricConfig{
            pathConfig_.invalidPredictionCostPerVoxel,
            FiberLocalSmoothnessConfig{
                pathConfig_.smoothnessWeight,
                pathConfig_.smoothnessNormalWeight,
                pathConfig_.smoothnessTangentWeight,
                pathConfig_.smoothnessFreeAngleDegrees * static_cast<float>(3.14159265358979323846 / 180.0)}});
    return FiberletReplaySourceTransition{incoming, outgoingId, pathCost(local)};
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
