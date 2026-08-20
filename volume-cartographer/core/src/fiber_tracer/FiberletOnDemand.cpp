#include "vc/fiber_tracer/FiberletOnDemand.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <tuple>

namespace vc::fiber_tracer
{
namespace
{

std::int64_t floorDiv(std::int64_t numerator, std::int64_t denominator)
{
    if (denominator <= 0)
        throw std::invalid_argument("fiberlet chunk divisor must be positive");
    std::int64_t quotient = numerator / denominator;
    const std::int64_t remainder = numerator % denominator;
    if (remainder < 0)
        --quotient;
    return quotient;
}

std::vector<std::array<std::size_t, 3>> ownedCells(
    const FiberletStorageCodecConfig& codec, const FiberletDatasetMetadata& metadata, const FiberPredictionGridInfo& grid, int cellSide)
{
    if (cellSide <= 0)
        throw std::invalid_argument("fiberlet anchor cell side must be positive");
    const std::array<std::size_t, 3>
        cellShape{(grid.shapeZYX[0] + static_cast<std::size_t>(cellSide) - 1) / static_cast<std::size_t>(cellSide), (grid.shapeZYX[1] + static_cast<std::size_t>(cellSide) - 1) / static_cast<std::size_t>(cellSide), (grid.shapeZYX[2] + static_cast<std::size_t>(cellSide) - 1) / static_cast<std::size_t>(cellSide)};
    std::vector<std::array<std::size_t, 3>> result;
    std::array<std::int64_t, 3> end{};
    for (std::size_t axis = 0; axis < 3; ++axis)
        end[axis] = codec.coordinateOriginZYX[axis] + metadata.coordinateUnitsPerChunkZYX[axis];
    for (std::int64_t z = codec.coordinateOriginZYX[0]; z < end[0]; ++z) {
        for (std::int64_t y = codec.coordinateOriginZYX[1]; y < end[1]; ++y) {
            for (std::int64_t x = codec.coordinateOriginZYX[2]; x < end[2]; ++x) {
                if (z >= 0 && y >= 0 && x >= 0 && static_cast<std::uint64_t>(z) < cellShape[0] &&
                    static_cast<std::uint64_t>(y) < cellShape[1] && static_cast<std::uint64_t>(x) < cellShape[2])
                    result.push_back({static_cast<std::size_t>(z), static_cast<std::size_t>(y), static_cast<std::size_t>(x)});
            }
        }
    }
    return result;
}

FiberletStorageKey storageKey(const FiberletAnchorId& id)
{
    if (id.componentIndex > 1)
        throw std::invalid_argument("fiberlet component index must be zero or one");
    return {{static_cast<std::int64_t>(id.cellZYX[0]), static_cast<std::int64_t>(id.cellZYX[1]), static_cast<std::int64_t>(id.cellZYX[2])}, static_cast<std::uint8_t>(id.componentIndex)};
}

float pathLength(std::span<const cv::Vec3f> points)
{
    double length = 0.0;
    for (std::size_t index = 1; index < points.size(); ++index) {
        const auto delta = points[index] - points[index - 1];
        length += std::sqrt(static_cast<double>(delta.dot(delta)));
    }
    const float result = static_cast<float>(length);
    if (!(result > 0.0F) || !std::isfinite(result))
        throw std::invalid_argument("fiberlet path has invalid length");
    return result;
}

const char* chunkStatusName(vc::render::ChunkStatus status) noexcept
{
    switch (status) {
    case vc::render::ChunkStatus::MissQueued:
        return "miss_queued";
    case vc::render::ChunkStatus::Data:
        return "data";
    case vc::render::ChunkStatus::AllFill:
        return "all_fill";
    case vc::render::ChunkStatus::Missing:
        return "missing";
    case vc::render::ChunkStatus::Error:
        return "error";
    }
    return "unknown";
}

LoadedFiberAnchorArtifact loadedAnchors(const FiberPredictionGridInfo& grid, const FiberAnchorConfig& config, std::vector<FiberletStoredAnchor> anchors)
{
    LoadedFiberAnchorArtifact loaded;
    loaded.report.grid = grid;
    loaded.report.config = config;
    std::map<std::array<std::size_t, 3>, FiberCellAnchorResult> cells;
    for (const auto& stored : anchors) {
        std::array<std::size_t, 3> cell{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (stored.key.coordinateZYX[axis] < 0 ||
                static_cast<std::uint64_t>(stored.key.coordinateZYX[axis]) > std::numeric_limits<std::size_t>::max())
                throw std::invalid_argument("cached fiberlet anchor cell cannot be represented by size_t");
            cell[axis] = static_cast<std::size_t>(stored.key.coordinateZYX[axis]);
        }
        auto& result = cells[cell];
        result.cellZYX = cell;
        auto& component = result.components.at(stored.key.variant);
        if (component.retained)
            throw std::invalid_argument("cached fiberlet anchor key is duplicated");
        component.retained = true;
        component.anchor.cellZYX = cell;
        component.anchor.positionPredictionXYZ = stored.positionPredictionXYZ;
        component.anchor.axisXYZ = stored.fittedAxisXYZ;
        ++result.retainedAnchorCount;
    }
    for (auto& [cell, result] : cells) {
        (void)cell;
        loaded.report.nonEmptyCells.push_back(std::move(result));
    }
    loaded.report.diagnostics.totalCells = cells.size();
    loaded.report.diagnostics.oneAnchorCells =
        std::count_if(loaded.report.nonEmptyCells.begin(), loaded.report.nonEmptyCells.end(), [](const auto& cell) {
            return cell.retainedAnchorCount == 1;
        });
    loaded.report.diagnostics.twoAnchorCells = loaded.report.nonEmptyCells.size() - loaded.report.diagnostics.oneAnchorCells;
    return loaded;
}

}  // namespace

struct FiberletOnDemandPreprocessor::State {
    explicit State(FiberletOnDemandConfig input) : config(std::move(input)) {}

    FiberletOnDemandConfig config;
    std::shared_ptr<FiberletChunkDataset> anchorDataset;
    std::shared_ptr<FiberletChunkDataset> fiberletDataset;
    std::shared_ptr<vc::render::ChunkCache> anchorCache;
    std::shared_ptr<vc::render::ChunkCache> fiberletCache;
    std::array<std::mutex, 64> fiberletGenerationStripes;
};

FiberletOnDemandPreprocessor::FiberletOnDemandPreprocessor(FiberletOnDemandConfig config)
    : state_(std::make_shared<State>(std::move(config)))
{
}

std::shared_ptr<FiberletOnDemandPreprocessor> FiberletOnDemandPreprocessor::create(FiberletOnDemandConfig config)
{
    auto result = std::shared_ptr<FiberletOnDemandPreprocessor>(new FiberletOnDemandPreprocessor(std::move(config)));
    result->initialize();
    return result;
}

void FiberletOnDemandPreprocessor::initialize()
{
    auto& config = state_->config;
    if (!config.predictionSampler || !config.normalSampler)
        throw std::invalid_argument("on-demand fiberlet preprocessing requires prediction and normal samplers");
    if (config.anchorMetadata.kind != FiberletDatasetKind::Anchors || config.fiberletMetadata.kind != FiberletDatasetKind::Fiberlets)
        throw std::invalid_argument("on-demand fiberlet preprocessing dataset kinds are invalid");
    if (config.anchorMetadata.profile != FiberletStorageProfile::Float32Cache || config.fiberletMetadata.profile != FiberletStorageProfile::Float32Cache)
        throw std::invalid_argument("on-demand extraction currently requires float32 cache profiles");
    if (config.anchorMetadata.chunkGridShapeZYX != config.fiberletMetadata.chunkGridShapeZYX ||
        config.anchorMetadata.coordinateOriginZYX != config.fiberletMetadata.coordinateOriginZYX ||
        config.anchorMetadata.coordinateUnitsPerChunkZYX != config.fiberletMetadata.coordinateUnitsPerChunkZYX)
        throw std::invalid_argument("anchor and fiberlet cache grids must agree");
    validateFiberAnchorConfig(config.anchorConfig);
    validateFiberletPathConfig(config.pathConfig);
    state_->anchorDataset = FiberletChunkDataset::createOrOpen(config.anchorRoot, config.anchorMetadata);
    state_->fiberletDataset = FiberletChunkDataset::createOrOpen(config.fiberletRoot, config.fiberletMetadata);
    config.anchorCacheOptions.schedulerLane = "fiberlet-anchors";
    config.fiberletCacheOptions.schedulerLane = "fiberlet-paths";
    const std::weak_ptr<FiberletOnDemandPreprocessor> weak = shared_from_this();
    state_->anchorCache = createGeneratedFiberletChunkCache(
        state_->anchorDataset,
        [weak](FiberletStorageChunkKind, const vc::render::ChunkKey& key, const FiberletStorageCodecConfig& codec) {
            const auto self = weak.lock();
            if (!self)
                throw std::runtime_error("on-demand fiberlet preprocessor was destroyed");
            return self->generateAnchorChunk(key, codec);
        },
        config.anchorCacheOptions);
    state_->fiberletCache = createGeneratedFiberletChunkCache(
        state_->fiberletDataset,
        [weak](FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, const FiberletStorageCodecConfig& codec) {
            const auto self = weak.lock();
            if (!self)
                throw std::runtime_error("on-demand fiberlet preprocessor was destroyed");
            return self->generateFiberletChunk(kind, key, codec);
        },
        config.fiberletCacheOptions);
}

const std::shared_ptr<vc::render::ChunkCache>& FiberletOnDemandPreprocessor::anchorCache() const noexcept
{
    return state_->anchorCache;
}
const std::shared_ptr<vc::render::ChunkCache>& FiberletOnDemandPreprocessor::fiberletCache() const noexcept
{
    return state_->fiberletCache;
}
const std::shared_ptr<FiberletChunkDataset>& FiberletOnDemandPreprocessor::anchorDataset() const noexcept
{
    return state_->anchorDataset;
}
const std::shared_ptr<FiberletChunkDataset>& FiberletOnDemandPreprocessor::fiberletDataset() const noexcept
{
    return state_->fiberletDataset;
}
const FiberPredictionGridInfo& FiberletOnDemandPreprocessor::grid() const noexcept
{
    return state_->config.grid;
}
const FiberAnchorConfig& FiberletOnDemandPreprocessor::anchorConfig() const noexcept
{
    return state_->config.anchorConfig;
}

std::vector<std::byte> FiberletOnDemandPreprocessor::generateAnchorChunk(const vc::render::ChunkKey& key, const FiberletStorageCodecConfig& codec)
{
    const auto start = std::chrono::steady_clock::now();
    const auto& config = state_->config;
    const auto cells = ownedCells(codec, config.anchorMetadata, config.grid, config.anchorConfig.cellSizePredictionVoxels);
    if (config.progress)
        config.progress(FiberletOnDemandProgress{
            .stage = "anchors", .status = "started", .key = key,
            .inputCount = cells.size()});
    if (cells.empty()) {
        if (config.progress)
            config.progress(FiberletOnDemandProgress{
                .stage = "anchors", .status = "completed", .key = key});
        return serializeFiberletAnchors(codec, {});
    }
    auto extracted = extractFiberAnchorsForCells(config.grid, config.anchorConfig, config.predictionSampler, cells, {}, {}, false);
    std::vector<FiberletStoredAnchor> stored;
    for (const auto& cell : extracted.nonEmptyCells) {
        for (std::size_t component = 0; component < cell.components.size(); ++component) {
            if (!cell.components[component].retained)
                continue;
            stored.push_back(
                {storageKey({cell.cellZYX, component}),
                 cell.components[component].anchor.positionPredictionXYZ,
                 cell.components[component].anchor.axisXYZ});
        }
    }
    std::sort(stored.begin(), stored.end(), [](const auto& left, const auto& right) { return left.key < right.key; });
    if (config.progress) {
        config.progress(FiberletOnDemandProgress{
            .stage = "anchors",
            .status = "completed",
            .key = key,
            .inputCount = cells.size(),
            .outputCount = stored.size(),
            .elapsedSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count(),
            .cpuSeconds = extracted.profile.elapsedCpuSeconds});
    }
    return serializeFiberletAnchors(codec, stored);
}

std::vector<vc::render::ChunkKey> FiberletOnDemandPreprocessor::anchorDependencies(const vc::render::ChunkKey& fiberletChunk) const
{
    const auto& metadata = state_->config.anchorMetadata;
    const auto codec = state_->fiberletDataset->codecConfig(FiberletStorageChunkKind::FiberletPrefix, fiberletChunk);
    std::array<int, 3> minimumOffset{0, 0, 0};
    std::array<int, 3> maximumOffset{0, 0, 0};
    for (const auto& offset : fiberletCellNeighborhoodOffsets(state_->config.pathConfig.cellRadius, state_->config.pathConfig.neighborhoodMarginCells)) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            minimumOffset[axis] = std::min(minimumOffset[axis], offset[axis]);
            maximumOffset[axis] = std::max(maximumOffset[axis], offset[axis]);
        }
    }
    std::array<std::int64_t, 3> minimumCell{};
    std::array<std::int64_t, 3> maximumCell{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        minimumCell[axis] = codec.coordinateOriginZYX[axis] + minimumOffset[axis];
        maximumCell[axis] = codec.coordinateOriginZYX[axis] + metadata.coordinateUnitsPerChunkZYX[axis] - 1 + maximumOffset[axis];
    }
    std::array<std::int64_t, 3> beginChunk{};
    std::array<std::int64_t, 3> endChunk{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        beginChunk[axis] = floorDiv(minimumCell[axis] - metadata.coordinateOriginZYX[axis], metadata.coordinateUnitsPerChunkZYX[axis]);
        endChunk[axis] = floorDiv(maximumCell[axis] - metadata.coordinateOriginZYX[axis], metadata.coordinateUnitsPerChunkZYX[axis]);
    }
    std::vector<vc::render::ChunkKey> result;
    for (std::int64_t z = beginChunk[0]; z <= endChunk[0]; ++z) {
        for (std::int64_t y = beginChunk[1]; y <= endChunk[1]; ++y) {
            for (std::int64_t x = beginChunk[2]; x <= endChunk[2]; ++x) {
                if (z >= 0 && y >= 0 && x >= 0 && z < metadata.chunkGridShapeZYX[0] && y < metadata.chunkGridShapeZYX[1] &&
                    x < metadata.chunkGridShapeZYX[2])
                    result.push_back({0, static_cast<int>(z), static_cast<int>(y), static_cast<int>(x)});
            }
        }
    }
    return result;
}

std::vector<FiberletScheduledChunk> FiberletOnDemandPreprocessor::referenceChunkSchedule(
    const PolylineArcGeometry& reference, double beginArcBase, double endArcBase, double radiusBaseVoxels) const
{
    if (!(beginArcBase >= 0.0) || !(endArcBase > beginArcBase) || endArcBase > reference.length() || !(radiusBaseVoxels > 0.0) ||
        !std::isfinite(radiusBaseVoxels)) {
        throw std::invalid_argument("fiberlet reference chunk schedule is invalid");
    }
    const auto& config = state_->config;
    const auto line = slicePolylineArc(reference, beginArcBase, endArcBase);
    const auto cells = fiberAnchorCellsNearPolyline(line, radiusBaseVoxels, config.grid, config.anchorConfig.cellSizePredictionVoxels);
    std::map<std::array<int, 4>, FiberletScheduledChunk> unique;
    for (const auto& cell : cells) {
        vc::render::ChunkKey key{0, 0, 0, 0};
        std::array<int, 3> chunkZYX{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto coordinate = static_cast<std::int64_t>(cell[axis]);
            const auto chunk =
                floorDiv(coordinate - config.fiberletMetadata.coordinateOriginZYX[axis], config.fiberletMetadata.coordinateUnitsPerChunkZYX[axis]);
            if (chunk < 0 || chunk >= config.fiberletMetadata.chunkGridShapeZYX[axis]) {
                key.level = -1;
                break;
            }
            chunkZYX[axis] = static_cast<int>(chunk);
        }
        if (key.level < 0)
            continue;
        key = {0, chunkZYX[0], chunkZYX[1], chunkZYX[2]};
        cv::Vec3d centerBaseXYZ;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto gridAxis = 2 - axis;
            const auto cellBegin = cell[gridAxis] * static_cast<std::size_t>(config.anchorConfig.cellSizePredictionVoxels);
            const auto cellEnd =
                std::min(cellBegin + static_cast<std::size_t>(config.anchorConfig.cellSizePredictionVoxels), config.grid.shapeZYX[gridAxis]);
            centerBaseXYZ[static_cast<int>(axis)] = 0.5 * static_cast<double>(cellBegin + cellEnd - 1) * config.grid.predictionToBaseScale;
        }
        const auto projection = projectPointToPolylineArc(reference, centerBaseXYZ, beginArcBase, endArcBase);
        FiberletScheduledChunk candidate{key, projection.arc, projection.distance};
        auto [found, inserted] = unique.emplace(std::array<int, 4>{key.level, key.iz, key.iy, key.ix}, candidate);
        if (!inserted && std::tie(candidate.nearestReferenceArcBase, candidate.nearestReferenceDistanceBase) <
                             std::tie(found->second.nearestReferenceArcBase, found->second.nearestReferenceDistanceBase)) {
            found->second = candidate;
        }
    }
    std::vector<FiberletScheduledChunk> result;
    result.reserve(unique.size());
    for (auto& [key, chunk] : unique) {
        (void)key;
        result.push_back(std::move(chunk));
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
        return std::
                   tuple{left.nearestReferenceArcBase, left.nearestReferenceDistanceBase, left.key.level, left.key.iz, left.key.iy, left.key.ix} <
               std::tuple{
                   right.nearestReferenceArcBase, right.nearestReferenceDistanceBase, right.key.level, right.key.iz, right.key.iy, right.key.ix};
    });
    return result;
}

void FiberletOnDemandPreprocessor::prefetchScheduled(std::span<const FiberletScheduledChunk> schedule, std::size_t begin, std::size_t count, bool wait) const
{
    if (begin > schedule.size())
        throw std::out_of_range("fiberlet prefetch schedule offset is invalid");
    const auto end = std::min(schedule.size(), begin + std::min(count, schedule.size() - begin));
    std::map<std::array<int, 4>, std::pair<vc::render::ChunkKey, int>> anchorPriorities;
    for (std::size_t index = begin; index < end; ++index) {
        const int priority = static_cast<int>(std::min<std::size_t>(index - begin, static_cast<std::size_t>(std::numeric_limits<int>::max())));
        for (const auto& dependency : anchorDependencies(schedule[index].key)) {
            const std::array<int, 4> dependencyId{dependency.level, dependency.iz, dependency.iy, dependency.ix};
            auto [found, inserted] = anchorPriorities.emplace(dependencyId, std::pair{dependency, priority});
            if (!inserted)
                found->second.second = std::min(found->second.second, priority);
        }
    }
    for (const auto& [id, request] : anchorPriorities) {
        (void)id;
        const auto& [key, priority] = request;
        state_->anchorCache->prefetchChunks({key}, false, priority);
    }
    for (std::size_t index = begin; index < end; ++index) {
        const int priority = static_cast<int>(std::min<std::size_t>(index - begin, static_cast<std::size_t>(std::numeric_limits<int>::max())));
        state_->fiberletCache->prefetchChunks({schedule[index].key}, false, priority);
    }
    if (wait) {
        for (std::size_t index = begin; index < end; ++index) {
            const auto& key = schedule[index].key;
            const auto resolved = state_->fiberletCache->getChunkBlocking(key.level, key.iz, key.iy, key.ix);
            if (resolved.status != vc::render::ChunkStatus::Data)
                throw std::runtime_error("scheduled fiberlet chunk did not resolve to data");
        }
    }
}

std::vector<std::byte> FiberletOnDemandPreprocessor::generateFiberletChunk(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, const FiberletStorageCodecConfig& codec)
{
    const auto start = std::chrono::steady_clock::now();
    auto prefixKey = key;
    prefixKey.level = 0;
    auto routeKey = key;
    routeKey.level = 1;
    const std::array<int, 3> keyZYX{key.iz, key.iy, key.ix};
    std::size_t stripe = 1469598103934665603ULL;
    for (const auto coordinate : keyZYX) {
        stripe ^= static_cast<std::uint32_t>(coordinate);
        stripe *= 1099511628211ULL;
    }
    std::lock_guard keyLock(state_->fiberletGenerationStripes[stripe % state_->fiberletGenerationStripes.size()]);
    const auto& requestedKey = kind == FiberletStorageChunkKind::FiberletPrefix ? prefixKey : routeKey;
    if (auto cached = state_->fiberletDataset->readChunk(kind, requestedKey))
        return *cached;

    const auto& config = state_->config;
    std::vector<FiberletStoredAnchor> storedAnchors;
    for (const auto& dependency : anchorDependencies(prefixKey)) {
        const auto chunk = state_->anchorCache->getChunkBlocking(dependency.level, dependency.iz, dependency.iy, dependency.ix);
        if (chunk.status != vc::render::ChunkStatus::Data || !chunk.bytes) {
            std::string message =
                "required fiberlet anchor chunk " +
                std::to_string(dependency.level) + '/' +
                std::to_string(dependency.iz) + '/' +
                std::to_string(dependency.iy) + '/' +
                std::to_string(dependency.ix) +
                " resolved as " + chunkStatusName(chunk.status);
            if (!chunk.error.empty())
                message += ": " + chunk.error;
            throw std::runtime_error(std::move(message));
        }
        auto decoded = deserializeFiberletAnchors(*chunk.bytes);
        storedAnchors.insert(storedAnchors.end(), std::make_move_iterator(decoded.anchors.begin()), std::make_move_iterator(decoded.anchors.end()));
    }
    std::sort(storedAnchors.begin(), storedAnchors.end(), [](const auto& left, const auto& right) { return left.key < right.key; });
    const auto duplicate = std::adjacent_find(storedAnchors.begin(), storedAnchors.end(), [](const auto& left, const auto& right) {
        return left.key == right.key;
    });
    if (duplicate != storedAnchors.end())
        throw std::invalid_argument("anchor dependency chunks contain duplicate stable keys");
    if (config.progress) {
        config.progress(FiberletOnDemandProgress{
            .stage = "fiberlets", .status = "started", .key = prefixKey,
            .inputCount = storedAnchors.size()});
    }
    const std::size_t inputAnchorCount = storedAnchors.size();
    auto loaded = loadedAnchors(config.grid, config.anchorConfig, std::move(storedAnchors));
    const auto ownerBegin = codec.coordinateOriginZYX;
    std::array<std::int64_t, 3> ownerEnd{};
    for (std::size_t axis = 0; axis < 3; ++axis)
        ownerEnd[axis] = ownerBegin[axis] + config.fiberletMetadata.coordinateUnitsPerChunkZYX[axis];
    auto report =
        traceFiberletPaths(loaded, config.grid, config.pathConfig,
            config.predictionSampler, *config.normalSampler,
            [&](const FiberletPathProgress& progress) {
                if (config.progress) {
                    config.progress(FiberletOnDemandProgress{
                        .stage = "fiberlets",
                        .status = "running",
                        .phase = progress.phase,
                        .key = prefixKey,
                        .inputCount = inputAnchorCount,
                        .phaseCompleted = progress.completed,
                        .phaseTotal = progress.total,
                        .elapsedSeconds = progress.elapsedSeconds});
                }
            }, {}, {},
            [&](const FiberletAnchorId& first) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                const auto coordinate = static_cast<std::int64_t>(first.cellZYX[axis]);
                if (coordinate < ownerBegin[axis] || coordinate >= ownerEnd[axis])
                    return false;
            }
            return true;
        });

    struct StoredPair {
        FiberletStoredPrefix prefix;
        FiberletStoredRoute route;
    };
    std::vector<StoredPair> pairs;
    for (const auto& candidate : report.candidates) {
        if (!candidate.success)
            continue;
        if (candidate.routeLatticeUV.size() > std::numeric_limits<std::uint16_t>::max())
            throw std::overflow_error("fiberlet route has too many interior layers");
        StoredPair stored;
        stored.prefix.id = {storageKey(candidate.start), storageKey(candidate.target)};
        stored.prefix.interiorPointCount = static_cast<std::uint16_t>(candidate.routeLatticeUV.size());
        if (!candidate.routeLatticeUV.empty()) {
            stored.prefix.entryUV = candidate.routeLatticeUV.front();
            stored.prefix.exitUV = candidate.routeLatticeUV.back();
        }
        if (candidate.routeLatticeUV.size() > 2) {
            stored.route.middleUV.assign(candidate.routeLatticeUV.begin() + 1, candidate.routeLatticeUV.end() - 1);
        }
        stored.prefix.pathLengthPredictionVoxels = pathLength(candidate.pointsPredictionXYZ);
        stored.prefix.totalCost = candidate.cost.total();
        pairs.push_back(std::move(stored));
    }
    std::sort(pairs.begin(), pairs.end(), [](const auto& left, const auto& right) { return left.prefix.id < right.prefix.id; });
    std::vector<FiberletStoredPrefix> prefixes;
    std::vector<FiberletStoredRoute> routes;
    prefixes.reserve(pairs.size());
    routes.reserve(pairs.size());
    for (auto& pair : pairs) {
        prefixes.push_back(std::move(pair.prefix));
        routes.push_back(std::move(pair.route));
    }
    const auto prefixCodec = state_->fiberletDataset->codecConfig(FiberletStorageChunkKind::FiberletPrefix, prefixKey);
    const auto routeCodec = state_->fiberletDataset->codecConfig(FiberletStorageChunkKind::FiberletRoutes, routeKey);
    auto prefixBytes = serializeFiberletPrefixes(prefixCodec, prefixes);
    auto routeBytes = serializeFiberletRoutes(routeCodec, routes);
    state_->fiberletDataset->publishChunk(FiberletStorageChunkKind::FiberletPrefix, prefixKey, prefixBytes);
    state_->fiberletDataset->publishChunk(FiberletStorageChunkKind::FiberletRoutes, routeKey, routeBytes);
    if (config.progress) {
        config.progress(FiberletOnDemandProgress{
            .stage = "fiberlets",
            .status = "completed",
            .key = prefixKey,
            .inputCount = inputAnchorCount,
            .outputCount = prefixes.size(),
            .elapsedSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count(),
            .cpuSeconds = report.elapsedCpuSeconds,
            .candidateGenerationWorkers =
                report.candidateGenerationWorkers,
            .candidateGenerationSeconds = report.candidateGenerationSeconds,
            .candidateGenerationCpuSeconds =
                report.candidateGenerationCpuSeconds});
    }
    return kind == FiberletStorageChunkKind::FiberletPrefix ? std::move(prefixBytes) : std::move(routeBytes);
}

}  // namespace vc::fiber_tracer
