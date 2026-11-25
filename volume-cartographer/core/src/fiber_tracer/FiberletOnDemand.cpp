#include "vc/fiber_tracer/FiberletOnDemand.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <future>
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

std::size_t ownedCellCount(const FiberletStorageCodecConfig& codec, const FiberletDatasetMetadata& metadata, const FiberPredictionGridInfo& grid, int cellSide)
{
    if (cellSide <= 0)
        throw std::invalid_argument("fiberlet anchor cell side must be positive");
    const std::array<std::size_t, 3>
        cellShape{(grid.shapeZYX[0] + static_cast<std::size_t>(cellSide) - 1) / static_cast<std::size_t>(cellSide), (grid.shapeZYX[1] + static_cast<std::size_t>(cellSide) - 1) / static_cast<std::size_t>(cellSide), (grid.shapeZYX[2] + static_cast<std::size_t>(cellSide) - 1) / static_cast<std::size_t>(cellSide)};
    std::size_t result = 1;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto begin = std::clamp<std::int64_t>(codec.coordinateOriginZYX[axis], 0, static_cast<std::int64_t>(cellShape[axis]));
        const auto end =
            std::clamp<std::int64_t>(codec.coordinateOriginZYX[axis] + metadata.coordinateUnitsPerChunkZYX[axis], 0, static_cast<std::int64_t>(cellShape[axis]));
        result *= static_cast<std::size_t>(std::max<std::int64_t>(0, end - begin));
    }
    return result;
}

std::vector<std::array<std::size_t, 3>> selectedOwnedCells(
    const FiberletStorageCodecConfig& codec, const FiberletDatasetMetadata& metadata, const FiberPredictionGridInfo& grid, int cellSide, const FiberletAnchorCellPredicate& predicate)
{
    const size_t side = static_cast<size_t>(cellSide);
    const std::array<size_t, 3> cellShape{
        (grid.shapeZYX[0] + side - 1) / side,
        (grid.shapeZYX[1] + side - 1) / side,
        (grid.shapeZYX[2] + side - 1) / side,
    };
    std::array<size_t, 3> begin{};
    std::array<size_t, 3> end{};
    for (size_t axis = 0; axis < 3; ++axis) {
        begin[axis] = static_cast<size_t>(std::clamp<int64_t>(codec.coordinateOriginZYX[axis], 0, static_cast<int64_t>(cellShape[axis])));
        end[axis] = static_cast<size_t>(
            std::clamp<int64_t>(codec.coordinateOriginZYX[axis] + metadata.coordinateUnitsPerChunkZYX[axis], 0, static_cast<int64_t>(cellShape[axis])));
    }
    std::vector<std::array<size_t, 3>> result;
    result.reserve((end[0] - begin[0]) * (end[1] - begin[1]) * (end[2] - begin[2]));
    for (size_t z = begin[0]; z < end[0]; ++z) {
        for (size_t y = begin[1]; y < end[1]; ++y) {
            for (size_t x = begin[2]; x < end[2]; ++x) {
                const std::array<size_t, 3> cell{z, y, x};
                if (predicate(cell))
                    result.push_back(cell);
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

std::vector<FiberletChunkKey> fiberletOutputChunksForNonemptyPresence(
    const FiberPresenceChunkScanReport& presence, const FiberletDatasetMetadata& outputMetadata, int anchorCellSizePredictionVoxels)
{
    if (anchorCellSizePredictionVoxels <= 0)
        throw std::invalid_argument("fiberlet sparse selection requires a positive anchor cell size");
    for (size_t axis = 0; axis < 3; ++axis) {
        if (presence.shapeZYX[axis] == 0 || presence.chunksZYX[axis] == 0 || outputMetadata.coordinateUnitsPerChunkZYX[axis] <= 0 ||
            outputMetadata.chunkGridShapeZYX[axis] <= 0) {
            throw std::invalid_argument("fiberlet sparse selection has invalid grid metadata");
        }
    }

    std::set<std::array<int, 3>> owners;
    const auto cellSide = static_cast<size_t>(anchorCellSizePredictionVoxels);
    for (const auto& source : presence.nonemptyChunksZYX) {
        std::array<size_t, 3> firstOwner{};
        std::array<size_t, 3> lastOwner{};
        for (size_t axis = 0; axis < 3; ++axis) {
            if (source[axis] >= presence.chunkGridShapeZYX[axis])
                throw std::invalid_argument("fiber presence scan contains an out-of-grid chunk");
            const size_t begin = source[axis] * presence.chunksZYX[axis];
            const size_t end = std::min(presence.shapeZYX[axis], begin + presence.chunksZYX[axis]);
            if (begin >= end)
                throw std::invalid_argument("fiber presence scan contains an empty chunk extent");
            const size_t firstCell = begin / cellSide;
            const size_t endCell = (end + cellSide - 1) / cellSide;
            const auto units = static_cast<size_t>(outputMetadata.coordinateUnitsPerChunkZYX[axis]);
            firstOwner[axis] = firstCell / units;
            lastOwner[axis] = (endCell - 1) / units;
            const auto grid = static_cast<size_t>(outputMetadata.chunkGridShapeZYX[axis]);
            if (firstOwner[axis] >= grid)
                throw std::invalid_argument("fiber presence chunk lies outside the output grid");
            lastOwner[axis] = std::min(lastOwner[axis], grid - 1);
        }
        for (size_t z = firstOwner[0]; z <= lastOwner[0]; ++z) {
            for (size_t y = firstOwner[1]; y <= lastOwner[1]; ++y) {
                for (size_t x = firstOwner[2]; x <= lastOwner[2]; ++x) {
                    owners.insert({static_cast<int>(z), static_cast<int>(y), static_cast<int>(x)});
                }
            }
        }
    }

    std::vector<FiberletChunkKey> result;
    result.reserve(owners.size());
    for (const auto& owner : owners)
        result.push_back({0, owner[0], owner[1], owner[2]});
    return result;
}

struct FiberletPreprocessSchedule::Impl {
    using Key = std::array<int, 4>;

    enum class State {
        Pending,
        Running,
        Complete,
    };

    struct Output {
        FiberletChunkKey key;
        State state = State::Pending;
        std::size_t missingAnchors = 0;
    };

    struct Anchor {
        FiberletChunkKey key;
        State state = State::Pending;
        std::vector<std::size_t> dependents;
        std::size_t firstDependent = 0;
        std::size_t firstIncompleteDependent = std::numeric_limits<std::size_t>::max();
    };

    static Key id(const FiberletChunkKey& key) { return {key.level, key.iz, key.iy, key.ix}; }

    void advanceOutputZ()
    {
        currentZ.reset();
        for (const auto& output : outputs) {
            if (output.state != State::Complete) {
                currentZ = output.key.iz;
                return;
            }
        }
    }

    std::vector<Output> outputs;
    std::map<Key, std::size_t> outputByKey;
    std::map<Key, Anchor> anchors;
    std::set<std::pair<std::size_t, Key>> pendingAnchors;
    std::set<std::size_t> readyOutputs;
    std::optional<int> currentZ;
    std::size_t completedAnchorCount = 0;
    std::size_t completedOutputCount = 0;
};

FiberletPreprocessSchedule::FiberletPreprocessSchedule(
    std::vector<FiberletChunkKey> outputChunks,
    std::vector<std::vector<FiberletChunkKey>> anchorDependencies,
    std::span<const FiberletChunkKey> completedOutputs,
    std::span<const FiberletChunkKey> availableAnchors)
    : impl_(std::make_unique<Impl>())
{
    if (outputChunks.size() != anchorDependencies.size())
        throw std::invalid_argument("fiberlet preprocess outputs and dependency lists differ in size");
    const auto keyLess = [](const auto& left, const auto& right) { return Impl::id(left) < Impl::id(right); };
    if (!std::is_sorted(outputChunks.begin(), outputChunks.end(), keyLess) ||
        std::adjacent_find(outputChunks.begin(), outputChunks.end()) != outputChunks.end()) {
        throw std::invalid_argument("fiberlet preprocess outputs must be ordered and unique");
    }

    std::set<Impl::Key> completedOutputIds;
    for (const auto& key : completedOutputs)
        completedOutputIds.insert(Impl::id(key));
    std::set<Impl::Key> availableAnchorIds;
    for (const auto& key : availableAnchors)
        availableAnchorIds.insert(Impl::id(key));

    impl_->outputs.reserve(outputChunks.size());
    for (std::size_t index = 0; index < outputChunks.size(); ++index) {
        auto& dependencies = anchorDependencies[index];
        std::sort(dependencies.begin(), dependencies.end(), keyLess);
        dependencies.erase(std::unique(dependencies.begin(), dependencies.end()), dependencies.end());
        const auto outputId = Impl::id(outputChunks[index]);
        const bool complete = completedOutputIds.contains(outputId);
        if (!impl_->outputByKey.emplace(outputId, index).second)
            throw std::invalid_argument("fiberlet preprocess output is duplicated");
        impl_->outputs.push_back({outputChunks[index], complete ? Impl::State::Complete : Impl::State::Pending, 0});
        impl_->completedOutputCount += complete;
        for (const auto& dependency : dependencies) {
            const auto dependencyId = Impl::id(dependency);
            auto [found, inserted] = impl_->anchors.try_emplace(
                dependencyId,
                Impl::Anchor{
                    dependency,
                    Impl::State::Pending,
                    {},
                    index,
                    complete ? std::numeric_limits<std::size_t>::max() : index});
            found->second.firstDependent = std::min(found->second.firstDependent, index);
            if (!complete)
                found->second.firstIncompleteDependent = std::min(found->second.firstIncompleteDependent, index);
            found->second.dependents.push_back(index);
        }
    }
    if (completedOutputIds.size() != impl_->completedOutputCount)
        throw std::invalid_argument("fiberlet preprocess completed output is not in the active output set");

    for (auto& [anchorId, anchor] : impl_->anchors) {
        if (availableAnchorIds.contains(anchorId)) {
            anchor.state = Impl::State::Complete;
            ++impl_->completedAnchorCount;
        } else {
            const auto priority = anchor.firstIncompleteDependent != std::numeric_limits<std::size_t>::max()
                ? anchor.firstIncompleteDependent
                : impl_->outputs.size() + anchor.firstDependent;
            impl_->pendingAnchors.emplace(priority, anchorId);
            for (const auto dependent : anchor.dependents) {
                if (impl_->outputs[dependent].state != Impl::State::Complete)
                    ++impl_->outputs[dependent].missingAnchors;
            }
        }
    }
    for (std::size_t index = 0; index < impl_->outputs.size(); ++index) {
        const auto& output = impl_->outputs[index];
        if (output.state == Impl::State::Pending && output.missingAnchors == 0)
            impl_->readyOutputs.insert(index);
    }
    impl_->advanceOutputZ();
}

FiberletPreprocessSchedule::~FiberletPreprocessSchedule() = default;
FiberletPreprocessSchedule::FiberletPreprocessSchedule(FiberletPreprocessSchedule&&) noexcept = default;
FiberletPreprocessSchedule& FiberletPreprocessSchedule::operator=(FiberletPreprocessSchedule&&) noexcept = default;

std::optional<FiberletPreprocessWork> FiberletPreprocessSchedule::takeNext()
{
    if (impl_->currentZ && !impl_->readyOutputs.empty()) {
        const auto ready = impl_->readyOutputs.begin();
        auto& output = impl_->outputs[*ready];
        if (output.key.iz == *impl_->currentZ) {
            const auto work = FiberletPreprocessWork{FiberletPreprocessWorkKind::Fiberlet, output.key};
            output.state = Impl::State::Running;
            impl_->readyOutputs.erase(ready);
            return work;
        }
    }
    if (!impl_->pendingAnchors.empty()) {
        const auto pending = impl_->pendingAnchors.begin();
        const auto anchorId = pending->second;
        auto& anchor = impl_->anchors.at(anchorId);
        anchor.state = Impl::State::Running;
        impl_->pendingAnchors.erase(pending);
        return FiberletPreprocessWork{FiberletPreprocessWorkKind::Anchor, anchor.key};
    }
    return std::nullopt;
}

void FiberletPreprocessSchedule::complete(const FiberletPreprocessWork& work)
{
    const auto id = Impl::id(work.key);
    if (work.kind == FiberletPreprocessWorkKind::Anchor) {
        auto found = impl_->anchors.find(id);
        if (found == impl_->anchors.end() || found->second.state != Impl::State::Running)
            throw std::logic_error("completed fiberlet preprocess anchor was not running");
        found->second.state = Impl::State::Complete;
        ++impl_->completedAnchorCount;
        for (const auto dependent : found->second.dependents) {
            auto& output = impl_->outputs[dependent];
            if (output.state == Impl::State::Complete)
                continue;
            if (output.missingAnchors == 0)
                throw std::logic_error("fiberlet preprocess dependency count underflow");
            --output.missingAnchors;
            if (output.missingAnchors == 0 && output.state == Impl::State::Pending)
                impl_->readyOutputs.insert(dependent);
        }
        return;
    }

    const auto found = impl_->outputByKey.find(id);
    if (found == impl_->outputByKey.end())
        throw std::logic_error("completed fiberlet preprocess output is unknown");
    auto& output = impl_->outputs[found->second];
    if (output.state != Impl::State::Running)
        throw std::logic_error("completed fiberlet preprocess output was not running");
    output.state = Impl::State::Complete;
    ++impl_->completedOutputCount;
    impl_->advanceOutputZ();
}

bool FiberletPreprocessSchedule::done() const noexcept
{
    return impl_->completedAnchorCount == impl_->anchors.size() && impl_->completedOutputCount == impl_->outputs.size();
}

std::optional<int> FiberletPreprocessSchedule::currentOutputZ() const noexcept { return impl_->currentZ; }
std::size_t FiberletPreprocessSchedule::anchorTotal() const noexcept { return impl_->anchors.size(); }
std::size_t FiberletPreprocessSchedule::anchorsCompleted() const noexcept { return impl_->completedAnchorCount; }
std::size_t FiberletPreprocessSchedule::outputTotal() const noexcept { return impl_->outputs.size(); }
std::size_t FiberletPreprocessSchedule::outputsCompleted() const noexcept { return impl_->completedOutputCount; }

struct FiberletOnDemandPreprocessor::State {
    explicit State(FiberletOnDemandConfig input) : config(std::move(input)) {}

    FiberletOnDemandConfig config;
    std::shared_ptr<FiberletChunkDataset> anchorDataset;
    std::shared_ptr<FiberletChunkDataset> fiberletDataset;
    FiberletChunkRead anchorRead;
    FiberletChunkRead fiberletRead;
    std::map<std::array<int, 3>, std::vector<std::array<std::size_t, 3>>> selectedCellsByChunk;
    struct EvaluationAnchorChunkEntry {
        std::shared_future<std::shared_ptr<const std::vector<FiberletStoredAnchor>>> future;
        std::size_t bytes = 0;
        std::uint64_t touch = 0;
    };
    mutable std::mutex evaluationAnchorMutex;
    mutable std::map<std::array<int, 4>, EvaluationAnchorChunkEntry> evaluationAnchorChunks;
    mutable std::size_t evaluationAnchorBytes = 0;
    mutable std::uint64_t evaluationAnchorTouch = 0;
    using Pair = std::pair<FiberletChunkDataset::MaterializedChunk, FiberletChunkDataset::MaterializedChunk>;
    std::mutex generationMutex;
    std::set<std::array<int, 3>> emptyOwners;
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
    if (!config.predictionSampler || !config.normalSampler || !config.anchorRetainPredicate || !config.pointPredicate) {
        throw std::invalid_argument("on-demand fiberlet preprocessing requires samplers and corridor predicates");
    }
    if (config.anchorMetadata.kind != FiberletDatasetKind::Anchors ||
        (config.fiberletMetadata.kind != FiberletDatasetKind::Fiberlets && config.fiberletMetadata.kind != FiberletDatasetKind::Combined))
        throw std::invalid_argument("on-demand fiberlet preprocessing dataset kinds are invalid");
    if (config.anchorMetadata.profile != FiberletStorageProfile::Float32Cache)
        throw std::invalid_argument("on-demand anchor extraction requires a float32 intermediate cache");
    if (config.anchorMetadata.chunkGridShapeZYX != config.fiberletMetadata.chunkGridShapeZYX ||
        config.anchorMetadata.coordinateOriginZYX != config.fiberletMetadata.coordinateOriginZYX ||
        config.anchorMetadata.coordinateUnitsPerChunkZYX != config.fiberletMetadata.coordinateUnitsPerChunkZYX)
        throw std::invalid_argument("anchor and fiberlet cache grids must agree");
    validateFiberAnchorConfig(config.anchorConfig);
    validateFiberletPathConfig(config.pathConfig);
    const auto& quantization = config.geometryQuantization;
    (void)fiberletPositionBinCountForEvaluation(
        static_cast<int>(config.fiberletMetadata.spatialChunkSideBaseVoxels),
        quantization.positionQuantumBaseVoxels);
    if (config.selectedAnchorCellsZYX.empty() == !config.anchorCellPredicate) {
        throw std::invalid_argument("on-demand fiberlet preprocessing requires exactly one cell selector");
    }
    if (!config.selectedAnchorCellsZYX.empty() &&
        (!std::is_sorted(config.selectedAnchorCellsZYX.begin(), config.selectedAnchorCellsZYX.end()) ||
         std::adjacent_find(config.selectedAnchorCellsZYX.begin(), config.selectedAnchorCellsZYX.end()) != config.selectedAnchorCellsZYX.end())) {
        throw std::invalid_argument("on-demand fiberlet selected cells must be nonempty, ordered, and unique");
    }
    const auto cellSide = static_cast<std::size_t>(config.anchorConfig.cellSizePredictionVoxels);
    const std::array<std::size_t, 3>
        cellShape{(config.grid.shapeZYX[0] + cellSide - 1) / cellSide, (config.grid.shapeZYX[1] + cellSide - 1) / cellSide, (config.grid.shapeZYX[2] + cellSide - 1) / cellSide};
    for (const auto& cell : config.selectedAnchorCellsZYX) {
        std::array<int, 3> owner{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (cell[axis] >= cellShape[axis])
                throw std::invalid_argument("on-demand fiberlet selected cell lies outside the prediction grid");
            const auto chunk =
                floorDiv(static_cast<std::int64_t>(cell[axis]) - config.anchorMetadata.coordinateOriginZYX[axis], config.anchorMetadata.coordinateUnitsPerChunkZYX[axis]);
            if (chunk < 0 || chunk >= config.anchorMetadata.chunkGridShapeZYX[axis]) {
                throw std::invalid_argument("on-demand fiberlet selected cell lies outside the cache grid");
            }
            owner[axis] = static_cast<int>(chunk);
        }
        state_->selectedCellsByChunk[owner].push_back(cell);
    }
    if(config.cachedOnly){
        state_->anchorDataset=FiberletChunkDataset::openExisting(config.anchorRoot);
        state_->fiberletDataset=FiberletChunkDataset::openExisting(config.fiberletRoot);
        if(state_->anchorDataset->metadata().datasetFingerprint!=config.anchorMetadata.datasetFingerprint ||
           state_->fiberletDataset->metadata().datasetFingerprint!=config.fiberletMetadata.datasetFingerprint)
            throw std::invalid_argument("cached-only Fiberlet metadata mismatch");
    }else{
        state_->anchorDataset = FiberletChunkDataset::createOrOpen(config.anchorRoot, config.anchorMetadata);
        state_->fiberletDataset = FiberletChunkDataset::createOrOpen(config.fiberletRoot, config.fiberletMetadata);
    }
    const std::weak_ptr<FiberletOnDemandPreprocessor> weak = shared_from_this();
    state_->anchorRead = fiberletGeneratedChunkRead(
        state_->anchorDataset,
        [weak](FiberletStorageChunkKind, const FiberletChunkKey& key, const FiberletStorageCodecConfig& codec) {
            const auto self = weak.lock();
            if (!self)
                throw std::runtime_error("on-demand fiberlet preprocessor was destroyed");
            if(self->state_->config.cachedOnly){
                const auto& c=self->state_->config;
                if(ownedCellCount(codec,c.anchorMetadata,c.grid,c.anchorConfig.cellSizePredictionVoxels)!=0)
                    throw std::runtime_error("cached-only anchor chunk missing; prepare anchors before tracing");
                // Canonically outside the cell grid: generation returns an empty
                // payload without sampling or publishing any file.
            }
            return self->generateAnchorChunk(key, codec);
        },
        config.memoryCache);
    state_->fiberletRead = fiberletGeneratedChunkRead(
        state_->fiberletDataset,
        [weak](FiberletStorageChunkKind kind, const FiberletChunkKey& key, const FiberletStorageCodecConfig& codec) {
            const auto self = weak.lock();
            if (!self)
                throw std::runtime_error("on-demand fiberlet preprocessor was destroyed");
            if(self->state_->config.cachedOnly)throw std::runtime_error("cached-only fiberlet chunk missing");
            return self->generateFiberletChunk(kind, key, codec);
        },
        config.memoryCache);
}

const FiberletChunkRead& FiberletOnDemandPreprocessor::anchorRead() const noexcept
{
    return state_->anchorRead;
}
const FiberletChunkRead& FiberletOnDemandPreprocessor::fiberletRead() const noexcept
{
    return state_->fiberletRead;
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

std::shared_ptr<const std::vector<FiberletStoredAnchor>> FiberletOnDemandPreprocessor::evaluationAnchorChunk(
    const FiberletChunkKey& key, std::shared_ptr<const FiberletAnchorChunkPayload> canonicalChunk) const
{
    if (!canonicalChunk)
        throw std::invalid_argument("fiberlet evaluation anchor chunk is null");
    const auto& config = state_->config;
    const auto& quantization = config.geometryQuantization;
    if (!quantization.enabled()) {
        const auto* anchors = &canonicalChunk->anchors;
        return {std::move(canonicalChunk), anchors};
    }

    const std::array<int, 4> cacheKey{key.level, key.iz, key.iy, key.ix};
    std::shared_ptr<std::promise<std::shared_ptr<const std::vector<FiberletStoredAnchor>>>> promise;
    std::shared_future<std::shared_ptr<const std::vector<FiberletStoredAnchor>>> future;
    {
        std::lock_guard lock(state_->evaluationAnchorMutex);
        if (const auto found = state_->evaluationAnchorChunks.find(cacheKey); found != state_->evaluationAnchorChunks.end()) {
            found->second.touch = ++state_->evaluationAnchorTouch;
            future = found->second.future;
        } else {
            promise = std::make_shared<std::promise<std::shared_ptr<const std::vector<FiberletStoredAnchor>>>>();
            future = promise->get_future().share();
            state_->evaluationAnchorChunks.emplace(cacheKey, State::EvaluationAnchorChunkEntry{future, 0, ++state_->evaluationAnchorTouch});
        }
    }
    if (!promise)
        return future.get();

    try {
        auto transformed = std::make_shared<std::vector<FiberletStoredAnchor>>(canonicalChunk->anchors);
        for (auto& anchor : *transformed) {
            anchor.positionPredictionXYZ =
                quantizeFiberletPositionForEvaluation(anchor.positionPredictionXYZ, config.grid, quantization.positionQuantumBaseVoxels);
            if (quantization.compactDirections) {
                anchor.fittedAxisXYZ = quantizeFiberletDirectionForEvaluation(anchor.fittedAxisXYZ);
            }
        }
        if (!transformed->empty()) {
            std::vector<cv::Vec3f> scoringPoints;
            scoringPoints.reserve(transformed->size());
            for (const auto& anchor : *transformed)
                scoringPoints.push_back(anchor.positionPredictionXYZ);
            const auto scoring =
                sampleFiberletScoringPoints(scoringPoints, config.grid, config.pathConfig, config.predictionSampler, *config.normalSampler);
            if (scoring.size() != transformed->size()) {
                throw std::logic_error("fiberlet evaluation anchor scoring count is inconsistent");
            }
            for (std::size_t index = 0; index < transformed->size(); ++index) {
                auto& anchor = transformed->at(index);
                anchor.predictionAxisXYZ = scoring[index].prediction.direction;
                anchor.predictionPresence = scoring[index].prediction.presence;
                anchor.normalXYZ = scoring[index].normalXYZ;
                anchor.predictionValid = scoring[index].prediction.valid;
                anchor.predictionPresenceValid = scoring[index].prediction.presenceValid;
                anchor.normalValid = scoring[index].normalValid;
            }
        }

        const std::shared_ptr<const std::vector<FiberletStoredAnchor>> result = transformed;
        promise->set_value(result);
        const auto bytes = sizeof(*transformed) + transformed->capacity() * sizeof(FiberletStoredAnchor);
        {
            std::lock_guard lock(state_->evaluationAnchorMutex);
            const auto current = state_->evaluationAnchorChunks.find(cacheKey);
            if (current != state_->evaluationAnchorChunks.end()) {
                current->second.bytes = bytes;
                current->second.touch = ++state_->evaluationAnchorTouch;
                state_->evaluationAnchorBytes += bytes;
            }
            while (state_->evaluationAnchorBytes > config.evaluationAnchorCacheBytes && state_->evaluationAnchorChunks.size() > 1) {
                auto oldest = state_->evaluationAnchorChunks.end();
                for (auto iterator = state_->evaluationAnchorChunks.begin(); iterator != state_->evaluationAnchorChunks.end(); ++iterator) {
                    if (iterator->first == cacheKey || iterator->second.bytes == 0)
                        continue;
                    if (oldest == state_->evaluationAnchorChunks.end() || iterator->second.touch < oldest->second.touch)
                        oldest = iterator;
                }
                if (oldest == state_->evaluationAnchorChunks.end())
                    break;
                state_->evaluationAnchorBytes -= oldest->second.bytes;
                state_->evaluationAnchorChunks.erase(oldest);
            }
        }
        return result;
    } catch (...) {
        promise->set_exception(std::current_exception());
        std::lock_guard lock(state_->evaluationAnchorMutex);
        state_->evaluationAnchorChunks.erase(cacheKey);
        throw;
    }
}

bool FiberletOnDemandPreprocessor::isSelectedAnchorCell(const FiberletStorageKey& anchor) const noexcept
{
    std::array<std::size_t, 3> cell{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (anchor.coordinateZYX[axis] < 0 || static_cast<std::uint64_t>(anchor.coordinateZYX[axis]) > std::numeric_limits<std::size_t>::max()) {
            return false;
        }
        cell[axis] = static_cast<std::size_t>(anchor.coordinateZYX[axis]);
    }
    if (state_->config.anchorCellPredicate)
        return state_->config.anchorCellPredicate(cell);
    const auto& selected = state_->config.selectedAnchorCellsZYX;
    return std::binary_search(selected.begin(), selected.end(), cell);
}

FiberletChunkDataset::MaterializedChunk FiberletOnDemandPreprocessor::generateAnchorChunk(const FiberletChunkKey& key, const FiberletStorageCodecConfig& codec)
{
    const auto start = std::chrono::steady_clock::now();
    const auto& config = state_->config;
    std::vector<std::array<std::size_t, 3>> selectedCells;
    const auto work = config.acquireWork ? config.acquireWork() : nullptr;
    if (config.anchorCellPredicate) {
        selectedCells = selectedOwnedCells(codec, config.anchorMetadata, config.grid, config.anchorConfig.cellSizePredictionVoxels, config.anchorCellPredicate);
    } else {
        const auto found = state_->selectedCellsByChunk.find({key.iz, key.iy, key.ix});
        if (found != state_->selectedCellsByChunk.end())
            selectedCells = found->second;
    }
    const auto& cells = selectedCells;
    const auto unfilteredInputCount = ownedCellCount(codec, config.anchorMetadata, config.grid, config.anchorConfig.cellSizePredictionVoxels);
    if (config.progress)
        config.progress(FiberletOnDemandProgress{.stage = "anchors", .status = "started", .key = key, .inputCount = cells.size(), .unfilteredInputCount = unfilteredInputCount});
    if (cells.empty()) {
        if (config.progress)
            config.progress(FiberletOnDemandProgress{.stage = "anchors", .status = "completed", .key = key, .unfilteredInputCount = unfilteredInputCount});
        FiberletDecodedAnchors decoded{codec, {}};
        // Empty sparse chunks remain absent. The generated cache still receives
        // a canonical decoded empty payload for this process.
        return {serializeFiberletAnchors(codec, {}), std::make_shared<const FiberletAnchorChunkPayload>(std::move(decoded)), true};
    }
    auto extracted =
        extractFiberAnchorsForCells(config.grid, config.anchorConfig, config.predictionSampler, cells, config.anchorRetainPredicate, {}, false);
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
    std::vector<cv::Vec3f> scoringPoints;
    scoringPoints.reserve(stored.size());
    for (const auto& anchor : stored)
        scoringPoints.push_back(anchor.positionPredictionXYZ);
    const auto scoring = sampleFiberletScoringPoints(scoringPoints, config.grid, config.pathConfig, config.predictionSampler, *config.normalSampler);
    if (scoring.size() != stored.size())
        throw std::logic_error("fiberlet anchor scoring count is inconsistent");
    for (std::size_t index = 0; index < stored.size(); ++index) {
        stored[index].predictionAxisXYZ = scoring[index].prediction.direction;
        stored[index].predictionPresence = scoring[index].prediction.presence;
        stored[index].normalXYZ = scoring[index].normalXYZ;
        stored[index].predictionValid = scoring[index].prediction.valid;
        stored[index].predictionPresenceValid = scoring[index].prediction.presenceValid;
        stored[index].normalValid = scoring[index].normalValid;
    }
    if (config.progress) {
        config.progress(
            FiberletOnDemandProgress{
            .stage = "anchors",
            .status = "completed",
            .key = key,
            .inputCount = cells.size(),
            .unfilteredInputCount = unfilteredInputCount,
            .outputCount = stored.size(),
                .elapsedSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
            .cpuSeconds = extracted.profile.elapsedCpuSeconds});
    }
    auto bytes = serializeFiberletAnchors(codec, stored);
    FiberletDecodedAnchors decoded{codec, std::move(stored)};
    const bool sparseEmpty = decoded.anchors.empty();
    return {
        std::move(bytes),
        std::make_shared<const FiberletAnchorChunkPayload>(std::move(decoded)),
        sparseEmpty};
}

std::vector<FiberletChunkKey> FiberletOnDemandPreprocessor::anchorDependencies(const FiberletChunkKey& fiberletChunk) const
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
    std::vector<FiberletChunkKey> result;
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

FiberletChunkDataset::MaterializedChunk FiberletOnDemandPreprocessor::generateFiberletChunk(
    FiberletStorageChunkKind kind, const FiberletChunkKey& key, const FiberletStorageCodecConfig& codec)
{
    const std::array<int, 3> owner{key.iz, key.iy, key.ix};
    {
        std::lock_guard lock(state_->generationMutex);
        if (state_->emptyOwners.contains(owner)) {
            if (kind == FiberletStorageChunkKind::FiberletPrefix)
                return {serializeFiberletPrefixes(codec, {}),
                    std::make_shared<const FiberletPrefixChunkPayload>(FiberletDecodedPrefixes{codec, {}}), true};
            return {serializeFiberletRoutes(codec, {}),
                std::make_shared<const FiberletRouteChunkPayload>(FiberletDecodedRoutes{codec, {}}), true};
        }
    }
    auto prefixKey = key; prefixKey.level = 0;
    auto routeKey = key; routeKey.level = 1;
    auto prefix = state_->fiberletDataset->readMaterializedChunk(FiberletStorageChunkKind::FiberletPrefix, prefixKey);
    auto route = state_->fiberletDataset->readMaterializedChunk(FiberletStorageChunkKind::FiberletRoutes, routeKey);
    auto pair = prefix && route ? State::Pair{std::move(*prefix), std::move(*route)}
        : generateFiberletPair(prefixKey, state_->fiberletDataset->codecConfig(FiberletStorageChunkKind::FiberletPrefix, prefixKey));
    if (std::dynamic_pointer_cast<const FiberletPrefixChunkPayload>(pair.first.payload)->prefixes.empty()) {
        std::lock_guard lock(state_->generationMutex);
        state_->emptyOwners.insert(owner);
    }
    return kind == FiberletStorageChunkKind::FiberletPrefix ? std::move(pair.first) : std::move(pair.second);
}

std::pair<FiberletChunkDataset::MaterializedChunk, FiberletChunkDataset::MaterializedChunk>
FiberletOnDemandPreprocessor::generateFiberletPair(
    const FiberletChunkKey& key, const FiberletStorageCodecConfig& codec)
{
    const auto start = std::chrono::steady_clock::now();
    auto prefixKey = key;
    prefixKey.level = 0;
    auto routeKey = key;
    routeKey.level = 1;
    const auto& config = state_->config;
    std::vector<FiberletStoredAnchor> storedAnchors;
    const auto dependencies = anchorDependencies(prefixKey);
    for (const auto& dependency : dependencies) {
        const auto anchors = std::dynamic_pointer_cast<const FiberletAnchorChunkPayload>(state_->anchorRead(dependency));
        if (!anchors)
            throw std::runtime_error(
                "required fiberlet anchor chunk " + std::to_string(dependency.level) + '/' +
                std::to_string(dependency.iz) + '/' + std::to_string(dependency.iy) + '/' +
                std::to_string(dependency.ix) + " has the wrong payload type");
        const auto evaluated = evaluationAnchorChunk(dependency, anchors);
        storedAnchors.insert(storedAnchors.end(), evaluated->begin(), evaluated->end());
    }
    const auto work = config.acquireWork ? config.acquireWork() : nullptr;
    std::sort(storedAnchors.begin(), storedAnchors.end(), [](const auto& left, const auto& right) { return left.key < right.key; });
    const auto duplicate = std::adjacent_find(storedAnchors.begin(), storedAnchors.end(), [](const auto& left, const auto& right) {
        return left.key == right.key;
    });
    if (duplicate != storedAnchors.end())
        throw std::invalid_argument("anchor dependency chunks contain duplicate stable keys");
    if (config.progress) {
        config.progress(FiberletOnDemandProgress{.stage = "fiberlets", .status = "started", .key = prefixKey, .inputCount = storedAnchors.size()});
    }
    const std::size_t inputAnchorCount = storedAnchors.size();
    auto loaded = loadedAnchors(config.grid, config.anchorConfig, std::move(storedAnchors));
    const auto ownerBegin = codec.coordinateOriginZYX;
    std::array<std::int64_t, 3> ownerEnd{};
    for (std::size_t axis = 0; axis < 3; ++axis)
        ownerEnd[axis] = ownerBegin[axis] + config.fiberletMetadata.coordinateUnitsPerChunkZYX[axis];
    auto report = traceFiberletPaths(
        loaded,
        config.grid,
        config.pathConfig,
        config.predictionSampler,
        *config.normalSampler,
            [&](const FiberletPathProgress& progress) {
                if (config.progress) {
                config.progress(
                    FiberletOnDemandProgress{
                        .stage = "fiberlets",
                        .status = "running",
                        .phase = progress.phase,
                        .key = prefixKey,
                        .inputCount = inputAnchorCount,
                        .phaseCompleted = progress.completed,
                        .phaseTotal = progress.total,
                        .elapsedSeconds = progress.elapsedSeconds});
                }
        },
        config.pointPredicate,
        {},
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
        if (candidate.pointsPredictionXYZ.size() < 2)
            throw std::logic_error("successful fiberlet has no endpoint steps");
        if (candidate.segmentCosts.size() + 1 != candidate.pointsPredictionXYZ.size())
            throw std::logic_error("successful fiberlet segment costs differ from its geometry");
        stored.route.segmentCostDensities.reserve(candidate.segmentCosts.size());
        for (size_t segment = 0; segment < candidate.segmentCosts.size(); ++segment) {
            const float segmentLength = cv::norm(
                candidate.pointsPredictionXYZ[segment + 1] -
                candidate.pointsPredictionXYZ[segment]);
            if (!(segmentLength > 0.0F) || !std::isfinite(segmentLength))
                throw std::logic_error("successful fiberlet segment length is invalid");
            stored.route.segmentCostDensities.push_back(
                candidate.segmentCosts[segment].total() / segmentLength);
        }
        stored.prefix.pathLengthPredictionVoxels = fiberletCandidatePathLength(candidate);
        stored.prefix.cost = {
            candidate.cost.invalidPrediction,
            candidate.cost.alignment,
            candidate.cost.isotropicSmoothness,
            candidate.cost.tangentSmoothness,
            candidate.cost.normalSmoothness,
        };
        const float scale = static_cast<float>(config.grid.predictionToBaseScale);
        stored.prefix.firstStepBaseXYZ = candidate.pointsPredictionXYZ[1] * scale - candidate.pointsPredictionXYZ.front() * scale;
        stored.prefix.lastStepBaseXYZ =
            candidate.pointsPredictionXYZ.back() * scale - candidate.pointsPredictionXYZ[candidate.pointsPredictionXYZ.size() - 2] * scale;
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
    const bool sparseEmpty = prefixes.empty();
    const auto prefixCodec = state_->fiberletDataset->codecConfig(FiberletStorageChunkKind::FiberletPrefix, prefixKey);
    const auto routeCodec = state_->fiberletDataset->codecConfig(FiberletStorageChunkKind::FiberletRoutes, routeKey);
    auto prefixBytes = serializeFiberletPrefixes(prefixCodec, prefixes);
    auto routeBytes = serializeFiberletRoutes(routeCodec, routes);
    FiberletChunkDataset::MaterializedChunk
        prefixChunk{std::move(prefixBytes), std::make_shared<const FiberletPrefixChunkPayload>(FiberletDecodedPrefixes{prefixCodec, std::move(prefixes)}), true};
    FiberletChunkDataset::MaterializedChunk
        routeChunk{std::move(routeBytes), std::make_shared<const FiberletRouteChunkPayload>(FiberletDecodedRoutes{routeCodec, std::move(routes)}), true};
    if (!sparseEmpty)
        state_->fiberletDataset->publishFiberletChunkPair(prefixKey, prefixChunk, routeKey, routeChunk);
    if (config.progress) {
        config.progress(
            FiberletOnDemandProgress{
            .stage = "fiberlets",
            .status = "completed",
            .key = prefixKey,
            .inputCount = inputAnchorCount,
                .outputCount = std::dynamic_pointer_cast<const FiberletPrefixChunkPayload>(prefixChunk.payload)->prefixes.size(),
                .elapsedSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
            .cpuSeconds = report.elapsedCpuSeconds,
                .candidateGenerationWorkers = report.candidateGenerationWorkers,
            .candidateGenerationSeconds = report.candidateGenerationSeconds,
                .candidateGenerationCpuSeconds = report.candidateGenerationCpuSeconds});
    }
    return {std::move(prefixChunk), std::move(routeChunk)};
}

}  // namespace vc::fiber_tracer
