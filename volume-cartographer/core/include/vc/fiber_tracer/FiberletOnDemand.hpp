#pragma once

#include "vc/fiber_tracer/FiberletDataset.hpp"
#include "vc/fiber_tracer/FiberPaths.hpp"
#include "vc/fiber_tracer/FiberletQuantization.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace vc::fiber_tracer
{

struct FiberletOnDemandProgress {
    std::string stage;
    std::string status;
    std::string phase;
    vc::render::ChunkKey key;
    std::size_t inputCount = 0;
    std::size_t unfilteredInputCount = 0;
    std::size_t outputCount = 0;
    std::size_t phaseCompleted = 0;
    std::size_t phaseTotal = 0;
    double elapsedSeconds = 0.0;
    double cpuSeconds = 0.0;
    std::size_t candidateGenerationWorkers = 0;
    double candidateGenerationSeconds = 0.0;
    double candidateGenerationCpuSeconds = 0.0;
};

using FiberletOnDemandProgressCallback = std::function<void(const FiberletOnDemandProgress&)>;
using FiberletAnchorCellPredicate = std::function<bool(const std::array<std::size_t, 3>& cellZYX)>;

struct FiberletOnDemandConfig {
    bool cachedOnly = false; // Read existing immutable chunks; missing data is an error.
    std::filesystem::path anchorRoot;
    std::filesystem::path fiberletRoot;
    FiberletDatasetMetadata anchorMetadata;
    FiberletDatasetMetadata fiberletMetadata;
    FiberPredictionGridInfo grid;
    FiberAnchorConfig anchorConfig;
    FiberletPathConfig pathConfig;
    FiberStoredPredictionBatchSampler predictionSampler;
    std::shared_ptr<const vc::lasagna::NormalSampler> normalSampler;
    std::vector<std::array<std::size_t, 3>> selectedAnchorCellsZYX;
    FiberletAnchorCellPredicate anchorCellPredicate;
    FiberAnchorRetainPredicate anchorRetainPredicate;
    FiberletPointPredicate pointPredicate;
    FiberletGeometryQuantization geometryQuantization;
    std::size_t evaluationAnchorCacheBytes = 256ULL * 1024ULL * 1024ULL;
    FiberletChunkCacheOptions anchorCacheOptions;
    FiberletChunkCacheOptions fiberletCacheOptions;
    FiberletOnDemandProgressCallback progress;
    FiberletChunkResolvedCallback chunkResolved;
    // Optional shared CPU admission. Acquired only for generation computation,
    // never while waiting for anchor dependencies. Lifetime releases admission.
    std::function<std::shared_ptr<void>()> acquireWork;
};

struct FiberletScheduledChunk {
    vc::render::ChunkKey key;
    double nearestReferenceArcBase = 0.0;
    double nearestReferenceDistanceBase = 0.0;
};

struct FiberletReferencePrefetchCandidate {
    FiberletScheduledChunk scheduled;
    double remainingReferenceBase = 0.0;
    std::size_t stableOrder = 0;
};

// Deduplicate one physical preprocessor's reference schedules and order the
// retained chunks from greatest remaining reference distance to least.
[[nodiscard]] std::vector<FiberletScheduledChunk>
orderFiberletReferencePrefetch(
    std::span<const FiberletReferencePrefetchCandidate> candidates);

[[nodiscard]] std::string fiberletScheduledResolutionError(
    const vc::render::ChunkKey& key,
    const vc::render::ChunkResult& resolved);

[[nodiscard]] std::vector<vc::render::ChunkKey> fiberletOutputChunksForNonemptyPresence(
    const FiberPresenceChunkScanReport& presence, const FiberletDatasetMetadata& outputMetadata, int anchorCellSizePredictionVoxels);

enum class FiberletPreprocessWorkKind {
    Anchor,
    Fiberlet,
};

struct FiberletPreprocessWork {
    FiberletPreprocessWorkKind kind = FiberletPreprocessWorkKind::Anchor;
    vc::render::ChunkKey key;
};

class FiberletPreprocessSchedule final
{
public:
    FiberletPreprocessSchedule(
        std::vector<vc::render::ChunkKey> outputChunks,
        std::vector<std::vector<vc::render::ChunkKey>> anchorDependencies,
        std::span<const vc::render::ChunkKey> completedOutputs,
        std::span<const vc::render::ChunkKey> availableAnchors);
    ~FiberletPreprocessSchedule();

    FiberletPreprocessSchedule(const FiberletPreprocessSchedule&) = delete;
    FiberletPreprocessSchedule& operator=(const FiberletPreprocessSchedule&) = delete;
    FiberletPreprocessSchedule(FiberletPreprocessSchedule&&) noexcept;
    FiberletPreprocessSchedule& operator=(FiberletPreprocessSchedule&&) noexcept;

    // Ready fiberlets in the current Z slab always precede anchor lookahead.
    [[nodiscard]] std::optional<FiberletPreprocessWork> takeNext();
    void complete(const FiberletPreprocessWork& work);

    [[nodiscard]] bool done() const noexcept;
    [[nodiscard]] std::optional<int> currentOutputZ() const noexcept;
    [[nodiscard]] std::size_t anchorTotal() const noexcept;
    [[nodiscard]] std::size_t anchorsCompleted() const noexcept;
    [[nodiscard]] std::size_t outputTotal() const noexcept;
    [[nodiscard]] std::size_t outputsCompleted() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class FiberletOnDemandPreprocessor : public std::enable_shared_from_this<FiberletOnDemandPreprocessor>
{
public:
    static std::shared_ptr<FiberletOnDemandPreprocessor> create(FiberletOnDemandConfig config);

    [[nodiscard]] const std::shared_ptr<vc::render::ChunkCache>& anchorCache() const noexcept;
    [[nodiscard]] const std::shared_ptr<vc::render::ChunkCache>& fiberletCache() const noexcept;
    [[nodiscard]] const std::shared_ptr<FiberletChunkDataset>& anchorDataset() const noexcept;
    [[nodiscard]] const std::shared_ptr<FiberletChunkDataset>& fiberletDataset() const noexcept;
    [[nodiscard]] const FiberPredictionGridInfo& grid() const noexcept;
    [[nodiscard]] const FiberAnchorConfig& anchorConfig() const noexcept;
    [[nodiscard]] std::shared_ptr<const std::vector<FiberletStoredAnchor>> evaluationAnchorChunk(
        const vc::render::ChunkKey& key, std::shared_ptr<const FiberletAnchorChunkPayload> canonicalChunk) const;
    [[nodiscard]] bool isSelectedAnchorCell(const FiberletStorageKey& anchor) const noexcept;

    [[nodiscard]] std::vector<vc::render::ChunkKey> anchorDependencies(const vc::render::ChunkKey& fiberletChunk) const;
    [[nodiscard]] std::vector<FiberletScheduledChunk> referenceChunkSchedule(
        const PolylineArcGeometry& reference, double beginArcBase, double endArcBase, double radiusBaseVoxels) const;
    // Includes owner-cell displacement padding, but no path-connection halo.
    [[nodiscard]] std::vector<FiberletScheduledChunk> anchorChunkScheduleNearReference(
        const PolylineArcGeometry& reference, double beginArcBase, double endArcBase, double radiusBaseVoxels) const;
    void prefetchScheduled(
        std::span<const FiberletScheduledChunk> schedule,
        std::size_t begin,
        std::size_t count,
        bool wait = false,
        int priorityOffset = 0) const;
    // Stop speculative generation in dependency order and drain writes. Call
    // this before releasing a batch-owned preprocessor.
    void shutdown();

private:
    explicit FiberletOnDemandPreprocessor(FiberletOnDemandConfig config);
    void initialize();
    FiberletChunkDataset::MaterializedChunk generateAnchorChunk(const vc::render::ChunkKey& key, const FiberletStorageCodecConfig& codec);
    FiberletChunkDataset::MaterializedChunk generateFiberletChunk(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, const FiberletStorageCodecConfig& codec);
    std::pair<FiberletChunkDataset::MaterializedChunk, FiberletChunkDataset::MaterializedChunk>
        generateFiberletPair(const vc::render::ChunkKey& key, const FiberletStorageCodecConfig& codec);

    struct State;
    std::shared_ptr<State> state_;
};

}  // namespace vc::fiber_tracer
