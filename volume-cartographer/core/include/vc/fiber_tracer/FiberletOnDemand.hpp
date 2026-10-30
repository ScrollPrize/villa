#pragma once

#include "vc/fiber_tracer/FiberletDataset.hpp"
#include "vc/fiber_tracer/FiberPaths.hpp"
#include "vc/fiber_tracer/FiberletQuantization.hpp"

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
    FiberletChunkKey key;
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
    std::shared_ptr<vcr::MemoryCache> memoryCache;
    FiberletOnDemandProgressCallback progress;
    // Optional shared CPU admission. Acquired only for generation computation,
    // never while waiting for anchor dependencies. Lifetime releases admission.
    std::function<std::shared_ptr<void>()> acquireWork;
};

[[nodiscard]] std::vector<FiberletChunkKey> fiberletOutputChunksForNonemptyPresence(
    const FiberPresenceChunkScanReport& presence, const FiberletDatasetMetadata& outputMetadata, int anchorCellSizePredictionVoxels);

enum class FiberletPreprocessWorkKind {
    Anchor,
    Fiberlet,
};

struct FiberletPreprocessWork {
    FiberletPreprocessWorkKind kind = FiberletPreprocessWorkKind::Anchor;
    FiberletChunkKey key;
};

class FiberletPreprocessSchedule final
{
public:
    FiberletPreprocessSchedule(
        std::vector<FiberletChunkKey> outputChunks,
        std::vector<std::vector<FiberletChunkKey>> anchorDependencies,
        std::span<const FiberletChunkKey> completedOutputs,
        std::span<const FiberletChunkKey> availableAnchors);
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

    // Blocking reads: chunks come from the datasets; absent ones are generated
    // and published. Fiberlet keys: level 0 prefixes, level 1 routes.
    [[nodiscard]] const FiberletChunkRead& anchorRead() const noexcept;
    [[nodiscard]] const FiberletChunkRead& fiberletRead() const noexcept;
    [[nodiscard]] const std::shared_ptr<FiberletChunkDataset>& anchorDataset() const noexcept;
    [[nodiscard]] const std::shared_ptr<FiberletChunkDataset>& fiberletDataset() const noexcept;
    [[nodiscard]] const FiberPredictionGridInfo& grid() const noexcept;
    [[nodiscard]] const FiberAnchorConfig& anchorConfig() const noexcept;
    [[nodiscard]] std::shared_ptr<const std::vector<FiberletStoredAnchor>> evaluationAnchorChunk(
        const FiberletChunkKey& key, std::shared_ptr<const FiberletAnchorChunkPayload> canonicalChunk) const;
    [[nodiscard]] bool isSelectedAnchorCell(const FiberletStorageKey& anchor) const noexcept;

    [[nodiscard]] std::vector<FiberletChunkKey> anchorDependencies(const FiberletChunkKey& fiberletChunk) const;

private:
    explicit FiberletOnDemandPreprocessor(FiberletOnDemandConfig config);
    void initialize();
    FiberletChunkDataset::MaterializedChunk generateAnchorChunk(const FiberletChunkKey& key, const FiberletStorageCodecConfig& codec);
    FiberletChunkDataset::MaterializedChunk generateFiberletChunk(FiberletStorageChunkKind kind, const FiberletChunkKey& key, const FiberletStorageCodecConfig& codec);
    std::pair<FiberletChunkDataset::MaterializedChunk, FiberletChunkDataset::MaterializedChunk>
        generateFiberletPair(const FiberletChunkKey& key, const FiberletStorageCodecConfig& codec);

    struct State;
    std::shared_ptr<State> state_;
};

}  // namespace vc::fiber_tracer
