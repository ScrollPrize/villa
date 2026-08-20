#pragma once

#include "vc/fiber_tracer/FiberletDataset.hpp"
#include "vc/fiber_tracer/FiberPaths.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

#include <functional>
#include <memory>
#include <string>

namespace vc::fiber_tracer
{

struct FiberletOnDemandProgress {
    std::string stage;
    std::string status;
    std::string phase;
    vc::render::ChunkKey key;
    std::size_t inputCount = 0;
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

struct FiberletOnDemandConfig {
    std::filesystem::path anchorRoot;
    std::filesystem::path fiberletRoot;
    FiberletDatasetMetadata anchorMetadata;
    FiberletDatasetMetadata fiberletMetadata;
    FiberPredictionGridInfo grid;
    FiberAnchorConfig anchorConfig;
    FiberletPathConfig pathConfig;
    FiberStoredPredictionBatchSampler predictionSampler;
    std::shared_ptr<const vc::lasagna::NormalSampler> normalSampler;
    vc::render::ChunkCache::Options anchorCacheOptions;
    vc::render::ChunkCache::Options fiberletCacheOptions;
    FiberletOnDemandProgressCallback progress;
};

struct FiberletScheduledChunk {
    vc::render::ChunkKey key;
    double nearestReferenceArcBase = 0.0;
    double nearestReferenceDistanceBase = 0.0;
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

    [[nodiscard]] std::vector<vc::render::ChunkKey> anchorDependencies(const vc::render::ChunkKey& fiberletChunk) const;
    [[nodiscard]] std::vector<FiberletScheduledChunk> referenceChunkSchedule(
        const PolylineArcGeometry& reference, double beginArcBase, double endArcBase, double radiusBaseVoxels) const;
    void prefetchScheduled(std::span<const FiberletScheduledChunk> schedule, std::size_t begin, std::size_t count, bool wait = false) const;

private:
    explicit FiberletOnDemandPreprocessor(FiberletOnDemandConfig config);
    void initialize();
    std::vector<std::byte> generateAnchorChunk(const vc::render::ChunkKey& key, const FiberletStorageCodecConfig& codec);
    std::vector<std::byte> generateFiberletChunk(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, const FiberletStorageCodecConfig& codec);

    struct State;
    std::shared_ptr<State> state_;
};

}  // namespace vc::fiber_tracer
