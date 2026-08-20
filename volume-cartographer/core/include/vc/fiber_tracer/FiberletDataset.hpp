#pragma once

#include "vc/core/render/ChunkCache.hpp"
#include "vc/fiber_tracer/FiberletStorage.hpp"

#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vc::fiber_tracer
{

enum class FiberletDatasetKind {
    Anchors,
    Fiberlets,
};

struct FiberletDatasetMetadata {
    FiberletDatasetKind kind = FiberletDatasetKind::Anchors;
    FiberletStorageProfile profile = FiberletStorageProfile::Float32Cache;
    std::array<std::int32_t, 3> chunkGridShapeZYX{0, 0, 0};
    std::array<std::int64_t, 3> coordinateOriginZYX{0, 0, 0};
    std::array<std::int64_t, 3> coordinateUnitsPerChunkZYX{0, 0, 0};
    std::array<std::int64_t, 3> maximumEndpointReachCoordinateUnitsZYX{0, 0, 0};
    std::array<std::uint8_t, 32> datasetFingerprint{};
    std::uint32_t spatialChunkSideBaseVoxels = 512;
    std::uint8_t coordinateBits = 16;
    std::uint8_t deltaBits = 16;
    std::uint8_t routeCountBits = 16;
    std::uint8_t routeLatticeBits = 16;
    std::uint8_t costBits = 32;
    std::uint32_t positionQuantumBaseVoxels = 0;
    double predictionToBaseScale = 1.0;
    std::string algorithmFingerprint;
    std::string fiberManifest;
    std::string fiberManifestHash;
    std::string normalManifest;
    std::string normalManifestHash;
};

class FiberletChunkDataset
{
public:
    static std::shared_ptr<FiberletChunkDataset> createOrOpen(std::filesystem::path root, const FiberletDatasetMetadata& metadata);

    [[nodiscard]] const std::filesystem::path& root() const noexcept;
    [[nodiscard]] const FiberletDatasetMetadata& metadata() const noexcept;
    [[nodiscard]] FiberletStorageCodecConfig codecConfig(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key) const;
    [[nodiscard]] std::filesystem::path chunkPath(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key) const;
    [[nodiscard]] std::optional<std::vector<std::byte>> readChunk(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key) const;
    void publishChunk(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, std::span<const std::byte> bytes) const;
    void validateChunk(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, std::span<const std::byte> bytes) const;

private:
    FiberletChunkDataset(std::filesystem::path root, FiberletDatasetMetadata metadata);

    std::filesystem::path root_;
    FiberletDatasetMetadata metadata_;
};

using FiberletChunkGenerator =
    std::function<std::vector<std::byte>(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, const FiberletStorageCodecConfig& config)>;

[[nodiscard]] std::shared_ptr<vc::render::ChunkCache> createGeneratedFiberletChunkCache(
    std::shared_ptr<FiberletChunkDataset> dataset, FiberletChunkGenerator generator, vc::render::ChunkCache::Options options = {});

}  // namespace vc::fiber_tracer
