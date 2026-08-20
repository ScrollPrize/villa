#pragma once

#include "vc/core/render/ChunkCache.hpp"
#include "vc/fiber_tracer/FiberletStorage.hpp"

#include <array>
#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vc::fiber_tracer
{

struct FiberletIncidentPrefix {
    DirectedFiberletStorageId id;
    FiberletStoredPrefix prefix;
};

class FiberletAnchorChunkPayload final : public vc::render::DecodedChunkPayload
{
public:
    explicit FiberletAnchorChunkPayload(FiberletDecodedAnchors decoded);

    [[nodiscard]] std::size_t residentBytes() const noexcept override;
    [[nodiscard]] const FiberletStoredAnchor* find(
        const FiberletStorageKey& key) const noexcept;

    FiberletStorageCodecConfig config;
    std::vector<FiberletStoredAnchor> anchors;
};

class FiberletPrefixChunkPayload final : public vc::render::DecodedChunkPayload
{
public:
    explicit FiberletPrefixChunkPayload(FiberletDecodedPrefixes decoded);

    [[nodiscard]] std::size_t residentBytes() const noexcept override;
    [[nodiscard]] const FiberletStoredPrefix* find(
        const FiberletStorageId& id) const noexcept;
    [[nodiscard]] std::vector<FiberletIncidentPrefix> incident(
        const FiberletStorageKey& key) const;

    FiberletStorageCodecConfig config;
    std::vector<FiberletStoredPrefix> prefixes;

private:
    [[nodiscard]] const FiberletStorageKey& endpoint(
        std::uint32_t encoded) const noexcept;

    std::vector<std::uint32_t> incidentOrder_;
};

class FiberletRouteChunkPayload final : public vc::render::DecodedChunkPayload
{
public:
    explicit FiberletRouteChunkPayload(FiberletDecodedRoutes decoded);

    [[nodiscard]] std::size_t residentBytes() const noexcept override;

    FiberletStorageCodecConfig config;
    std::vector<FiberletStoredRoute> routes;
};

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
    struct MaterializationStats {
        std::size_t anchorDecodes = 0;
        std::size_t prefixDecodes = 0;
        std::size_t routeDecodes = 0;
    };

    static std::shared_ptr<FiberletChunkDataset> createOrOpen(std::filesystem::path root, const FiberletDatasetMetadata& metadata);

    [[nodiscard]] const std::filesystem::path& root() const noexcept;
    [[nodiscard]] const FiberletDatasetMetadata& metadata() const noexcept;
    [[nodiscard]] MaterializationStats materializationStats() const noexcept;
    [[nodiscard]] FiberletStorageCodecConfig codecConfig(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key) const;
    [[nodiscard]] std::filesystem::path chunkPath(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key) const;
    [[nodiscard]] std::optional<std::vector<std::byte>> readChunk(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key) const;
    struct MaterializedChunk {
        std::vector<std::byte> bytes;
        std::shared_ptr<const vc::render::DecodedChunkPayload> payload;
        bool alreadyPublished = false;
    };
    [[nodiscard]] std::optional<MaterializedChunk> readMaterializedChunk(
        FiberletStorageChunkKind kind,
        const vc::render::ChunkKey& key) const;
    void publishChunk(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, std::span<const std::byte> bytes) const;
    void publishMaterializedChunk(
        FiberletStorageChunkKind kind,
        const vc::render::ChunkKey& key,
        const MaterializedChunk& chunk) const;
    void publishFiberletChunkPair(
        const vc::render::ChunkKey& prefixKey,
        const MaterializedChunk& prefix,
        const vc::render::ChunkKey& routeKey,
        const MaterializedChunk& routes) const;
    void validateChunk(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, std::span<const std::byte> bytes) const;

private:
    FiberletChunkDataset(std::filesystem::path root, FiberletDatasetMetadata metadata);

    std::filesystem::path root_;
    FiberletDatasetMetadata metadata_;
    mutable std::array<std::atomic_size_t, 3> materializationDecodes_{};
};

using FiberletChunkGenerator =
    std::function<FiberletChunkDataset::MaterializedChunk(
        FiberletStorageChunkKind kind,
        const vc::render::ChunkKey& key,
        const FiberletStorageCodecConfig& config)>;

using FiberletChunkResolvedCallback = std::function<void(
    FiberletStorageChunkKind kind,
    const vc::render::ChunkKey& key,
    vc::render::ChunkFetchStatus status)>;

[[nodiscard]] std::shared_ptr<vc::render::ChunkCache> createGeneratedFiberletChunkCache(
    std::shared_ptr<FiberletChunkDataset> dataset,
    FiberletChunkGenerator generator,
    vc::render::ChunkCache::Options options = {},
    FiberletChunkResolvedCallback resolved = {});

[[nodiscard]] std::shared_ptr<const vc::render::DecodedChunkPayload>
decodeFiberletChunkPayload(
    FiberletStorageChunkKind kind,
    std::span<const std::byte> bytes);

}  // namespace vc::fiber_tracer
