#pragma once

#include "vc/core/render/ChunkCache.hpp"
#include "vc/fiber_tracer/FiberletStorage.hpp"

#include <array>
#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

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
    Combined,
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
    // Stable producer/data identities and hashes. Runtime filesystem paths are
    // deliberately excluded from both this object and its fingerprints.
    nlohmann::json sources = nlohmann::json::object();
    // Complete effective scientific processing, selection, layout, storage,
    // and codec settings required to interpret or reproduce this dataset.
    nlohmann::json processing = nlohmann::json::object();
};

// Canonically derive the algorithm and dataset fingerprints from the
// structured metadata. Call this after all effective values are resolved.
void finalizeFiberletDatasetIdentity(FiberletDatasetMetadata& metadata);

class FiberletChunkDataset
{
public:
    struct MaterializationStats {
        std::size_t anchorDecodes = 0;
        std::size_t prefixDecodes = 0;
        std::size_t routeDecodes = 0;
    };

    static std::shared_ptr<FiberletChunkDataset> createOrOpen(std::filesystem::path root, const FiberletDatasetMetadata& metadata);
    // Open and validate an existing dataset using its authoritative metadata;
    // callers do not supply a duplicate configuration.
    static std::shared_ptr<FiberletChunkDataset> openExisting(
        std::filesystem::path root);

    [[nodiscard]] const std::filesystem::path& root() const noexcept;
    [[nodiscard]] const FiberletDatasetMetadata& metadata() const noexcept;
    [[nodiscard]] bool datasetComplete() const;
    void configureExpectedChunks(std::span<const vc::render::ChunkKey> chunks);
    [[nodiscard]] const std::vector<vc::render::ChunkKey>& expectedChunks() const noexcept;
    [[nodiscard]] bool isExpectedChunk(const vc::render::ChunkKey& key) const;
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
    std::vector<vc::render::ChunkKey> expectedChunks_;
    bool expectedChunksConfigured_ = false;
    mutable std::array<std::atomic_size_t, 3> materializationDecodes_{};
};

using FiberletChunkGenerator =
    std::function<FiberletChunkDataset::MaterializedChunk(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, const FiberletStorageCodecConfig& config)>;

using FiberletChunkResolvedCallback =
    std::function<void(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, vc::render::ChunkFetchStatus status)>;

struct FiberletChunkCacheOptions {
    vc::render::ChunkCache::Options cache;
    vc::render::ChunkCacheService::Options service;
};

[[nodiscard]] std::shared_ptr<vc::render::ChunkCache> createGeneratedFiberletChunkCache(
    std::shared_ptr<FiberletChunkDataset> dataset,
    FiberletChunkGenerator generator,
    FiberletChunkCacheOptions options = {},
    FiberletChunkResolvedCallback resolved = {});

[[nodiscard]] std::shared_ptr<vc::render::ChunkCache> createStoredFiberletAnchorChunkCache(
    std::shared_ptr<FiberletChunkDataset> dataset, FiberletChunkCacheOptions options = {});

[[nodiscard]] std::shared_ptr<vc::render::ChunkCache> createStoredFiberletPathChunkCache(
    std::shared_ptr<FiberletChunkDataset> dataset, FiberletChunkCacheOptions options = {});

// Expose a sparse dataset layer over a compatible lower cache. A materialized
// upper chunk shadows the lower layer, including an explicitly empty chunk;
// an absent upper chunk falls through without being copied into the layer.
[[nodiscard]] std::shared_ptr<vc::render::ChunkCache>
createOverlayFiberletAnchorChunkCache(
    std::shared_ptr<FiberletChunkDataset> layer,
    std::shared_ptr<FiberletChunkDataset> lowerDataset,
    std::shared_ptr<vc::render::ChunkCache> lower,
    FiberletChunkCacheOptions options = {});

[[nodiscard]] std::shared_ptr<vc::render::ChunkCache>
createOverlayFiberletPathChunkCache(
    std::shared_ptr<FiberletChunkDataset> layer,
    std::shared_ptr<FiberletChunkDataset> lowerDataset,
    std::shared_ptr<vc::render::ChunkCache> lower,
    FiberletChunkCacheOptions options = {});

// Reduction overlays are temporary mutable layers. Callers must guarantee
// exclusive access and discard every decoded view of the replaced coordinate
// before using the new bytes.
void replaceFiberletOverlayChunk(
    const std::shared_ptr<FiberletChunkDataset>& layer,
    FiberletStorageChunkKind kind,
    const vc::render::ChunkKey& key,
    const FiberletChunkDataset::MaterializedChunk& current,
    const FiberletChunkDataset::MaterializedChunk& chunk);

void replaceFiberletOverlayChunkPair(
    const std::shared_ptr<FiberletChunkDataset>& layer,
    const vc::render::ChunkKey& prefixKey,
    const FiberletChunkDataset::MaterializedChunk& currentPrefix,
    const FiberletChunkDataset::MaterializedChunk& prefix,
    const vc::render::ChunkKey& routeKey,
    const FiberletChunkDataset::MaterializedChunk& currentRoutes,
    const FiberletChunkDataset::MaterializedChunk& routes);

[[nodiscard]] std::shared_ptr<const vc::render::DecodedChunkPayload> decodeFiberletChunkPayload(FiberletStorageChunkKind kind, std::span<const std::byte> bytes);

}  // namespace vc::fiber_tracer
