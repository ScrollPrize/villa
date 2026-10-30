#pragma once

#include "vc/fiber_tracer/FiberletStorage.hpp"
#include "vc_reader/native/memory_cache.hpp"

#include <array>
#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace vc::lasagna { class LasagnaDataset; }

#include <nlohmann/json.hpp>

namespace vc::fiber_tracer
{
struct FiberPredictionGridInfo;

// Bump whenever current anchor or Fiberlet generation can produce different
// authoritative payloads for otherwise identical effective inputs.
inline constexpr std::uint32_t kFiberletGenerationContractVersion = 3;

// One sparse chunk of a fiberlet dataset. Path datasets keep prefixes at level 0
// and routes at level 1 of the same coordinates.
struct FiberletChunkKey {
    int level = 0;
    int iz = 0;
    int iy = 0;
    int ix = 0;

    friend bool operator==(const FiberletChunkKey&, const FiberletChunkKey&) = default;
};

class FiberletChunkPayload
{
public:
    virtual ~FiberletChunkPayload() = default;
};

struct FiberletIncidentPrefix {
    DirectedFiberletStorageId id;
    FiberletStoredPrefix prefix;
};

class FiberletAnchorChunkPayload final : public FiberletChunkPayload
{
public:
    explicit FiberletAnchorChunkPayload(FiberletDecodedAnchors decoded);
    [[nodiscard]] const FiberletStoredAnchor* find(
        const FiberletStorageKey& key) const noexcept;

    FiberletStorageCodecConfig config;
    std::vector<FiberletStoredAnchor> anchors;
};

class FiberletPrefixChunkPayload final : public FiberletChunkPayload
{
public:
    explicit FiberletPrefixChunkPayload(FiberletDecodedPrefixes decoded);
    [[nodiscard]] const FiberletStoredPrefix* find(
        const FiberletStorageId& id) const noexcept;
    [[nodiscard]] std::vector<FiberletIncidentPrefix> incident(
        const FiberletStorageKey& key) const;

    FiberletStorageCodecConfig config;
    std::vector<FiberletStoredPrefix> prefixes;
    // Format-3 owner chunk's complete endpoint table. Not a global anchor list:
    // referenced anchors must not become duplicate seeds in anchorsInChunk().
    std::shared_ptr<const FiberletAnchorChunkPayload> endpoints;

private:
    [[nodiscard]] const FiberletStorageKey& endpoint(
        std::uint32_t encoded) const noexcept;

    std::vector<std::uint32_t> incidentOrder_;
};

class FiberletRouteChunkPayload final : public FiberletChunkPayload
{
public:
    explicit FiberletRouteChunkPayload(FiberletDecodedRoutes decoded);

    FiberletStorageCodecConfig config;
    std::vector<FiberletStoredRoute> routes;
};

class FiberletTraceChunkPayload final : public FiberletChunkPayload {
public:
  explicit FiberletTraceChunkPayload(FiberletDecodedTraces decoded);

  FiberletStorageCodecConfig config;
  std::vector<FiberletStoredTrace> traces;
};

class FiberletPublishedChunkPayload final : public FiberletChunkPayload
{
public:
    explicit FiberletPublishedChunkPayload(FiberletDecodedPublishedChunk decoded);

    FiberletDecodedPublishedChunk decoded;
};

enum class FiberletDatasetKind {
    Anchors,
    Fiberlets,
    Combined,
    Published,
    Traces,
};

enum class FiberletDatasetBuildState {
    Partial,
    Complete,
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
    // Publication lifecycle fields are intentionally excluded from identity.
    FiberletDatasetBuildState buildState = FiberletDatasetBuildState::Partial;
    std::optional<std::uint64_t> expectedChunkCount;
    std::optional<std::string> expectedChunkDigest;
};

// Canonically derive the algorithm and dataset fingerprints from the
// structured metadata. Call this after all effective values are resolved.
void finalizeFiberletDatasetIdentity(FiberletDatasetMetadata& metadata);

[[nodiscard]] std::string fiberletContentHash(
    const std::filesystem::path& path);

void validateFiberletNormalDatasetCompatibility(
    const FiberletDatasetMetadata& metadata,
    const vc::lasagna::LasagnaDataset& normals);
// Validate CLI inputs before constructing storage metadata or generating routes.
void validateFiberletNormalDatasetCompatibility(
    const FiberPredictionGridInfo& grid,
    const vc::lasagna::LasagnaDataset& normals);

// Producer diagnostics do not affect cache compatibility. This also permits
// reopening version-3 caches written before producer toolchains were removed
// from the scientific identity.
[[nodiscard]] bool fiberletDatasetMetadataCompatible(
    const FiberletDatasetMetadata& left,
    const FiberletDatasetMetadata& right);

class FiberletChunkDataset
{
public:
    struct MaterializationStats {
        std::size_t anchorDecodes = 0;
        std::size_t prefixDecodes = 0;
        std::size_t routeDecodes = 0;
    std::size_t traceDecodes = 0;
    };

    static std::shared_ptr<FiberletChunkDataset> createOrOpen(
        std::filesystem::path root,
        const FiberletDatasetMetadata& metadata);
    // Open and validate an existing dataset using its authoritative metadata;
    // callers do not supply a duplicate configuration.
    static std::shared_ptr<FiberletChunkDataset> openExisting(
        std::filesystem::path root);

    [[nodiscard]] const std::filesystem::path& root() const noexcept;
    [[nodiscard]] const FiberletDatasetMetadata& metadata() const noexcept;
    [[nodiscard]] bool datasetComplete() const;
    void configureExpectedChunks(std::span<const FiberletChunkKey> chunks);
    void markComplete();
    void markPartial();
    [[nodiscard]] const std::vector<FiberletChunkKey>& expectedChunks() const noexcept;
    [[nodiscard]] bool isExpectedChunk(const FiberletChunkKey& key) const;
    [[nodiscard]] MaterializationStats materializationStats() const noexcept;
    [[nodiscard]] FiberletStorageCodecConfig codecConfig(FiberletStorageChunkKind kind, const FiberletChunkKey& key) const;
    [[nodiscard]] FiberletChunkKey chunkOwningAnchor(
        const FiberletStorageKey& anchor, int level = 0) const;
    [[nodiscard]] std::filesystem::path chunkPath(FiberletStorageChunkKind kind, const FiberletChunkKey& key) const;
    [[nodiscard]] std::optional<std::vector<std::byte>> readChunk(FiberletStorageChunkKind kind, const FiberletChunkKey& key) const;
    struct MaterializedChunk {
        std::vector<std::byte> bytes;
        std::shared_ptr<const FiberletChunkPayload> payload;
        bool alreadyPublished = false;
    };
    [[nodiscard]] std::optional<MaterializedChunk> readMaterializedChunk(
        FiberletStorageChunkKind kind,
        const FiberletChunkKey& key) const;
    void publishChunk(FiberletStorageChunkKind kind, const FiberletChunkKey& key, std::span<const std::byte> bytes) const;
    void publishMaterializedChunk(
        FiberletStorageChunkKind kind,
        const FiberletChunkKey& key,
        const MaterializedChunk& chunk) const;
    void publishFiberletChunkPair(
        const FiberletChunkKey& prefixKey,
        const MaterializedChunk& prefix,
        const FiberletChunkKey& routeKey,
        const MaterializedChunk& routes) const;
    void publishPublishedChunk(
        const FiberletChunkKey& key,
        std::span<const std::byte> bytes) const;
    [[nodiscard]] std::optional<FiberletDecodedPublishedChunk>
    readPublishedChunk(const FiberletChunkKey& key) const;
    void validateChunk(FiberletStorageChunkKind kind, const FiberletChunkKey& key, std::span<const std::byte> bytes) const;

private:
    FiberletChunkDataset(
        std::filesystem::path root,
        FiberletDatasetMetadata metadata);

    std::filesystem::path root_;
    FiberletDatasetMetadata metadata_;
    std::vector<FiberletChunkKey> expectedChunks_;
    bool expectedChunksConfigured_ = false;
  mutable std::array<std::atomic_size_t, 5> materializationDecodes_{};
};

using FiberletChunkGenerator =
    std::function<FiberletChunkDataset::MaterializedChunk(FiberletStorageChunkKind kind, const FiberletChunkKey& key, const FiberletStorageCodecConfig& config)>;

// Blocking read of one decoded chunk; throws when it cannot be produced.
using FiberletChunkRead =
    std::function<std::shared_ptr<const FiberletChunkPayload>(const FiberletChunkKey&)>;

// All reads keep chunk bytes in `cache` (optional, shared across datasets) and
// decode them on use. Generated reads take chunks from the dataset, generating
// and publishing absent ones. Stored reads treat absent sparse chunks as empty.
[[nodiscard]] FiberletChunkRead fiberletGeneratedChunkRead(
    std::shared_ptr<FiberletChunkDataset> dataset,
    FiberletChunkGenerator generator,
    std::shared_ptr<vcr::MemoryCache> cache = {});

[[nodiscard]] FiberletChunkRead fiberletStoredAnchorRead(
    std::shared_ptr<FiberletChunkDataset> dataset,
    std::shared_ptr<vcr::MemoryCache> cache = {});

[[nodiscard]] FiberletChunkRead fiberletStoredPathRead(
    std::shared_ptr<FiberletChunkDataset> dataset,
    std::shared_ptr<vcr::MemoryCache> cache = {});

[[nodiscard]] std::shared_ptr<const FiberletChunkPayload> decodeFiberletChunkPayload(FiberletStorageChunkKind kind, std::span<const std::byte> bytes);

}  // namespace vc::fiber_tracer
