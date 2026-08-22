#pragma once

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

enum class FiberletStorageProfile : std::uint8_t {
    Float32Cache = 1,
    CompactQuantized = 2,
    CompactDirectionsFixedCost = 3,
};

enum class FiberletStorageChunkKind : std::uint8_t {
    Anchors = 1,
    FiberletPrefix = 2,
    FiberletRoutes = 3,
};

struct FiberletStorageKey {
    // Float and compact-direction profiles: global anchor-cell Z/Y/X.
    // Compact-quantized profile: global quantized base-coordinate Z/Y/X.
    // The profile gives the interpretation.
    std::array<std::int64_t, 3> coordinateZYX{0, 0, 0};
    std::uint8_t variant = 0;

    auto operator<=>(const FiberletStorageKey&) const = default;
};

struct FiberletStorageId {
    FiberletStorageKey first;
    FiberletStorageKey second;

    auto operator<=>(const FiberletStorageId&) const = default;
};

struct DirectedFiberletStorageId {
    FiberletStorageId fiberlet;
    bool reverse = false;

    auto operator<=>(const DirectedFiberletStorageId&) const = default;
};

struct FiberletStoredAnchor {
    FiberletStorageKey key;
    cv::Vec3f positionPredictionXYZ{0.0F, 0.0F, 0.0F};
    cv::Vec3f fittedAxisXYZ{1.0F, 0.0F, 0.0F};
    cv::Vec3f predictionAxisXYZ{0.0F, 0.0F, 0.0F};
    float predictionPresence = 0.0F;
    cv::Vec3f normalXYZ{0.0F, 0.0F, 0.0F};
    bool predictionValid = false;
    bool predictionPresenceValid = false;
    bool normalValid = false;
};

struct FiberletStoredPathCost {
    float invalidPrediction = 0.0F;
    float alignment = 0.0F;
    float isotropicSmoothness = 0.0F;
    float tangentSmoothness = 0.0F;
    float normalSmoothness = 0.0F;

    [[nodiscard]] float total() const noexcept
    {
        return invalidPrediction + alignment + isotropicSmoothness +
            tangentSmoothness + normalSmoothness;
    }
};

struct FiberletStoredPrefix {
    FiberletStorageId id;
    std::uint16_t interiorPointCount = 0;
    std::array<std::int16_t, 2> entryUV{0, 0};
    std::array<std::int16_t, 2> exitUV{0, 0};
    float pathLengthPredictionVoxels = 0.0F;
    FiberletStoredPathCost cost;
    cv::Vec3f firstStepBaseXYZ{0.0F, 0.0F, 0.0F};
    cv::Vec3f lastStepBaseXYZ{0.0F, 0.0F, 0.0F};
};

struct FiberletStoredRoute {
    std::vector<std::array<std::int16_t, 2>> middleUV;
    // Total metric cost per prediction-voxel for every reconstructed geometry
    // segment, decoded from the fixed sqrt-density uint16 representation.
    std::vector<float> segmentCostDensities;
};

inline constexpr float kFiberletStoredCostDensityMaximum = 256.0F;

[[nodiscard]] std::uint16_t encodeFiberletStoredCostDensity(float density);

[[nodiscard]] float decodeFiberletStoredCostDensity(std::uint16_t code);

struct FiberletStorageCodecConfig {
    FiberletStorageProfile profile = FiberletStorageProfile::Float32Cache;
    std::array<std::int32_t, 3> chunkZYX{0, 0, 0};
    std::array<std::uint8_t, 32> datasetFingerprint{};
    std::array<std::int64_t, 3> coordinateOriginZYX{0, 0, 0};
    std::uint8_t coordinateBits = 16;
    std::uint8_t deltaBits = 16;
    std::uint8_t routeCountBits = 16;
    std::uint8_t routeLatticeBits = 16;
    std::uint8_t costBits = 32;
    std::uint32_t positionQuantumBaseVoxels = 0;
    double predictionToBaseScale = 1.0;
};

struct FiberletDecodedAnchors {
    FiberletStorageCodecConfig config;
    std::vector<FiberletStoredAnchor> anchors;
};

struct FiberletDecodedPrefixes {
    FiberletStorageCodecConfig config;
    std::vector<FiberletStoredPrefix> prefixes;
};

struct FiberletDecodedRoutes {
    FiberletStorageCodecConfig config;
    std::vector<FiberletStoredRoute> routes;
};

[[nodiscard]] std::vector<std::byte> serializeFiberletAnchors(
    const FiberletStorageCodecConfig& config,
    std::span<const FiberletStoredAnchor> anchors);

[[nodiscard]] std::vector<std::byte> serializeFiberletPrefixes(
    const FiberletStorageCodecConfig& config,
    std::span<const FiberletStoredPrefix> prefixes);

[[nodiscard]] std::vector<std::byte> serializeFiberletRoutes(
    const FiberletStorageCodecConfig& config,
    std::span<const FiberletStoredRoute> routes);

[[nodiscard]] FiberletDecodedAnchors deserializeFiberletAnchors(std::span<const std::byte> bytes);

[[nodiscard]] FiberletDecodedPrefixes deserializeFiberletPrefixes(std::span<const std::byte> bytes);

[[nodiscard]] FiberletDecodedRoutes deserializeFiberletRoutes(std::span<const std::byte> bytes);

// Decode field compression while retaining the same strict payload envelope.
// Generated ChunkCache instances keep this byte form in their decoded LRU and
// keep the compressed form only in the authoritative sparse dataset.
[[nodiscard]] std::vector<std::byte> materializeFiberletPayload(std::span<const std::byte> bytes);

}  // namespace vc::fiber_tracer
