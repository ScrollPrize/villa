#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/AtomicFile.hpp"
#include "vc/fiber_tracer/FiberletStorage.hpp"
#include "vc/fiber_tracer/FiberletDataset.hpp"
#include "vc/fiber_tracer/FiberletChunkGraph.hpp"
#include "vc/fiber_tracer/FiberletQuantization.hpp"
#include "vc/fiber_tracer/FiberletOnDemand.hpp"

#include <cmath>
#include <atomic>
#include <array>
#include <bit>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <random>
#include <set>

using namespace vc::fiber_tracer;

namespace
{

FiberletStorageCodecConfig floatConfig()
{
    FiberletStorageCodecConfig config;
    config.profile = FiberletStorageProfile::Float32Cache;
    config.chunkZYX = {2, 3, 4};
    config.coordinateOriginZYX = {100, 200, 300};
    config.coordinateBits = 8;
    config.deltaBits = 8;
    config.routeCountBits = 8;
    config.routeLatticeBits = 8;
    config.costBits = 32;
    config.datasetFingerprint[0] = 91;
    return config;
}

FiberletStorageCodecConfig compactConfig()
{
    auto config = floatConfig();
    config.profile = FiberletStorageProfile::CompactQuantized;
    config.costBits = 8;
    config.positionQuantumBaseVoxels = 4;
    config.predictionToBaseScale = 8.0;
    return config;
}

FiberletStorageCodecConfig compactDefaultConfig()
{
    auto config = floatConfig();
    config.profile = FiberletStorageProfile::CompactDirectionsFixedCost;
    config.costBits = 16;
    return config;
}

FiberletStorageKey key(std::int64_t z, std::int64_t y, std::int64_t x, std::uint8_t variant = 0)
{
    return {{z, y, x}, variant};
}

FiberletChunkDataset::MaterializedChunk materialized(FiberletStorageChunkKind kind, std::vector<std::byte> bytes)
{
    FiberletChunkDataset::MaterializedChunk result;
    result.payload = decodeFiberletChunkPayload(kind, bytes);
    result.bytes = std::move(bytes);
    return result;
}

}  // namespace

TEST_CASE("Fiberlet endpoint reconstruction accepts only finite float roundoff")
{
    FiberletStoredAnchor cached;
    cached.predictionAxisXYZ = {0.25F, 0.5F, 0.75F};
    cached.predictionPresence = 0.625F;
    cached.normalXYZ = {0.282699347F, 0.807072759F, 0.518376827F};
    cached.predictionValid = true;
    cached.predictionPresenceValid = true;
    cached.normalValid = true;

    FiberletEndpointScoring reconstructed{
        {cached.predictionAxisXYZ, cached.predictionPresence, true, true},
        {0.282699376F, 0.807072759F, 0.518376887F},
        true};
    CHECK(fiberletEndpointScoringEquivalent(cached, reconstructed));

    reconstructed.normalXYZ = cached.normalXYZ;
    reconstructed.normalXYZ[0] =
        cached.normalXYZ[0] + 8.0F *
            std::numeric_limits<float>::epsilon();
    CHECK(fiberletEndpointScoringEquivalent(cached, reconstructed));
    reconstructed.normalXYZ[0] =
        cached.normalXYZ[0] + 9.0F *
            std::numeric_limits<float>::epsilon();
    CHECK_FALSE(fiberletEndpointScoringEquivalent(cached, reconstructed));

    reconstructed.normalXYZ = cached.normalXYZ;
    reconstructed.normalXYZ[1] -=
        8.0F * std::numeric_limits<float>::epsilon();
    CHECK(fiberletEndpointScoringEquivalent(cached, reconstructed));
    reconstructed.normalXYZ[1] =
        cached.normalXYZ[1] - 9.0F *
            std::numeric_limits<float>::epsilon();
    CHECK_FALSE(fiberletEndpointScoringEquivalent(cached, reconstructed));

    reconstructed.normalXYZ = -cached.normalXYZ;
    CHECK_FALSE(fiberletEndpointScoringEquivalent(cached, reconstructed));

    reconstructed.normalXYZ = cached.normalXYZ;
    reconstructed.normalValid = false;
    CHECK_FALSE(fiberletEndpointScoringEquivalent(cached, reconstructed));

    reconstructed.normalValid = true;
    reconstructed.prediction.presence = -0.0F;
    cached.predictionPresence = 0.0F;
    CHECK(fiberletEndpointScoringEquivalent(cached, reconstructed));
    reconstructed.prediction.presence =
        8.0F * std::numeric_limits<float>::epsilon();
    CHECK(fiberletEndpointScoringEquivalent(cached, reconstructed));
    reconstructed.prediction.presence =
        9.0F * std::numeric_limits<float>::epsilon();
    CHECK_FALSE(fiberletEndpointScoringEquivalent(cached, reconstructed));

    cached.predictionPresence = 2.0F;
    reconstructed.prediction.presence =
        2.0F + 16.0F * std::numeric_limits<float>::epsilon();
    CHECK(fiberletEndpointScoringEquivalent(cached, reconstructed));
    reconstructed.prediction.presence =
        2.0F + 18.0F * std::numeric_limits<float>::epsilon();
    CHECK_FALSE(fiberletEndpointScoringEquivalent(cached, reconstructed));

    reconstructed.prediction.presence =
        std::numeric_limits<float>::infinity();
    CHECK_FALSE(fiberletEndpointScoringEquivalent(cached, reconstructed));
    cached.predictionPresence = reconstructed.prediction.presence;
    CHECK_FALSE(fiberletEndpointScoringEquivalent(cached, reconstructed));
    reconstructed.prediction.presence =
        std::numeric_limits<float>::quiet_NaN();
    cached.predictionPresence = reconstructed.prediction.presence;
    CHECK_FALSE(fiberletEndpointScoringEquivalent(cached, reconstructed));
}

TEST_CASE("Fiberlet scheduled resolution errors preserve key status and cause")
{
    const vc::render::ChunkKey key{0, 427, 139, 187};
    vc::render::ChunkResult resolved;
    resolved.status = vc::render::ChunkStatus::Error;
    resolved.error = "endpoint scoring mismatch";
    CHECK(
        fiberletScheduledResolutionError(key, resolved) ==
        "scheduled fiberlet chunk 0/427/139/187 resolved as error: "
        "endpoint scoring mismatch");

    resolved.status = vc::render::ChunkStatus::Missing;
    resolved.error.clear();
    CHECK(
        fiberletScheduledResolutionError(key, resolved) ==
        "scheduled fiberlet chunk 0/427/139/187 resolved as missing");
}

TEST_CASE("Fiberlet evaluation quantization uses base positions and shared codecs")
{
    FiberPredictionGridInfo grid;
    grid.predictionToBaseScale = 8.0F;
    grid.shapeZYX = {16, 16, 16};
    const auto position = quantizeFiberletPositionForEvaluation({1.2F, 2.3F, 3.6F}, grid, 4);
    CHECK(position[0] == doctest::Approx(1.0));
    CHECK(position[1] == doctest::Approx(2.5));
    CHECK(position[2] == doctest::Approx(3.5));

    const auto fractional = quantizeFiberletPositionForEvaluation(
        {static_cast<float>(8.0625 / 8.0),
         static_cast<float>(8.0624 / 8.0),
         static_cast<float>(8.0626 / 8.0)},
        grid,
        0.125);
    CHECK(fractional[0] == doctest::Approx(8.125 / 8.0));
    CHECK(fractional[1] == doctest::Approx(8.0 / 8.0));
    CHECK(fractional[2] == doctest::Approx(8.125 / 8.0));
    CHECK(fiberletPositionBinCountForEvaluation(512, 0.125) == 4096);
    CHECK(fiberletPositionBinCountForEvaluation(512, 0.0) == 0);
    CHECK_THROWS_AS(
        fiberletPositionBinCountForEvaluation(512, 0.3),
        std::invalid_argument);
    CHECK_THROWS_AS(
        fiberletPositionBinCountForEvaluation(
            512, std::numeric_limits<double>::denorm_min()),
        std::invalid_argument);
    CHECK_THROWS_AS(
        quantizeFiberletPositionForEvaluation(
            {1.0F, 1.0F, 1.0F}, grid, -0.125),
        std::invalid_argument);
    CHECK_THROWS_AS(
        quantizeFiberletPositionForEvaluation(
            {1.0F, 1.0F, 1.0F}, grid,
            std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument);
    CHECK_THROWS_AS(
        quantizeFiberletPositionForEvaluation(
            {1.0F, 1.0F, 1.0F}, grid,
            std::numeric_limits<double>::infinity()),
        std::invalid_argument);
    CHECK_THROWS_AS(
        quantizeFiberletPositionForEvaluation(
            {static_cast<float>(127.95 / 8.0), 1.0F, 1.0F}, grid,
            0.125),
        std::invalid_argument);

    const cv::Vec3f inputDirection{0.25F, -0.4F, 0.881759F};
    const auto direction = quantizeFiberletDirectionForEvaluation(inputDirection);
    CHECK(cv::norm(direction) == doctest::Approx(1.0).epsilon(1.0e-5));
    CHECK(std::abs(direction.dot(inputDirection)) > 0.99F);

    const std::array<float, 4> costs{1.0F, 2.0F, 5.0F, 9.0F};
    const auto decoded = quantizeFiberletCostsForEvaluation(costs, 8);
    REQUIRE(decoded.size() == costs.size());
    const std::array<std::uint32_t, 4> expectedBits{
        0x3f800000U, 0x40004040U, 0x409f7f80U, 0x41100000U};
    for (size_t index = 0; index < decoded.size(); ++index)
        CHECK(std::bit_cast<std::uint32_t>(decoded[index]) == expectedBits[index]);
}

TEST_CASE("Fiberlet sqrt cost-density evaluation uses a fixed global range")
{
    const std::array<float, 3> costs{0.5F, 2.0F, 512.0F};
    const std::array<float, 3> lengths{2.0F, 8.0F, 2.0F};
    const auto decoded = quantizeFiberletCostsForEvaluation(
        costs,
        lengths,
        16,
        FiberletCostQuantizationDomain::SqrtPerPredictionVoxel,
        256.0F);
    REQUIRE(decoded.size() == costs.size());
    const float code = 2048.0F;
    const float decodedDensity = 256.0F *
        (code / 65535.0F) * (code / 65535.0F);
    CHECK(decoded[0] == doctest::Approx(decodedDensity * lengths[0]));
    CHECK(decoded[1] == doctest::Approx(decodedDensity * lengths[1]));
    CHECK(decoded[2] == doctest::Approx(512.0F));

    const std::array<float, 1> isolatedCost{costs[0]};
    const std::array<float, 1> isolatedLength{lengths[0]};
    const auto isolated = quantizeFiberletCostsForEvaluation(
        isolatedCost, isolatedLength, 16,
        FiberletCostQuantizationDomain::SqrtPerPredictionVoxel,
        256.0F);
    CHECK(std::bit_cast<std::uint32_t>(isolated[0]) ==
          std::bit_cast<std::uint32_t>(decoded[0]));
    CHECK_THROWS_AS(
        fiberletCostQuantizationValueForEvaluation(
            1.0F, 0.0F,
            FiberletCostQuantizationDomain::SqrtPerPredictionVoxel,
            256.0F),
        std::invalid_argument);
    CHECK_THROWS_AS(
        fiberletCostQuantizationValueForEvaluation(
            1.0F, std::numeric_limits<float>::infinity(),
            FiberletCostQuantizationDomain::SqrtPerPredictionVoxel,
            256.0F),
        std::invalid_argument);
    CHECK_THROWS_AS(
        fiberletCostQuantizationValueForEvaluation(
            1.0F, 1.0F,
            FiberletCostQuantizationDomain::SqrtPerPredictionVoxel,
            0.0F),
        std::invalid_argument);
}

TEST_CASE("Fiberlet geometry caches ignore replay cost representation")
{
    const FiberletEvaluationQuantization baseline{0, false, 0, 512};
    const FiberletEvaluationQuantization costOnly{0, false, 8, 512};
    const FiberletEvaluationQuantization q4Float{4, true, 0, 512};
    const FiberletEvaluationQuantization q4U8{4, true, 8, 512};
    const FiberletEvaluationQuantization q4U16{4, true, 16, 512};
    const FiberletEvaluationQuantization compactAxisFloat{0, true, 0, 512};
    const FiberletEvaluationQuantization compactAxisU8{0, true, 8, 512};
    const FiberletEvaluationQuantization compactAxisU16{0, true, 16, 512};
    FiberletEvaluationQuantization compactAxisSqrtU16{0, true, 16, 512};
    compactAxisSqrtU16.costDomain =
        FiberletCostQuantizationDomain::SqrtPerPredictionVoxel;
    compactAxisSqrtU16.costDensityMaximum = 256.0F;
    FiberletEvaluationQuantization qOneEighthFloat{
        0.125, true, 0, 512};
    FiberletEvaluationQuantization qOneEighthSqrtU16{
        0.125, true, 16, 512};
    qOneEighthSqrtU16.costDomain =
        FiberletCostQuantizationDomain::SqrtPerPredictionVoxel;
    qOneEighthSqrtU16.costDensityMaximum = 256.0F;

    const FiberletGeometryCacheProfile exact{{0, false}, 0, 512};
    const FiberletGeometryCacheProfile compactAxis{{0, true}, 8, 512};
    const FiberletGeometryCacheProfile legacyQ4U8{{4, true}, 8, 512};
    const FiberletGeometryCacheProfile qOneEighth{{0.125, true}, 8, 512};
    CHECK(fiberletGeometryCacheProfile(baseline) == exact);
    CHECK(fiberletGeometryCacheProfile(costOnly) == exact);
    CHECK(fiberletGeometryCacheProfile(q4Float) == legacyQ4U8);
    CHECK(fiberletGeometryCacheProfile(q4U8) == legacyQ4U8);
    CHECK(fiberletGeometryCacheProfile(q4U16) == legacyQ4U8);
    CHECK(fiberletGeometryCacheProfile(compactAxisFloat) == compactAxis);
    CHECK(fiberletGeometryCacheProfile(compactAxisU8) == compactAxis);
    CHECK(fiberletGeometryCacheProfile(compactAxisU16) == compactAxis);
    CHECK(fiberletGeometryCacheProfile(compactAxisSqrtU16) == compactAxis);
    CHECK(fiberletGeometryCacheProfile(qOneEighthFloat) == qOneEighth);
    CHECK(fiberletGeometryCacheProfile(qOneEighthSqrtU16) == qOneEighth);
    CHECK(qOneEighth != compactAxis);
    CHECK(qOneEighth != legacyQ4U8);
    FiberletEvaluationQuantization invalidQuantum;
    invalidQuantum.positionQuantumBaseVoxels =
        std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(
        fiberletGeometryCacheProfile(invalidQuantum),
        std::invalid_argument);
    invalidQuantum.positionQuantumBaseVoxels = 0.3;
    CHECK_THROWS_AS(
        fiberletGeometryCacheProfile(invalidQuantum),
        std::invalid_argument);
    CHECK(compactAxisFloat.costBits == 0);
    CHECK(compactAxisU8.costBits == 8);
    CHECK(compactAxisU16.costBits == 16);
    CHECK(compactAxisSqrtU16.costDomain ==
          FiberletCostQuantizationDomain::SqrtPerPredictionVoxel);
}

TEST_CASE("Fiberlet replay profiles distinguish the compact default from the exact oracle")
{
    const auto compact = defaultFiberletReplayQuantization(512);
    CHECK(compact.positionQuantumBaseVoxels == 0.0);
    CHECK(compact.compactDirections);
    CHECK(compact.costBits == 16);
    CHECK(compact.storageChunkSideBaseVoxels == 512);
    CHECK(compact.costDomain ==
          FiberletCostQuantizationDomain::SqrtPerPredictionVoxel);
    CHECK(compact.costDensityMaximum == 256.0F);
    CHECK(fiberletGeometryCacheProfile(compact).enabled());

    const auto exact = exactFiberletReplayQuantization(512);
    CHECK(exact.positionQuantumBaseVoxels == 0.0);
    CHECK_FALSE(exact.compactDirections);
    CHECK(exact.costBits == 0);
    CHECK(exact.storageChunkSideBaseVoxels == 512);
    CHECK(exact.costDomain == FiberletCostQuantizationDomain::RawTotal);
    CHECK(exact.costDensityMaximum == 0.0F);
    CHECK_FALSE(exact.enabled());
    CHECK_FALSE(fiberletGeometryCacheProfile(exact).enabled());

    CHECK_THROWS_AS(defaultFiberletReplayQuantization(0),
                    std::invalid_argument);
    CHECK_THROWS_AS(exactFiberletReplayQuantization(-1),
                    std::invalid_argument);
}

TEST_CASE("Fiberlet standard quantization matrix includes compact-axis cost views")
{
    const auto scenarios = standardFiberletQuantizationScenarios();
    REQUIRE(scenarios.size() == 19);
    const std::array<std::string, 4> names{
        "compact_axis", "compact_axis_cost_u8", "compact_axis_cost_u16",
        "compact_axis_cost_sqrt_u16_max256"};
    const std::array<int, 4> costBits{0, 8, 16, 16};
    for (std::size_t index = 0; index < names.size(); ++index) {
        const auto found = std::find_if(
            scenarios.begin(), scenarios.end(), [&](const auto& scenario) {
                return scenario.name == names[index];
            });
        REQUIRE(found != scenarios.end());
        CHECK(found->positionQuantumBaseVoxels == 0);
        CHECK(found->compactAxes);
        CHECK(found->costBits == costBits[index]);
        CHECK(found->costDomain ==
              (names[index].starts_with("compact_axis_cost_sqrt_u16")
                   ? FiberletCostQuantizationDomain::SqrtPerPredictionVoxel
                   : FiberletCostQuantizationDomain::RawTotal));
        if (names[index].starts_with("compact_axis_cost_sqrt_u16"))
            CHECK(found->costDensityMaximum == doctest::Approx(256.0F));
    }
    const auto fractional = std::find_if(
        scenarios.begin(), scenarios.end(), [](const auto& scenario) {
            return scenario.name ==
                "position_q1_8_compact_axis_cost_sqrt_u16_max256";
        });
    REQUIRE(fractional != scenarios.end());
    CHECK(fractional->positionQuantumBaseVoxels == doctest::Approx(0.125));
    CHECK(fractional->compactAxes);
    CHECK(fractional->costBits == 16);
    CHECK(fractional->costDomain ==
          FiberletCostQuantizationDomain::SqrtPerPredictionVoxel);
    CHECK(fractional->costDensityMaximum == doctest::Approx(256.0F));
    CHECK(std::none_of(
        scenarios.begin(), scenarios.end(), [](const auto& scenario) {
            return scenario.name == "combined_q1_axis_cost_u8";
        }));
}

TEST_CASE("Fiberlet storage float anchors round trip exact float bits")
{
    const auto config = floatConfig();
    const std::vector<FiberletStoredAnchor> anchors{
        {key(101, 202, 303), {3.25F, 2.5F, 1.75F}, {0.0F, 0.6F, 0.8F}, {0.3F, 0.4F, 0.8660254F}, 0.625F, {0.0F, 1.0F, 0.0F}, true, true, true},
        {key(101, 202, 303, 1), {3.5F, 2.75F, 1.5F}, {1.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}, 0.125F, {0.0F, 0.0F, 0.0F}, false, true, false},
    };
    const auto bytes = serializeFiberletAnchors(config, anchors);
    const auto decoded = deserializeFiberletAnchors(bytes);
    REQUIRE(decoded.anchors.size() == anchors.size());
    CHECK(decoded.anchors[0].key == anchors[0].key);
    CHECK(decoded.anchors[0].positionPredictionXYZ == anchors[0].positionPredictionXYZ);
    CHECK(decoded.anchors[0].fittedAxisXYZ == anchors[0].fittedAxisXYZ);
    CHECK(decoded.anchors[0].predictionAxisXYZ == anchors[0].predictionAxisXYZ);
    CHECK(decoded.anchors[0].predictionPresence == anchors[0].predictionPresence);
    CHECK(decoded.anchors[0].normalXYZ == anchors[0].normalXYZ);
    CHECK(decoded.anchors[0].predictionValid == anchors[0].predictionValid);
    CHECK(decoded.anchors[0].predictionPresenceValid == anchors[0].predictionPresenceValid);
    CHECK(decoded.anchors[0].normalValid == anchors[0].normalValid);
    CHECK_FALSE(decoded.anchors[1].predictionValid);
    CHECK(decoded.anchors[1].predictionPresenceValid);
    CHECK_FALSE(decoded.anchors[1].normalValid);
    CHECK(serializeFiberletAnchors(config, anchors) == bytes);
}

TEST_CASE("Fiberlet storage compact anchors use quantized keys and compact axes")
{
    const auto config = compactConfig();
    const std::vector<FiberletStoredAnchor> anchors{
        {key(101, 202, 303), {}, {0.25F, -0.4F, 0.881759F}},
    };
    const auto decoded = deserializeFiberletAnchors(serializeFiberletAnchors(config, anchors));
    REQUIRE(decoded.anchors.size() == 1);
    CHECK(decoded.anchors[0].key == anchors[0].key);
    CHECK(decoded.anchors[0].positionPredictionXYZ[0] == doctest::Approx(151.5));
    CHECK(decoded.anchors[0].positionPredictionXYZ[1] == doctest::Approx(101.0));
    CHECK(decoded.anchors[0].positionPredictionXYZ[2] == doctest::Approx(50.5));
    CHECK(std::abs(decoded.anchors[0].fittedAxisXYZ.dot(anchors[0].fittedAxisXYZ)) > 0.99F);
}

TEST_CASE("Fiberlet compact default keeps float positions and fixed uint16 costs")
{
    const auto config = compactDefaultConfig();
    const std::vector<FiberletStoredAnchor> anchors{
        {key(101, 202, 303), {3.25F, 2.5F, 1.75F},
            {0.25F, -0.4F, 0.881759F}, {0.3F, 0.4F, 0.8660254F},
            0.625F, {0.0F, 1.0F, 0.0F}, true, true, true},
    };
    const auto decodedAnchors =
        deserializeFiberletAnchors(serializeFiberletAnchors(config, anchors));
    REQUIRE(decodedAnchors.anchors.size() == 1);
    CHECK(decodedAnchors.anchors[0].key == anchors[0].key);
    CHECK(decodedAnchors.anchors[0].positionPredictionXYZ ==
          anchors[0].positionPredictionXYZ);
    CHECK(std::abs(decodedAnchors.anchors[0].fittedAxisXYZ.dot(
              anchors[0].fittedAxisXYZ)) > 0.99F);

    const std::vector<FiberletStoredPrefix> prefixes{
        {.id = {key(101, 202, 303), key(102, 203, 304)},
         .interiorPointCount = 4,
         .entryUV = {-1, 2},
         .exitUV = {3, -4},
         .pathLengthPredictionVoxels = 8.0F,
         .cost = {0.25F, 0.5F, 0.375F, 0.625F, 0.25F},
         .firstStepBaseXYZ = {1.0F, 0.25F, 0.0F},
         .lastStepBaseXYZ = {0.75F, -0.25F, 0.0F}},
    };
    const auto decodedPrefixes = deserializeFiberletPrefixes(
        serializeFiberletPrefixes(config, prefixes));
    REQUIRE(decodedPrefixes.prefixes.size() == 1);
    const float encoded = std::round(
        std::sqrt((2.0F / 8.0F) / 256.0F) * 65535.0F);
    const float expected =
        256.0F * (encoded / 65535.0F) * (encoded / 65535.0F) * 8.0F;
    CHECK(decodedPrefixes.prefixes[0].cost.total() ==
          doctest::Approx(expected));
}

TEST_CASE("Fiberlet storage prefixes and independently cached routes round trip")
{
    auto config = floatConfig();
    const std::vector<FiberletStoredPrefix> prefixes{
        {.id = {key(101, 202, 303), key(102, 203, 304)},
         .interiorPointCount = 4,
         .entryUV = {-1, 2},
         .exitUV = {3, -4},
         .pathLengthPredictionVoxels = 7.5F,
         .cost = {0.25F, 0.5F, 0.375F, 0.625F, 0.5F},
         .firstStepBaseXYZ = {1.0F, 0.25F, 0.0F},
         .lastStepBaseXYZ = {0.75F, -0.25F, 0.0F}},
        {.id = {key(101, 202, 303), key(104, 204, 305)},
         .interiorPointCount = 2,
         .entryUV = {0, 1},
         .exitUV = {0, 1},
         .pathLengthPredictionVoxels = 9.0F,
         .cost = {1.0F, 2.0F, 1.5F, 2.5F, 1.5F},
         .firstStepBaseXYZ = {1.0F, 0.0F, 0.25F},
         .lastStepBaseXYZ = {1.0F, 0.0F, -0.25F}},
    };
    const std::vector<FiberletStoredRoute> routes{
        {.middleUV = {{0, 1}, {1, 1}},
         .segmentCostDensities = {0.25F, 0.5F, 0.75F, 1.0F, 1.25F}},
        {.middleUV = {}, .segmentCostDensities = {2.0F, 3.0F, 4.0F}},
    };
    const auto decodedPrefixes = deserializeFiberletPrefixes(serializeFiberletPrefixes(config, prefixes));
    const auto decodedRoutes = deserializeFiberletRoutes(serializeFiberletRoutes(config, routes));
    REQUIRE(decodedPrefixes.prefixes.size() == 2);
    CHECK(decodedPrefixes.prefixes[0].id == prefixes[0].id);
    CHECK(decodedPrefixes.prefixes[0].entryUV == prefixes[0].entryUV);
    CHECK(decodedPrefixes.prefixes[0].cost.invalidPrediction == prefixes[0].cost.invalidPrediction);
    CHECK(decodedPrefixes.prefixes[0].cost.alignment == prefixes[0].cost.alignment);
    CHECK(decodedPrefixes.prefixes[0].cost.isotropicSmoothness == prefixes[0].cost.isotropicSmoothness);
    CHECK(decodedPrefixes.prefixes[0].cost.tangentSmoothness == prefixes[0].cost.tangentSmoothness);
    CHECK(decodedPrefixes.prefixes[0].cost.normalSmoothness == prefixes[0].cost.normalSmoothness);
    CHECK(decodedPrefixes.prefixes[0].firstStepBaseXYZ == prefixes[0].firstStepBaseXYZ);
    CHECK(decodedPrefixes.prefixes[0].lastStepBaseXYZ == prefixes[0].lastStepBaseXYZ);
    REQUIRE(decodedRoutes.routes.size() == 2);
    CHECK(decodedRoutes.routes[0].middleUV == routes[0].middleUV);
    CHECK(decodedRoutes.routes[1].middleUV.empty());
    REQUIRE(decodedRoutes.routes[0].segmentCostDensities.size() == 5);
    for (size_t index = 0; index < routes[0].segmentCostDensities.size(); ++index) {
        CHECK(decodedRoutes.routes[0].segmentCostDensities[index] ==
              doctest::Approx(routes[0].segmentCostDensities[index]).epsilon(2.0e-4));
    }
}

TEST_CASE("Fiberlet stored cost density uses fixed sqrt uint16 quantization")
{
    for (const float density : {0.0F, 0.25F, 1.0F, 16.0F, 255.0F, 256.0F}) {
        const auto code = encodeFiberletStoredCostDensity(density);
        const float decoded = decodeFiberletStoredCostDensity(code);
        CHECK(decoded == doctest::Approx(density).epsilon(5.0e-4));
    }
    CHECK(encodeFiberletStoredCostDensity(0.0F) == 0);
    CHECK(encodeFiberletStoredCostDensity(256.0F) ==
          std::numeric_limits<std::uint16_t>::max());
    CHECK(encodeFiberletStoredCostDensity(1000.0F) ==
          std::numeric_limits<std::uint16_t>::max());
    CHECK(decodeFiberletStoredCostDensity(
              std::numeric_limits<std::uint16_t>::max()) ==
          doctest::Approx(256.0F));
    CHECK_THROWS_AS(
        encodeFiberletStoredCostDensity(-1.0F), std::invalid_argument);
    CHECK_THROWS_AS(
        encodeFiberletStoredCostDensity(
            std::numeric_limits<float>::infinity()),
        std::invalid_argument);
}

TEST_CASE("Fiberlet route reconstruction restores unoriented endpoint axes")
{
    FiberletPathConfig config;
    config.longitudinalStepPredictionVoxels = 2.0F;
    const std::vector<std::array<std::int16_t, 2>> lattice{{0, 0}, {0, 0}, {0, 0}};
    const auto points = reconstructFiberletRoutePoints({0, 0, 0}, {-1, 0, 0}, {8, 0, 0}, {-1, 0, 0}, lattice, config);
    REQUIRE(points.size() == 5);
    CHECK(points.front() == cv::Vec3f{0, 0, 0});
    CHECK(points.back() == cv::Vec3f{8, 0, 0});
    CHECK(points[1][0] == doctest::Approx(2.0F));
}

TEST_CASE("Fiberlet endpoint steps exactly match full route reconstruction")
{
    FiberletPathConfig config;
    config.longitudinalStepPredictionVoxels = 2.0F;
    const cv::Vec3f firstPosition{1.0F, 2.0F, 3.0F};
    const cv::Vec3f axis{1.0F, 0.0F, 0.0F};
    for (const auto lattice : std::vector<std::vector<std::array<std::int16_t, 2>>>{{}, {{1, -1}}, {{1, -1}, {2, 0}, {-1, 1}}}) {
        const cv::Vec3f secondPosition = firstPosition + cv::Vec3f{2.0F * static_cast<float>(lattice.size() + 1), 0.0F, 0.0F};
        const std::array<std::int16_t, 2> entry = lattice.empty() ? std::array<std::int16_t, 2>{} : lattice.front();
        const std::array<std::int16_t, 2> exit = lattice.empty() ? std::array<std::int16_t, 2>{} : lattice.back();
        const auto endpoints = reconstructFiberletRouteEndpointSteps(firstPosition, axis, secondPosition, axis, lattice.size(), entry, exit, config);
        const auto points = reconstructFiberletRoutePoints(firstPosition, axis, secondPosition, axis, lattice, config);
        REQUIRE(points.size() >= 2);
        CHECK(endpoints.firstPredictionXYZ == points[1] - points[0]);
        CHECK(endpoints.lastPredictionXYZ == points.back() - points[points.size() - 2]);
        CHECK(-endpoints.lastPredictionXYZ == points[points.size() - 2] - points.back());
        CHECK(-endpoints.firstPredictionXYZ == points[0] - points[1]);
    }
}

TEST_CASE("Fiberlet storage compact cost is decoded from the authoritative chunk range")
{
    auto config = compactConfig();
    const std::vector<FiberletStoredPrefix> prefixes{
        {.id = {key(101, 202, 303), key(102, 203, 304)},
         .pathLengthPredictionVoxels = 2.0F,
         .cost = {0, 10.0F},
         .firstStepBaseXYZ = {1, 0, 0},
         .lastStepBaseXYZ = {1, 0, 0}},
        {.id = {key(101, 202, 303), key(104, 204, 305)},
         .pathLengthPredictionVoxels = 3.0F,
         .cost = {0, 20.0F},
         .firstStepBaseXYZ = {1, 0, 0},
         .lastStepBaseXYZ = {1, 0, 0}},
    };
    const auto decoded = deserializeFiberletPrefixes(serializeFiberletPrefixes(config, prefixes));
    CHECK(decoded.prefixes[0].cost.total() == doctest::Approx(10.0F));
    CHECK(decoded.prefixes[1].cost.total() == doctest::Approx(20.0F));
}

TEST_CASE("Fiberlet storage rejects corruption and noncanonical input")
{
    const auto config = floatConfig();
    std::vector<FiberletStoredAnchor> anchors{{key(101, 202, 303), {1, 2, 3}, {1, 0, 0}}};
    auto bytes = serializeFiberletAnchors(config, anchors);
    auto oldMagic = bytes;
    oldMagic[6] = std::byte{'1'};
    CHECK_THROWS_AS(deserializeFiberletAnchors(oldMagic), std::invalid_argument);
    bytes.back() ^= std::byte{1};
    CHECK_THROWS_AS(deserializeFiberletAnchors(bytes), std::invalid_argument);

    anchors.push_back(anchors.front());
    CHECK_THROWS_AS(serializeFiberletAnchors(config, anchors), std::invalid_argument);
}

TEST_CASE("Fiberlet sparse dataset generates, publishes, and reuses opaque chunks")
{
    std::mt19937_64 random(std::random_device{}());
    const auto root = std::filesystem::temp_directory_path() / ("vc_fiberlet_dataset_" + std::to_string(random()));
    FiberletDatasetMetadata metadata;
    metadata.kind = FiberletDatasetKind::Anchors;
    metadata.profile = FiberletStorageProfile::Float32Cache;
    metadata.chunkGridShapeZYX = {2, 2, 2};
    metadata.coordinateUnitsPerChunkZYX = {8, 8, 8};
    metadata.maximumEndpointReachCoordinateUnitsZYX = {4, 4, 4};
    metadata.coordinateBits = 8;
    metadata.deltaBits = 8;
    metadata.routeCountBits = 8;
    metadata.routeLatticeBits = 8;
    metadata.costBits = 32;
    finalizeFiberletDatasetIdentity(metadata);
    auto dataset = FiberletChunkDataset::createOrOpen(root, metadata);
    std::atomic<int> generated{0};
    std::atomic<int> generatedResolutions{0};
    auto cache = createGeneratedFiberletChunkCache(
        dataset,
        [&](FiberletStorageChunkKind kind, const vc::render::ChunkKey&, const FiberletStorageCodecConfig& config) {
            ++generated;
            CHECK(kind == FiberletStorageChunkKind::Anchors);
            const auto origin = config.coordinateOriginZYX;
            const std::vector<FiberletStoredAnchor> anchors{{key(origin[0], origin[1], origin[2]), {1, 2, 3}, {1, 0, 0}}};
            return materialized(kind, serializeFiberletAnchors(config, anchors));
        },
        {},
        [&](FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, vc::render::ChunkFetchStatus status) {
            CHECK(kind == FiberletStorageChunkKind::Anchors);
            CHECK(key.level == 0);
            CHECK(key.iz == 1);
            CHECK(key.iy == 0);
            CHECK(key.ix == 1);
            CHECK(status == vc::render::ChunkFetchStatus::Found);
            ++generatedResolutions;
            throw std::runtime_error("observer failure must be isolated");
        });
    auto first = cache->getChunkBlocking(0, 1, 0, 1);
    REQUIRE(first.status == vc::render::ChunkStatus::Data);
    CHECK(generated.load() == 1);
    CHECK(generatedResolutions.load() == 1);
    cache.reset();

    auto reopened = FiberletChunkDataset::createOrOpen(root, metadata);
    std::atomic<int> persistedResolutions{0};
    auto secondCache = createGeneratedFiberletChunkCache(
        reopened,
        [&](FiberletStorageChunkKind, const vc::render::ChunkKey&, const FiberletStorageCodecConfig&) -> FiberletChunkDataset::MaterializedChunk {
            ++generated;
            throw std::runtime_error("existing chunk should have been reused");
        },
        {},
        [&](FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, vc::render::ChunkFetchStatus status) {
            CHECK(kind == FiberletStorageChunkKind::Anchors);
            CHECK(key.level == 0);
            CHECK(key.iz == 1);
            CHECK(key.iy == 0);
            CHECK(key.ix == 1);
            CHECK(status == vc::render::ChunkFetchStatus::Found);
            ++persistedResolutions;
        });
    auto second = secondCache->getChunkBlocking(0, 1, 0, 1);
    REQUIRE(second.status == vc::render::ChunkStatus::Data);
    CHECK(generated.load() == 1);
    CHECK(persistedResolutions.load() == 1);
    const auto firstPayload = std::dynamic_pointer_cast<const FiberletAnchorChunkPayload>(first.payload);
    const auto secondPayload = std::dynamic_pointer_cast<const FiberletAnchorChunkPayload>(second.payload);
    REQUIRE(firstPayload);
    REQUIRE(secondPayload);
    REQUIRE(secondPayload->anchors.size() == firstPayload->anchors.size());
    CHECK(secondPayload->anchors.front().key == firstPayload->anchors.front().key);
    CHECK(secondPayload->anchors.front().positionPredictionXYZ == firstPayload->anchors.front().positionPredictionXYZ);
    CHECK(secondPayload->anchors.front().fittedAxisXYZ == firstPayload->anchors.front().fittedAxisXYZ);

    const auto selfDescribing = FiberletChunkDataset::openExisting(root);
    CHECK(selfDescribing->metadata().algorithmFingerprint == metadata.algorithmFingerprint);
    CHECK(selfDescribing->metadata().datasetFingerprint == metadata.datasetFingerprint);

    std::ifstream attributesInput(root / ".zattrs");
    auto attributes = nlohmann::json::parse(attributesInput);
    attributesInput.close();
    attributes["unexpected"] = true;
    {
        std::ofstream output(root / ".zattrs", std::ios::binary | std::ios::trunc);
        output << attributes.dump(2) << '\n';
    }
    CHECK_THROWS_AS(FiberletChunkDataset::createOrOpen(root, metadata), std::invalid_argument);
    std::filesystem::remove_all(root);
}

TEST_CASE("Generated sparse empty chunks remain absent on disk")
{
    std::mt19937_64 random(std::random_device{}());
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_fiberlet_generated_sparse_empty_" + std::to_string(random()));
    FiberletDatasetMetadata metadata;
    metadata.kind = FiberletDatasetKind::Anchors;
    metadata.profile = FiberletStorageProfile::Float32Cache;
    metadata.chunkGridShapeZYX = {2, 2, 2};
    metadata.coordinateUnitsPerChunkZYX = {8, 8, 8};
    metadata.maximumEndpointReachCoordinateUnitsZYX = {4, 4, 4};
    metadata.coordinateBits = 8;
    metadata.deltaBits = 8;
    metadata.routeCountBits = 8;
    metadata.routeLatticeBits = 8;
    metadata.costBits = 32;
    finalizeFiberletDatasetIdentity(metadata);
    auto dataset = FiberletChunkDataset::createOrOpen(root, metadata);
    const vc::render::ChunkKey key{0, 1, 0, 1};
    auto cache = createGeneratedFiberletChunkCache(
        dataset,
        [](FiberletStorageChunkKind kind, const vc::render::ChunkKey&,
           const FiberletStorageCodecConfig& config) {
            auto chunk = materialized(
                kind, serializeFiberletAnchors(config, {}));
            chunk.alreadyPublished = true;
            return chunk;
        });

    const auto fetched = cache->getChunkBlocking(
        key.level, key.iz, key.iy, key.ix);
    REQUIRE(fetched.status == vc::render::ChunkStatus::Data);
    const auto payload = std::dynamic_pointer_cast<
        const FiberletAnchorChunkPayload>(fetched.payload);
    REQUIRE(payload);
    CHECK(payload->anchors.empty());
    CHECK_FALSE(std::filesystem::exists(
        dataset->chunkPath(FiberletStorageChunkKind::Anchors, key)));

    cache->cancelPendingAndWait();
    std::filesystem::remove_all(root);
}

TEST_CASE("Fiberlet dataset identity is path-independent and source-sensitive")
{
    FiberletDatasetMetadata first;
    first.kind = FiberletDatasetKind::Combined;
    first.profile = FiberletStorageProfile::CompactDirectionsFixedCost;
    first.chunkGridShapeZYX = {2, 3, 4};
    first.coordinateUnitsPerChunkZYX = {8, 8, 8};
    first.maximumEndpointReachCoordinateUnitsZYX = {4, 4, 4};
    first.sources = {
        {"source_volume", {{"sample_id", "PHerc0001"}, {"volume_id", "20260101000000"}}},
        {"fiber_prediction", {{"run_uuid", "fiber-run"}, {"manifest_sha256", "fiber-sha"}}},
        {"normal_prediction", {{"run_uuid", "normal-run"}, {"manifest_sha256", "normal-sha"}}},
    };
    first.processing = {
        {"anchors", {{"cell_size_prediction", 8}, {"threshold", 0.5}}},
        {"storage", {{"chunk_side_base", 512}}},
    };
    finalizeFiberletDatasetIdentity(first);

    auto relocated = first;
    finalizeFiberletDatasetIdentity(relocated);
    CHECK(relocated.algorithmFingerprint == first.algorithmFingerprint);
    CHECK(relocated.datasetFingerprint == first.datasetFingerprint);

    auto otherSource = first;
    otherSource.sources["fiber_prediction"]["run_uuid"] = "other-fiber-run";
    finalizeFiberletDatasetIdentity(otherSource);
    CHECK(otherSource.algorithmFingerprint == first.algorithmFingerprint);
    CHECK(otherSource.datasetFingerprint != first.datasetFingerprint);
}

TEST_CASE("Fiberlet generation contract changes invalidate persisted chunks")
{
    CHECK(kFiberletGenerationContractVersion == 3);
    for (const auto kind : {FiberletDatasetKind::Anchors,
                            FiberletDatasetKind::Fiberlets,
                            FiberletDatasetKind::Combined}) {
        FiberletDatasetMetadata previous;
        previous.kind = kind;
        previous.profile = FiberletStorageProfile::Float32Cache;
        previous.chunkGridShapeZYX = {2, 3, 4};
        previous.coordinateUnitsPerChunkZYX = {4, 4, 4};
        previous.maximumEndpointReachCoordinateUnitsZYX = {4, 4, 4};
        previous.sources = {
            {"fiber_prediction", {{"manifest_hash", "same"}}}};
        previous.processing = {{"contract_version", 2}};
        finalizeFiberletDatasetIdentity(previous);

        auto current = previous;
        current.processing["contract_version"] =
            kFiberletGenerationContractVersion;
        finalizeFiberletDatasetIdentity(current);

        CHECK(current.algorithmFingerprint != previous.algorithmFingerprint);
        CHECK(current.datasetFingerprint != previous.datasetFingerprint);

        current.processing["producer_toolchain"] = {
            {"compiler_id", "GNU"},
            {"compiler_version", "16.0"},
            {"build_config", "Release"}};
        finalizeFiberletDatasetIdentity(current);
        auto otherToolchain = current;
        otherToolchain.processing["producer_toolchain"]["compiler_id"] =
            "Clang";
        finalizeFiberletDatasetIdentity(otherToolchain);
        CHECK(otherToolchain.algorithmFingerprint !=
              current.algorithmFingerprint);
        CHECK(otherToolchain.datasetFingerprint !=
              current.datasetFingerprint);
    }

    const auto root = std::filesystem::temp_directory_path() /
        ("vc_fiberlet_contract_" +
         std::to_string(std::mt19937_64{std::random_device{}()}()));
    FiberletDatasetMetadata previous;
    previous.kind = FiberletDatasetKind::Anchors;
    previous.profile = FiberletStorageProfile::Float32Cache;
    previous.chunkGridShapeZYX = {1, 1, 1};
    previous.coordinateUnitsPerChunkZYX = {4, 4, 4};
    previous.maximumEndpointReachCoordinateUnitsZYX = {4, 4, 4};
    previous.processing = {{"contract_version", 2}};
    finalizeFiberletDatasetIdentity(previous);
    auto oldDataset = FiberletChunkDataset::createOrOpen(root, previous);
    oldDataset.reset();
    const auto marker = root / "old-v2-marker";
    std::ofstream(marker) << "untouched";

    auto current = previous;
    current.processing["contract_version"] =
        kFiberletGenerationContractVersion;
    finalizeFiberletDatasetIdentity(current);
    CHECK_THROWS_AS(
        FiberletChunkDataset::createOrOpen(root, current),
        std::invalid_argument);
    CHECK(std::filesystem::exists(marker));
    CHECK(FiberletChunkDataset::openExisting(root)->metadata()
              .algorithmFingerprint == previous.algorithmFingerprint);
    std::filesystem::remove_all(root);
}

TEST_CASE("Fiberlet single-cell tube test agrees with the canonical selector")
{
    FiberPredictionGridInfo grid;
    grid.shapeZYX = {12, 12, 12};
    grid.predictionToBaseScale = 2.0F;
    const std::vector<cv::Vec3d> reference{{-2.0, 7.0, 7.0}, {26.0, 7.0, 7.0}};
    const double radius = 1.5;
    const int cellSide = 4;
    const auto selected = fiberAnchorCellsNearPolyline(reference, radius, grid, cellSide);
    const std::set<std::array<size_t, 3>> selectedSet(selected.begin(), selected.end());
    for (size_t z = 0; z < 3; ++z) {
        for (size_t y = 0; y < 3; ++y) {
            for (size_t x = 0; x < 3; ++x) {
                const std::array<size_t, 3> cell{z, y, x};
                CHECK(fiberAnchorCellIntersectsPolylineTube(cell, reference, radius, grid, cellSide) == selectedSet.contains(cell));
            }
        }
    }
}

TEST_CASE("Fiberlet prefix and routes become visible only as a complete payload pair")
{
    std::mt19937_64 random(std::random_device{}());
    const auto root = std::filesystem::temp_directory_path() / ("vc_fiberlet_pair_" + std::to_string(random()));
    FiberletDatasetMetadata metadata;
    metadata.kind = FiberletDatasetKind::Fiberlets;
    metadata.profile = FiberletStorageProfile::Float32Cache;
    metadata.chunkGridShapeZYX = {1, 1, 1};
    metadata.coordinateUnitsPerChunkZYX = {8, 8, 8};
    metadata.maximumEndpointReachCoordinateUnitsZYX = {4, 4, 4};
    metadata.coordinateBits = 8;
    metadata.deltaBits = 8;
    metadata.routeCountBits = 8;
    metadata.routeLatticeBits = 8;
    metadata.costBits = 32;
    finalizeFiberletDatasetIdentity(metadata);
    auto dataset = FiberletChunkDataset::createOrOpen(root, metadata);
    const vc::render::ChunkKey prefixKey{0, 0, 0, 0};
    const vc::render::ChunkKey routeKey{1, 0, 0, 0};
    const auto prefix = serializeFiberletPrefixes(dataset->codecConfig(FiberletStorageChunkKind::FiberletPrefix, prefixKey), {});
    const auto routes = serializeFiberletRoutes(dataset->codecConfig(FiberletStorageChunkKind::FiberletRoutes, routeKey), {});
    dataset->publishChunk(FiberletStorageChunkKind::FiberletPrefix, prefixKey, prefix);
    CHECK_FALSE(dataset->readChunk(FiberletStorageChunkKind::FiberletPrefix, prefixKey).has_value());
    dataset->publishChunk(FiberletStorageChunkKind::FiberletRoutes, routeKey, routes);
    CHECK(dataset->readChunk(FiberletStorageChunkKind::FiberletPrefix, prefixKey).has_value());
    CHECK(dataset->readChunk(FiberletStorageChunkKind::FiberletRoutes, routeKey).has_value());
    std::filesystem::remove_all(root);
}

TEST_CASE("Fiberlet sparse output selection conservatively covers nonempty presence chunks")
{
    FiberPresenceChunkScanReport presence;
    presence.shapeZYX = {9, 9, 9};
    presence.chunksZYX = {4, 4, 4};
    presence.chunkGridShapeZYX = {3, 3, 3};
    presence.nonemptyChunksZYX = {{0, 0, 0}, {1, 1, 1}, {2, 2, 2}};
    FiberletDatasetMetadata output;
    output.chunkGridShapeZYX = {3, 3, 3};
    output.coordinateUnitsPerChunkZYX = {2, 2, 2};

    const auto selected = fiberletOutputChunksForNonemptyPresence(presence, output, 2);
    REQUIRE(selected.size() == 3);
    CHECK((std::array{selected[0].iz, selected[0].iy, selected[0].ix} == std::array{0, 0, 0}));
    CHECK((std::array{selected[1].iz, selected[1].iy, selected[1].ix} == std::array{1, 1, 1}));
    CHECK((std::array{selected[2].iz, selected[2].iy, selected[2].ix} == std::array{2, 2, 2}));
}

TEST_CASE("Whole-volume scheduler prioritizes ready fiberlets within a Z frontier")
{
    const vc::render::ChunkKey outputA{0, 3, 0, 0};
    const vc::render::ChunkKey outputB{0, 3, 0, 1};
    const vc::render::ChunkKey outputC{0, 4, 0, 0};
    const vc::render::ChunkKey anchorA{0, 2, 0, 0};
    const vc::render::ChunkKey anchorB{0, 3, 0, 0};
    const vc::render::ChunkKey anchorC{0, 4, 0, 0};
    FiberletPreprocessSchedule schedule(
        {outputA, outputB, outputC},
        {{anchorA, anchorB}, {anchorB, anchorB}, {anchorC}},
        {},
        std::array{anchorA, anchorB});

    CHECK(schedule.anchorTotal() == 3);
    CHECK(schedule.anchorsCompleted() == 2);
    CHECK(schedule.outputTotal() == 3);
    CHECK(schedule.currentOutputZ() == 3);

    const auto first = schedule.takeNext();
    REQUIRE(first);
    CHECK(first->kind == FiberletPreprocessWorkKind::Fiberlet);
    CHECK(first->key == outputA);
    const auto second = schedule.takeNext();
    REQUIRE(second);
    CHECK(second->kind == FiberletPreprocessWorkKind::Fiberlet);
    CHECK(second->key == outputB);
    const auto third = schedule.takeNext();
    REQUIRE(third);
    CHECK(third->kind == FiberletPreprocessWorkKind::Anchor);
    CHECK(third->key == anchorC);

    schedule.complete(*third);
    CHECK_FALSE(schedule.takeNext().has_value());
    schedule.complete(*second);
    CHECK_FALSE(schedule.takeNext().has_value());
    schedule.complete(*first);
    CHECK(schedule.currentOutputZ() == 4);
    const auto fourth = schedule.takeNext();
    REQUIRE(fourth);
    CHECK(fourth->kind == FiberletPreprocessWorkKind::Fiberlet);
    CHECK(fourth->key == outputC);
    schedule.complete(*fourth);
    CHECK(schedule.done());
}

TEST_CASE("Whole-volume scheduler retains missing anchor-cache work for resumed outputs")
{
    const vc::render::ChunkKey output{0, 8, 2, 1};
    const vc::render::ChunkKey anchor{0, 8, 2, 1};
    FiberletPreprocessSchedule schedule({output}, {{anchor}}, std::array{output}, {});
    CHECK(schedule.outputsCompleted() == 1);
    CHECK_FALSE(schedule.currentOutputZ().has_value());
    const auto work = schedule.takeNext();
    REQUIRE(work);
    CHECK(work->kind == FiberletPreprocessWorkKind::Anchor);
    CHECK(work->key == anchor);
    schedule.complete(*work);
    CHECK(schedule.done());
}

TEST_CASE("Whole-volume scheduler prioritizes anchors that unblock incomplete outputs")
{
    const vc::render::ChunkKey completedOutput{0, 7, 0, 0};
    const vc::render::ChunkKey pendingOutput{0, 8, 0, 0};
    const vc::render::ChunkKey cacheRepairAnchor{0, 7, 0, 0};
    const vc::render::ChunkKey blockingAnchor{0, 8, 0, 0};
    FiberletPreprocessSchedule schedule(
        {completedOutput, pendingOutput},
        {{cacheRepairAnchor}, {blockingAnchor}},
        std::array{completedOutput},
        {});

    const auto first = schedule.takeNext();
    REQUIRE(first);
    CHECK(first->kind == FiberletPreprocessWorkKind::Anchor);
    CHECK(first->key == blockingAnchor);
    schedule.complete(*first);
    const auto second = schedule.takeNext();
    REQUIRE(second);
    CHECK(second->kind == FiberletPreprocessWorkKind::Fiberlet);
    CHECK(second->key == pendingOutput);
    schedule.complete(*second);
    const auto third = schedule.takeNext();
    REQUIRE(third);
    CHECK(third->kind == FiberletPreprocessWorkKind::Anchor);
    CHECK(third->key == cacheRepairAnchor);
    schedule.complete(*third);
    CHECK(schedule.done());
}

TEST_CASE("Combined fiberlet dataset exposes complete sparse graph facets")
{
    std::mt19937_64 random(std::random_device{}());
    const auto root = std::filesystem::temp_directory_path() / ("vc_fiberlet_combined_" + std::to_string(random()));
    FiberletDatasetMetadata metadata;
    metadata.kind = FiberletDatasetKind::Combined;
    metadata.profile = FiberletStorageProfile::CompactDirectionsFixedCost;
    metadata.chunkGridShapeZYX = {2, 2, 2};
    metadata.coordinateUnitsPerChunkZYX = {8, 8, 8};
    metadata.maximumEndpointReachCoordinateUnitsZYX = {4, 4, 4};
    metadata.coordinateBits = 8;
    metadata.deltaBits = 8;
    metadata.routeCountBits = 8;
    metadata.routeLatticeBits = 8;
    metadata.costBits = 16;
    finalizeFiberletDatasetIdentity(metadata);
    auto dataset = FiberletChunkDataset::createOrOpen(root, metadata);
    const vc::render::ChunkKey owner{0, 0, 0, 0};
    dataset->configureExpectedChunks(std::span<const vc::render::ChunkKey>(&owner, 1));
    auto incompleteAnchorCache = createStoredFiberletAnchorChunkCache(dataset);
    auto incompletePathCache = createStoredFiberletPathChunkCache(dataset);
    incompletePathCache->cancelPendingAndWait();
    incompleteAnchorCache->cancelPendingAndWait();

    const auto first = key(6, 7, 7);
    const auto second = key(7, 7, 7);
    const std::vector<FiberletStoredAnchor> anchors{{first, {1, 2, 3}, {1, 0, 0}}, {second, {2, 2, 3}, {1, 0, 0}}};
    dataset->publishChunk(FiberletStorageChunkKind::Anchors, owner, serializeFiberletAnchors(dataset->codecConfig(FiberletStorageChunkKind::Anchors, owner), anchors));
    CHECK_FALSE(dataset->datasetComplete());
    const std::vector<FiberletStoredPrefix> prefixes{
        {.id = {first, second},
         .pathLengthPredictionVoxels = 1.0F,
         .cost = {0.0F, 1.0F, 0.0F, 0.0F, 0.0F},
         .firstStepBaseXYZ = {1, 0, 0},
         .lastStepBaseXYZ = {1, 0, 0}}};
    const std::vector<FiberletStoredRoute> routes{{
        .middleUV = {},
        .segmentCostDensities = {1.0F},
    }};
    const vc::render::ChunkKey routeKey{1, 0, 0, 0};
    const auto prefixBytes = serializeFiberletPrefixes(dataset->codecConfig(FiberletStorageChunkKind::FiberletPrefix, owner), prefixes);
    dataset->publishChunk(FiberletStorageChunkKind::FiberletPrefix, owner, prefixBytes);
    CHECK_FALSE(dataset->datasetComplete());
    dataset->publishFiberletChunkPair(
        owner,
        materialized(FiberletStorageChunkKind::FiberletPrefix, prefixBytes),
        routeKey,
        materialized(FiberletStorageChunkKind::FiberletRoutes, serializeFiberletRoutes(dataset->codecConfig(FiberletStorageChunkKind::FiberletRoutes, routeKey), routes)));
    CHECK(dataset->datasetComplete());
    CHECK_FALSE(std::filesystem::exists(root / "active_chunks.bin"));
    CHECK_FALSE(std::filesystem::exists(root / "dataset.complete"));
    CHECK_FALSE(std::filesystem::exists(root / "complete"));

    {
        std::ofstream(root / "active_chunks.bin") << "legacy";
        std::ofstream(root / "dataset.complete") << "legacy";
        std::filesystem::create_directories(root / "complete");
        std::ofstream(root / "complete" / "0.0.0") << "legacy";
    }

    auto reopened = FiberletChunkDataset::createOrOpen(root, metadata);
    CHECK_FALSE(std::filesystem::exists(root / "active_chunks.bin"));
    CHECK_FALSE(std::filesystem::exists(root / "dataset.complete"));
    CHECK_FALSE(std::filesystem::exists(root / "complete"));
    CHECK_THROWS_AS(reopened->datasetComplete(), std::invalid_argument);
    reopened->configureExpectedChunks(std::span<const vc::render::ChunkKey>(&owner, 1));
    CHECK(reopened->datasetComplete());
    const vc::render::ChunkKey inactive{0, 1, 1, 1};
    const vc::render::ChunkKey inactiveRoute{1, 1, 1, 1};
    const auto emptyAnchorsBytes = serializeFiberletAnchors(reopened->codecConfig(FiberletStorageChunkKind::Anchors, inactive), {});
    const auto emptyPrefixesBytes = serializeFiberletPrefixes(reopened->codecConfig(FiberletStorageChunkKind::FiberletPrefix, inactive), {});
    const auto emptyRoutesBytes = serializeFiberletRoutes(reopened->codecConfig(FiberletStorageChunkKind::FiberletRoutes, inactiveRoute), {});
    reopened->publishChunk(FiberletStorageChunkKind::Anchors, inactive, emptyAnchorsBytes);
    reopened->publishFiberletChunkPair(
        inactive,
        materialized(FiberletStorageChunkKind::FiberletPrefix, emptyPrefixesBytes),
        inactiveRoute,
        materialized(FiberletStorageChunkKind::FiberletRoutes, emptyRoutesBytes));
    CHECK(std::filesystem::exists(reopened->chunkPath(FiberletStorageChunkKind::Anchors, inactive)));
    const auto empty = reopened->readMaterializedChunk(FiberletStorageChunkKind::Anchors, inactive);
    REQUIRE(empty.has_value());
    const auto emptyAnchors = std::dynamic_pointer_cast<const FiberletAnchorChunkPayload>(empty->payload);
    REQUIRE(emptyAnchors);
    CHECK(emptyAnchors->anchors.empty());

    auto anchorCache = createStoredFiberletAnchorChunkCache(reopened);
    auto pathCache = createStoredFiberletPathChunkCache(reopened);
    FiberletChunkGraphSource graph(reopened, anchorCache, reopened, pathCache);
    const auto incident = graph.incidentEdges(second, true);
    REQUIRE(incident.status == FiberletGraphQueryStatus::Ready);
    REQUIRE(incident.value.edges.size() == 1);
    CHECK(incident.value.edges.front().id.fiberlet == FiberletStorageId{first, second});
    const auto loaded = graph.anchor(second, true);
    REQUIRE(loaded.status == FiberletGraphQueryStatus::Ready);
    CHECK(loaded.value.anchor.key == second);

    pathCache->cancelPendingAndWait();
    anchorCache->cancelPendingAndWait();
    std::filesystem::remove_all(root);
}

TEST_CASE("Combined fiberlet dataset with no expected chunks is complete from found payloads")
{
    std::mt19937_64 random(std::random_device{}());
    const auto root = std::filesystem::temp_directory_path() / ("vc_fiberlet_combined_empty_" + std::to_string(random()));
    FiberletDatasetMetadata metadata;
    metadata.kind = FiberletDatasetKind::Combined;
    metadata.profile = FiberletStorageProfile::CompactDirectionsFixedCost;
    metadata.chunkGridShapeZYX = {1, 1, 1};
    metadata.coordinateUnitsPerChunkZYX = {8, 8, 8};
    metadata.maximumEndpointReachCoordinateUnitsZYX = {4, 4, 4};
    metadata.coordinateBits = 8;
    metadata.deltaBits = 8;
    metadata.routeCountBits = 8;
    metadata.routeLatticeBits = 8;
    metadata.costBits = 16;
    finalizeFiberletDatasetIdentity(metadata);
    auto dataset = FiberletChunkDataset::createOrOpen(root, metadata);
    dataset->configureExpectedChunks(std::span<const vc::render::ChunkKey>{});
    CHECK(dataset->datasetComplete());
    auto anchorCache = createStoredFiberletAnchorChunkCache(dataset);
    auto pathCache = createStoredFiberletPathChunkCache(dataset);
    pathCache->cancelPendingAndWait();
    anchorCache->cancelPendingAndWait();
    std::filesystem::remove_all(root);
}

TEST_CASE("Stored combined fiberlet caches interpret absent sparse chunks as empty")
{
    std::mt19937_64 random(std::random_device{}());
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_fiberlet_combined_sparse_" + std::to_string(random()));
    FiberletDatasetMetadata metadata;
    metadata.kind = FiberletDatasetKind::Combined;
    metadata.profile = FiberletStorageProfile::CompactDirectionsFixedCost;
    metadata.chunkGridShapeZYX = {2, 2, 2};
    metadata.coordinateUnitsPerChunkZYX = {8, 8, 8};
    metadata.maximumEndpointReachCoordinateUnitsZYX = {4, 4, 4};
    metadata.coordinateBits = 8;
    metadata.deltaBits = 8;
    metadata.routeCountBits = 8;
    metadata.routeLatticeBits = 8;
    metadata.costBits = 16;
    finalizeFiberletDatasetIdentity(metadata);
    auto dataset = FiberletChunkDataset::createOrOpen(root, metadata);

    auto anchorCache = createStoredFiberletAnchorChunkCache(dataset);
    auto pathCache = createStoredFiberletPathChunkCache(dataset);
    const auto anchors = anchorCache->getChunkBlocking(0, 1, 0, 1);
    const auto prefixes = pathCache->getChunkBlocking(0, 1, 0, 1);
    const auto routes = pathCache->getChunkBlocking(1, 1, 0, 1);
    REQUIRE(anchors.status == vc::render::ChunkStatus::Data);
    REQUIRE(prefixes.status == vc::render::ChunkStatus::Data);
    REQUIRE(routes.status == vc::render::ChunkStatus::Data);
    const auto anchorPayload = std::dynamic_pointer_cast<
        const FiberletAnchorChunkPayload>(anchors.payload);
    const auto prefixPayload = std::dynamic_pointer_cast<
        const FiberletPrefixChunkPayload>(prefixes.payload);
    const auto routePayload = std::dynamic_pointer_cast<
        const FiberletRouteChunkPayload>(routes.payload);
    REQUIRE(anchorPayload);
    REQUIRE(prefixPayload);
    REQUIRE(routePayload);
    CHECK(anchorPayload->anchors.empty());
    CHECK(prefixPayload->prefixes.empty());
    CHECK(routePayload->routes.empty());

    pathCache->cancelPendingAndWait();
    anchorCache->cancelPendingAndWait();
    std::filesystem::remove_all(root);
}

TEST_CASE("Fiberlet atomic temporary cleanup removes only exact abandoned write names")
{
    std::mt19937_64 random(std::random_device{}());
    const auto root = std::filesystem::temp_directory_path() / ("vc_fiberlet_atomic_cleanup_" + std::to_string(random()));
    std::filesystem::create_directories(root / "nested");
    std::ofstream(root / "anchors" ".tmp.123.0") << "abandoned";
    std::ofstream(root / "nested" / "routes.tmp.456.789") << "abandoned";
    std::ofstream(root / "keep.tmp.invalid.0") << "keep";
    std::ofstream(root / "keep.tmp.1.invalid") << "keep";

    {
        vc::core::util::ExclusiveDirectoryLock lock(root);
        CHECK_THROWS_AS(
            vc::core::util::ExclusiveDirectoryLock(root),
            std::filesystem::filesystem_error);
        CHECK(vc::core::util::cleanupAtomicWriteTemporaryFiles(root) == 2);
        CHECK_FALSE(std::filesystem::exists(root / "anchors.tmp.123.0"));
        CHECK_FALSE(std::filesystem::exists(root / "nested" / "routes.tmp.456.789"));
        CHECK(std::filesystem::exists(root / "keep.tmp.invalid.0"));
        CHECK(std::filesystem::exists(root / "keep.tmp.1.invalid"));
    }
    std::filesystem::create_directories(root / "occupied-target");
    CHECK_THROWS(vc::core::util::atomicWriteString(root / "occupied-target", "cannot replace a directory"));
    CHECK(vc::core::util::cleanupAtomicWriteTemporaryFiles(root) == 0);
    std::filesystem::remove_all(root);
}

TEST_CASE("Fiberlet chunk graph loads complete cross-chunk adjacency and routes")
{
    std::mt19937_64 random(std::random_device{}());
    const auto root = std::filesystem::temp_directory_path() / ("vc_fiberlet_graph_" + std::to_string(random()));
    FiberletDatasetMetadata anchorsMetadata;
    anchorsMetadata.kind = FiberletDatasetKind::Anchors;
    anchorsMetadata.profile = FiberletStorageProfile::Float32Cache;
    anchorsMetadata.chunkGridShapeZYX = {4, 4, 4};
    anchorsMetadata.coordinateUnitsPerChunkZYX = {8, 8, 8};
    anchorsMetadata.maximumEndpointReachCoordinateUnitsZYX = {16, 16, 16};
    anchorsMetadata.coordinateBits = 8;
    anchorsMetadata.deltaBits = 8;
    anchorsMetadata.routeCountBits = 8;
    anchorsMetadata.routeLatticeBits = 8;
    anchorsMetadata.costBits = 32;
    finalizeFiberletDatasetIdentity(anchorsMetadata);
    auto fiberletsMetadata = anchorsMetadata;
    fiberletsMetadata.kind = FiberletDatasetKind::Fiberlets;
    finalizeFiberletDatasetIdentity(fiberletsMetadata);

    const auto first = key(7, 15, 15);
    const auto second = key(8, 7, 7);
    const FiberletStorageId edgeId{first, second};
    auto anchorsDataset = FiberletChunkDataset::createOrOpen(root / "anchors", anchorsMetadata);
    auto fiberletsDataset = FiberletChunkDataset::createOrOpen(root / "fiberlets", fiberletsMetadata);

    auto anchorCache =
        createGeneratedFiberletChunkCache(anchorsDataset, [=](FiberletStorageChunkKind kind, const vc::render::ChunkKey&, const FiberletStorageCodecConfig& config) {
            std::vector<FiberletStoredAnchor> anchors;
            const auto appendIfOwned = [&](const FiberletStorageKey& candidate, const cv::Vec3f& position) {
                bool owned = true;
                for (std::size_t axis = 0; axis < 3; ++axis)
                    owned = owned && candidate.coordinateZYX[axis] >= config.coordinateOriginZYX[axis] &&
                            candidate.coordinateZYX[axis] < config.coordinateOriginZYX[axis] + 8;
                if (owned)
                    anchors.push_back({candidate, position, {1, 0, 0}});
            };
            appendIfOwned(first, {1, 2, 3});
            appendIfOwned(second, {2, 2, 3});
            std::sort(anchors.begin(), anchors.end(), [](const auto& left, const auto& right) { return left.key < right.key; });
            return materialized(kind, serializeFiberletAnchors(config, anchors));
        });
    FiberletChunkCacheOptions fiberletCacheOptions;
    fiberletCacheOptions.service.decodedByteCapacity = 1024;
    std::atomic<int> routeRequests{0};
    auto fiberletCache = createGeneratedFiberletChunkCache(
        fiberletsDataset,
        [=, &routeRequests](FiberletStorageChunkKind kind, const vc::render::ChunkKey& chunk, const FiberletStorageCodecConfig& config) {
            const bool owner = chunk.iz == 0 && chunk.iy == 1 && chunk.ix == 1;
            if (kind == FiberletStorageChunkKind::FiberletPrefix) {
                const std::vector<FiberletStoredPrefix> prefixes =
                    owner
                        ? std::vector<FiberletStoredPrefix>{{.id = edgeId, .pathLengthPredictionVoxels = 1.0F, .cost = {0.25F, 4.0F, 1.0F, 2.0F, 2.0F}, .firstStepBaseXYZ = {1, 0, 0}, .lastStepBaseXYZ = {1, 0, 0}}}
                          : std::vector<FiberletStoredPrefix>{};
                return materialized(kind, serializeFiberletPrefixes(config, prefixes));
            }
            ++routeRequests;
            const std::vector<FiberletStoredRoute> routes =
                owner ? std::vector<FiberletStoredRoute>{{
                            .middleUV = {},
                            .segmentCostDensities = {9.25F}}}
                      : std::vector<FiberletStoredRoute>{};
            return materialized(kind, serializeFiberletRoutes(config, routes));
        },
        fiberletCacheOptions);
    FiberletChunkGraphSource graph(anchorsDataset, anchorCache, fiberletsDataset, fiberletCache);

    auto incident = graph.incidentEdges(second, true);
    REQUIRE(incident.status == FiberletGraphQueryStatus::Ready);
    REQUIRE(incident.value.edges.size() == 1);
    CHECK(incident.value.edges.front().id.fiberlet == edgeId);
    CHECK(incident.value.edges.front().id.reverse);
    CHECK_FALSE(incident.value.payloadLeases.empty());
    CHECK(fiberletCache->stats().decodedBytes >
          fiberletCacheOptions.service.decodedByteCapacity);

    const auto loadedAnchor = graph.anchor(second, true);
    REQUIRE(loadedAnchor.status == FiberletGraphQueryStatus::Ready);
    CHECK((loadedAnchor.value.anchor.positionPredictionXYZ == cv::Vec3f{2, 2, 3}));

    const auto edge = graph.edge(edgeId, true);
    REQUIRE(edge.status == FiberletGraphQueryStatus::Ready);
    CHECK(edge.value.prefix.cost.total() == 9.25F);
    CHECK((edge.value.prefix.firstStepBaseXYZ == cv::Vec3f{1, 0, 0}));
    CHECK(routeRequests.load() == 0);

    auto route = graph.route(edgeId, true);
    REQUIRE(route.status == FiberletGraphQueryStatus::Ready);
    CHECK(route.value.prefix.cost.total() == 9.25F);
    CHECK(route.value.route.middleUV.empty());
    CHECK(route.value.pointsPredictionXYZ.size() == 2);
    CHECK(routeRequests.load() == 1);

    FiberletChunkGraphSource transformedGraph(anchorsDataset, anchorCache, fiberletsDataset, fiberletCache, {}, [](const vc::render::ChunkKey&, std::shared_ptr<const FiberletAnchorChunkPayload> canonical) {
        auto transformed = std::make_shared<std::vector<FiberletStoredAnchor>>(canonical->anchors);
        for (auto& anchor : *transformed)
            anchor.positionPredictionXYZ[0] += 10.0F;
        return std::shared_ptr<const std::vector<FiberletStoredAnchor>>(std::move(transformed));
    });
    const auto transformedAnchor = transformedGraph.anchor(second, true);
    REQUIRE(transformedAnchor.status == FiberletGraphQueryStatus::Ready);
    CHECK((transformedAnchor.value.anchor.positionPredictionXYZ == cv::Vec3f{12, 2, 3}));
    const auto transformedChunk = transformedGraph.anchorsInChunk({0, 1, 0, 0}, true);
    REQUIRE(transformedChunk.status == FiberletGraphQueryStatus::Ready);
    REQUIRE(transformedChunk.value.anchors->size() == 1);
    CHECK((transformedChunk.value.anchors->front().positionPredictionXYZ == cv::Vec3f{12, 2, 3}));
    const auto transformedEdge = transformedGraph.edge(edgeId, true);
    REQUIRE(transformedEdge.status == FiberletGraphQueryStatus::Ready);
    CHECK((transformedEdge.value.firstAnchor.positionPredictionXYZ == cv::Vec3f{11, 2, 3}));
    CHECK((transformedEdge.value.secondAnchor.positionPredictionXYZ == cv::Vec3f{12, 2, 3}));
    const auto transformedRoute = transformedGraph.route(edgeId, true);
    REQUIRE(transformedRoute.status == FiberletGraphQueryStatus::Ready);
    REQUIRE(transformedRoute.value.pointsPredictionXYZ.size() == 2);
    CHECK((transformedRoute.value.pointsPredictionXYZ.front() == cv::Vec3f{11, 2, 3}));
    CHECK((transformedRoute.value.pointsPredictionXYZ.back() == cv::Vec3f{12, 2, 3}));

    const vc::render::ChunkKey owner{0, 0, 1, 1};
    const vc::render::ChunkKey distant{0, 3, 3, 3};
    auto ownerBeforeRelease = fiberletCache->getChunkIfCached(owner.level, owner.iz, owner.iy, owner.ix);
    CHECK(ownerBeforeRelease.status == vc::render::ChunkStatus::Data);
    // Release every graph lease, then force LRU enforcement with another
    // chunk. Connectivity must be reconstructible from stable IDs after the
    // original owner payload is evicted.
    incident.value = {};
    route.value = {};
    ownerBeforeRelease.payload.reset();
    const auto distantChunk = fiberletCache->getChunkBlocking(distant.level, distant.iz, distant.iy, distant.ix);
    REQUIRE(distantChunk.status == vc::render::ChunkStatus::Data);
    CHECK(fiberletCache->stats().decodedBytes <=
          fiberletCacheOptions.service.decodedByteCapacity);
    const auto reloadedIncident = graph.incidentEdges(second, true);
    REQUIRE(reloadedIncident.status == FiberletGraphQueryStatus::Ready);
    REQUIRE(reloadedIncident.value.edges.size() == 1);
    CHECK(reloadedIncident.value.edges.front().id.fiberlet == edgeId);
    std::filesystem::remove_all(root);
}

TEST_CASE("Fiberlet chunk route analysis finds exact simple entry-to-exit optima")
{
    std::mt19937_64 random(std::random_device{}());
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_fiberlet_chunk_routes_" + std::to_string(random()));
    FiberletDatasetMetadata metadata;
    metadata.kind = FiberletDatasetKind::Anchors;
    metadata.profile = FiberletStorageProfile::Float32Cache;
    metadata.chunkGridShapeZYX = {1, 1, 1};
    metadata.coordinateUnitsPerChunkZYX = {64, 64, 64};
    metadata.maximumEndpointReachCoordinateUnitsZYX = {32, 32, 32};
    metadata.spatialChunkSideBaseVoxels = 64;
    metadata.predictionToBaseScale = 1.0;
    finalizeFiberletDatasetIdentity(metadata);
    auto fiberletMetadata = metadata;
    fiberletMetadata.kind = FiberletDatasetKind::Fiberlets;
    finalizeFiberletDatasetIdentity(fiberletMetadata);
    auto anchorDataset = FiberletChunkDataset::createOrOpen(
        root / "anchors", metadata);
    auto fiberletDataset = FiberletChunkDataset::createOrOpen(
        root / "fiberlets", fiberletMetadata);
    const vc::render::ChunkKey owner{0, 0, 0, 0};
    const vc::render::ChunkKey routeOwner{1, 0, 0, 0};

    const auto outsideLeft = key(12, 12, 8);
    const auto a = key(12, 12, 11);
    const auto loopFirst = key(12, 12, 12);
    const auto b = key(12, 12, 13);
    const auto loopSecond = key(12, 12, 14);
    const auto c = key(12, 12, 15);
    const auto d = key(12, 12, 17);
    const auto dead = key(12, 12, 18, 1);
    const auto outsideB = key(12, 12, 21);
    const auto outsideC = key(12, 12, 22);
    const auto outsideD = key(12, 12, 23);
    const auto makeAnchor = [](FiberletStorageKey id) {
        FiberletStoredAnchor anchor;
        anchor.key = id;
        anchor.positionPredictionXYZ = {
            static_cast<float>(id.coordinateZYX[2]),
            static_cast<float>(id.coordinateZYX[1]),
            static_cast<float>(id.coordinateZYX[0])};
        anchor.fittedAxisXYZ = {1.0F, 0.0F, 0.0F};
        anchor.predictionAxisXYZ = {1.0F, 0.0F, 0.0F};
        anchor.predictionPresence = 1.0F;
        anchor.normalXYZ = {0.0F, 0.0F, 1.0F};
        anchor.predictionValid = true;
        anchor.predictionPresenceValid = true;
        anchor.normalValid = true;
        return anchor;
    };
    std::vector<FiberletStoredAnchor> anchors;
    for (const auto& id : {outsideLeft, a, loopFirst, b, loopSecond, c, d,
                           dead, outsideB, outsideC, outsideD}) {
        anchors.push_back(makeAnchor(id));
    }
    std::sort(anchors.begin(), anchors.end(),
              [](const auto& left, const auto& right) {
                  return left.key < right.key;
              });

    const auto makePrefix = [](FiberletStorageKey first,
                               FiberletStorageKey second, float loss) {
        FiberletStoredPrefix prefix;
        prefix.id = {std::min(first, second), std::max(first, second)};
        prefix.pathLengthPredictionVoxels = 1.0F;
        prefix.cost = {loss, 0.0F, 0.0F, 0.0F, 0.0F};
        prefix.firstStepBaseXYZ = {1.0F, 0.0F, 0.0F};
        prefix.lastStepBaseXYZ = {1.0F, 0.0F, 0.0F};
        return prefix;
    };
    std::vector<FiberletStoredPrefix> prefixes{
        makePrefix(outsideLeft, a, 1.0F),
        makePrefix(a, b, 1.0F),
        makePrefix(b, outsideB, 10.0F),
        makePrefix(a, c, 2.0F),
        makePrefix(c, outsideC, 2.0F),
        makePrefix(a, d, 2.0F),
        makePrefix(d, outsideD, 2.0F),
        makePrefix(a, dead, 0.01F),
        makePrefix(b, c, 100.0F),
        makePrefix(a, loopFirst, 0.1F),
        makePrefix(loopFirst, loopSecond, 0.1F),
        makePrefix(a, loopSecond, 0.1F),
    };
    prefixes.back().firstStepBaseXYZ = {-1.0F, 0.0F, 0.0F};
    prefixes.back().lastStepBaseXYZ = {-1.0F, 0.0F, 0.0F};
    std::sort(prefixes.begin(), prefixes.end(),
              [](const auto& left, const auto& right) {
                  return left.id < right.id;
              });
    std::vector<FiberletStoredRoute> routes(
        prefixes.size(), FiberletStoredRoute{{}, {1.0F}});
    std::atomic<int> generatedAnchors{0};
    std::atomic<int> generatedPrefixes{0};
    std::atomic<int> generatedRoutes{0};
    const auto makeGeneratedAnchorCache = [&](bool generationAllowed) {
        return createGeneratedFiberletChunkCache(
            anchorDataset,
            [&, generationAllowed](FiberletStorageChunkKind kind,
                                    const vc::render::ChunkKey& chunk,
                                    const FiberletStorageCodecConfig& codec) {
                if (!generationAllowed)
                    throw std::runtime_error(
                        "hot chunk-route analysis must not regenerate data");
                ++generatedAnchors;
                CHECK(kind == FiberletStorageChunkKind::Anchors);
                CHECK(chunk == owner);
                return materialized(
                    kind, serializeFiberletAnchors(codec, anchors));
            });
    };
    const auto makeGeneratedFiberletCache = [&](bool generationAllowed) {
        return createGeneratedFiberletChunkCache(
            fiberletDataset,
            [&, generationAllowed](FiberletStorageChunkKind kind,
                                    const vc::render::ChunkKey& chunk,
                                    const FiberletStorageCodecConfig& codec) {
                if (!generationAllowed)
                    throw std::runtime_error(
                        "hot chunk-route analysis must not regenerate data");
                if (kind == FiberletStorageChunkKind::FiberletPrefix) {
                    ++generatedPrefixes;
                    CHECK(chunk == owner);
                    return materialized(
                        kind, serializeFiberletPrefixes(codec, prefixes));
                }
                ++generatedRoutes;
                CHECK(chunk == routeOwner);
                return materialized(
                    kind, serializeFiberletRoutes(codec, routes));
            });
    };

    auto generatedAnchorCache = makeGeneratedAnchorCache(true);
    auto generatedFiberletCache = makeGeneratedFiberletCache(true);
    FiberletPathConfig paths;
    paths.smoothnessWeight = 0.0F;
    paths.smoothnessNormalWeight = 0.0F;
    paths.smoothnessTangentWeight = 0.0F;
    FiberletChunkGraphSource graph(
        anchorDataset, generatedAnchorCache,
        fiberletDataset, generatedFiberletCache, paths,
        [](const vc::render::ChunkKey&,
           std::shared_ptr<const FiberletAnchorChunkPayload> payload) {
            auto transformed =
                std::make_shared<std::vector<FiberletStoredAnchor>>(
                    payload->anchors);
            for (auto& anchor : *transformed)
                anchor.positionPredictionXYZ[1] += 0.25F;
            return transformed;
        });
    const auto routeChunk = graph.routesInChunk(owner, true);
    REQUIRE(routeChunk.status == FiberletGraphQueryStatus::Ready);
    REQUIRE(routeChunk.value.payloadLease);
    REQUIRE(routeChunk.value.payloadLease->routes.size() == routes.size());
    for (std::size_t index = 0; index < routes.size(); ++index) {
        const auto pointRoute = graph.storedRoute(prefixes[index].id, true);
        REQUIRE(pointRoute.status == FiberletGraphQueryStatus::Ready);
        CHECK(routeChunk.value.payloadLease->routes[index].middleUV ==
              pointRoute.value.route.middleUV);
        CHECK(routeChunk.value.payloadLease->routes[index]
                  .segmentCostDensities ==
              pointRoute.value.route.segmentCostDensities);
    }
    FiberletChunkRouteAnalysisConfig config;
    config.minimumBaseXYZ = {10.0, 10.0, 10.0};
    config.maximumBaseXYZ = {20.0, 20.0, 20.0};
    config.maximumJoinAngleDegrees = 45.0F;
    config.parallelThreads = 4;
    const auto report = analyzeFiberletChunkRoutes(graph, config);
    auto leftConfig = config;
    leftConfig.minimumBaseXYZ[0] = 10.0;
    leftConfig.maximumBaseXYZ[0] = 15.0;
    auto rightConfig = config;
    rightConfig.minimumBaseXYZ[0] = 15.0;
    rightConfig.maximumBaseXYZ[0] = 20.0;
    const auto leftPopulation = collectFiberletChunkRoutePopulation(
        graph, leftConfig);
    const auto rightPopulation = collectFiberletChunkRoutePopulation(
        graph, rightConfig);
    std::set<FiberletStorageId> stageAll;
    stageAll.insert(
        leftPopulation.physicalFiberletIds.begin(),
        leftPopulation.physicalFiberletIds.end());
    stageAll.insert(
        rightPopulation.physicalFiberletIds.begin(),
        rightPopulation.physicalFiberletIds.end());
    std::set<FiberletStorageId> stageInterior;
    stageInterior.insert(
        leftPopulation.internalFiberletIds.begin(),
        leftPopulation.internalFiberletIds.end());
    stageInterior.insert(
        rightPopulation.internalFiberletIds.begin(),
        rightPopulation.internalFiberletIds.end());
    const FiberletStorageId betweenBoxes{
        std::min(a, c), std::max(a, c)};
    CHECK(stageAll.contains(betweenBoxes));
    CHECK_FALSE(stageInterior.contains(betweenBoxes));
    CHECK(stageAll.size() > stageInterior.size());
    auto serialConfig = config;
    serialConfig.parallelThreads = 1;
    const auto serialReduction = analyzeAndSimplifyFiberletChunkRoutes(
        graph, serialConfig);
    const auto parallelReduction = analyzeAndSimplifyFiberletChunkRoutes(
        graph, config);
    CHECK(serialReduction.analysis.physicalFiberletIds ==
          parallelReduction.analysis.physicalFiberletIds);
    CHECK(serialReduction.analysis.internalFiberletIds ==
          parallelReduction.analysis.internalFiberletIds);
    CHECK(serialReduction.analysis.retainedPhysicalFiberlets ==
          parallelReduction.analysis.retainedPhysicalFiberlets);
    CHECK(serialReduction.analysis.generatedSearchStates ==
          parallelReduction.analysis.generatedSearchStates);
    CHECK(serialReduction.analysis.expandedSearchStates ==
          parallelReduction.analysis.expandedSearchStates);
    CHECK(serialReduction.simplification.livePhysicalFiberletIds ==
          parallelReduction.simplification.livePhysicalFiberletIds);
    CHECK(serialReduction.simplification.livePhysicalDirections ==
          parallelReduction.simplification.livePhysicalDirections);
    CHECK(serialReduction.simplification.retainedInsideAnchorIds ==
          parallelReduction.simplification.retainedInsideAnchorIds);
    CHECK(serialReduction.simplification.boundaryPortalIds ==
          parallelReduction.simplification.boundaryPortalIds);
    REQUIRE(serialReduction.simplification.macros.size() ==
            parallelReduction.simplification.macros.size());
    for (std::size_t index = 0;
         index < serialReduction.simplification.macros.size(); ++index) {
        for (std::size_t reverse = 0; reverse < 2; ++reverse) {
            const auto& serial = serialReduction.simplification.macros[index]
                .directions[reverse];
            const auto& parallel =
                parallelReduction.simplification.macros[index]
                    .directions[reverse];
            CHECK(serial.live == parallel.live);
            CHECK(serial.physicalFiberlets == parallel.physicalFiberlets);
            CHECK(serial.anchors == parallel.anchors);
            CHECK(serial.edgeLosses == parallel.edgeLosses);
            CHECK(serial.internalJoinLosses == parallel.internalJoinLosses);
        }
    }
    CHECK(report.insideAnchors == 7);
    CHECK(report.physicalFiberlets == prefixes.size());
    CHECK(report.physicalFiberletIds.size() == report.physicalFiberlets);
    CHECK(std::is_sorted(
        report.physicalFiberletIds.begin(),
        report.physicalFiberletIds.end()));
    CHECK(report.internalFiberlets == 8);
    CHECK(report.internalFiberletIds.size() == report.internalFiberlets);
    CHECK(std::is_sorted(
        report.internalFiberletIds.begin(),
        report.internalFiberletIds.end()));
    CHECK(report.crossingFiberlets == 4);
    CHECK(report.directedEntries == 4);
    CHECK(report.directedExits == 4);
    CHECK(report.reachableEntries == 4);
    CHECK(report.unreachableEntries == 0);
    CHECK(report.tiedOptimalEntries >= 1);
    CHECK(report.usedInsideAnchors < report.insideAnchors);
    CHECK(report.usedPhysicalFiberlets < report.physicalFiberlets);
    CHECK(report.retainedPhysicalFiberlets.size() ==
          report.usedPhysicalFiberlets);
    CHECK(std::is_sorted(
        report.retainedPhysicalFiberlets.begin(),
        report.retainedPhysicalFiberlets.end()));
    CHECK(report.usedInternalFiberlets < report.internalFiberlets);
    CHECK(report.usedInternalFiberlets + report.unusedInternalFiberlets ==
          report.internalFiberlets);
    CHECK(report.rejectedVisitedTargets > 0);
    CHECK(report.routeLosses.count == report.optimalRoutes);
    CHECK(report.routeLengthsPredictionVoxels.count == report.optimalRoutes);
    const auto generatedRoute = generatedFiberletCache->getChunkBlocking(
        routeOwner.level, routeOwner.iz, routeOwner.iy, routeOwner.ix);
    REQUIRE(generatedRoute.status == vc::render::ChunkStatus::Data);
    CHECK(generatedAnchors.load() == 1);
    CHECK(generatedPrefixes.load() == 1);
    CHECK(generatedRoutes.load() == 1);
    const auto anchorPath = anchorDataset->chunkPath(
        FiberletStorageChunkKind::Anchors, owner);
    const auto prefixPath = fiberletDataset->chunkPath(
        FiberletStorageChunkKind::FiberletPrefix, owner);
    const auto routePath = fiberletDataset->chunkPath(
        FiberletStorageChunkKind::FiberletRoutes, routeOwner);
    const auto anchorTime = std::filesystem::last_write_time(anchorPath);
    const auto prefixTime = std::filesystem::last_write_time(prefixPath);
    const auto routeTime = std::filesystem::last_write_time(routePath);
    CHECK(std::filesystem::last_write_time(anchorPath) == anchorTime);
    CHECK(std::filesystem::last_write_time(prefixPath) == prefixTime);

    const auto population = collectFiberletChunkRoutePopulation(graph, config);
    CHECK(population.insideAnchors == report.insideAnchors);
    CHECK(population.physicalFiberletIds == report.physicalFiberletIds);
    CHECK(population.internalFiberletIds == report.internalFiberletIds);

    auto reducedMetadata = fiberletMetadata;
    reducedMetadata.processing["reduction"] = {
        {"contract", "test_exact_entry_to_exit_reduction"},
        {"chunk_size_base_voxels", 64},
    };
    finalizeFiberletDatasetIdentity(reducedMetadata);
    auto reducedDataset = FiberletChunkDataset::createOrOpen(
        root / "reduced", reducedMetadata);
    const auto reducedWrite = writeReducedFiberletChunk(
        graph, reducedDataset, owner, report.physicalFiberletIds,
        report.retainedPhysicalFiberlets);
    CHECK(reducedWrite.owner == owner);
    CHECK(reducedWrite.inputFiberlets == report.physicalFiberlets);
    CHECK(reducedWrite.retainedFiberlets == report.usedPhysicalFiberlets);
    CHECK_FALSE(reducedWrite.reused);
    const auto reducedPrefixPath = reducedDataset->chunkPath(
        FiberletStorageChunkKind::FiberletPrefix, owner);
    const auto reducedRoutePath = reducedDataset->chunkPath(
        FiberletStorageChunkKind::FiberletRoutes, routeOwner);
    const auto reducedPrefixTime =
        std::filesystem::last_write_time(reducedPrefixPath);
    const auto reducedRouteTime =
        std::filesystem::last_write_time(reducedRoutePath);
    const auto reducedHotWrite = writeReducedFiberletChunk(
        graph, reducedDataset, owner, report.physicalFiberletIds,
        report.retainedPhysicalFiberlets);
    CHECK(reducedHotWrite.reused);
    CHECK(std::filesystem::last_write_time(reducedPrefixPath) ==
          reducedPrefixTime);
    CHECK(std::filesystem::last_write_time(reducedRoutePath) ==
          reducedRouteTime);
    CHECK(std::filesystem::last_write_time(prefixPath) == prefixTime);
    CHECK(std::filesystem::last_write_time(routePath) == routeTime);

    auto reducedCache = createStoredFiberletPathChunkCache(reducedDataset);
    FiberletChunkGraphSource reducedGraph(
        anchorDataset, generatedAnchorCache,
        reducedDataset, reducedCache, paths);
    const auto reducedPopulation = collectFiberletChunkRoutePopulation(
        reducedGraph, config);
    CHECK(reducedPopulation.physicalFiberletIds ==
          report.retainedPhysicalFiberlets);
    const auto simplified = simplifyFiberletChunkRoutes(
        reducedGraph, config, report.retainedPhysicalFiberlets);
    CHECK(simplified.inputPhysicalFiberlets ==
          report.retainedPhysicalFiberlets.size());
    CHECK(simplified.livePhysicalFiberlets +
              simplified.deadPhysicalFiberletsRemoved ==
          simplified.inputPhysicalFiberlets);
    CHECK(simplified.liveDirectedStates +
              simplified.deadDirectedStatesRemoved ==
          simplified.inputDirectedStates);
    CHECK(simplified.retainedAnchors + simplified.unusedAnchorsRemoved ==
          simplified.inputAnchors);
    CHECK(simplified.retainedAnchors ==
          simplified.retainedInsideAnchors + simplified.boundaryPortals);
    CHECK(simplified.retainedInsideAnchors +
              simplified.unusedInsideAnchorsRemoved ==
          simplified.inputInsideAnchors);
    CHECK(simplified.physicalMacros +
              simplified.physicalFiberletsMerged ==
          simplified.livePhysicalFiberlets);
    CHECK(simplified.zeroContinuationStates +
              simplified.forcedContinuationStates +
              simplified.branchingStates ==
          simplified.liveDirectedMacros);
    CHECK(simplified.directedChainMacros +
              simplified.directedMacrosMerged ==
          simplified.liveDirectedMacros);
    CHECK(simplified.deterministicRollouts == simplified.rollouts.size());
    CHECK(simplified.structuralDuplicateFiberlets == 0);
    CHECK(std::is_sorted(
        simplified.livePhysicalFiberletIds.begin(),
        simplified.livePhysicalFiberletIds.end()));
    CHECK(std::is_sorted(
        simplified.retainedInsideAnchorIds.begin(),
        simplified.retainedInsideAnchorIds.end()));
    CHECK(std::is_sorted(
        simplified.boundaryPortalIds.begin(),
        simplified.boundaryPortalIds.end()));

    std::vector<FiberletStorageId> expandedPhysical;
    for (const auto& macro : simplified.macros) {
        const auto& forward = macro.directions[0];
        const auto& reverse = macro.directions[1];
        REQUIRE(forward.anchors.size() ==
                forward.physicalFiberlets.size() + 1);
        REQUIRE(reverse.anchors.size() ==
                reverse.physicalFiberlets.size() + 1);
        CHECK(forward.edgeLosses.size() ==
              forward.physicalFiberlets.size());
        CHECK(forward.edgeLengthsPredictionVoxels.size() ==
              forward.physicalFiberlets.size());
        CHECK(forward.internalJoinLosses.size() + 1 ==
              forward.physicalFiberlets.size());
        CHECK(reverse.edgeLosses.size() ==
              reverse.physicalFiberlets.size());
        CHECK(reverse.internalJoinLosses.size() + 1 ==
              reverse.physicalFiberlets.size());
        CHECK(std::vector<FiberletStorageKey>(
                  forward.anchors.rbegin(), forward.anchors.rend()) ==
              reverse.anchors);
        for (std::size_t index = 0;
             index < forward.physicalFiberlets.size(); ++index) {
            const auto& forwardId = forward.physicalFiberlets[index];
            const auto& reverseId = reverse.physicalFiberlets[
                reverse.physicalFiberlets.size() - index - 1];
            CHECK(forwardId.fiberlet == reverseId.fiberlet);
            CHECK(forwardId.reverse != reverseId.reverse);
            expandedPhysical.push_back(forwardId.fiberlet);
        }
        if (forward.live) {
            CHECK(appendFiberletChunkRouteMacroLoss(0.0, 0.0, forward) ==
                  doctest::Approx(forward.diagnosticLoss));
            CHECK(canAppendFiberletChunkRouteMacro(forward, {}));
            std::vector<FiberletStorageKey> blocked{
                forward.anchors.back()};
            CHECK_FALSE(canAppendFiberletChunkRouteMacro(forward, blocked));
        }
    }
    std::sort(expandedPhysical.begin(), expandedPhysical.end());
    CHECK(expandedPhysical == simplified.livePhysicalFiberletIds);
    CHECK(simplified.physicalFiberletsPerMacro.count ==
          simplified.physicalMacros);
    CHECK(simplified.physicalFiberletsMerged > 0);
    for (const auto& transition : simplified.transitions) {
        REQUIRE(transition.incoming.macro < simplified.macros.size());
        REQUIRE(transition.outgoing.macro < simplified.macros.size());
        CHECK(simplified.macros[transition.incoming.macro]
                  .directions[static_cast<std::size_t>(
                      transition.incoming.reverse)]
                  .live);
        CHECK(simplified.macros[transition.outgoing.macro]
                  .directions[static_cast<std::size_t>(
                      transition.outgoing.reverse)]
                  .live);
    }
    for (const auto& rollout : simplified.rollouts) {
        REQUIRE(rollout.macros.size() > 1);
        REQUIRE(rollout.transitionJoinLosses.size() + 1 ==
                rollout.macros.size());
        std::set<FiberletStorageKey> uniqueAnchors(
            rollout.anchors.begin(), rollout.anchors.end());
        CHECK(uniqueAnchors.size() == rollout.anchors.size());
        const auto& first = simplified.macros[rollout.macros.front().macro]
            .directions[static_cast<std::size_t>(
                rollout.macros.front().reverse)];
        double expandedLoss = first.diagnosticLoss;
        double expandedLength = first.diagnosticLengthPredictionVoxels;
        for (std::size_t index = 1; index < rollout.macros.size(); ++index) {
            const auto id = rollout.macros[index];
            const auto& direction = simplified.macros[id.macro]
                .directions[static_cast<std::size_t>(id.reverse)];
            expandedLoss = appendFiberletChunkRouteMacroLoss(
                expandedLoss, rollout.transitionJoinLosses[index - 1],
                direction);
            expandedLength += direction.diagnosticLengthPredictionVoxels;
        }
        CHECK(expandedLoss == doctest::Approx(rollout.diagnosticLoss));
        CHECK(expandedLength ==
              doctest::Approx(
                  rollout.diagnosticLengthPredictionVoxels));
    }
    reducedCache->cancelPendingAndWait();

    auto repairedMetadata = reducedMetadata;
    repairedMetadata.processing["reduction"]["contract"] =
        "test_partial_pair_repair";
    finalizeFiberletDatasetIdentity(repairedMetadata);
    auto repairedDataset = FiberletChunkDataset::createOrOpen(
        root / "repaired", repairedMetadata);
    repairedDataset->publishChunk(
        FiberletStorageChunkKind::FiberletPrefix, owner,
        serializeFiberletPrefixes(
            repairedDataset->codecConfig(
                FiberletStorageChunkKind::FiberletPrefix, owner),
            {}));
    CHECK_FALSE(std::filesystem::exists(repairedDataset->chunkPath(
        FiberletStorageChunkKind::FiberletRoutes, routeOwner)));
    const auto repaired = writeReducedFiberletChunk(
        graph, repairedDataset, owner, report.physicalFiberletIds,
        report.retainedPhysicalFiberlets);
    CHECK_FALSE(repaired.reused);
    REQUIRE(repairedDataset->readMaterializedChunk(
        FiberletStorageChunkKind::FiberletPrefix, owner));
    REQUIRE(repairedDataset->readMaterializedChunk(
        FiberletStorageChunkKind::FiberletRoutes, routeOwner));

    config.maximumGeneratedStatesPerEntry = 1;
    CHECK_THROWS_AS(analyzeFiberletChunkRoutes(graph, config),
                    std::runtime_error);
    config.maximumGeneratedStatesPerEntry = 1'000'000;
    config.maximumJoinAngleDegrees = 0.0F;
    const auto strictAngle = analyzeFiberletChunkRoutes(graph, config);
    CHECK(strictAngle.reachableEntries == 0);
    CHECK(strictAngle.unreachableEntries == strictAngle.directedEntries);
    generatedFiberletCache->cancelPendingAndWait();
    generatedAnchorCache->cancelPendingAndWait();

    anchorDataset = FiberletChunkDataset::openExisting(root / "anchors");
    fiberletDataset = FiberletChunkDataset::openExisting(root / "fiberlets");
    auto hotAnchorCache = makeGeneratedAnchorCache(false);
    auto hotFiberletCache = makeGeneratedFiberletCache(false);
    FiberletChunkGraphSource hotGraph(
        anchorDataset, hotAnchorCache,
        fiberletDataset, hotFiberletCache, paths);
    config.maximumJoinAngleDegrees = 45.0F;
    const auto hotReport = analyzeFiberletChunkRoutes(hotGraph, config);
    CHECK(hotReport.insideAnchors == report.insideAnchors);
    CHECK(hotReport.physicalFiberlets == report.physicalFiberlets);
    CHECK(hotReport.usedInsideAnchors == report.usedInsideAnchors);
    CHECK(hotReport.usedPhysicalFiberlets == report.usedPhysicalFiberlets);
    CHECK(hotReport.usedInternalFiberlets == report.usedInternalFiberlets);
    CHECK(generatedAnchors.load() == 1);
    CHECK(generatedPrefixes.load() == 1);
    CHECK(generatedRoutes.load() == 1);
    CHECK(std::filesystem::last_write_time(anchorPath) == anchorTime);
    CHECK(std::filesystem::last_write_time(prefixPath) == prefixTime);
    CHECK(std::filesystem::last_write_time(routePath) == routeTime);
    hotFiberletCache->cancelPendingAndWait();
    hotAnchorCache->cancelPendingAndWait();
    std::filesystem::remove_all(root);
}

TEST_CASE("Fiberlet sparse overlays fall through and enforce monotone records")
{
    std::mt19937_64 random(std::random_device{}());
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_fiberlet_overlays_" + std::to_string(random()));
    FiberletDatasetMetadata anchorMetadata;
    anchorMetadata.kind = FiberletDatasetKind::Anchors;
    anchorMetadata.profile = FiberletStorageProfile::Float32Cache;
    anchorMetadata.chunkGridShapeZYX = {1, 1, 2};
    anchorMetadata.coordinateUnitsPerChunkZYX = {16, 16, 16};
    anchorMetadata.maximumEndpointReachCoordinateUnitsZYX = {8, 8, 8};
    anchorMetadata.spatialChunkSideBaseVoxels = 16;
    anchorMetadata.predictionToBaseScale = 1.0;
    finalizeFiberletDatasetIdentity(anchorMetadata);
    auto pathMetadata = anchorMetadata;
    pathMetadata.kind = FiberletDatasetKind::Fiberlets;
    finalizeFiberletDatasetIdentity(pathMetadata);
    const vc::render::ChunkKey owner{0, 0, 0, 0};
    const vc::render::ChunkKey routeOwner{1, 0, 0, 0};
    const FiberletStoredAnchor first{
        key(1, 1, 1), {1, 1, 1}, {1, 0, 0}, {1, 0, 0}, 1.0F,
        {0, 1, 0}, true, true, true};
    const FiberletStoredAnchor second{
        key(1, 1, 2), {2, 1, 1}, {1, 0, 0}, {1, 0, 0}, 1.0F,
        {0, 1, 0}, true, true, true};
    FiberletStoredPrefix prefix;
    prefix.id = {first.key, second.key};
    prefix.pathLengthPredictionVoxels = 1.0F;
    prefix.firstStepBaseXYZ = {1, 0, 0};
    prefix.lastStepBaseXYZ = {1, 0, 0};
    FiberletStoredRoute route;
    route.segmentCostDensities = {1.0F};

    auto baseAnchors = FiberletChunkDataset::createOrOpen(
        root / "base-anchors", anchorMetadata);
    auto basePaths = FiberletChunkDataset::createOrOpen(
        root / "base-paths", pathMetadata);
    baseAnchors->publishChunk(
        FiberletStorageChunkKind::Anchors, owner,
        serializeFiberletAnchors(
            baseAnchors->codecConfig(
                FiberletStorageChunkKind::Anchors, owner),
            std::array{first, second}));
    FiberletChunkDataset::MaterializedChunk basePrefix = materialized(
        FiberletStorageChunkKind::FiberletPrefix,
        serializeFiberletPrefixes(
            basePaths->codecConfig(
                FiberletStorageChunkKind::FiberletPrefix, owner),
            std::array{prefix}));
    FiberletChunkDataset::MaterializedChunk baseRoute = materialized(
        FiberletStorageChunkKind::FiberletRoutes,
        serializeFiberletRoutes(
            basePaths->codecConfig(
                FiberletStorageChunkKind::FiberletRoutes, routeOwner),
            std::array{route}));
    basePaths->publishFiberletChunkPair(
        owner, basePrefix, routeOwner, baseRoute);
    auto baseAnchorCache = createStoredFiberletAnchorChunkCache(baseAnchors);
    auto basePathCache = createStoredFiberletPathChunkCache(basePaths);

    auto layerAnchorMetadata = anchorMetadata;
    layerAnchorMetadata.processing["reduction"] = "layer1";
    finalizeFiberletDatasetIdentity(layerAnchorMetadata);
    auto layerPathMetadata = pathMetadata;
    layerPathMetadata.processing["reduction"] = "layer1";
    finalizeFiberletDatasetIdentity(layerPathMetadata);
    auto layerAnchors = FiberletChunkDataset::createOrOpen(
        root / "layer1-anchors", layerAnchorMetadata);
    auto layerPaths = FiberletChunkDataset::createOrOpen(
        root / "layer1-paths", layerPathMetadata);
    auto layerAnchorCache = createOverlayFiberletAnchorChunkCache(
        layerAnchors, baseAnchors, baseAnchorCache);
    auto layerPathCache = createOverlayFiberletPathChunkCache(
        layerPaths, basePaths, basePathCache);
    auto inheritedAnchors = layerAnchorCache->getChunkBlocking(0, 0, 0, 0);
    auto inheritedPaths = layerPathCache->getChunkBlocking(0, 0, 0, 0);
    REQUIRE(inheritedAnchors.status == vc::render::ChunkStatus::Data);
    REQUIRE(inheritedPaths.status == vc::render::ChunkStatus::Data);
    REQUIRE(std::dynamic_pointer_cast<const FiberletAnchorChunkPayload>(
                inheritedAnchors.payload)->anchors.size() == 2);
    REQUIRE(std::dynamic_pointer_cast<const FiberletPrefixChunkPayload>(
                inheritedPaths.payload)->prefixes.size() == 1);
    layerAnchorCache->cancelPendingAndWait();
    layerPathCache->cancelPendingAndWait();

    const auto currentAnchors = materialized(
        FiberletStorageChunkKind::Anchors,
        serializeFiberletAnchors(
            layerAnchors->codecConfig(
                FiberletStorageChunkKind::Anchors, owner),
            std::array{first, second}));
    const auto emptyAnchors = materialized(
        FiberletStorageChunkKind::Anchors,
        serializeFiberletAnchors(
            layerAnchors->codecConfig(
                FiberletStorageChunkKind::Anchors, owner), {}));
    replaceFiberletOverlayChunk(
        layerAnchors, FiberletStorageChunkKind::Anchors, owner,
        currentAnchors, emptyAnchors);
    const auto currentPrefix = materialized(
        FiberletStorageChunkKind::FiberletPrefix,
        serializeFiberletPrefixes(
            layerPaths->codecConfig(
                FiberletStorageChunkKind::FiberletPrefix, owner),
            std::array{prefix}));
    const auto currentRoute = materialized(
        FiberletStorageChunkKind::FiberletRoutes,
        serializeFiberletRoutes(
            layerPaths->codecConfig(
                FiberletStorageChunkKind::FiberletRoutes, routeOwner),
            std::array{route}));
    const auto emptyPrefix = materialized(
        FiberletStorageChunkKind::FiberletPrefix,
        serializeFiberletPrefixes(
            layerPaths->codecConfig(
                FiberletStorageChunkKind::FiberletPrefix, owner), {}));
    const auto emptyRoute = materialized(
        FiberletStorageChunkKind::FiberletRoutes,
        serializeFiberletRoutes(
            layerPaths->codecConfig(
                FiberletStorageChunkKind::FiberletRoutes, routeOwner), {}));
    replaceFiberletOverlayChunkPair(
        layerPaths, owner, currentPrefix, emptyPrefix, routeOwner,
        currentRoute, emptyRoute);

    auto layer2AnchorMetadata = layerAnchorMetadata;
    layer2AnchorMetadata.processing["reduction"] = "layer2";
    finalizeFiberletDatasetIdentity(layer2AnchorMetadata);
    auto layer2PathMetadata = layerPathMetadata;
    layer2PathMetadata.processing["reduction"] = "layer2";
    finalizeFiberletDatasetIdentity(layer2PathMetadata);
    auto layer2Anchors = FiberletChunkDataset::createOrOpen(
        root / "layer2-anchors", layer2AnchorMetadata);
    auto layer2Paths = FiberletChunkDataset::createOrOpen(
        root / "layer2-paths", layer2PathMetadata);
    auto emptyAnchorCache = createOverlayFiberletAnchorChunkCache(
        layerAnchors, baseAnchors, baseAnchorCache);
    auto emptyPathCache = createOverlayFiberletPathChunkCache(
        layerPaths, basePaths, basePathCache);
    auto layer2AnchorCache = createOverlayFiberletAnchorChunkCache(
        layer2Anchors, layerAnchors, emptyAnchorCache);
    auto layer2PathCache = createOverlayFiberletPathChunkCache(
        layer2Paths, layerPaths, emptyPathCache);
    const auto shadowedAnchors = layer2AnchorCache->getChunkBlocking(0, 0, 0, 0);
    const auto shadowedPaths = layer2PathCache->getChunkBlocking(0, 0, 0, 0);
    REQUIRE(shadowedAnchors.status == vc::render::ChunkStatus::Data);
    REQUIRE(shadowedPaths.status == vc::render::ChunkStatus::Data);
    CHECK(std::dynamic_pointer_cast<const FiberletAnchorChunkPayload>(
              shadowedAnchors.payload)->anchors.empty());
    CHECK(std::dynamic_pointer_cast<const FiberletPrefixChunkPayload>(
              shadowedPaths.payload)->prefixes.empty());

    FiberletStoredAnchor mutated = first;
    mutated.predictionPresence = 0.5F;
    const auto mutatedAnchors = materialized(
        FiberletStorageChunkKind::Anchors,
        serializeFiberletAnchors(
            layerAnchors->codecConfig(
                FiberletStorageChunkKind::Anchors, owner),
            std::array{mutated}));
    CHECK_THROWS_AS(
        replaceFiberletOverlayChunk(
            layerAnchors, FiberletStorageChunkKind::Anchors, owner,
            currentAnchors, mutatedAnchors),
        std::invalid_argument);
    CHECK_THROWS_AS(
        replaceFiberletOverlayChunk(
            layerAnchors, FiberletStorageChunkKind::Anchors, owner,
            emptyAnchors, currentAnchors),
        std::invalid_argument);
    auto mutatedRouteValue = route;
    mutatedRouteValue.segmentCostDensities[0] = 0.5F;
    const auto mutatedRoute = materialized(
        FiberletStorageChunkKind::FiberletRoutes,
        serializeFiberletRoutes(
            layerPaths->codecConfig(
                FiberletStorageChunkKind::FiberletRoutes, routeOwner),
            std::array{mutatedRouteValue}));
    CHECK_THROWS_AS(
        replaceFiberletOverlayChunkPair(
            layerPaths, owner, currentPrefix, currentPrefix, routeOwner,
            currentRoute, mutatedRoute),
        std::invalid_argument);

    auto incompatibleMetadata = anchorMetadata;
    incompatibleMetadata.sources = {{"identity", "different"}};
    finalizeFiberletDatasetIdentity(incompatibleMetadata);
    auto incompatible = FiberletChunkDataset::createOrOpen(
        root / "incompatible", incompatibleMetadata);
    CHECK_THROWS_AS(
        createOverlayFiberletAnchorChunkCache(
            incompatible, baseAnchors, baseAnchorCache),
        std::invalid_argument);

    auto partialMetadata = pathMetadata;
    partialMetadata.processing["reduction"] = "partial";
    finalizeFiberletDatasetIdentity(partialMetadata);
    auto partial = FiberletChunkDataset::createOrOpen(
        root / "partial", partialMetadata);
    partial->publishChunk(
        FiberletStorageChunkKind::FiberletPrefix, owner,
        serializeFiberletPrefixes(
            partial->codecConfig(
                FiberletStorageChunkKind::FiberletPrefix, owner), {}));
    auto partialCache = createOverlayFiberletPathChunkCache(
        partial, basePaths, basePathCache);
    const auto partialResult = partialCache->getChunkBlocking(0, 0, 0, 0);
    CHECK(partialResult.status != vc::render::ChunkStatus::Data);

    partialCache->cancelPendingAndWait();
    layer2AnchorCache->cancelPendingAndWait();
    layer2PathCache->cancelPendingAndWait();
    emptyAnchorCache->cancelPendingAndWait();
    emptyPathCache->cancelPendingAndWait();
    baseAnchorCache->cancelPendingAndWait();
    basePathCache->cancelPendingAndWait();
    std::filesystem::remove_all(root);
}

TEST_CASE("Fiberlet overlay write-back cache retains and spills serialized chunks")
{
    std::mt19937_64 random(std::random_device{}());
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_fiberlet_write_back_" + std::to_string(random()));

    FiberletDatasetMetadata anchorMetadata;
    anchorMetadata.kind = FiberletDatasetKind::Anchors;
    anchorMetadata.profile = FiberletStorageProfile::Float32Cache;
    anchorMetadata.chunkGridShapeZYX = {1, 1, 1};
    anchorMetadata.coordinateUnitsPerChunkZYX = {16, 16, 16};
    anchorMetadata.maximumEndpointReachCoordinateUnitsZYX = {8, 8, 8};
    anchorMetadata.spatialChunkSideBaseVoxels = 16;
    anchorMetadata.predictionToBaseScale = 1.0;
    finalizeFiberletDatasetIdentity(anchorMetadata);
    auto pathMetadata = anchorMetadata;
    pathMetadata.kind = FiberletDatasetKind::Fiberlets;
    finalizeFiberletDatasetIdentity(pathMetadata);

    const vc::render::ChunkKey owner{0, 0, 0, 0};
    const vc::render::ChunkKey routeOwner{1, 0, 0, 0};
    const auto budget =
        std::make_shared<vc::render::DecodedChunkCacheBudget>(1U << 20);
    const auto retained = FiberletChunkWriteBackCache::create({
        1U << 20, 2, budget, {}});
    auto anchors = FiberletChunkDataset::createOrOpen(
        root / "retained-anchors", anchorMetadata, retained);
    auto paths = FiberletChunkDataset::createOrOpen(
        root / "retained-paths", pathMetadata, retained);
    const auto anchor = materialized(
        FiberletStorageChunkKind::Anchors,
        serializeFiberletAnchors(
            anchors->codecConfig(FiberletStorageChunkKind::Anchors, owner),
            {}));
    const auto prefix = materialized(
        FiberletStorageChunkKind::FiberletPrefix,
        serializeFiberletPrefixes(
            paths->codecConfig(
                FiberletStorageChunkKind::FiberletPrefix, owner),
            {}));
    const auto routes = materialized(
        FiberletStorageChunkKind::FiberletRoutes,
        serializeFiberletRoutes(
            paths->codecConfig(
                FiberletStorageChunkKind::FiberletRoutes, routeOwner),
            {}));

    anchors->replaceOverlayChunk(
        FiberletStorageChunkKind::Anchors, owner, anchor);
    paths->replaceOverlayChunkPair(owner, prefix, routeOwner, routes);
    CHECK_FALSE(std::filesystem::exists(anchors->chunkPath(
        FiberletStorageChunkKind::Anchors, owner)));
    CHECK_FALSE(std::filesystem::exists(paths->chunkPath(
        FiberletStorageChunkKind::FiberletPrefix, owner)));
    CHECK_FALSE(std::filesystem::exists(paths->chunkPath(
        FiberletStorageChunkKind::FiberletRoutes, routeOwner)));
    REQUIRE(anchors->readMaterializedChunk(
        FiberletStorageChunkKind::Anchors, owner));
    REQUIRE(paths->readMaterializedChunk(
        FiberletStorageChunkKind::FiberletPrefix, owner));
    REQUIRE(paths->readMaterializedChunk(
        FiberletStorageChunkKind::FiberletRoutes, routeOwner));
    CHECK(paths->pairPresence(owner) ==
          FiberletChunkDataset::PairPresence::Complete);
    const auto retainedStats = retained->stats();
    CHECK(retainedStats.residentEntries == 2);
    CHECK(retainedStats.spills == 0);
    CHECK(retainedStats.memoryHits == 3);
    CHECK(retained->logicalFiles(root).size() == 3);
    CHECK(budget->maximumBytes() < (1U << 20));
    retained->finish();
    CHECK(budget->maximumBytes() == (1U << 20));

    const auto spilled = FiberletChunkWriteBackCache::create({1, 2, {}, {}});
    auto spilledAnchors = FiberletChunkDataset::createOrOpen(
        root / "spilled-anchors", anchorMetadata, spilled);
    auto spilledPaths = FiberletChunkDataset::createOrOpen(
        root / "spilled-paths", pathMetadata, spilled);
    spilledAnchors->replaceOverlayChunk(
        FiberletStorageChunkKind::Anchors, owner, anchor);
    spilledPaths->replaceOverlayChunkPair(owner, prefix, routeOwner, routes);
    spilled->waitForSpills();
    CHECK(std::filesystem::exists(spilledAnchors->chunkPath(
        FiberletStorageChunkKind::Anchors, owner)));
    CHECK(std::filesystem::exists(spilledPaths->chunkPath(
        FiberletStorageChunkKind::FiberletPrefix, owner)));
    CHECK(std::filesystem::exists(spilledPaths->chunkPath(
        FiberletStorageChunkKind::FiberletRoutes, routeOwner)));
    CHECK(spilledPaths->pairPresence(owner) ==
          FiberletChunkDataset::PairPresence::Complete);
    CHECK(spilled->stats().spills == 2);
    REQUIRE(spilledAnchors->readMaterializedChunk(
        FiberletStorageChunkKind::Anchors, owner));
    REQUIRE(spilledPaths->readMaterializedChunk(
        FiberletStorageChunkKind::FiberletPrefix, owner));
    REQUIRE(spilledPaths->readMaterializedChunk(
        FiberletStorageChunkKind::FiberletRoutes, routeOwner));
    spilled->finish();

    std::filesystem::remove_all(root);
}

TEST_CASE("Fiberlet overlay write-back cache never exposes a partial pair")
{
    std::mt19937_64 random(std::random_device{}());
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_fiberlet_write_back_failure_" + std::to_string(random()));
    FiberletDatasetMetadata metadata;
    metadata.kind = FiberletDatasetKind::Fiberlets;
    metadata.profile = FiberletStorageProfile::Float32Cache;
    metadata.chunkGridShapeZYX = {1, 1, 1};
    metadata.coordinateUnitsPerChunkZYX = {16, 16, 16};
    metadata.maximumEndpointReachCoordinateUnitsZYX = {8, 8, 8};
    metadata.spatialChunkSideBaseVoxels = 16;
    metadata.predictionToBaseScale = 1.0;
    finalizeFiberletDatasetIdentity(metadata);

    const vc::render::ChunkKey owner{0, 0, 0, 0};
    const vc::render::ChunkKey routeOwner{1, 0, 0, 0};
    auto write = [](const std::filesystem::path& path,
                    std::span<const std::byte> bytes) {
        if (path.parent_path().filename() == "routes")
            throw std::runtime_error("injected route write failure");
        vc::core::util::atomicWriteBytes(path, bytes);
    };
    const auto store = FiberletChunkWriteBackCache::create({1, 1, {}, write});
    auto paths = FiberletChunkDataset::createOrOpen(root, metadata, store);
    const auto prefix = materialized(
        FiberletStorageChunkKind::FiberletPrefix,
        serializeFiberletPrefixes(
            paths->codecConfig(
                FiberletStorageChunkKind::FiberletPrefix, owner),
            {}));
    const auto routes = materialized(
        FiberletStorageChunkKind::FiberletRoutes,
        serializeFiberletRoutes(
            paths->codecConfig(
                FiberletStorageChunkKind::FiberletRoutes, routeOwner),
            {}));
    CHECK_THROWS_WITH_AS(
        paths->replaceOverlayChunkPair(owner, prefix, routeOwner, routes),
        doctest::Contains("injected route write failure"),
        std::runtime_error);
    CHECK_FALSE(std::filesystem::exists(paths->chunkPath(
        FiberletStorageChunkKind::FiberletPrefix, owner)));
    CHECK_FALSE(std::filesystem::exists(paths->chunkPath(
        FiberletStorageChunkKind::FiberletRoutes, routeOwner)));
    CHECK_THROWS_WITH_AS(
        store->finish(), doctest::Contains("injected route write failure"),
        std::runtime_error);
    std::filesystem::remove_all(root);
}

TEST_CASE("Fiberlet overlay write-back LRU remains readable during async spill")
{
    std::mt19937_64 random(std::random_device{}());
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_fiberlet_write_back_lru_" + std::to_string(random()));
    FiberletDatasetMetadata metadata;
    metadata.kind = FiberletDatasetKind::Anchors;
    metadata.profile = FiberletStorageProfile::Float32Cache;
    metadata.chunkGridShapeZYX = {1, 1, 3};
    metadata.coordinateUnitsPerChunkZYX = {16, 16, 16};
    metadata.maximumEndpointReachCoordinateUnitsZYX = {8, 8, 8};
    metadata.spatialChunkSideBaseVoxels = 16;
    metadata.predictionToBaseScale = 1.0;
    finalizeFiberletDatasetIdentity(metadata);
    const auto probe = FiberletChunkDataset::createOrOpen(
        root / "probe", metadata);
    const std::array<vc::render::ChunkKey, 3> keys{
        vc::render::ChunkKey{0, 0, 0, 0},
        vc::render::ChunkKey{0, 0, 0, 1},
        vc::render::ChunkKey{0, 0, 0, 2}};
    std::array<FiberletChunkDataset::MaterializedChunk, 3> chunks;
    for (std::size_t index = 0; index < keys.size(); ++index) {
        chunks[index] = materialized(
            FiberletStorageChunkKind::Anchors,
            serializeFiberletAnchors(
                probe->codecConfig(
                    FiberletStorageChunkKind::Anchors, keys[index]),
                {}));
    }

    std::promise<void> writerEntered;
    std::promise<void> releaseWriter;
    const auto release = releaseWriter.get_future().share();
    std::atomic_bool firstWrite{true};
    std::mutex writesMutex;
    std::vector<std::filesystem::path> writes;
    auto write = [&](const std::filesystem::path& path,
                     std::span<const std::byte> bytes) {
        {
            std::lock_guard lock(writesMutex);
            writes.push_back(path);
        }
        if (firstWrite.exchange(false)) {
            writerEntered.set_value();
            release.wait();
        }
        vc::core::util::atomicWriteBytes(path, bytes);
    };
    constexpr std::size_t entryOverhead = 256;
    const std::size_t entryBytes = chunks[0].bytes.size() + entryOverhead;
    const auto store = FiberletChunkWriteBackCache::create(
        {entryBytes * 2, 1, {}, write});
    auto dataset = FiberletChunkDataset::createOrOpen(
        root / "cached", metadata, store);
    dataset->replaceOverlayChunk(
        FiberletStorageChunkKind::Anchors, keys[0], chunks[0]);
    dataset->replaceOverlayChunk(
        FiberletStorageChunkKind::Anchors, keys[1], chunks[1]);
    REQUIRE(dataset->readMaterializedChunk(
        FiberletStorageChunkKind::Anchors, keys[0]));

    auto pressure = std::async(std::launch::async, [&] {
        dataset->replaceOverlayChunk(
            FiberletStorageChunkKind::Anchors, keys[2], chunks[2]);
    });
    writerEntered.get_future().wait();
    REQUIRE(dataset->readMaterializedChunk(
        FiberletStorageChunkKind::Anchors, keys[1]));
    releaseWriter.set_value();
    pressure.get();
    store->waitForSpills();

    {
        std::lock_guard lock(writesMutex);
        REQUIRE(writes.size() == 2);
        CHECK(writes[0].filename() == "0.0.1");
        CHECK(writes[1].filename() == "0.0.0");
    }
    CHECK(store->stats().spills == 2);
    CHECK(std::filesystem::exists(dataset->chunkPath(
        FiberletStorageChunkKind::Anchors, keys[0])));
    CHECK(std::filesystem::exists(dataset->chunkPath(
        FiberletStorageChunkKind::Anchors, keys[1])));
    CHECK_FALSE(std::filesystem::exists(dataset->chunkPath(
        FiberletStorageChunkKind::Anchors, keys[2])));
    store->finish();
    std::filesystem::remove_all(root);
}
