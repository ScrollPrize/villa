#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberletStorage.hpp"
#include "vc/fiber_tracer/FiberletDataset.hpp"
#include "vc/fiber_tracer/FiberletChunkGraph.hpp"
#include "vc/fiber_tracer/FiberletQuantization.hpp"

#include <cmath>
#include <atomic>
#include <array>
#include <bit>
#include <filesystem>
#include <fstream>
#include <limits>
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
    metadata.algorithmFingerprint = "test-algorithm";
    metadata.datasetFingerprint[0] = 12;
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

TEST_CASE("Fiberlet prefix and routes become visible through one completion marker")
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
    metadata.algorithmFingerprint = "pair-test";
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
    anchorsMetadata.algorithmFingerprint = "graph-test";
    anchorsMetadata.datasetFingerprint[0] = 77;
    auto fiberletsMetadata = anchorsMetadata;
    fiberletsMetadata.kind = FiberletDatasetKind::Fiberlets;
    fiberletsMetadata.datasetFingerprint[0] = 78;

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
    vc::render::ChunkCache::Options fiberletCacheOptions;
    fiberletCacheOptions.decodedByteCapacity = 1024;
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
    CHECK(fiberletCache->stats().decodedBytes > fiberletCacheOptions.decodedByteCapacity);

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
    CHECK(fiberletCache->stats().decodedBytes <= fiberletCacheOptions.decodedByteCapacity);
    const auto reloadedIncident = graph.incidentEdges(second, true);
    REQUIRE(reloadedIncident.status == FiberletGraphQueryStatus::Ready);
    REQUIRE(reloadedIncident.value.edges.size() == 1);
    CHECK(reloadedIncident.value.edges.front().id.fiberlet == edgeId);
    std::filesystem::remove_all(root);
}
