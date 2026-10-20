#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/AtomicFile.hpp"
#include "vc/fiber_tracer/FiberletStorage.hpp"
#include "vc/fiber_tracer/FiberletDataset.hpp"
#include "vc/fiber_tracer/FiberletQuantization.hpp"
#include "vc/fiber_tracer/FiberletOnDemand.hpp"
#include "vc/fiber_tracer/FiberletPipeline.hpp"

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

TEST_CASE("Canonical on-demand preprocessing generates and reopens without sampling")
{
    class Normals final : public vc::lasagna::NormalSampler {
    public:
        vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override
        { return {{0, 0, 1}, true, {}}; }
    };
    std::mt19937_64 random(std::random_device{}());
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_fiberlet_pipeline_" + std::to_string(random()));
    const FiberPredictionGridInfo grid{{12, 12, 12}, 1.0};
    FiberAnchorConfig anchors;
    FiberletPathConfig paths;
    paths.cellRadius = 1;
    const auto anchorMetadata = makeFiberletDatasetMetadata(
        FiberletDatasetKind::Anchors, grid, anchors, paths, 12, nlohmann::json::object());
    const auto pathMetadata = makeFiberletDatasetMetadata(
        FiberletDatasetKind::Fiberlets, grid, anchors, paths, 12, nlohmann::json::object());
    std::atomic<int> sampled{0};
    auto config = makeCanonicalFiberletOnDemandConfig(
        root / "anchors", root / "paths", anchorMetadata, pathMetadata,
        grid, anchors, paths,
        [&](const auto& indices, int, auto& samples) {
            ++sampled;
            samples.clear();
            for (const auto& index : indices) {
                const double y = static_cast<double>(index[1]) - 5.5;
                const double z = static_cast<double>(index[0]) - 5.5;
                samples.push_back({{1, 0, 0}, std::exp(-(y*y + z*z) / 2.0), true, true});
            }
        }, std::make_shared<Normals>(), {});
    auto producer = FiberletOnDemandPreprocessor::create(config);
    const auto result = producer->fiberletCache()->getChunkBlocking(0, 0, 0, 0);
    REQUIRE(result.status == vc::render::ChunkStatus::Data);
    const auto prefixes = std::dynamic_pointer_cast<const FiberletPrefixChunkPayload>(result.payload);
    REQUIRE(prefixes);
    CHECK(!prefixes->prefixes.empty());
    CHECK(sampled.load() > 0);
    producer->shutdown();
    producer.reset();
    const auto calls = sampled.load();
    config.cachedOnly = true;
    auto reopened = FiberletOnDemandPreprocessor::create(config);
    const auto cached = reopened->fiberletCache()->getChunkBlocking(0, 0, 0, 0);
    REQUIRE(cached.status == vc::render::ChunkStatus::Data);
    CHECK(sampled.load() == calls);
    reopened->shutdown();
    reopened.reset();
    std::filesystem::remove_all(root);
}

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

FiberletStorageCodecConfig traceConfig()
{
    FiberletStorageCodecConfig config;
    config.profile = FiberletStorageProfile::Float64Traces;
    config.chunkZYX = {1, 2, 3};
    config.coordinateOriginZYX = {512, 1024, 1536};
    config.coordinateBits = 32;
    config.deltaBits = 32;
    config.routeCountBits = 32;
    config.routeLatticeBits = 32;
    config.costBits = 64;
    config.predictionToBaseScale = 8.0;
    config.datasetFingerprint[0] = 73;
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

TEST_CASE("Fiber trace payload round-trips float64 geometry and global costs exactly")
{
    const auto config = traceConfig();
    const std::vector<FiberletStoredTrace> traces{
        {7,
         {1024.25, 2048.5, 4096.75},
         0.625F,
         123.125,
         45.5,
         {{900.125, 1900.25, 3900.5},
          {1024.25, 2048.5, 4096.75},
          {1100.875, 2150.125, 4200.25}}},
        {9,
         {-1.5, 2.25, 3.75},
         0.25F,
         0.0,
         1.25,
         {{-1.5, 2.25, 3.75}, {-0.25, 2.5, 4.0}}},
    };
    const auto bytes = serializeFiberletTraces(config, traces);
    const auto decoded = deserializeFiberletTraces(bytes);
    CHECK(decoded.config.profile == FiberletStorageProfile::Float64Traces);
    CHECK(decoded.config.chunkZYX == config.chunkZYX);
    REQUIRE(decoded.traces.size() == traces.size());
    for (std::size_t index = 0; index < traces.size(); ++index) {
        CHECK(decoded.traces[index].ordinal == traces[index].ordinal);
        CHECK(decoded.traces[index].seedBaseXYZ == traces[index].seedBaseXYZ);
        CHECK(decoded.traces[index].seedPresence == traces[index].seedPresence);
        CHECK(decoded.traces[index].totalMetricCost == traces[index].totalMetricCost);
        CHECK(decoded.traces[index].pathLengthPredictionVoxels ==
              traces[index].pathLengthPredictionVoxels);
        CHECK(decoded.traces[index].pointsBaseXYZ == traces[index].pointsBaseXYZ);
    }
    CHECK_THROWS_AS(serializeFiberletAnchors(config, {}), std::invalid_argument);
    auto invalid = traces.front();
    invalid.pointsBaseXYZ = {{0, 0, 0}, {1, 0, 0}};
    CHECK_THROWS_AS(
        serializeFiberletTraces(config, std::span{&invalid, 1}),
        std::invalid_argument);
    invalid = traces.front();
    invalid.totalMetricCost = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(
        serializeFiberletTraces(config, std::span{&invalid, 1}),
        std::invalid_argument);
}

TEST_CASE("Fiberlet scheduled resolution errors preserve key status and cause")
{
    const vc::render::ChunkKey key{0, 427, 139, 187};
    vc::render::ChunkResult resolved;
    resolved.status = vc::render::ChunkStatus::Error;
    resolved.error = "synthetic generation failure";
    CHECK(
        fiberletScheduledResolutionError(key, resolved) ==
        "scheduled fiberlet chunk 0/427/139/187 resolved as error: "
        "synthetic generation failure");

    resolved.status = vc::render::ChunkStatus::Missing;
    resolved.error.clear();
    CHECK(
        fiberletScheduledResolutionError(key, resolved) ==
        "scheduled fiberlet chunk 0/427/139/187 resolved as missing");
}

TEST_CASE("Reference prefetch deduplicates physical chunks and orders remaining work")
{
    const auto scheduled = [](int z, int y, int x, double arc, double distance) {
        return FiberletScheduledChunk{{0, z, y, x}, arc, distance};
    };
    const std::vector<FiberletReferencePrefetchCandidate> candidates{
        {scheduled(1, 2, 3, 2.0, 2.0), 10.0, 1},
        {scheduled(1, 2, 3, 3.0, 5.0), 12.0, 2},
        {scheduled(4, 5, 6, 1.0, 1.0), 12.0, 5},
        {scheduled(4, 5, 6, 4.0, 1.0), 12.0, 3},
        {scheduled(7, 8, 9, 0.0, 0.0), 5.0, 0},
    };
    const auto ordered = orderFiberletReferencePrefetch(candidates);
    REQUIRE(ordered.size() == 3);
    CHECK(ordered[0].key == (vc::render::ChunkKey{0, 4, 5, 6}));
    CHECK(ordered[0].nearestReferenceArcBase == 4.0);
    CHECK(ordered[1].key == (vc::render::ChunkKey{0, 1, 2, 3}));
    CHECK(ordered[1].nearestReferenceArcBase == 3.0);
    CHECK(ordered[2].key == (vc::render::ChunkKey{0, 7, 8, 9}));

    auto invalid = candidates;
    invalid.front().remainingReferenceBase = -1.0;
    CHECK_THROWS_AS(
        orderFiberletReferencePrefetch(invalid),
        std::invalid_argument);
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
    REQUIRE(first.payload);
    CHECK(cache->stats().decodedChunkBytes == first.payload->residentBytes());
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
    REQUIRE(second.payload);
    CHECK(secondCache->stats().decodedChunkBytes ==
          second.payload->residentBytes());
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
        CHECK(otherToolchain.algorithmFingerprint ==
              current.algorithmFingerprint);
        CHECK(otherToolchain.datasetFingerprint ==
              current.datasetFingerprint);
        CHECK(fiberletDatasetMetadataCompatible(current, otherToolchain));
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

TEST_CASE("Fiberlet caches are compatible across producer toolchains")
{
    FiberletDatasetMetadata first;
    first.kind = FiberletDatasetKind::Anchors;
    first.profile = FiberletStorageProfile::Float32Cache;
    first.chunkGridShapeZYX = {1, 1, 1};
    first.coordinateUnitsPerChunkZYX = {4, 4, 4};
    first.maximumEndpointReachCoordinateUnitsZYX = {4, 4, 4};
    first.processing = {
        {"contract_version", kFiberletGenerationContractVersion},
        {"producer_toolchain", {
            {"compiler_id", "GNU"},
            {"compiler_version", "16.0"},
            {"build_config", "QuickBuild"},
        }},
    };
    finalizeFiberletDatasetIdentity(first);

    auto second = first;
    second.processing["producer_toolchain"] = {
        {"compiler_id", "Clang"},
        {"compiler_version", "22.0"},
        {"build_config", "Release"},
    };
    finalizeFiberletDatasetIdentity(second);
    REQUIRE(fiberletDatasetMetadataCompatible(first, second));

    const auto root = std::filesystem::temp_directory_path() /
        ("vc_fiberlet_toolchain_" +
         std::to_string(std::mt19937_64{std::random_device{}()}()));
    auto created = FiberletChunkDataset::createOrOpen(root, first);
    created.reset();
    const auto reopened = FiberletChunkDataset::createOrOpen(root, second);
    CHECK(reopened->metadata().processing.at("producer_toolchain")
              .at("compiler_id") == "GNU");
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

TEST_CASE("Combined fiberlet dataset publishes and validates complete sparse tuples")
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
    metadata.processing["paths"] = {
        {"cell_radius", 4},
        {"neighborhood_margin_cells", 0.5F},
        {"longitudinal_step_prediction", 2.0F},
        {"transverse_step_prediction", 0.5F},
        {"maximum_endpoint_angle_degrees", 45.0F},
        {"maximum_prediction_deviation_degrees", 25.0F},
        {"corridor_radius_prediction", 0.0F},
        {"invalid_prediction_cost_per_voxel", 4.0F},
        {"smoothness_weight", 2.0F},
        {"smoothness_normal_weight", 0.1F},
        {"smoothness_tangent_weight", 10.0F},
        {"smoothness_free_angle_degrees", 0.0F},
    };
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
    auto third = second;
    third.variant = 1;
    const auto orphan = key(5, 7, 7);
    const std::vector<FiberletStoredAnchor> anchors{
        {orphan, {0, 2, 3}, {1, 0, 0}, {1, 0, 0}, 0.5F,
         {0, 1, 0}, true, true, true},
        {first, {1, 2, 3}, {1, 0, 0}, {1, 0, 0}, 1.0F,
         {0, 1, 0}, true, true, true},
        {second, {2, 2, 3}, {1, 0, 0}, {1, 0, 0}, 1.0F,
         {0, 1, 0}, true, true, true},
        {third, {3, 2, 3}, {1, 0, 0}, {1, 0, 0}, 1.0F,
         {0, 1, 0}, true, true, true}};
    dataset->publishChunk(FiberletStorageChunkKind::Anchors, owner, serializeFiberletAnchors(dataset->codecConfig(FiberletStorageChunkKind::Anchors, owner), anchors));
    CHECK_FALSE(dataset->datasetComplete());
    const std::vector<FiberletStoredPrefix> prefixes{
        {.id = {first, second},
         .pathLengthPredictionVoxels = 1.0F,
         .cost = {0.0F, 1.0F, 0.0F, 0.0F, 0.0F},
         .firstStepBaseXYZ = {1, 0, 0},
         .lastStepBaseXYZ = {1, 0, 0}},
        {.id = {second, third},
         .pathLengthPredictionVoxels = 1.0F,
         .cost = {0.0F, 1.0F, 0.0F, 0.0F, 0.0F},
         .firstStepBaseXYZ = {1, 0, 0},
         .lastStepBaseXYZ = {1, 0, 0}}};
    const std::vector<FiberletStoredRoute> routes{
        {.middleUV = {}, .segmentCostDensities = {1.0F}},
        {.middleUV = {}, .segmentCostDensities = {1.0F}}};
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

    auto combinedMetadata = anchorMetadata;
    combinedMetadata.kind = FiberletDatasetKind::Combined;
    combinedMetadata.profile =
        FiberletStorageProfile::CompactDirectionsFixedCost;
    combinedMetadata.costBits = 16;
    finalizeFiberletDatasetIdentity(combinedMetadata);
    auto combined = FiberletChunkDataset::createOrOpen(
        root / "combined", combinedMetadata);
    combined->publishChunk(
        FiberletStorageChunkKind::Anchors, owner,
        serializeFiberletAnchors(
            combined->codecConfig(FiberletStorageChunkKind::Anchors, owner),
            std::array{first, second}));
    combined->publishFiberletChunkPair(
        owner,
        materialized(
            FiberletStorageChunkKind::FiberletPrefix,
            serializeFiberletPrefixes(
                combined->codecConfig(
                    FiberletStorageChunkKind::FiberletPrefix, owner),
                std::array{prefix})),
        routeOwner,
        materialized(
            FiberletStorageChunkKind::FiberletRoutes,
            serializeFiberletRoutes(
                combined->codecConfig(
                    FiberletStorageChunkKind::FiberletRoutes, routeOwner),
                std::array{route})));
    auto combinedAnchorCache = createStoredFiberletAnchorChunkCache(combined);
    auto combinedPathCache = createStoredFiberletPathChunkCache(combined);
    auto combinedOverlayAnchors = createOverlayFiberletAnchorChunkCache(
        layerAnchors, combined, combinedAnchorCache);
    auto combinedOverlayPaths = createOverlayFiberletPathChunkCache(
        layerPaths, combined, combinedPathCache);
    CHECK(combinedOverlayAnchors->getChunkBlocking(0, 0, 0, 0).status ==
          vc::render::ChunkStatus::Data);
    CHECK(combinedOverlayPaths->getChunkBlocking(0, 0, 0, 0).status ==
          vc::render::ChunkStatus::Data);
    combinedOverlayAnchors->cancelPendingAndWait();
    combinedOverlayPaths->cancelPendingAndWait();
    combinedAnchorCache->cancelPendingAndWait();
    combinedPathCache->cancelPendingAndWait();

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
