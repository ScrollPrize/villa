#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/AtomicFile.hpp"
#include "vc/fiber_tracer/FiberletStorage.hpp"
#include "vc/fiber_tracer/FiberletDataset.hpp"
#include "vc/fiber_tracer/FiberletQuantization.hpp"
#include "vc/fiber_tracer/FiberletOnDemand.hpp"
#include "vc/fiber_tracer/FiberletPipeline.hpp"
#include "vc/fiber_tracer/FiberletChunkGraph.hpp"
#include "vc/fiber_tracer/FiberletCropTrace.hpp"
#include "vc/lasagna/Dataset.hpp"

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
#include <cstdlib>
#include <iostream>

using namespace vc::fiber_tracer;

// Compare the immutable crop against the shared stored adapter, including joins.
// Geometry and metric costs must not change when switching to parallel tracing.
static void checkMaterializedGraph(
    const FiberletStoredReplayGraphSource& stored,
    const FiberletMaterializedReplayGraph& materialized)
{
    REQUIRE(materialized.graph);
    CHECK(materialized.graph->supportsConcurrentQueries());
    for (const auto& seed : materialized.insideAnchors) {
        const auto ids = materialized.graph->outgoing(seed.key);
        CHECK(ids == stored.outgoing(seed.key));
        for (const auto& id : ids) {
            CHECK(materialized.graph->routePoints(id) == stored.routePoints(id));
            const auto profile = materialized.graph->costProfile(id);
            const auto expected = stored.costProfile(id);
            CHECK(profile.segmentCostDensities == expected.segmentCostDensities);
            CHECK(profile.segmentLengthsPredictionVoxels == expected.segmentLengthsPredictionVoxels);
            const auto outgoing = materialized.graph->arc(id);
            CHECK(outgoing.cost.total() == stored.arc(id).cost.total());
            const auto incoming = materialized.graph->arc({id.fiberlet, !id.reverse});
            for (const auto& next : ids) {
                const auto actual = materialized.graph->transition(incoming, materialized.graph->arc(next));
                const auto wanted = stored.transition(incoming, stored.arc(next));
                REQUIRE(actual.has_value() == wanted.has_value());
                if (actual)
                    CHECK(actual->cost.total() == wanted->cost.total());
            }
        }
    }
}

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
    const auto result = producer->fiberletRead()({0, 0, 0, 0});
    const auto prefixes = std::dynamic_pointer_cast<const FiberletPrefixChunkPayload>(result);
    REQUIRE(prefixes);
    CHECK(!prefixes->prefixes.empty());
    CHECK(sampled.load() > 0);
    const auto recovered = fiberletPathConfigFromDatasetMetadata(pathMetadata);
    CHECK(recovered.cellRadius == paths.cellRadius);
    {
        FiberletStoredReplayGraphSource source(
            producer->anchorDataset(), producer->anchorRead(),
            producer->fiberletDataset(), producer->fiberletRead(), recovered, 45,
            [producer](const auto& chunk, auto payload) {
                return producer->evaluationAnchorChunk(chunk, std::move(payload));
            });
        const DirectedFiberletStorageId id{prefixes->prefixes.front().id, false};
        CHECK_FALSE(source.routePoints(id).empty());
        const auto forward = source.arc(id);
        const auto reverse = source.arc({id.fiberlet, true});
        CHECK(forward.source == reverse.target);
        CHECK(forward.target == reverse.source);
        CHECK(source.costProfile(id).segmentCostDensities.size() > 0);
        CHECK(source.anchorsInBaseBox({1e100, 1e100, 1e100},
                                     {2e100, 2e100, 2e100}).empty());
        CHECK(source.anchorsInBaseBox({-2e100, -2e100, -2e100},
                                     {-1e100, -1e100, -1e100}).empty());
        CHECK_THROWS_AS(source.anchorsInBaseBox({}, {}), std::invalid_argument);
        const auto materialized = source.materializeBaseBox({}, {12, 12, 12}, 4);
        REQUIRE_FALSE(materialized.insideAnchors.empty());
        checkMaterializedGraph(source, materialized);
        const auto serial = source.materializeBaseBox({}, {12, 12, 12}, 1);
        checkMaterializedGraph(source, serial);
        const auto empty = source.materializeBaseBox({1e100, 1e100, 1e100},
                                                     {2e100, 2e100, 2e100}, 2);
        CHECK(empty.insideAnchors.empty());
        CHECK_THROWS_AS(source.materializeBaseBox({}, {12, 12, 12}, 0), std::invalid_argument);
        CHECK_THROWS_AS(source.materializeBaseBoxForSeeds({}, {12, 12, 12},
            {-1, 0, 0}, {12, 12, 12}), std::invalid_argument);
    }
    producer.reset();
    const auto calls = sampled.load();
    config.cachedOnly = true;
    auto reopened = FiberletOnDemandPreprocessor::create(config);
    REQUIRE(reopened->fiberletRead()({0, 0, 0, 0}));
    CHECK(sampled.load() == calls);
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
    auto memory = std::make_shared<vcr::MemoryCache>(1U << 20);
    auto read = fiberletGeneratedChunkRead(
        dataset,
        [&](FiberletStorageChunkKind kind, const FiberletChunkKey&, const FiberletStorageCodecConfig& config) {
            ++generated;
            CHECK(kind == FiberletStorageChunkKind::Anchors);
            const auto origin = config.coordinateOriginZYX;
            const std::vector<FiberletStoredAnchor> anchors{{key(origin[0], origin[1], origin[2]), {1, 2, 3}, {1, 0, 0}}};
            return materialized(kind, serializeFiberletAnchors(config, anchors));
        },
        memory);
    auto first = read({0, 1, 0, 1});
    REQUIRE(first);
    CHECK(generated.load() == 1);
    CHECK(memory->size() == 1);
    CHECK(memory->nbytes() > 0);
    // Repeat reads decode the bytes the memory cache keeps, without generating.
    const auto repeated = std::dynamic_pointer_cast<const FiberletAnchorChunkPayload>(read({0, 1, 0, 1}));
    REQUIRE(repeated);
    CHECK(repeated->anchors.size() == 1);
    CHECK(generated.load() == 1);
    CHECK(memory->hits() == 1);

    auto reopened = FiberletChunkDataset::createOrOpen(root, metadata);
    auto secondRead = fiberletGeneratedChunkRead(
        reopened,
        [&](FiberletStorageChunkKind, const FiberletChunkKey&, const FiberletStorageCodecConfig&) -> FiberletChunkDataset::MaterializedChunk {
            ++generated;
            throw std::runtime_error("existing chunk should have been reused");
        });
    auto second = secondRead({0, 1, 0, 1});
    REQUIRE(second);
    CHECK(generated.load() == 1);
    const auto firstPayload = std::dynamic_pointer_cast<const FiberletAnchorChunkPayload>(first);
    const auto secondPayload = std::dynamic_pointer_cast<const FiberletAnchorChunkPayload>(second);
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
    const FiberletChunkKey key{0, 1, 0, 1};
    auto read = fiberletGeneratedChunkRead(
        dataset,
        [](FiberletStorageChunkKind kind, const FiberletChunkKey&,
           const FiberletStorageCodecConfig& config) {
            auto chunk = materialized(
                kind, serializeFiberletAnchors(config, {}));
            chunk.alreadyPublished = true;
            return chunk;
        });

    const auto payload = std::dynamic_pointer_cast<
        const FiberletAnchorChunkPayload>(read(key));
    REQUIRE(payload);
    CHECK(payload->anchors.empty());
    CHECK_FALSE(std::filesystem::exists(
        dataset->chunkPath(FiberletStorageChunkKind::Anchors, key)));

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
    const FiberletChunkKey prefixKey{0, 0, 0, 0};
    const FiberletChunkKey routeKey{1, 0, 0, 0};
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
    const FiberletChunkKey outputA{0, 3, 0, 0};
    const FiberletChunkKey outputB{0, 3, 0, 1};
    const FiberletChunkKey outputC{0, 4, 0, 0};
    const FiberletChunkKey anchorA{0, 2, 0, 0};
    const FiberletChunkKey anchorB{0, 3, 0, 0};
    const FiberletChunkKey anchorC{0, 4, 0, 0};
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
    const FiberletChunkKey output{0, 8, 2, 1};
    const FiberletChunkKey anchor{0, 8, 2, 1};
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
    const FiberletChunkKey completedOutput{0, 7, 0, 0};
    const FiberletChunkKey pendingOutput{0, 8, 0, 0};
    const FiberletChunkKey cacheRepairAnchor{0, 7, 0, 0};
    const FiberletChunkKey blockingAnchor{0, 8, 0, 0};
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
    const FiberletChunkKey owner{0, 0, 0, 0};
    dataset->configureExpectedChunks(std::span<const FiberletChunkKey>(&owner, 1));
    (void)fiberletStoredAnchorRead(dataset);
    (void)fiberletStoredPathRead(dataset);

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
    const FiberletChunkKey routeKey{1, 0, 0, 0};
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
    reopened->configureExpectedChunks(std::span<const FiberletChunkKey>(&owner, 1));
    CHECK(reopened->datasetComplete());
    const FiberletChunkKey inactive{0, 1, 1, 1};
    const FiberletChunkKey inactiveRoute{1, 1, 1, 1};
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
    dataset->configureExpectedChunks(std::span<const FiberletChunkKey>{});
    CHECK(dataset->datasetComplete());
    (void)fiberletStoredAnchorRead(dataset);
    (void)fiberletStoredPathRead(dataset);
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

    auto anchorRead = fiberletStoredAnchorRead(dataset);
    auto pathRead = fiberletStoredPathRead(dataset);
    const auto anchorPayload = std::dynamic_pointer_cast<
        const FiberletAnchorChunkPayload>(anchorRead({0, 1, 0, 1}));
    const auto prefixPayload = std::dynamic_pointer_cast<
        const FiberletPrefixChunkPayload>(pathRead({0, 1, 0, 1}));
    const auto routePayload = std::dynamic_pointer_cast<
        const FiberletRouteChunkPayload>(pathRead({1, 1, 0, 1}));
    REQUIRE(anchorPayload);
    REQUIRE(prefixPayload);
    REQUIRE(routePayload);
    CHECK(anchorPayload->anchors.empty());
    CHECK(prefixPayload->prefixes.empty());
    CHECK(routePayload->routes.empty());

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

    auto anchorRead =
        fiberletGeneratedChunkRead(anchorsDataset, [=](FiberletStorageChunkKind kind, const FiberletChunkKey&, const FiberletStorageCodecConfig& config) {
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
    auto fiberletMemory = std::make_shared<vcr::MemoryCache>(1024);
    std::atomic<int> routeRequests{0};
    auto fiberletRead = fiberletGeneratedChunkRead(
        fiberletsDataset,
        [=, &routeRequests](FiberletStorageChunkKind kind, const FiberletChunkKey& chunk, const FiberletStorageCodecConfig& config) {
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
        fiberletMemory);
    FiberletChunkGraphSource graph(anchorsDataset, anchorRead, fiberletsDataset, fiberletRead);

    auto incident = graph.incidentEdges(second);
    REQUIRE(incident.status == FiberletGraphQueryStatus::Ready);
    REQUIRE(incident.value.edges.size() == 1);
    CHECK(incident.value.edges.front().id.fiberlet == edgeId);
    CHECK(incident.value.edges.front().id.reverse);
    CHECK_FALSE(incident.value.payloadLeases.empty());
    REQUIRE(incident.value.payloadLeases.front());

    const auto loadedAnchor = graph.anchor(second);
    REQUIRE(loadedAnchor.status == FiberletGraphQueryStatus::Ready);
    CHECK((loadedAnchor.value.anchor.positionPredictionXYZ == cv::Vec3f{2, 2, 3}));

    const auto edge = graph.edge(edgeId);
    REQUIRE(edge.status == FiberletGraphQueryStatus::Ready);
    CHECK(edge.value.prefix.cost.total() == 9.25F);
    CHECK((edge.value.prefix.firstStepBaseXYZ == cv::Vec3f{1, 0, 0}));
    CHECK(routeRequests.load() == 0);

    auto route = graph.route(edgeId);
    REQUIRE(route.status == FiberletGraphQueryStatus::Ready);
    CHECK(route.value.prefix.cost.total() == 9.25F);
    CHECK(route.value.route.middleUV.empty());
    CHECK(route.value.pointsPredictionXYZ.size() == 2);
    CHECK(routeRequests.load() == 1);

    FiberletChunkGraphSource transformedGraph(anchorsDataset, anchorRead, fiberletsDataset, fiberletRead, {}, [](const FiberletChunkKey&, std::shared_ptr<const FiberletAnchorChunkPayload> canonical) {
        auto transformed = std::make_shared<std::vector<FiberletStoredAnchor>>(canonical->anchors);
        for (auto& anchor : *transformed)
            anchor.positionPredictionXYZ[0] += 10.0F;
        return std::shared_ptr<const std::vector<FiberletStoredAnchor>>(std::move(transformed));
    });
    const auto transformedAnchor = transformedGraph.anchor(second);
    REQUIRE(transformedAnchor.status == FiberletGraphQueryStatus::Ready);
    CHECK((transformedAnchor.value.anchor.positionPredictionXYZ == cv::Vec3f{12, 2, 3}));
    const auto transformedChunk = transformedGraph.anchorsInChunk({0, 1, 0, 0});
    REQUIRE(transformedChunk.status == FiberletGraphQueryStatus::Ready);
    REQUIRE(transformedChunk.value.anchors->size() == 1);
    CHECK((transformedChunk.value.anchors->front().positionPredictionXYZ == cv::Vec3f{12, 2, 3}));
    const auto transformedEdge = transformedGraph.edge(edgeId);
    REQUIRE(transformedEdge.status == FiberletGraphQueryStatus::Ready);
    CHECK((transformedEdge.value.firstAnchor.positionPredictionXYZ == cv::Vec3f{11, 2, 3}));
    CHECK((transformedEdge.value.secondAnchor.positionPredictionXYZ == cv::Vec3f{12, 2, 3}));
    const auto transformedRoute = transformedGraph.route(edgeId);
    REQUIRE(transformedRoute.status == FiberletGraphQueryStatus::Ready);
    REQUIRE(transformedRoute.value.pointsPredictionXYZ.size() == 2);
    CHECK((transformedRoute.value.pointsPredictionXYZ.front() == cv::Vec3f{11, 2, 3}));
    CHECK((transformedRoute.value.pointsPredictionXYZ.back() == cv::Vec3f{12, 2, 3}));

    // Release every graph lease, then evict with another chunk. Connectivity
    // must be reconstructible from stable IDs after the owner payload is evicted.
    const FiberletChunkKey distant{0, 3, 3, 3};
    incident.value = {};
    route.value = {};
    REQUIRE(fiberletRead(distant));
    CHECK(fiberletMemory->nbytes() <= fiberletMemory->max_bytes());
    const auto reloadedIncident = graph.incidentEdges(second);
    REQUIRE(reloadedIncident.status == FiberletGraphQueryStatus::Ready);
    REQUIRE(reloadedIncident.value.edges.size() == 1);
    CHECK(reloadedIncident.value.edges.front().id.fiberlet == edgeId);
    std::filesystem::remove_all(root);
}

TEST_CASE("Fiberlet crop materialization ignores partial tuples reached only by irrelevant halo edges")
{
    std::mt19937_64 random(std::random_device{}());
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_fiberlet_crop_partial_halo_" + std::to_string(random()));
    FiberletDatasetMetadata metadata;
    metadata.kind = FiberletDatasetKind::Combined;
    metadata.profile = FiberletStorageProfile::CompactDirectionsFixedCost;
    metadata.chunkGridShapeZYX = {1, 1, 4};
    metadata.coordinateUnitsPerChunkZYX = {8, 8, 8};
    metadata.maximumEndpointReachCoordinateUnitsZYX = {8, 8, 8};
    metadata.spatialChunkSideBaseVoxels = 8;
    metadata.predictionToBaseScale = 1.0;
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

    const auto inside = key(4, 4, 12);
    const auto relevantOutside = key(4, 4, 17);
    const auto irrelevantFirst = key(4, 4, 20);
    const auto irrelevantSecond = key(4, 4, 25);
    const auto isolated = key(4, 4, 9);
    const auto makeAnchor = [](FiberletStorageKey id, float x) {
        FiberletStoredAnchor anchor;
        anchor.key = id;
        anchor.positionPredictionXYZ = {
            x,
            static_cast<float>(id.coordinateZYX[1]),
            static_cast<float>(id.coordinateZYX[0])};
        anchor.fittedAxisXYZ = {1.0F, 0.0F, 0.0F};
        anchor.predictionAxisXYZ = {1.0F, 0.0F, 0.0F};
        anchor.predictionPresence = 1.0F;
        anchor.normalXYZ = {0.0F, 1.0F, 0.0F};
        anchor.predictionValid = true;
        anchor.predictionPresenceValid = true;
        anchor.normalValid = true;
        return anchor;
    };
    const auto makePrefix = [](FiberletStorageKey first,
                               FiberletStorageKey second) {
        FiberletStoredPrefix prefix;
        prefix.id = {first, second};
        prefix.pathLengthPredictionVoxels = 1.0F;
        prefix.cost = {0.0F, 1.0F, 0.0F, 0.0F, 0.0F};
        prefix.firstStepBaseXYZ = {1.0F, 0.0F, 0.0F};
        prefix.lastStepBaseXYZ = {1.0F, 0.0F, 0.0F};
        return prefix;
    };
    const auto publishComplete = [&dataset](
                                     const FiberletChunkKey& owner,
                                     const std::vector<FiberletStoredAnchor>& anchors,
                                     const std::vector<FiberletStoredPrefix>& prefixes) {
        const FiberletChunkKey routeOwner{
            1, owner.iz, owner.iy, owner.ix};
        dataset->publishChunk(
            FiberletStorageChunkKind::Anchors, owner,
            serializeFiberletAnchors(
                dataset->codecConfig(
                    FiberletStorageChunkKind::Anchors, owner),
                anchors));
        const auto prefixBytes = serializeFiberletPrefixes(
            dataset->codecConfig(
                FiberletStorageChunkKind::FiberletPrefix, owner),
            prefixes);
        std::vector<FiberletStoredRoute> routes(
            prefixes.size(), FiberletStoredRoute{{}, {1.0F}});
        dataset->publishFiberletChunkPair(
            owner,
            materialized(FiberletStorageChunkKind::FiberletPrefix,
                         prefixBytes),
            routeOwner,
            materialized(
                FiberletStorageChunkKind::FiberletRoutes,
                serializeFiberletRoutes(
                    dataset->codecConfig(
                        FiberletStorageChunkKind::FiberletRoutes,
                        routeOwner),
                    routes)));
    };

    const FiberletChunkKey insideOwner{0, 0, 0, 1};
    const FiberletChunkKey haloOwner{0, 0, 0, 2};
    const FiberletChunkKey partialOwner{0, 0, 0, 3};
    publishComplete(
        insideOwner, {makeAnchor(isolated, 9.0F), makeAnchor(inside, 15.0F)},
        {makePrefix(inside, relevantOutside)});
    publishComplete(
        haloOwner,
        {makeAnchor(relevantOutside, 16.0F),
         makeAnchor(irrelevantFirst, 20.0F)},
        {makePrefix(irrelevantFirst, irrelevantSecond)});
    const std::array partialAnchors{makeAnchor(irrelevantSecond, 21.0F)};
    dataset->publishChunk(
        FiberletStorageChunkKind::Anchors, partialOwner,
        serializeFiberletAnchors(
            dataset->codecConfig(
                FiberletStorageChunkKind::Anchors, partialOwner),
            partialAnchors));
    dataset->publishChunk(
        FiberletStorageChunkKind::FiberletPrefix, partialOwner,
        serializeFiberletPrefixes(
            dataset->codecConfig(
                FiberletStorageChunkKind::FiberletPrefix, partialOwner),
            {}));

    FiberletStoredReplayGraphSource stored(dataset);
    const auto crop = stored.materializeBaseBox(
        {8.0, 0.0, 0.0}, {16.0, 8.0, 8.0}, 4);
    REQUIRE(crop.graph);
    REQUIRE(crop.insideAnchors.size() == 2);
    CHECK(crop.graph->outgoing(isolated).empty());
    const auto outgoing = crop.graph->outgoing(inside);
    REQUIRE(outgoing.size() == 1);
    CHECK(outgoing.front().fiberlet ==
          FiberletStorageId{inside, relevantOutside});
    CHECK_THROWS_AS(
        crop.graph->arc(
            {{irrelevantFirst, irrelevantSecond}, false}),
        std::out_of_range);

    CHECK_THROWS_AS(
        stored.materializeBaseBoxForSeeds(
            {8.0, 0.0, 0.0},
            {24.0, 8.0, 8.0},
            {8.0, 0.0, 0.0},
            {16.0, 8.0, 8.0},
            4),
        std::runtime_error);
    CHECK_THROWS_AS(
        stored.materializeBaseBox(
            {24.0, 0.0, 0.0}, {32.0, 8.0, 8.0}, 4),
        std::runtime_error);
    std::filesystem::remove_all(root);
}

TEST_CASE("Published Fiberlet dataset stores self-contained slash-separated chunks")
{
    std::mt19937_64 random(std::random_device{}());
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_fiberlet_published_" + std::to_string(random()));
    FiberletDatasetMetadata metadata;
    metadata.kind = FiberletDatasetKind::Published;
    metadata.processing = makeFiberletDatasetMetadata(
        FiberletDatasetKind::Anchors, {{12, 12, 12}, 1.0},
        FiberAnchorConfig{}, FiberletPathConfig{}, 12,
        nlohmann::json::object()).processing;
    metadata.profile = FiberletStorageProfile::CompactDirectionsFixedCost;
    metadata.chunkGridShapeZYX = {2, 1, 1};
    metadata.coordinateUnitsPerChunkZYX = {8, 8, 8};
    metadata.maximumEndpointReachCoordinateUnitsZYX = {4, 4, 4};
    metadata.coordinateBits = 8;
    metadata.deltaBits = 8;
    metadata.routeCountBits = 8;
    metadata.routeLatticeBits = 8;
    metadata.costBits = 16;
    finalizeFiberletDatasetIdentity(metadata);
    auto dataset = FiberletChunkDataset::createOrOpen(root, metadata);
    const std::array<FiberletChunkKey, 2> expected{
        FiberletChunkKey{0, 0, 0, 0},
        FiberletChunkKey{0, 1, 0, 0}};
    dataset->configureExpectedChunks(expected);

    const auto first = key(7, 2, 2);
    const auto second = key(8, 2, 2);
    const std::vector<FiberletStoredAnchor> firstOwned{
        {first, {1, 2, 3}, {1, 0, 0}}};
    const std::vector<FiberletStoredAnchor> firstReferenced{
        {second, {2, 2, 3}, {1, 0, 0}}};
    const std::vector<FiberletStoredPrefix> prefixes{{
        .id = {first, second},
        .interiorPointCount = 0,
        .entryUV = {0, 0},
        .exitUV = {0, 0},
        .pathLengthPredictionVoxels = 1.0F,
        .cost = {0.0F, 1.0F, 0.0F, 0.0F, 0.0F},
        .firstStepBaseXYZ = {1, 0, 0},
        .lastStepBaseXYZ = {1, 0, 0}}};
    const std::vector<FiberletStoredRoute> routes{{
        .middleUV = {}, .segmentCostDensities = {1.0F}}};

    const auto firstCodec = dataset->codecConfig(
        FiberletStorageChunkKind::PublishedFiberlets, expected[0]);
    auto referencedCodec = firstCodec;
    referencedCodec.coordinateOriginZYX = {8, 2, 2};
    const auto ownedBytes = serializeFiberletAnchors(firstCodec, firstOwned);
    const auto referencedBytes = serializeFiberletAnchors(
        referencedCodec, firstReferenced);
    const auto prefixBytes = serializeFiberletPrefixes(firstCodec, prefixes);
    const auto routeBytes = serializeFiberletRoutes(firstCodec, routes);
    const auto firstFinal = serializePublishedFiberletChunk(
        firstCodec, ownedBytes, referencedBytes, prefixBytes, routeBytes);
    CHECK(std::to_integer<unsigned>(firstFinal.at(12)) == 4);
    CHECK(deserializePublishedFiberletChunk(
        materializeFiberletPayload(firstFinal)).routePayload == routeBytes);
    CHECK_THROWS_AS(deserializeFiberletTraces(firstFinal), std::invalid_argument);
    CHECK_THROWS_AS(serializePublishedFiberletChunk(
        firstCodec, ownedBytes, ownedBytes, prefixBytes, routeBytes),
        std::invalid_argument);
    const auto decoded = deserializePublishedFiberletChunk(firstFinal);
    CHECK(decoded.ownedAnchorPayload == ownedBytes);
    CHECK(decoded.referencedAnchorPayload == referencedBytes);
    CHECK(decoded.prefixPayload == prefixBytes);
    CHECK(decoded.routePayload == routeBytes);
    REQUIRE(decoded.ownedAnchors.anchors.size() == 1);
    REQUIRE(decoded.referencedAnchors.anchors.size() == 1);
    CHECK(decoded.ownedAnchors.anchors.front().key == first);
    CHECK(decoded.referencedAnchors.anchors.front().key == second);
    dataset->publishPublishedChunk(expected[0], firstFinal);

    const auto secondCodec = dataset->codecConfig(
        FiberletStorageChunkKind::PublishedFiberlets, expected[1]);
    const auto secondOwned = serializeFiberletAnchors(
        secondCodec,
        std::array<FiberletStoredAnchor, 1>{
            FiberletStoredAnchor{second, {2, 2, 3}, {1, 0, 0}}});
    const auto secondEmptyAnchors = serializeFiberletAnchors(secondCodec, {});
    const auto secondEmptyPrefixes = serializeFiberletPrefixes(secondCodec, {});
    const auto secondEmptyRoutes = serializeFiberletRoutes(secondCodec, {});
    dataset->publishPublishedChunk(
        expected[1],
        serializePublishedFiberletChunk(
            secondCodec, secondOwned, secondEmptyAnchors,
            secondEmptyPrefixes, secondEmptyRoutes));
    CHECK_FALSE(dataset->datasetComplete());
    CHECK_THROWS_AS(
        (void)fiberletStoredAnchorRead(dataset), std::invalid_argument);
    dataset->markComplete();
    CHECK(dataset->datasetComplete());
    {
        FiberletStoredReplayGraphSource stored(dataset);
        CHECK(stored.routePoints({prefixes.front().id, false}).size() == 2);
        CHECK(stored.arc({prefixes.front().id, true}).target == first);
        const auto crop = stored.materializeBaseBox({}, {1e6, 1e6, 1e6}, 2);
        REQUIRE(crop.insideAnchors.size() == 2);
        checkMaterializedGraph(stored, crop);
    }
    auto reopened = FiberletChunkDataset::createOrOpen(root, metadata);
    reopened->configureExpectedChunks(expected);
    CHECK(reopened->datasetComplete());
    CHECK(reopened->readPublishedChunk(expected[0])->routePayload == routeBytes);
    CHECK_THROWS_AS(dataset->publishChunk(
        FiberletStorageChunkKind::Anchors, expected[0], ownedBytes),
        std::invalid_argument);
    CHECK(std::filesystem::exists(root / "fiberlets" / "0" / "0" / "0"));
    CHECK_FALSE(std::filesystem::exists(root / "anchors"));
    CHECK_FALSE(std::filesystem::exists(root / "prefix"));
    CHECK_FALSE(std::filesystem::exists(root / "routes"));
    const auto ownedFacet = dataset->readMaterializedChunk(
        FiberletStorageChunkKind::Anchors, expected[0]);
    REQUIRE(ownedFacet);
    const auto ownedFacetPayload = std::dynamic_pointer_cast<
        const FiberletAnchorChunkPayload>(ownedFacet->payload);
    REQUIRE(ownedFacetPayload);
    REQUIRE(ownedFacetPayload->anchors.size() == 1);
    CHECK(ownedFacetPayload->anchors.front().key == first);

    auto anchorRead = fiberletStoredAnchorRead(dataset);
    auto pathRead = fiberletStoredPathRead(dataset);
    REQUIRE(anchorRead({0, 1, 0, 0}));
    REQUIRE(std::dynamic_pointer_cast<const FiberletRouteChunkPayload>(pathRead({1, 0, 0, 0})));

    // Open while complete, then remove the neighboring owner's chunk. An
    // already-open route remains self-contained; new opens must still reject
    // the now-incomplete dataset.
    auto graphAnchors = fiberletStoredAnchorRead(dataset);
    auto graphPaths = fiberletStoredPathRead(dataset);
    FiberletChunkGraphSource graph(dataset, graphAnchors, dataset, graphPaths);
    std::filesystem::remove(root / "fiberlets" / "1" / "0" / "0");
    const auto edge = graph.edge(prefixes.front().id);
    REQUIRE(edge.status == FiberletGraphQueryStatus::Ready);
    CHECK(edge.value.firstAnchor.key == first);
    CHECK(edge.value.secondAnchor.key == second);
    REQUIRE(edge.value.prefixPayloadLease->endpoints);
    const auto graphRoute = graph.route(prefixes.front().id);
    REQUIRE(graphRoute.status == FiberletGraphQueryStatus::Ready);
    REQUIRE(graphRoute.value.pointsPredictionXYZ.size() == 2);
    CHECK(graphRoute.value.pointsPredictionXYZ.front() == firstOwned.front().positionPredictionXYZ);
    CHECK(graphRoute.value.pointsPredictionXYZ.back() == firstReferenced.front().positionPredictionXYZ);
    CHECK_FALSE(dataset->datasetComplete());
    CHECK_THROWS_AS(
        (void)fiberletStoredPathRead(dataset), std::invalid_argument);
    dataset->configureExpectedChunks(expected);
    CHECK(dataset->metadata().buildState == FiberletDatasetBuildState::Partial);
    dataset->publishPublishedChunk(
        expected[1],
        serializePublishedFiberletChunk(
            secondCodec, secondOwned, secondEmptyAnchors,
            secondEmptyPrefixes, secondEmptyRoutes));
    dataset->markComplete();
    CHECK(dataset->datasetComplete());
    std::filesystem::create_directories(root / "fiberlets" / "0" / "0");
    std::ofstream(root / "fiberlets" / "0" / "0" / "1") << "stale";
    CHECK_FALSE(dataset->datasetComplete());
    CHECK_THROWS_AS(dataset->markComplete(), std::invalid_argument);
    std::filesystem::remove(root / "fiberlets" / "0" / "0" / "1");
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
