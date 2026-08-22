#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberletStorage.hpp"
#include "vc/fiber_tracer/FiberletDataset.hpp"
#include "vc/fiber_tracer/FiberletChunkGraph.hpp"

#include <cmath>
#include <atomic>
#include <filesystem>
#include <fstream>
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

FiberletChunkDataset::MaterializedChunk materialized(
    FiberletStorageChunkKind kind, std::vector<std::byte> bytes)
{
    FiberletChunkDataset::MaterializedChunk result;
    result.payload = decodeFiberletChunkPayload(kind, bytes);
    result.bytes = std::move(bytes);
    return result;
}

}  // namespace

TEST_CASE("Fiberlet storage float anchors round trip exact float bits")
{
    const auto config = floatConfig();
    const std::vector<FiberletStoredAnchor> anchors{
        {key(101, 202, 303), {3.25F, 2.5F, 1.75F},
         {0.0F, 0.6F, 0.8F}, {0.3F, 0.4F, 0.8660254F}, 0.625F,
         {0.0F, 1.0F, 0.0F}, true, true, true},
        {key(101, 202, 303, 1), {3.5F, 2.75F, 1.5F},
         {1.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}, 0.125F,
         {0.0F, 0.0F, 0.0F}, false, true, false},
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
    CHECK(decoded.anchors[0].predictionPresenceValid ==
          anchors[0].predictionPresenceValid);
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
         .interiorPointCount = 4, .entryUV = {-1, 2}, .exitUV = {3, -4},
         .pathLengthPredictionVoxels = 7.5F,
         .cost = {0.25F, 0.5F, 0.375F, 0.625F, 0.5F},
         .firstStepBaseXYZ = {1.0F, 0.25F, 0.0F},
         .lastStepBaseXYZ = {0.75F, -0.25F, 0.0F}},
        {.id = {key(101, 202, 303), key(104, 204, 305)},
         .interiorPointCount = 2, .entryUV = {0, 1}, .exitUV = {0, 1},
         .pathLengthPredictionVoxels = 9.0F,
         .cost = {1.0F, 2.0F, 1.5F, 2.5F, 1.5F},
         .firstStepBaseXYZ = {1.0F, 0.0F, 0.25F},
         .lastStepBaseXYZ = {1.0F, 0.0F, -0.25F}},
    };
    const std::vector<FiberletStoredRoute> routes{{{{0, 1}, {1, 1}}}, {}};
    const auto decodedPrefixes = deserializeFiberletPrefixes(serializeFiberletPrefixes(config, prefixes));
    const auto decodedRoutes = deserializeFiberletRoutes(serializeFiberletRoutes(config, routes));
    REQUIRE(decodedPrefixes.prefixes.size() == 2);
    CHECK(decodedPrefixes.prefixes[0].id == prefixes[0].id);
    CHECK(decodedPrefixes.prefixes[0].entryUV == prefixes[0].entryUV);
    CHECK(decodedPrefixes.prefixes[0].cost.invalidPrediction ==
          prefixes[0].cost.invalidPrediction);
    CHECK(decodedPrefixes.prefixes[0].cost.alignment ==
          prefixes[0].cost.alignment);
    CHECK(decodedPrefixes.prefixes[0].cost.isotropicSmoothness ==
          prefixes[0].cost.isotropicSmoothness);
    CHECK(decodedPrefixes.prefixes[0].cost.tangentSmoothness ==
          prefixes[0].cost.tangentSmoothness);
    CHECK(decodedPrefixes.prefixes[0].cost.normalSmoothness ==
          prefixes[0].cost.normalSmoothness);
    CHECK(decodedPrefixes.prefixes[0].firstStepBaseXYZ ==
          prefixes[0].firstStepBaseXYZ);
    CHECK(decodedPrefixes.prefixes[0].lastStepBaseXYZ ==
          prefixes[0].lastStepBaseXYZ);
    REQUIRE(decodedRoutes.routes.size() == 2);
    CHECK(decodedRoutes.routes[0].middleUV == routes[0].middleUV);
    CHECK(decodedRoutes.routes[1].middleUV.empty());
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
    for (const auto lattice : std::vector<std::vector<std::array<std::int16_t, 2>>>{
             {}, {{1, -1}}, {{1, -1}, {2, 0}, {-1, 1}}}) {
        const cv::Vec3f secondPosition =
            firstPosition + cv::Vec3f{
                                2.0F * static_cast<float>(lattice.size() + 1),
                                0.0F, 0.0F};
        const std::array<std::int16_t, 2> entry =
            lattice.empty() ? std::array<std::int16_t, 2>{} : lattice.front();
        const std::array<std::int16_t, 2> exit =
            lattice.empty() ? std::array<std::int16_t, 2>{} : lattice.back();
        const auto endpoints = reconstructFiberletRouteEndpointSteps(
            firstPosition, axis, secondPosition, axis,
            lattice.size(), entry, exit, config);
        const auto points = reconstructFiberletRoutePoints(
            firstPosition, axis, secondPosition, axis,
            lattice, config);
        REQUIRE(points.size() >= 2);
        CHECK(endpoints.firstPredictionXYZ == points[1] - points[0]);
        CHECK(endpoints.lastPredictionXYZ ==
              points.back() - points[points.size() - 2]);
        CHECK(-endpoints.lastPredictionXYZ ==
              points[points.size() - 2] - points.back());
        CHECK(-endpoints.firstPredictionXYZ == points[0] - points[1]);
    }
}

TEST_CASE("Fiberlet storage compact cost is decoded from the authoritative chunk range")
{
    auto config = compactConfig();
    const std::vector<FiberletStoredPrefix> prefixes{
        {.id = {key(101, 202, 303), key(102, 203, 304)},
         .pathLengthPredictionVoxels = 2.0F, .cost = {0, 10.0F},
         .firstStepBaseXYZ = {1, 0, 0}, .lastStepBaseXYZ = {1, 0, 0}},
        {.id = {key(101, 202, 303), key(104, 204, 305)},
         .pathLengthPredictionVoxels = 3.0F, .cost = {0, 20.0F},
         .firstStepBaseXYZ = {1, 0, 0}, .lastStepBaseXYZ = {1, 0, 0}},
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
    auto cache =
        createGeneratedFiberletChunkCache(dataset, [&](FiberletStorageChunkKind kind, const vc::render::ChunkKey&, const FiberletStorageCodecConfig& config) {
            ++generated;
            CHECK(kind == FiberletStorageChunkKind::Anchors);
            const auto origin = config.coordinateOriginZYX;
            const std::vector<FiberletStoredAnchor> anchors{{key(origin[0], origin[1], origin[2]), {1, 2, 3}, {1, 0, 0}}};
            return materialized(kind, serializeFiberletAnchors(config, anchors));
        }, {}, [&](FiberletStorageChunkKind kind, const vc::render::ChunkKey& key,
                   vc::render::ChunkFetchStatus status) {
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
    auto secondCache =
        createGeneratedFiberletChunkCache(reopened, [&](FiberletStorageChunkKind, const vc::render::ChunkKey&, const FiberletStorageCodecConfig&) -> FiberletChunkDataset::MaterializedChunk {
            ++generated;
            throw std::runtime_error("existing chunk should have been reused");
        }, {}, [&](FiberletStorageChunkKind kind, const vc::render::ChunkKey& key,
                   vc::render::ChunkFetchStatus status) {
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
    CHECK(secondPayload->anchors.front().positionPredictionXYZ ==
          firstPayload->anchors.front().positionPredictionXYZ);
    CHECK(secondPayload->anchors.front().fittedAxisXYZ ==
          firstPayload->anchors.front().fittedAxisXYZ);

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
                    owner ? std::vector<FiberletStoredPrefix>{{
                        .id = edgeId,
                        .pathLengthPredictionVoxels = 1.0F,
                        .cost = {0.25F, 4.0F, 1.0F, 2.0F, 2.0F},
                        .firstStepBaseXYZ = {1, 0, 0},
                        .lastStepBaseXYZ = {1, 0, 0}}}
                          : std::vector<FiberletStoredPrefix>{};
                return materialized(kind, serializeFiberletPrefixes(config, prefixes));
            }
            ++routeRequests;
            const std::vector<FiberletStoredRoute> routes = owner ? std::vector<FiberletStoredRoute>{{}} : std::vector<FiberletStoredRoute>{};
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
