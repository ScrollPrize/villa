#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/io/PolylineObj.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include "vc/fiber_tracer/FiberletCropTrace.hpp"
#include "vc/fiber_tracer/FiberletCropTraceArtifact.hpp"
#include "vc/fiber_tracer/FiberletDataset.hpp"
#include "vc/fiber_tracer/FiberTraceBeliefPropagation.hpp"
#include "vc/fiber_tracer/FiberTraceConstraints.hpp"
#include "vc/fiber_tracer/FiberTraceConsensus.hpp"
#include "vc/fiber_tracer/FiberTraceLabeling.hpp"
#include "vc/lasagna/Dataset.hpp"

#include "utils/zarr.hpp"

#include <filesystem>
#include <fstream>
#include <limits>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <thread>

namespace
{

using namespace vc::fiber_tracer;

FiberletStorageKey key(std::int64_t x, std::uint8_t variant = 0)
{
    return {{0, 0, x}, variant};
}

class ZNormalSampler final : public vc::lasagna::NormalSampler
{
public:
    [[nodiscard]] vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override { return {{0, 0, 1}, true, {}}; }
};

class TestGraph final : public FiberletReplayGraphSource
{
public:
    void addAnchor(FiberletStorageKey id, cv::Vec3d point) { positions[id] = point; }

    void connect(FiberletStorageKey first, FiberletStorageKey second)
    {
        FiberletStorageId physical{std::min(first, second), std::max(first, second)};
        const bool reverse = first != physical.first;
        DirectedFiberletStorageId directed{physical, reverse};
        adjacency[first].push_back(directed);
        directed.reverse = !directed.reverse;
        adjacency[second].push_back(directed);
    }

    void setCostDensity(
        FiberletStorageKey first,
        FiberletStorageKey second,
        float density)
    {
        edgeCostDensity[{std::min(first, second), std::max(first, second)}] =
            density;
    }

    [[nodiscard]] bool supportsConcurrentQueries() const noexcept override { return concurrentQueries; }
    [[nodiscard]] float predictionToBaseScale() const noexcept override { return 1.0F; }
    [[nodiscard]] int anchorCellSizePredictionVoxels() const noexcept override { return 4; }
    [[nodiscard]] float maximumJoinAngleDegrees() const noexcept override { return 45.0F; }
    [[nodiscard]] std::vector<FiberletReplaySourceAnchor> anchorsNearReference(const PolylineArcGeometry&, double, double, double) const override
    {
        return {};
    }
    [[nodiscard]] std::vector<DirectedFiberletStorageId> outgoing(const FiberletStorageKey& anchor) const override
    {
        struct ActiveQuery {
            explicit ActiveQuery(const TestGraph& owner) : owner_(owner)
            {
                if (!owner_.observeConcurrency)
                    return;
                const int active = owner_.activeQueries.fetch_add(1) + 1;
                int maximum = owner_.maximumConcurrentQueries.load();
                while (active > maximum && !owner_.maximumConcurrentQueries.compare_exchange_weak(maximum, active)) {
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            ~ActiveQuery()
            {
                if (owner_.observeConcurrency)
                    owner_.activeQueries.fetch_sub(1);
            }
            const TestGraph& owner_;
        } activeQuery(*this);
        if (const auto delay = queryDelay.find(anchor); delay != queryDelay.end())
            std::this_thread::sleep_for(delay->second);
        if (const auto failure = queryFailure.find(anchor); failure != queryFailure.end())
            throw std::runtime_error(failure->second);
        auto found = adjacency.find(anchor);
        if (found == adjacency.end())
            return {};
        auto result = found->second;
        std::sort(result.begin(), result.end());
        return result;
    }
    [[nodiscard]] FiberletReplaySourceArc arc(const DirectedFiberletStorageId& id) const override
    {
        const auto source = id.reverse ? id.fiberlet.second : id.fiberlet.first;
        const auto target = id.reverse ? id.fiberlet.first : id.fiberlet.second;
        const cv::Vec3d delta = positions.at(target) - positions.at(source);
        const float edgeLength = static_cast<float>(cv::norm(delta));
        const auto found = edgeCostDensity.find(id.fiberlet);
        const float density = found == edgeCostDensity.end()
            ? 1.0F
            : found->second;
        return {id, source, target, positions.at(source), positions.at(target), cv::Vec3f(delta), cv::Vec3f(delta), edgeLength, {0, edgeLength * density, 0, 0, 0}, std::nullopt, std::nullopt};
    }
    [[nodiscard]] FiberletReplaySourceCostProfile costProfile(const DirectedFiberletStorageId& id) const override
    {
        const auto edge = arc(id);
        return {{edge.pathLengthPredictionVoxels},
                {edge.cost.total() / edge.pathLengthPredictionVoxels}};
    }
    [[nodiscard]] std::vector<cv::Vec3d> routePoints(const DirectedFiberletStorageId& id) const override
    {
        const auto edge = arc(id);
        return {edge.sourcePositionBaseXYZ, edge.targetPositionBaseXYZ};
    }
    [[nodiscard]] std::optional<FiberletReplaySourceTransition> transition(const FiberletReplaySourceArc& incoming, const FiberletReplaySourceArc& outgoing) const override
    {
        if (incoming.target != outgoing.source || incoming.id.fiberlet == outgoing.id.fiberlet) {
            return std::nullopt;
        }
        const cv::Vec3d left = incoming.endStepBaseXYZ / cv::norm(incoming.endStepBaseXYZ);
        const cv::Vec3d right = outgoing.startStepBaseXYZ / cv::norm(outgoing.startStepBaseXYZ);
        if (!(left.dot(right) > std::cos(45.0 * 3.14159265358979323846 / 180.0)))
            return std::nullopt;
        return FiberletReplaySourceTransition{incoming.id, outgoing.id, {0.0F, joinCost, 0.0F, 0.0F, 0.0F}, std::nullopt};
    }

    std::map<FiberletStorageKey, cv::Vec3d> positions;
    std::map<FiberletStorageKey, std::vector<DirectedFiberletStorageId>> adjacency;
    std::map<FiberletStorageKey, std::chrono::milliseconds> queryDelay;
    std::map<FiberletStorageKey, std::string> queryFailure;
    std::map<FiberletStorageId, float> edgeCostDensity;
    bool concurrentQueries = false;
    bool observeConcurrency = false;
    mutable std::atomic<int> activeQueries{0};
    mutable std::atomic<int> maximumConcurrentQueries{0};
    float joinCost = 0.0F;
};

FiberletStoredAnchor anchor(FiberletStorageKey id, cv::Vec3f point, cv::Vec3f axis, float presence)
{
    FiberletStoredAnchor result;
    result.key = id;
    result.positionPredictionXYZ = point;
    result.fittedAxisXYZ = axis;
    result.predictionPresence = presence;
    result.predictionValid = true;
    result.predictionPresenceValid = true;
    return result;
}

struct TemporaryDirectory {
    explicit TemporaryDirectory(std::string_view tag)
    {
        std::mt19937_64 random(std::random_device{}());
        path = std::filesystem::temp_directory_path() / ("vc_fiberlet_normal_compat_" + std::string(tag) + "_" + std::to_string(random()));
        std::filesystem::create_directories(path);
    }
    ~TemporaryDirectory() { std::filesystem::remove_all(path); }
    std::filesystem::path path;
};

void createU8Zarr(const std::filesystem::path& path, const std::array<std::size_t, 3>& shape, const std::array<std::size_t, 3>& chunks)
{
    utils::ZarrMetadata metadata;
    metadata.version = utils::ZarrVersion::v2;
    metadata.shape = {shape[0], shape[1], shape[2]};
    metadata.chunks = {chunks[0], chunks[1], chunks[2]};
    metadata.dtype = utils::ZarrDtype::uint8;
    metadata.compressor_id.clear();
    metadata.fill_value = 0.0;
    (void)utils::ZarrArray::create(path, metadata);
}

FiberletDatasetMetadata normalCompatibilityMetadata(const std::array<std::size_t, 3>& predictionShapeZYX, double predictionToBase)
{
    FiberletDatasetMetadata metadata;
    metadata.predictionToBaseScale = predictionToBase;
    metadata.coordinateOriginZYX = {0, 0, 0};
    metadata.processing = {
        {"grid", {{"coordinate_order", "zyx_storage_xyz_vectors"}, {"prediction_to_base", predictionToBase}, {"shape_zyx", predictionShapeZYX}}},
    };
    metadata.sources = {
        {"normal_prediction", {{"manifest_content_hash", "deliberately-not-the-current-manifest"}}},
    };
    return metadata;
}

std::filesystem::path createNormalManifest(
    const TemporaryDirectory& directory,
    const std::array<std::size_t, 3>& baseShapeZYX,
    const std::array<std::size_t, 3>& nxShapeZYX,
    const std::array<std::size_t, 3>& nxChunksZYX,
    const std::array<std::size_t, 3>& nyShapeZYX,
    const std::array<std::size_t, 3>& nyChunksZYX,
    int nxScaledown = 1,
    int nyScaledown = 1,
    bool includeNy = true,
    double gradMagFactor = 1.0)
{
    createU8Zarr(directory.path / "nx.zarr", nxShapeZYX, nxChunksZYX);
    if (includeNy)
        createU8Zarr(directory.path / "ny.zarr", nyShapeZYX, nyChunksZYX);
    createU8Zarr(directory.path / "grad_mag.zarr", nxShapeZYX, nxChunksZYX);
    nlohmann::json groups{
        {"nx", {{"zarr", "nx.zarr"}, {"scaledown", nxScaledown}, {"channels", {"nx"}}}},
        {"grad_mag", {{"zarr", "grad_mag.zarr"}, {"scaledown", nxScaledown}, {"channels", {"grad_mag"}}}},
    };
    if (includeNy) {
        groups["ny"] = {
            {"zarr", "ny.zarr"},
            {"scaledown", nyScaledown},
            {"channels", {"ny"}},
        };
    }
    const nlohmann::json manifest{
        {"version", 2},
        {"source_to_base", 1.0},
        {"base_shape_zyx", baseShapeZYX},
        {"grad_mag_encode_scale", 255.0},
        {"grad_mag_factor", gradMagFactor},
        {"groups", std::move(groups)},
    };
    const auto path = directory.path / "dataset.lasagna.json";
    std::ofstream output(path);
    output << manifest.dump(2) << '\n';
    return path;
}

}  // namespace

TEST_CASE("Fiberlet crop search box expands every face by exact lookahead")
{
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {10.0, 20.0, 30.0};
    config.maximumBaseXYZ = {110.0, 220.0, 330.0};
    config.lookaheadDistanceBaseVoxels = 48.0;
    const auto search = fiberletCropTraceSearchBox(config);
    CHECK(search.minimumBaseXYZ == cv::Vec3d{-38.0, -28.0, -18.0});
    CHECK(search.maximumBaseXYZ == cv::Vec3d{158.0, 268.0, 378.0});

    config.lookaheadDistanceBaseVoxels = 0.0;
    CHECK_THROWS_AS(
        fiberletCropTraceSearchBox(config), std::invalid_argument);
    config.lookaheadDistanceBaseVoxels = 48.0;
    config.maximumBaseXYZ[1] = config.minimumBaseXYZ[1];
    CHECK_THROWS_AS(
        fiberletCropTraceSearchBox(config), std::invalid_argument);
}

TEST_CASE("Fiberlet crop tracing is bidirectional and uses anisotropic directional coverage")
{
    TestGraph graph;
    graph.joinCost = 2.0F;
    const auto outsideLeft = key(-10);
    const auto left = key(20);
    const auto seed = key(50);
    const auto right = key(80);
    const auto outsideRight = key(110);
    graph.addAnchor(outsideLeft, {-10, 0, 0});
    graph.addAnchor(left, {20, 0, 0});
    graph.addAnchor(seed, {50, 0, 0});
    graph.addAnchor(right, {80, 0, 0});
    graph.addAnchor(outsideRight, {110, 0, 0});
    graph.connect(outsideLeft, left);
    graph.connect(left, seed);
    graph.connect(seed, right);
    graph.connect(right, outsideRight);

    const auto parallel = key(51);
    const auto crossing = key(52);
    const auto normalFar = key(53);
    std::vector<FiberletStoredAnchor> anchors{
        anchor(seed, {50, 0, 0}, {1, 0, 0}, 1.0F),
        anchor(parallel, {50, 50, 0}, {1, 0, 0}, 0.8F),
        anchor(crossing, {50, 0, 0}, {0, 1, 0}, 0.7F),
        anchor(normalFar, {50, 0, 21}, {1, 0, 0}, 0.6F),
    };
    ZNormalSampler normals;
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {0, -100, -100};
    config.maximumBaseXYZ = {100, 100, 100};
    config.lookaheadDistanceBaseVoxels = 48;
    const auto result = traceFiberletCrop(graph, anchors, normals, 1.0, config);

    REQUIRE(result.lines.size() == 1);
    CHECK(result.bidirectionalLines == 1);
    CHECK(result.oneSidedLines == 0);
    CHECK(result.coveredAnchors == 1);
    CHECK(result.attemptedAnchors == 3);
    CHECK(result.noEdgeAnchors == 2);
    CHECK(result.lines.front().negativeTermination == "crop_boundary");
    CHECK(result.lines.front().positiveTermination == "crop_boundary");
    CHECK(result.lines.front().seedBaseXYZ == cv::Vec3d{50, 0, 0});
    REQUIRE(result.lines.front().pointsBaseXYZ.size() >= 5);
    CHECK(result.lines.front().pointsBaseXYZ.front()[0] == doctest::Approx(0));
    CHECK(result.lines.front().pointsBaseXYZ.back()[0] == doctest::Approx(100));
    CHECK(result.lines.front().pathLengthPredictionVoxels == doctest::Approx(100.0));
    CHECK(result.lines.front().totalMetricCost == doctest::Approx(106.0));

    auto limitedConfig = config;
    limitedConfig.maximumAttempts = 2;
    const auto limited = traceFiberletCrop(graph, anchors, normals, 1.0, limitedConfig);
    CHECK(limited.attemptedAnchors == 2);
    CHECK(limited.coveredAnchors == 1);
    CHECK(limited.lines.size() == 1);
    CHECK(limited.noEdgeAnchors == 1);
}

TEST_CASE("Fiberlet crop lookahead ranks beyond output boundary but clips output")
{
    TestGraph graph;
    const auto seed = key(50);
    const auto branch = key(90);
    const auto cheapFirst = key(110);
    const auto goodFirst = key(110, 1);
    const auto expensiveFuture = key(160);
    const auto goodFuture = key(160, 1);
    graph.addAnchor(seed, {50, 0, 0});
    graph.addAnchor(branch, {90, 0, 0});
    graph.addAnchor(cheapFirst, {110, 0, 0});
    graph.addAnchor(goodFirst, {110, 10, 0});
    graph.addAnchor(expensiveFuture, {160, 0, 0});
    graph.addAnchor(goodFuture, {160, 10, 0});
    graph.connect(seed, branch);
    graph.connect(branch, cheapFirst);
    graph.connect(branch, goodFirst);
    graph.connect(cheapFirst, expensiveFuture);
    graph.connect(goodFirst, goodFuture);
    graph.setCostDensity(branch, cheapFirst, 0.1F);
    graph.setCostDensity(cheapFirst, expensiveFuture, 10.0F);
    graph.setCostDensity(branch, goodFirst, 0.2F);
    graph.setCostDensity(goodFirst, goodFuture, 0.01F);

    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {0, -100, -100};
    config.maximumBaseXYZ = {100, 100, 100};
    config.lookaheadDistanceBaseVoxels = 40.0;
    config.maximumAttempts = 1;
    ZNormalSampler normals;
    const auto result = traceFiberletCrop(
        graph,
        {anchor(seed, {50, 0, 0}, {1, 0, 0}, 1.0F)},
        normals,
        1.0,
        config);

    REQUIRE(result.lines.size() == 1);
    const auto& line = result.lines.front();
    CHECK(line.positiveTermination == "crop_boundary");
    REQUIRE(line.pointsBaseXYZ.size() == 3);
    CHECK(line.pointsBaseXYZ[0] == cv::Vec3d{50, 0, 0});
    CHECK(line.pointsBaseXYZ[1] == cv::Vec3d{90, 0, 0});
    CHECK(line.pointsBaseXYZ[2][0] == doctest::Approx(100.0));
    CHECK(line.pointsBaseXYZ[2][1] == doctest::Approx(5.0));
}

TEST_CASE("Fiberlet crop lookahead preserves lexicographic pruning and state-cap ordering")
{
    TestGraph graph;
    const auto seed = key(0);
    const auto branch = key(10);
    graph.addAnchor(seed, {0, 0, 0});
    graph.addAnchor(branch, {10, 0, 0});
    graph.connect(seed, branch);
    for (std::int64_t index = 0; index < 65; ++index) {
        const auto target = key(100 + index);
        graph.addAnchor(target, {20, static_cast<double>(index) * 0.01, 0});
        graph.connect(branch, target);
    }

    ZNormalSampler normals;
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {-100, -100, -100};
    config.maximumBaseXYZ = {1'000, 100, 100};
    config.beamWidth = 1;
    config.lookaheadDistanceBaseVoxels = 100;
    config.maximumAttempts = 1;

    const auto pruned = traceFiberletCrop(graph, {anchor(seed, {0, 0, 0}, {1, 0, 0}, 1.0F)}, normals, 1.0, config);
    REQUIRE(pruned.lines.size() == 1);
    REQUIRE(pruned.lines.front().pointsBaseXYZ.size() == 3);
    CHECK(pruned.lines.front().pointsBaseXYZ.back() == cv::Vec3d{20, 0, 0});
    CHECK(pruned.lines.front().positiveTermination == "graph_exhausted");

    config.maximumGeneratedStatesPerStep = 1;
    const auto capped = traceFiberletCrop(graph, {anchor(seed, {0, 0, 0}, {1, 0, 0}, 1.0F)}, normals, 1.0, config);
    REQUIRE(capped.lines.size() == 1);
    CHECK(capped.lines.front().pointsBaseXYZ == pruned.lines.front().pointsBaseXYZ);
    CHECK(capped.lines.front().positiveTermination == "graph_exhausted");
}

TEST_CASE("Fiberlet crop tracing computes concurrently and integrates canonically")
{
    TestGraph graph;
    const auto outsideLeft = key(-10);
    const auto left = key(20);
    const auto seed = key(50);
    const auto right = key(80);
    const auto outsideRight = key(110);
    graph.addAnchor(outsideLeft, {-10, 0, 0});
    graph.addAnchor(left, {20, 0, 0});
    graph.addAnchor(seed, {50, 0, 0});
    graph.addAnchor(right, {80, 0, 0});
    graph.addAnchor(outsideRight, {110, 0, 0});
    graph.connect(outsideLeft, left);
    graph.connect(left, seed);
    graph.connect(seed, right);
    graph.connect(right, outsideRight);

    const auto covered = key(51);
    const auto crossing = key(52);
    const auto normalFar = key(53);
    const std::vector<FiberletStoredAnchor> anchors{
        anchor(seed, {50, 0, 0}, {1, 0, 0}, 1.0F),
        anchor(covered, {50, 50, 0}, {1, 0, 0}, 0.8F),
        anchor(crossing, {50, 0, 0}, {0, 1, 0}, 0.7F),
        anchor(normalFar, {50, 0, 21}, {1, 0, 0}, 0.6F),
    };
    ZNormalSampler normals;
    FiberletCropTraceConfig serialConfig;
    serialConfig.minimumBaseXYZ = {0, -100, -100};
    serialConfig.maximumBaseXYZ = {100, 100, 100};
    serialConfig.lookaheadDistanceBaseVoxels = 48;
    serialConfig.parallelThreads = 1;
    const auto serial = traceFiberletCrop(graph, anchors, normals, 1.0, serialConfig);

    graph.concurrentQueries = true;
    graph.observeConcurrency = true;
    graph.queryDelay[seed] = std::chrono::milliseconds(40);
    FiberletCropTraceConfig parallelConfig = serialConfig;
    parallelConfig.parallelThreads = 4;
    const auto parallel = traceFiberletCrop(graph, anchors, normals, 1.0, parallelConfig);
    CHECK(graph.maximumConcurrentQueries.load() >= 2);
    CHECK(parallel.candidateAnchors == serial.candidateAnchors);
    CHECK(parallel.attemptedAnchors == serial.attemptedAnchors);
    CHECK(parallel.coveredAnchors == serial.coveredAnchors);
    CHECK(parallel.noEdgeAnchors == serial.noEdgeAnchors);
    CHECK(parallel.oneSidedLines == serial.oneSidedLines);
    CHECK(parallel.bidirectionalLines == serial.bidirectionalLines);
    REQUIRE(parallel.lines.size() == serial.lines.size());
    for (std::size_t line = 0; line < serial.lines.size(); ++line) {
        CHECK(parallel.lines[line].seed == serial.lines[line].seed);
        CHECK(parallel.lines[line].seedBaseXYZ == serial.lines[line].seedBaseXYZ);
        CHECK(parallel.lines[line].seedPresence == serial.lines[line].seedPresence);
        CHECK(parallel.lines[line].negativeTermination == serial.lines[line].negativeTermination);
        CHECK(parallel.lines[line].positiveTermination == serial.lines[line].positiveTermination);
        CHECK(parallel.lines[line].negativeFiberlets == serial.lines[line].negativeFiberlets);
        CHECK(parallel.lines[line].positiveFiberlets == serial.lines[line].positiveFiberlets);
        CHECK(parallel.lines[line].pointsBaseXYZ == serial.lines[line].pointsBaseXYZ);
    }

    graph.maximumConcurrentQueries = 0;
    parallelConfig.parallelThreads = 0;
    parallelConfig.maximumAttempts = 2;
    const auto defaultWorkers = traceFiberletCrop(graph, anchors, normals, 1.0, parallelConfig);
    CHECK(defaultWorkers.attemptedAnchors == 2);
    CHECK(defaultWorkers.lines.size() == 1);

    graph.concurrentQueries = false;
    graph.maximumConcurrentQueries = 0;
    parallelConfig.parallelThreads = 4;
    const auto unsupportedSource = traceFiberletCrop(graph, anchors, normals, 1.0, parallelConfig);
    CHECK(unsupportedSource.attemptedAnchors == 2);
    CHECK(graph.maximumConcurrentQueries.load() == 1);
}

TEST_CASE("Fiberlet crop tracing reports speculative failures in canonical order")
{
    TestGraph graph;
    const auto first = key(10);
    const auto second = key(20);
    graph.addAnchor(first, {0, 0, 0});
    graph.addAnchor(second, {200, 0, 0});
    graph.concurrentQueries = true;
    graph.queryDelay[first] = std::chrono::milliseconds(30);
    graph.queryFailure[first] = "first canonical failure";
    graph.queryFailure[second] = "later speculative failure";

    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {-100, -100, -100};
    config.maximumBaseXYZ = {1'000, 100, 100};
    config.parallelThreads = 2;
    ZNormalSampler normals;
    CHECK_THROWS_WITH_AS(traceFiberletCrop(graph, {anchor(first, {0, 0, 0}, {1, 0, 0}, 1.0F), anchor(second, {200, 0, 0}, {1, 0, 0}, 0.9F)}, normals, 1.0, config), doctest::Contains("first canonical failure"), std::runtime_error);
}

TEST_CASE("Fiberlet crop tracing ignores failures beyond a serial fiber limit")
{
    TestGraph graph;
    const auto first = key(10);
    const auto firstTarget = key(11);
    const auto later = key(20);
    graph.addAnchor(first, {0, 0, 0});
    graph.addAnchor(firstTarget, {10, 0, 0});
    graph.addAnchor(later, {200, 0, 0});
    graph.connect(first, firstTarget);
    graph.concurrentQueries = true;
    graph.queryDelay[first] = std::chrono::milliseconds(30);
    graph.queryFailure[later] = "uncommitted speculative failure";

    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {-100, -100, -100};
    config.maximumBaseXYZ = {1'000, 100, 100};
    config.parallelThreads = 2;
    config.maximumFibers = 1;
    ZNormalSampler normals;
    const auto result =
        traceFiberletCrop(graph, {anchor(first, {0, 0, 0}, {1, 0, 0}, 1.0F), anchor(later, {200, 0, 0}, {1, 0, 0}, 0.9F)}, normals, 1.0, config);
    CHECK(result.lines.size() == 1);
    CHECK(result.lines.front().seed == first);
    CHECK(result.attemptedAnchors == 1);
    CHECK(result.computedCandidates == 2);
}

TEST_CASE("Owned Fiberlet replay views retain data after the source buffer dies")
{
    const auto points = FiberletReplayRoutePointView::owned({{1, 2, 3}, {4, 5, 6}}, true);
    REQUIRE(points.leased());
    REQUIRE(points.size() == 2);
    CHECK(points.front() == cv::Vec3d{4, 5, 6});
    CHECK(points.back() == cv::Vec3d{1, 2, 3});

    FiberletReplaySourceCostProfile profile;
    profile.segmentLengthsPredictionVoxels = {2.0F, 3.0F};
    profile.segmentCostDensities = {0.25F, 0.5F};
    const auto costs = FiberletReplayCostProfileView::owned(std::move(profile), true);
    REQUIRE(costs.leased());
    CHECK(costs.segmentLengthsPredictionVoxels[0] == 3.0F);
    CHECK(costs.segmentCostDensities[1] == 0.25F);
}

TEST_CASE("Fiberlet crop attempts follow strongest-first deterministic ordering")
{
    TestGraph graph;
    const auto first = key(10);
    const auto second = key(20);
    const auto third = key(30);
    graph.addAnchor(first, {0, 0, 0});
    graph.addAnchor(key(11), {10, 0, 0});
    graph.addAnchor(second, {200, 0, 0});
    graph.addAnchor(key(21), {210, 0, 0});
    graph.addAnchor(third, {400, 0, 0});
    graph.addAnchor(key(31), {410, 0, 0});
    graph.connect(first, key(11));
    graph.connect(second, key(21));
    graph.connect(third, key(31));
    const std::vector<FiberletStoredAnchor> anchors{
        anchor(third, {400, 0, 0}, {1, 0, 0}, 0.5F),
        anchor(second, {200, 0, 0}, {1, 0, 0}, 0.9F),
        anchor(first, {0, 0, 0}, {1, 0, 0}, 0.9F),
    };
    ZNormalSampler normals;
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {-100, -100, -100};
    config.maximumBaseXYZ = {1'000, 100, 100};
    config.maximumAttempts = 2;

    const auto result = traceFiberletCrop(graph, anchors, normals, 1.0, config);
    REQUIRE(result.lines.size() == 2);
    CHECK(result.attemptedAnchors == 2);
    CHECK(result.lines[0].seed == first);
    CHECK(result.lines[1].seed == second);

    config.maximumAttempts = 10;
    config.maximumFibers = 1;
    const auto acceptedLimited = traceFiberletCrop(graph, anchors, normals, 1.0, config);
    CHECK(acceptedLimited.attemptedAnchors == 1);
    CHECK(acceptedLimited.lines.size() == 1);

    TestGraph failureGraph;
    failureGraph.addAnchor(first, {0, 0, 0});
    failureGraph.addAnchor(second, {200, 0, 0});
    failureGraph.addAnchor(key(21), {210, 0, 0});
    failureGraph.connect(second, key(21));
    config.maximumAttempts = 1;
    config.maximumFibers = 0;
    const auto failureLimited =
        traceFiberletCrop(failureGraph, {anchor(second, {200, 0, 0}, {1, 0, 0}, 0.5F), anchor(first, {0, 0, 0}, {1, 0, 0}, 0.9F)}, normals, 1.0, config);
    CHECK(failureLimited.attemptedAnchors == 1);
    CHECK(failureLimited.noEdgeAnchors == 1);
    CHECK(failureLimited.lines.empty());
}

TEST_CASE("Polyline OBJ output uses explicit consecutive line indices")
{
    std::mt19937_64 random(std::random_device{}());
    const auto path = std::filesystem::temp_directory_path() / ("vc_fiberlet_crop_" + std::to_string(random()) + ".obj");
    vc::core::io::writePolylinesObj({{{"a"}, {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}}, {{"b"}, {{0, 1, 0}, {1, 1, 0}}}}, path);
    std::ifstream input(path);
    std::ostringstream text;
    text << input.rdbuf();
    CHECK(text.str().find("l 1 2\nl 2 3\n") != std::string::npos);
    CHECK(text.str().find("l 4 5\n") != std::string::npos);
    std::filesystem::remove(path);
}

TEST_CASE("Crop trace directions fit non-orthogonal local step modes")
{
    const double inverseRootTwo = 1.0 / std::sqrt(2.0);
    const cv::Vec3d diagonal{inverseRootTwo, inverseRootTwo, 0.0};
    const auto line = [](std::vector<cv::Vec3d> points) {
        FiberletCropTraceLine result;
        result.pointsBaseXYZ = std::move(points);
        return result;
    };
    std::vector<FiberletCropTraceLine> lines{
        line({{0, 0, 0}, {10, 0, 0}}),
        line({{10, 1, 0}, {0, 1, 0}}),
        line({{0, 2, 0}, cv::Vec3d{0, 2, 0} + diagonal * 10.0}),
        line({{0, 3, 0}, {1, 3, 0}, {2, 3, 0}, {3, 3, 0}, {4, 3, 0}, cv::Vec3d{4, 3, 0} + diagonal * 4.0}),
        line({{0, 4, 0}, {3, 4, 0}, cv::Vec3d{3, 4, 0} + diagonal}),
        line({{5, 5, 5}}),
    };

    const auto classified = classifyFiberletCropDirections(lines);
    CHECK(classified.dominanceFraction == doctest::Approx(0.75));
    CHECK(std::abs(classified.direction1BaseXYZ.dot(cv::Vec3d{1, 0, 0})) == doctest::Approx(1.0));
    CHECK(std::abs(classified.direction2BaseXYZ.dot(diagonal)) == doctest::Approx(1.0));
    REQUIRE(classified.lines.size() == lines.size());
    CHECK(classified.lines[0].group == FiberDirectionGroup::Direction1);
    CHECK(classified.lines[1].group == FiberDirectionGroup::Direction1);
    CHECK(classified.lines[2].group == FiberDirectionGroup::Direction2);
    CHECK(classified.lines[3].group == FiberDirectionGroup::Mixed);
    CHECK(classified.lines[4].group == FiberDirectionGroup::Direction1);
    CHECK(classified.lines[4].direction1SupportBaseVoxels == doctest::Approx(3.0));
    CHECK(classified.lines[4].direction2SupportBaseVoxels == doctest::Approx(1.0));
    CHECK(classified.lines[4].totalLengthBaseVoxels == doctest::Approx(4.0));
    CHECK(classified.lines[5].group == FiberDirectionGroup::Mixed);
    CHECK(classified.groupCounts == std::array<std::size_t, 3>{3, 1, 2});
    CHECK(classified.analyzedSteps == 10);
    CHECK(classified.analyzedLengthBaseVoxels == doctest::Approx(42.0));

    const auto stricter = classifyFiberletCropDirections(lines, 0.9);
    CHECK(stricter.dominanceFraction == doctest::Approx(0.9));
    CHECK(std::abs(stricter.direction1BaseXYZ.dot(
                       classified.direction1BaseXYZ)) == doctest::Approx(1.0));
    CHECK(std::abs(stricter.direction2BaseXYZ.dot(
                       classified.direction2BaseXYZ)) == doctest::Approx(1.0));
    CHECK(stricter.lines[4].group == FiberDirectionGroup::Mixed);
    CHECK(stricter.lines[4].direction1SupportBaseVoxels == doctest::Approx(3.0));
    CHECK(stricter.lines[4].direction2SupportBaseVoxels == doctest::Approx(1.0));
    CHECK(stricter.groupCounts == std::array<std::size_t, 3>{2, 1, 3});

    auto reversedLines = lines;
    for (auto& reversed : reversedLines)
        std::reverse(reversed.pointsBaseXYZ.begin(), reversed.pointsBaseXYZ.end());
    const auto reversed = classifyFiberletCropDirections(reversedLines);
    CHECK(reversed.groupCounts == classified.groupCounts);
    CHECK(std::abs(reversed.direction1BaseXYZ.dot(
                       classified.direction1BaseXYZ)) == doctest::Approx(1.0));
    CHECK(std::abs(reversed.direction2BaseXYZ.dot(
                       classified.direction2BaseXYZ)) == doctest::Approx(1.0));
    REQUIRE(reversed.lines.size() == classified.lines.size());
    for (std::size_t index = 0; index < classified.lines.size(); ++index) {
        CHECK(reversed.lines[index].group == classified.lines[index].group);
        CHECK(reversed.lines[index].direction1SupportBaseVoxels ==
              doctest::Approx(
                  classified.lines[index].direction1SupportBaseVoxels));
        CHECK(reversed.lines[index].direction2SupportBaseVoxels ==
              doctest::Approx(
                  classified.lines[index].direction2SupportBaseVoxels));
        CHECK(reversed.lines[index].totalLengthBaseVoxels ==
              doctest::Approx(classified.lines[index].totalLengthBaseVoxels));
    }

    CHECK_NOTHROW(classifyFiberletCropDirections(lines, 0.500001));
    CHECK_NOTHROW(classifyFiberletCropDirections(lines, 1.0));
    CHECK_THROWS_WITH_AS(
        classifyFiberletCropDirections(lines, 0.5),
        doctest::Contains("in (0.5, 1]"),
        std::invalid_argument);
    CHECK_THROWS_WITH_AS(
        classifyFiberletCropDirections(lines, 1.000001),
        doctest::Contains("in (0.5, 1]"),
        std::invalid_argument);
    CHECK_THROWS_WITH_AS(
        classifyFiberletCropDirections(
            lines, std::numeric_limits<double>::quiet_NaN()),
        doctest::Contains("in (0.5, 1]"),
        std::invalid_argument);
}

TEST_CASE("Crop trace direction groups use gradual segment support")
{
    const auto line = [](std::vector<cv::Vec3d> points) {
        FiberletCropTraceLine result;
        result.pointsBaseXYZ = std::move(points);
        return result;
    };
    const double x = std::sqrt(0.6);
    const double y = std::sqrt(0.4);
    std::vector<FiberletCropTraceLine> lines{
        line({{0, 0, 0}, {100, 0, 0}}),
        line({{100, 1, 0}, {0, 1, 0}}),
        line({{0, 2, 0}, {0, 102, 0}}),
        line({{0, 103, 0}, {0, 3, 0}}),
        line({{0, 4, 0}, {5 * x, 4 + 5 * y, 0}, {10 * x, 4, 0}}),
    };

    const auto classified = classifyFiberletCropDirections(lines);
    CHECK(classified.direction1BaseXYZ[0] == doctest::Approx(1.0));
    CHECK(classified.direction1BaseXYZ[1] == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(classified.direction2BaseXYZ[0] == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(classified.direction2BaseXYZ[1] == doctest::Approx(1.0));
    CHECK(classified.lines[4].direction1SupportBaseVoxels == doctest::Approx(6.0));
    CHECK(classified.lines[4].direction2SupportBaseVoxels == doctest::Approx(4.0));
    CHECK(classified.lines[4].totalLengthBaseVoxels == doctest::Approx(10.0));
    CHECK(classified.lines[4].group == FiberDirectionGroup::Mixed);

    const auto permissive = classifyFiberletCropDirections(lines, 0.59);
    CHECK(permissive.lines[4].group == FiberDirectionGroup::Direction1);
}

TEST_CASE("Crop trace non-orthogonal direction supports are independent")
{
    const auto line = [](cv::Vec3d direction, double length, double y) {
        FiberletCropTraceLine result;
        result.pointsBaseXYZ = {
            {0, y, 0},
            cv::Vec3d{0, y, 0} + direction * length,
        };
        return result;
    };
    const double rootThreeOverTwo = std::sqrt(3.0) / 2.0;
    const cv::Vec3d direction1{1, 0, 0};
    const cv::Vec3d direction2{0.5, rootThreeOverTwo, 0};
    const cv::Vec3d bisector{rootThreeOverTwo, 0.5, 0};
    const cv::Vec3d mirror{rootThreeOverTwo, -0.5, 0};
    const std::vector<FiberletCropTraceLine> lines{
        line(direction1, 100, 0),
        line(-direction1, 100, 1),
        line(direction2, 100, 2),
        line(-direction2, 100, 3),
        line(bisector, 10, 4),
        line(mirror, 10, 5),
    };

    const auto classified = classifyFiberletCropDirections(lines);
    const double cross = classified.direction1BaseXYZ.dot(
        classified.direction2BaseXYZ);
    const double q = cross * cross;
    const auto expectedSupport = [&](const cv::Vec3d& direction) {
        const double alignment = bisector.dot(direction);
        return 10.0 * std::clamp(
            (alignment * alignment - q) / (1.0 - q), 0.0, 1.0);
    };
    CHECK(std::abs(cross) == doctest::Approx(0.5));
    CHECK(classified.lines[4].direction1SupportBaseVoxels ==
          doctest::Approx(expectedSupport(classified.direction1BaseXYZ)));
    CHECK(classified.lines[4].direction2SupportBaseVoxels ==
          doctest::Approx(expectedSupport(classified.direction2BaseXYZ)));
    CHECK(classified.lines[4].totalLengthBaseVoxels == doctest::Approx(10.0));
    CHECK(classified.lines[4].direction1SupportBaseVoxels +
              classified.lines[4].direction2SupportBaseVoxels >
          classified.lines[4].totalLengthBaseVoxels);
    CHECK(classified.lines[4].group == FiberDirectionGroup::Mixed);
}

TEST_CASE("Mixed fiber ablation ranking is stable and support based")
{
    FiberDirectionClassification classification;
    classification.lines.resize(5);
    classification.lines[0].group = FiberDirectionGroup::Direction1;
    classification.lines[0].direction1SupportBaseVoxels = 10.0;
    classification.lines[0].totalLengthBaseVoxels = 10.0;
    classification.lines[1].direction1SupportBaseVoxels = 8.0;
    classification.lines[1].direction2SupportBaseVoxels = 2.0;
    classification.lines[1].totalLengthBaseVoxels = 10.0;
    classification.lines[2].direction1SupportBaseVoxels = 1.0;
    classification.lines[2].direction2SupportBaseVoxels = 8.0;
    classification.lines[2].totalLengthBaseVoxels = 10.0;
    classification.lines[3].direction1SupportBaseVoxels = 4.0;
    classification.lines[3].direction2SupportBaseVoxels = 4.0;
    classification.lines[3].totalLengthBaseVoxels = 5.0;
    classification.groupCounts = {1, 0, 4};

    const auto ranked = rankMixedFiberDirections(classification);
    REQUIRE(ranked.size() == 4);
    CHECK(ranked[0].lineIndex == 1);
    CHECK(ranked[1].lineIndex == 2);
    CHECK(ranked[2].lineIndex == 3);
    CHECK(ranked[3].lineIndex == 4);
    CHECK(ranked[0].confidence == doctest::Approx(0.8));
    CHECK(ranked[2].confidence == doctest::Approx(0.8));
    CHECK(ranked[3].confidence == doctest::Approx(0.0));
}

TEST_CASE("Crop trace direction support handles collapsed fitted axes")
{
    FiberletCropTraceLine forward;
    forward.pointsBaseXYZ = {{0, 0, 0}, {8, 0, 0}};
    FiberletCropTraceLine reverse;
    reverse.pointsBaseXYZ = {{8, 1, 0}, {0, 1, 0}};

    const auto classified =
        classifyFiberletCropDirections({forward, reverse}, 1.0);
    CHECK(classified.direction1BaseXYZ[0] == doctest::Approx(1.0));
    CHECK(classified.direction2BaseXYZ[0] == doctest::Approx(1.0));
    REQUIRE(classified.lines.size() == 2);
    for (const auto& line : classified.lines) {
        CHECK(line.direction1SupportBaseVoxels == doctest::Approx(8.0));
        CHECK(line.direction2SupportBaseVoxels == doctest::Approx(0.0));
        CHECK(line.totalLengthBaseVoxels == doctest::Approx(8.0));
        CHECK(line.group == FiberDirectionGroup::Direction1);
    }
    CHECK(classified.groupCounts == std::array<std::size_t, 3>{2, 0, 0});
}

TEST_CASE("Crop trace direction support guards nearly collapsed fitted axes")
{
    const auto classifySeparation = [](double angle) {
        FiberletCropTraceLine first;
        first.pointsBaseXYZ = {{0, 0, 0}, {10, 0, 0}};
        FiberletCropTraceLine second;
        second.pointsBaseXYZ = {
            {0, 1, 0},
            {10 * std::cos(angle), 1 + 10 * std::sin(angle), 0},
        };
        return classifyFiberletCropDirections({first, second}, 0.9);
    };

    const auto collapsed = classifySeparation(5.0e-5);
    const double collapsedCross = collapsed.direction1BaseXYZ.dot(
        collapsed.direction2BaseXYZ);
    CHECK(1.0 - collapsedCross * collapsedCross < 1.0e-8);
    CHECK(collapsed.groupCounts == std::array<std::size_t, 3>{2, 0, 0});
    for (const auto& line : collapsed.lines) {
        CHECK(std::isfinite(line.direction1SupportBaseVoxels));
        CHECK(line.direction2SupportBaseVoxels == doctest::Approx(0.0));
    }

    const auto separated = classifySeparation(5.0e-4);
    const double separatedCross = separated.direction1BaseXYZ.dot(
        separated.direction2BaseXYZ);
    CHECK(1.0 - separatedCross * separatedCross > 1.0e-8);
    CHECK(separated.groupCounts == std::array<std::size_t, 3>{1, 1, 0});
    for (const auto& line : separated.lines) {
        CHECK(std::isfinite(line.direction1SupportBaseVoxels));
        CHECK(std::isfinite(line.direction2SupportBaseVoxels));
    }
}

TEST_CASE("Crop direction OBJ artifacts preserve line and seed partitions")
{
    std::mt19937_64 random(std::random_device{}());
    const auto directory = std::filesystem::temp_directory_path() / ("vc_fiberlet_direction_objs_" + std::to_string(random()));
    std::filesystem::create_directories(directory);
    const auto output = directory / "crop.lines.obj";
    std::vector<FiberletCropTraceLine> lines(3);
    for (std::size_t index = 0; index < lines.size(); ++index) {
        lines[index].seedPresence = static_cast<float>(0.9 - index * 0.1);
        lines[index].seedBaseXYZ = {
            static_cast<double>(index + 10),
            static_cast<double>(index + 20),
            static_cast<double>(index + 30),
        };
        lines[index].pointsBaseXYZ = {
            {static_cast<double>(index), 0, 0},
            {static_cast<double>(index), 1, 0},
        };
    }
    FiberDirectionClassification classification;
    classification.lines.resize(3);
    classification.lines[0].group = FiberDirectionGroup::Direction1;
    classification.lines[1].group = FiberDirectionGroup::Direction2;
    classification.lines[2].group = FiberDirectionGroup::Mixed;
    writeFiberletCropDirectionObjs(lines, classification, output);

    const auto paths = fiberDirectionObjPaths(output);
    const auto read = [](const std::filesystem::path& path) {
        std::ifstream input(path);
        std::ostringstream text;
        text << input.rdbuf();
        return text.str();
    };
    const std::string all = read(paths.all);
    const std::string direction1 = read(paths.direction1);
    const std::string direction2 = read(paths.direction2);
    const std::string mixed = read(paths.mixed);
    CHECK(all.find("o fiber_000000_presence_0_9000") != std::string::npos);
    CHECK(all.find("o fiber_000001_presence_0_8000") != std::string::npos);
    CHECK(all.find("o fiber_000002_presence_0_7000") != std::string::npos);
    CHECK(direction1.find("fiber_000000") != std::string::npos);
    CHECK(direction1.find("fiber_000001") == std::string::npos);
    CHECK(direction2.find("fiber_000001") != std::string::npos);
    CHECK(mixed.find("fiber_000002") != std::string::npos);
    CHECK(read(paths.allAnchors).find("v 10 20 30\np 1\n") != std::string::npos);
    CHECK(read(paths.direction1Anchors).find("v 10 20 30\np 1\n") != std::string::npos);
    CHECK(read(paths.direction2Anchors).find("v 11 21 31\np 1\n") != std::string::npos);
    CHECK(read(paths.mixedAnchors).find("v 12 22 32\np 1\n") != std::string::npos);

    FiberDirectionClassification oneGroup;
    oneGroup.lines.resize(1);
    oneGroup.lines.front().group = FiberDirectionGroup::Direction1;
    const auto emptyOutput = directory / "empty-groups.obj";
    writeFiberletCropDirectionObjs({lines.front()}, oneGroup, emptyOutput);
    const auto emptyPaths = fiberDirectionObjPaths(emptyOutput);
    CHECK(read(emptyPaths.direction2).find("\nv ") == std::string::npos);
    CHECK(read(emptyPaths.mixedAnchors).find("\nv ") == std::string::npos);
    std::filesystem::remove_all(directory);
}

TEST_CASE("Crop quality deciles are stable and write every rank once")
{
    std::mt19937_64 random(std::random_device{}());
    const auto directory = std::filesystem::temp_directory_path() / ("vc_fiberlet_quality_objs_" + std::to_string(random()));
    std::filesystem::create_directories(directory);
    std::vector<FiberletCropTraceLine> lines(3);
    for (std::size_t index = 0; index < lines.size(); ++index) {
        lines[index].seedBaseXYZ = {static_cast<double>(index), 0, 0};
        lines[index].seedPresence = 1.0F;
        lines[index].pointsBaseXYZ = {lines[index].seedBaseXYZ, lines[index].seedBaseXYZ + cv::Vec3d{1, 0, 0}};
        lines[index].pathLengthPredictionVoxels = 2.0;
        lines[index].totalMetricCost = static_cast<double>(6 - index * 2);
    }
    const auto histogram = classifyFiberletCropQuality(lines);
    REQUIRE(histogram.bins[0].lineIndices == std::vector<std::size_t>{2});
    REQUIRE(histogram.bins[3].lineIndices == std::vector<std::size_t>{1});
    REQUIRE(histogram.bins[6].lineIndices == std::vector<std::size_t>{0});
    std::size_t total = 0;
    for (const auto& bin : histogram.bins)
        total += bin.lineIndices.size();
    CHECK(total == lines.size());
    CHECK(histogram.bins[0].minimumCostDensity == doctest::Approx(1.0));
    CHECK(histogram.bins[3].meanCostDensity == doctest::Approx(2.0));
    CHECK(histogram.bins[6].maximumCostDensity == doctest::Approx(3.0));

    const auto selected = selectFiberletCropQuality(lines, 0.34);
    CHECK(selected.inputLines == 3);
    CHECK(selected.lineIndices == std::vector<std::size_t>{1, 2});
    CHECK(selected.effectiveFraction == doctest::Approx(2.0 / 3.0));
    REQUIRE(selected.maximumRetainedCostDensity.has_value());
    CHECK(*selected.maximumRetainedCostDensity == doctest::Approx(2.0));
    CHECK(selectFiberletCropQuality(lines, 1.0).lineIndices ==
        std::vector<std::size_t>{0, 1, 2});

    auto tied = lines;
    for (auto& line : tied)
        line.totalMetricCost = line.pathLengthPredictionVoxels;
    CHECK(selectFiberletCropQuality(tied, 0.34).lineIndices ==
        std::vector<std::size_t>{0, 1});
    const auto empty = selectFiberletCropQuality({}, 0.5);
    CHECK(empty.lineIndices.empty());
    CHECK(empty.effectiveFraction == 0.0);
    CHECK_FALSE(empty.maximumRetainedCostDensity.has_value());
    CHECK_THROWS_AS(selectFiberletCropQuality(lines, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(selectFiberletCropQuality(lines, 1.1), std::invalid_argument);
    CHECK_THROWS_AS(
        selectFiberletCropQuality(
            lines, std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument);

    const auto output = directory / "crop.obj";
    writeFiberletCropQualityArtifacts(lines, histogram, output);
    const auto paths = fiberQualityObjPaths(output);
    for (const auto& path : paths.deciles)
        CHECK(std::filesystem::is_regular_file(path));
    CHECK(std::filesystem::is_regular_file(paths.histogramCsv));
    std::ifstream csv(paths.histogramCsv);
    std::ostringstream text;
    text << csv.rdbuf();
    CHECK(text.str().find("0,10,1,2,2,2,1,1,1") != std::string::npos);
    std::filesystem::remove_all(directory);
}

TEST_CASE("Crop trace artifact publishes sparse chunks and restores ordinal order")
{
    TemporaryDirectory directory("trace_artifact");
    FiberletDatasetMetadata source;
    source.kind = FiberletDatasetKind::Combined;
    source.profile = FiberletStorageProfile::CompactDirectionsFixedCost;
    source.chunkGridShapeZYX = {8, 8, 8};
    source.coordinateUnitsPerChunkZYX = {8, 8, 8};
    source.maximumEndpointReachCoordinateUnitsZYX = {4, 4, 4};
    source.spatialChunkSideBaseVoxels = 64;
    source.costBits = 16;
    source.predictionToBaseScale = 8.0;
    source.sources = {{"fixture", "trace-artifact"}};
    source.processing = {{"fixture", true}};
    finalizeFiberletDatasetIdentity(source);

    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {64, 128, 192};
    config.maximumBaseXYZ = {256, 320, 384};
    std::vector<FiberletCropTraceLine> lines(3);
    lines[0].seedBaseXYZ = {127.5, 191.5, 255.5};
    lines[1].seedBaseXYZ = {128.0, 192.0, 256.0};
    lines[2].seedBaseXYZ = {200.0, 260.0, 300.0};
    for (std::size_t index = 0; index < lines.size(); ++index) {
        lines[index].seedPresence = static_cast<float>(1.0 - index * 0.1);
        lines[index].totalMetricCost = 10.0 + index;
        lines[index].pathLengthPredictionVoxels = 20.0 + index;
        lines[index].pointsBaseXYZ = {
            lines[index].seedBaseXYZ - cv::Vec3d{80, 0, 0},
            lines[index].seedBaseXYZ,
            lines[index].seedBaseXYZ + cv::Vec3d{80, 0, 0},
        };
    }
    const auto output = directory.path / "crop-traces.zarr";
    writeFiberletCropTraceArtifact(output, source, nlohmann::json{{"version", 2}}, config, lines);
    const auto artifact = readFiberletCropTraceArtifact(output);
    CHECK(artifact.minimumBaseXYZ == config.minimumBaseXYZ);
    CHECK(artifact.maximumBaseXYZ == config.maximumBaseXYZ);
    REQUIRE(artifact.lines.size() == lines.size());
    for (std::size_t index = 0; index < lines.size(); ++index) {
        CHECK(artifact.lines[index].seedBaseXYZ == lines[index].seedBaseXYZ);
        CHECK(artifact.lines[index].seedPresence == lines[index].seedPresence);
        CHECK(artifact.lines[index].totalMetricCost == lines[index].totalMetricCost);
        CHECK(artifact.lines[index].pathLengthPredictionVoxels == lines[index].pathLengthPredictionVoxels);
        CHECK(artifact.lines[index].pointsBaseXYZ == lines[index].pointsBaseXYZ);
    }
    CHECK_THROWS_AS(writeFiberletCropTraceArtifact(output, source, nlohmann::json{{"version", 2}}, config, lines), std::invalid_argument);

    const auto empty = directory.path / "empty-traces.zarr";
    writeFiberletCropTraceArtifact(empty, source, nlohmann::json{{"version", 2}}, config, {});
    CHECK(readFiberletCropTraceArtifact(empty).lines.empty());
    CHECK(std::distance(std::filesystem::directory_iterator(empty / "traces"), std::filesystem::directory_iterator{}) == 1);

    const auto chunkFiles = [](const std::filesystem::path& root) {
        std::vector<std::filesystem::path> result;
        for (const auto& entry : std::filesystem::directory_iterator(root / "traces")) {
            if (entry.path().filename() != ".zarray")
                result.push_back(entry.path());
        }
        std::sort(result.begin(), result.end());
        return result;
    };

    const auto missing = directory.path / "missing-traces.zarr";
    writeFiberletCropTraceArtifact(missing, source, nlohmann::json{{"version", 2}}, config, lines);
    const auto missingFiles = chunkFiles(missing);
    REQUIRE_FALSE(missingFiles.empty());
    std::filesystem::remove(missingFiles.front());
    CHECK_THROWS_WITH_AS(readFiberletCropTraceArtifact(missing), doctest::Contains("inventory"), std::invalid_argument);

    const auto duplicate = directory.path / "duplicate-traces.zarr";
    writeFiberletCropTraceArtifact(duplicate, source, nlohmann::json{{"version", 2}}, config, lines);
    auto duplicateDataset = FiberletChunkDataset::openExisting(duplicate);
    const auto duplicateFiles = chunkFiles(duplicate);
    REQUIRE(duplicateFiles.size() >= 2);
    const auto duplicateName = duplicateFiles[1].filename().string();
    int z = 0;
    int y = 0;
    int x = 0;
    REQUIRE(std::sscanf(duplicateName.c_str(), "%d.%d.%d", &z, &y, &x) == 3);
    const vc::render::ChunkKey duplicateKey{0, z, y, x};
    const auto duplicateChunk = duplicateDataset->readMaterializedChunk(FiberletStorageChunkKind::FiberTraces, duplicateKey);
    const auto duplicatePayload = std::dynamic_pointer_cast<const FiberletTraceChunkPayload>(duplicateChunk->payload);
    REQUIRE(duplicatePayload);
    auto duplicateTraces = duplicatePayload->traces;
    REQUIRE_FALSE(duplicateTraces.empty());
    duplicateTraces.front().ordinal = 0;
    const auto duplicateBytes =
        serializeFiberletTraces(duplicateDataset->codecConfig(FiberletStorageChunkKind::FiberTraces, duplicateKey), duplicateTraces);
    vc::core::util::atomicWriteBytes(duplicateDataset->chunkPath(FiberletStorageChunkKind::FiberTraces, duplicateKey), duplicateBytes);
    duplicateDataset.reset();
    CHECK_THROWS_WITH_AS(readFiberletCropTraceArtifact(duplicate), doctest::Contains("ordinals"), std::invalid_argument);

    const auto wrongOwner = directory.path / "wrong-owner-traces.zarr";
    writeFiberletCropTraceArtifact(wrongOwner, source, nlohmann::json{{"version", 2}}, config, lines);
    auto wrongDataset = FiberletChunkDataset::openExisting(wrongOwner);
    const auto wrongFiles = chunkFiles(wrongOwner);
    REQUIRE_FALSE(wrongFiles.empty());
    REQUIRE(std::sscanf(wrongFiles.front().filename().string().c_str(), "%d.%d.%d", &z, &y, &x) == 3);
    const vc::render::ChunkKey wrongKey{0, z, y, x};
    const auto wrongChunk = wrongDataset->readMaterializedChunk(FiberletStorageChunkKind::FiberTraces, wrongKey);
    const auto wrongPayload = std::dynamic_pointer_cast<const FiberletTraceChunkPayload>(wrongChunk->payload);
    REQUIRE(wrongPayload);
    auto wrongTraces = wrongPayload->traces;
    REQUIRE_FALSE(wrongTraces.empty());
    const auto oldSeed = wrongTraces.front().seedBaseXYZ;
    wrongTraces.front().seedBaseXYZ[0] += 64.0;
    for (auto& point : wrongTraces.front().pointsBaseXYZ) {
        if (point == oldSeed)
            point = wrongTraces.front().seedBaseXYZ;
    }
    const auto wrongBytes = serializeFiberletTraces(wrongDataset->codecConfig(FiberletStorageChunkKind::FiberTraces, wrongKey), wrongTraces);
    vc::core::util::atomicWriteBytes(wrongDataset->chunkPath(FiberletStorageChunkKind::FiberTraces, wrongKey), wrongBytes);
    wrongDataset.reset();
    CHECK_THROWS_WITH_AS(readFiberletCropTraceArtifact(wrongOwner), doctest::Contains("wrong owner"), std::invalid_argument);
}

TEST_CASE("Fiberlet normal compatibility accepts structural identity and legal padding")
{
    TemporaryDirectory directory("compatible");
    const std::array<std::size_t, 3> baseShape{17, 18, 19};
    const std::array<std::size_t, 3> paddedNormalShape{11, 11, 12};
    const auto manifestPath = createNormalManifest(directory, baseShape, paddedNormalShape, {4, 4, 4}, paddedNormalShape, {3, 3, 3});
    const auto normals = vc::lasagna::LasagnaDataset::open(manifestPath, vc::lasagna::LasagnaDatasetOpenOptions{4.0});
    const auto metadata = normalCompatibilityMetadata({5, 5, 5}, 4.0);

    CHECK_NOTHROW(validateFiberletNormalDatasetCompatibility(metadata, normals));
}

TEST_CASE("Fiber trace normal compatibility uses base crop geometry rather than provenance identity")
{
    TemporaryDirectory directory("trace_compatible");
    const std::array<std::size_t, 3> baseShape{17, 18, 19};
    const auto manifestPath = createNormalManifest(
        directory,
        baseShape,
        {9, 9, 10},
        {4, 4, 4},
        {9, 9, 10},
        {4, 4, 4});
    FiberletCropTraceArtifact artifact;
    artifact.metadata.kind = FiberletDatasetKind::Traces;
    artifact.metadata.predictionToBaseScale = 4.0;
    artifact.metadata.sources = {{"normal_manifest", {{"different", true}}}};
    artifact.minimumBaseXYZ = {0, 0, 0};
    artifact.maximumBaseXYZ = {19, 18, 17};

    const auto normals = vc::lasagna::LasagnaDataset::open(
        manifestPath, vc::lasagna::LasagnaDatasetOpenOptions{1.0});
    CHECK_NOTHROW(validateFiberletCropTraceNormalDatasetCompatibility(
        artifact, normals));

    const auto wrongScale = vc::lasagna::LasagnaDataset::open(
        manifestPath, vc::lasagna::LasagnaDatasetOpenOptions{2.0});
    CHECK_THROWS_WITH_AS(
        validateFiberletCropTraceNormalDatasetCompatibility(
            artifact, wrongScale),
        doctest::Contains("base coordinates"),
        std::invalid_argument);

    artifact.maximumBaseXYZ[0] = 20.0;
    CHECK_THROWS_WITH_AS(
        validateFiberletCropTraceNormalDatasetCompatibility(
            artifact, normals),
        doctest::Contains("outside"),
        std::invalid_argument);
}

TEST_CASE("Fiberlet normal compatibility rejects incompatible frame shape and channels")
{
    const std::array<std::size_t, 3> baseShape{17, 18, 19};
    const auto metadata = normalCompatibilityMetadata({5, 5, 5}, 4.0);

    SUBCASE("excessive channel padding")
    {
        TemporaryDirectory directory("padding");
        const auto manifestPath = createNormalManifest(directory, baseShape, {14, 9, 10}, {4, 4, 4}, {14, 9, 10}, {4, 4, 4});
        const auto normals = vc::lasagna::LasagnaDataset::open(manifestPath, vc::lasagna::LasagnaDatasetOpenOptions{4.0});
        CHECK_THROWS_WITH_AS(validateFiberletNormalDatasetCompatibility(metadata, normals), doctest::Contains("shape is incompatible"), std::exception);
    }

    SUBCASE("missing normal component")
    {
        TemporaryDirectory directory("missing_ny");
        const auto manifestPath = createNormalManifest(directory, baseShape, {9, 9, 10}, {4, 4, 4}, {9, 9, 10}, {4, 4, 4}, 1, 1, false);
        const auto normals = vc::lasagna::LasagnaDataset::open(manifestPath, vc::lasagna::LasagnaDatasetOpenOptions{4.0});
        CHECK_THROWS_WITH_AS(validateFiberletNormalDatasetCompatibility(metadata, normals), doctest::Contains("missing required channel 'ny'"), std::exception);
    }

    SUBCASE("component scale mismatch")
    {
        TemporaryDirectory directory("scale");
        const auto manifestPath = createNormalManifest(directory, baseShape, {9, 9, 10}, {4, 4, 4}, {5, 5, 5}, {4, 4, 4}, 1, 2);
        const auto normals = vc::lasagna::LasagnaDataset::open(manifestPath, vc::lasagna::LasagnaDatasetOpenOptions{4.0});
        CHECK_THROWS_WITH_AS(validateFiberletNormalDatasetCompatibility(metadata, normals), doctest::Contains("matching shape and base scale"), std::exception);
    }

    SUBCASE("Fiberlet grid shape mismatch")
    {
        TemporaryDirectory directory("grid");
        const auto manifestPath = createNormalManifest(directory, baseShape, {9, 9, 10}, {4, 4, 4}, {9, 9, 10}, {4, 4, 4});
        const auto normals = vc::lasagna::LasagnaDataset::open(manifestPath, vc::lasagna::LasagnaDatasetOpenOptions{4.0});
        auto wrongMetadata = metadata;
        wrongMetadata.processing["grid"]["shape_zyx"] = {6, 5, 5};
        CHECK_THROWS_AS(validateFiberletNormalDatasetCompatibility(wrongMetadata, normals), std::invalid_argument);
    }

    SUBCASE("Fiberlet coordinate order mismatch")
    {
        TemporaryDirectory directory("coordinate_order");
        const auto manifestPath = createNormalManifest(directory, baseShape, {9, 9, 10}, {4, 4, 4}, {9, 9, 10}, {4, 4, 4});
        const auto normals = vc::lasagna::LasagnaDataset::open(manifestPath, vc::lasagna::LasagnaDatasetOpenOptions{4.0});
        auto wrongMetadata = metadata;
        wrongMetadata.processing["grid"]["coordinate_order"] = "xyz";
        CHECK_THROWS_WITH_AS(validateFiberletNormalDatasetCompatibility(wrongMetadata, normals), doctest::Contains("coordinate frame"), std::exception);
    }

    SUBCASE("Fiberlet duplicated scale mismatch")
    {
        TemporaryDirectory directory("structured_scale");
        const auto manifestPath = createNormalManifest(directory, baseShape, {9, 9, 10}, {4, 4, 4}, {9, 9, 10}, {4, 4, 4});
        const auto normals = vc::lasagna::LasagnaDataset::open(manifestPath, vc::lasagna::LasagnaDatasetOpenOptions{4.0});
        auto wrongMetadata = metadata;
        wrongMetadata.processing["grid"]["prediction_to_base"] = 8.0;
        CHECK_THROWS_WITH_AS(validateFiberletNormalDatasetCompatibility(wrongMetadata, normals), doctest::Contains("scale metadata is inconsistent"), std::exception);
    }

    SUBCASE("normal working scale mismatch")
    {
        TemporaryDirectory directory("working_scale");
        const auto manifestPath = createNormalManifest(directory, baseShape, {9, 9, 10}, {4, 4, 4}, {9, 9, 10}, {4, 4, 4});
        const auto normals = vc::lasagna::LasagnaDataset::open(manifestPath, vc::lasagna::LasagnaDatasetOpenOptions{2.0});
        CHECK_THROWS_WITH_AS(validateFiberletNormalDatasetCompatibility(metadata, normals), doctest::Contains("working scale"), std::exception);
    }

    SUBCASE("invalid gradient decode metadata")
    {
        TemporaryDirectory directory("gradient");
        const auto manifestPath = createNormalManifest(directory, baseShape, {9, 9, 10}, {4, 4, 4}, {9, 9, 10}, {4, 4, 4}, 1, 1, true, 0.0);
        const auto normals = vc::lasagna::LasagnaDataset::open(manifestPath, vc::lasagna::LasagnaDatasetOpenOptions{4.0});
        CHECK_THROWS_WITH_AS(validateFiberletNormalDatasetCompatibility(metadata, normals), doctest::Contains("grad_mag_factor"), std::exception);
    }
}

TEST_CASE("Trace constraints split evenly and hard-link consecutive pieces")
{
    FiberletCropTraceLine line;
    line.pointsBaseXYZ = {
        {0, 0, 0},
        {100, 0, 0},
        {100, 0, 0},
        {1000, 0, 0},
    };
    FiberletCropTraceLine degenerate;
    degenerate.pointsBaseXYZ = {{1, 2, 3}, {1, 2, 3}};
    FiberTraceConstraintConfig config;
    CHECK(config.maximumWindingDistance == 1.5);
    config.maximumDistanceBaseVoxels = 0.0;
    config.parallelThreads = 1;
    const auto report = extractFiberTraceConstraints(
        {line, degenerate},
        config,
        [](const cv::Vec3d&, const cv::Vec3d&, double) { return 0.0; });

    CHECK(report.inputTraces == 2);
    CHECK(report.skippedDegenerateTraces == 1);
    REQUIRE(report.pieces.size() == 3);
    const double expectedSpan = (1000.0 + 2.0 * 128.0) / 3.0;
    for (const auto& piece : report.pieces) {
        CHECK(piece.endArcBaseVoxels - piece.beginArcBaseVoxels ==
              doctest::Approx(expectedSpan));
        CHECK(piece.sampleArcsBaseVoxels.front() ==
              doctest::Approx(piece.beginArcBaseVoxels));
        CHECK(piece.sampleArcsBaseVoxels.back() ==
              doctest::Approx(piece.endArcBaseVoxels));
    }
    CHECK(report.pieces[0].endArcBaseVoxels -
              report.pieces[1].beginArcBaseVoxels == doctest::Approx(128.0));
    CHECK(report.pieces[1].endArcBaseVoxels -
              report.pieces[2].beginArcBaseVoxels == doctest::Approx(128.0));
    CHECK(report.hardConstraints == 2);
    REQUIRE(report.constraints.size() == 2);
    for (const auto& constraint : report.constraints) {
        CHECK(constraint.hardContinuity);
        CHECK(constraint.parallelScore == doctest::Approx(1.0));
        CHECK(constraint.perpendicularScore == doctest::Approx(0.0));
        CHECK(constraint.windingDistance == doctest::Approx(0.0));
        CHECK(constraint.pointABaseXYZ == constraint.pointBBaseXYZ);
    }
}

TEST_CASE("Trace constraint pieces retain short and exact target lengths")
{
    FiberTraceConstraintConfig config;
    config.maximumDistanceBaseVoxels = 0.0;
    config.parallelThreads = 1;
    const auto extract = [&](double length) {
        FiberletCropTraceLine line;
        line.pointsBaseXYZ = {{0, 0, 0}, {length, 0, 0}};
        return extractFiberTraceConstraints(
            {line},
            config,
            [](const cv::Vec3d&, const cv::Vec3d&, double) { return 0.0; });
    };

    const auto shortReport = extract(300.0);
    REQUIRE(shortReport.pieces.size() == 1);
    CHECK(shortReport.pieces.front().beginArcBaseVoxels == doctest::Approx(0.0));
    CHECK(shortReport.pieces.front().endArcBaseVoxels == doctest::Approx(300.0));
    CHECK(shortReport.hardConstraints == 0);

    const auto exactReport = extract(512.0);
    REQUIRE(exactReport.pieces.size() == 1);
    CHECK(exactReport.pieces.front().beginArcBaseVoxels == doctest::Approx(0.0));
    CHECK(exactReport.pieces.front().endArcBaseVoxels == doctest::Approx(512.0));
    CHECK(exactReport.hardConstraints == 0);
}

TEST_CASE("Constraint piece lines preserve dense source geometry")
{
    FiberletCropTraceLine source;
    source.pointsBaseXYZ = {
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {5.0, 0.0, 0.0},
        {7.0, 0.0, 0.0},
    };
    FiberTraceConstraintReport constraints;
    constraints.inputTraces = 1;
    constraints.pieces.resize(2);
    constraints.pieces[0].traceIndex = 0;
    constraints.pieces[0].beginArcBaseVoxels = 0.0;
    constraints.pieces[0].endArcBaseVoxels = 7.0;
    constraints.pieces[1].traceIndex = 0;
    constraints.pieces[1].beginArcBaseVoxels = 1.0;
    constraints.pieces[1].endArcBaseVoxels = 6.0;

    const auto pieces = makeFiberTraceConstraintPieceLines(
        {source}, constraints);
    REQUIRE(pieces.size() == 2);
    CHECK(pieces[0].pointsBaseXYZ == source.pointsBaseXYZ);
    REQUIRE(pieces[1].pointsBaseXYZ.size() == 4);
    CHECK(pieces[1].pointsBaseXYZ[0] == cv::Vec3d(1.0, 0.0, 0.0));
    CHECK(pieces[1].pointsBaseXYZ[1] == source.pointsBaseXYZ[1]);
    CHECK(pieces[1].pointsBaseXYZ[2] == source.pointsBaseXYZ[2]);
    CHECK(pieces[1].pointsBaseXYZ[3] == cv::Vec3d(6.0, 0.0, 0.0));
}

TEST_CASE("Trace constraints distinguish parallel and perpendicular neighbors deterministically")
{
    FiberletCropTraceLine first;
    first.pointsBaseXYZ = {{0, 0, 0}, {256, 0, 0}};
    FiberletCropTraceLine reversedParallel;
    reversedParallel.pointsBaseXYZ = {{256, 10, 0}, {0, 10, 0}};
    FiberletCropTraceLine perpendicular;
    perpendicular.pointsBaseXYZ = {{128, -128, 0}, {128, 128, 0}};
    FiberTraceConstraintConfig config;
    config.maximumDistanceBaseVoxels = 16.0;
    config.parallelThreads = 1;
    const auto winding = [](const cv::Vec3d& a, const cv::Vec3d& b, double) {
        return cv::norm(a - b) * 0.01;
    };
    const auto serial = extractFiberTraceConstraints(
        {first, reversedParallel, perpendicular}, config, winding);
    config.parallelThreads = 4;
    const auto parallel = extractFiberTraceConstraints(
        {first, reversedParallel, perpendicular}, config, winding);
    const auto batched = extractFiberTraceConstraints(
        {first, reversedParallel, perpendicular},
        config,
        winding,
        [&winding](
            const std::vector<std::pair<cv::Vec3d, cv::Vec3d>>& connectors,
            double step,
            int) {
            std::vector<double> result;
            result.reserve(connectors.size());
            for (const auto& connector : connectors)
                result.push_back(winding(connector.first, connector.second, step));
            return result;
        });

    REQUIRE(serial.constraints.size() == 3);
    REQUIRE(parallel.constraints.size() == serial.constraints.size());
    REQUIRE(batched.constraints.size() == serial.constraints.size());
    const auto findPair = [&](std::size_t traceA, std::size_t traceB) -> const FiberTraceConstraint& {
        for (const auto& constraint : serial.constraints) {
            const auto a = serial.pieces[constraint.pieceA].traceIndex;
            const auto b = serial.pieces[constraint.pieceB].traceIndex;
            if (a == traceA && b == traceB)
                return constraint;
        }
        throw std::runtime_error("missing test constraint");
    };
    const auto& aligned = findPair(0, 1);
    CHECK(aligned.parallelScore == doctest::Approx(1.0));
    CHECK(aligned.perpendicularScore == doctest::Approx(0.0));
    const auto& crossing = findPair(0, 2);
    CHECK(crossing.parallelScore == doctest::Approx(0.0).epsilon(1.0e-9));
    CHECK(crossing.perpendicularScore == doctest::Approx(1.0).epsilon(1.0e-9));
    for (std::size_t index = 0; index < serial.constraints.size(); ++index) {
        const auto& left = serial.constraints[index];
        const auto& right = parallel.constraints[index];
        CHECK(left.pieceA == right.pieceA);
        CHECK(left.pieceB == right.pieceB);
        CHECK(left.arcABaseVoxels == right.arcABaseVoxels);
        CHECK(left.arcBBaseVoxels == right.arcBBaseVoxels);
        CHECK(left.parallelScore == right.parallelScore);
        CHECK(left.perpendicularScore == right.perpendicularScore);
        CHECK(left.windingDistance == right.windingDistance);
        CHECK(left.windingDistance == batched.constraints[index].windingDistance);
    }
}

TEST_CASE("Reference constraint selection keeps measured cross-source links")
{
    FiberletCropTraceLine firstRun;
    firstRun.pointsBaseXYZ = {{0, 0, 0}, {256, 0, 0}};
    FiberletCropTraceLine secondRunSameSource;
    secondRunSameSource.pointsBaseXYZ = {{0, 8, 0}, {256, 8, 0}};
    FiberletCropTraceLine differentSource;
    differentSource.pointsBaseXYZ = {{0, 16, 0}, {256, 16, 0}};
    FiberletCropTraceLine multiPiece;
    multiPiece.pointsBaseXYZ = {{0, 1000, 0}, {1000, 1000, 0}};

    FiberTraceConstraintConfig config;
    config.maximumDistanceBaseVoxels = 20.0;
    config.parallelThreads = 4;
    std::size_t batchCalls = 0;
    std::size_t batchedConnectors = 0;
    const auto report = extractFiberTraceConstraints(
        {firstRun, secondRunSameSource, differentSource, multiPiece},
        config,
        {},
        [&](const std::vector<std::pair<cv::Vec3d, cv::Vec3d>>& connectors,
            double,
            int) {
            ++batchCalls;
            batchedConnectors += connectors.size();
            std::vector<double> windings;
            windings.reserve(connectors.size());
            for (const auto& [a, b] : connectors)
                windings.push_back(cv::norm(a - b) * 0.01);
            return windings;
        });

    CHECK(batchCalls == 1);
    CHECK(batchedConnectors == 3);
    CHECK(report.inputTraces == 4);
    CHECK(report.hardConstraints == 2);
    REQUIRE(report.constraints.size() == 5);

    const std::vector<std::size_t> sourceIds{0, 0, 1, 2};
    const auto ordered = orderMeasuredCrossSourceFiberTraceConstraints(
        report, sourceIds);
    REQUIRE(ordered.size() == 2);
    CHECK(ordered[0].constraintIndex < ordered[1].constraintIndex);
    for (const auto& link : ordered) {
        const auto& constraint = report.constraints[link.constraintIndex];
        CHECK_FALSE(constraint.hardContinuity);
        const std::size_t traceA = report.pieces[constraint.pieceA].traceIndex;
        const std::size_t traceB = report.pieces[constraint.pieceB].traceIndex;
        CHECK(sourceIds[traceA] != sourceIds[traceB]);
        CHECK(link.ownerSource < link.targetSource);
        CHECK(link.ownerSource == std::min(sourceIds[traceA], sourceIds[traceB]));
        CHECK(link.targetSource == std::max(sourceIds[traceA], sourceIds[traceB]));
        CHECK_FALSE(link.perpendicularDominant);
        CHECK(constraint.parallelScore == doctest::Approx(1.0));
        CHECK(constraint.perpendicularScore == doctest::Approx(0.0));
        CHECK(constraint.windingDistance > 0.0);
    }

    CHECK(orderMeasuredCrossSourceFiberTraceConstraints(
              report, {0, 0, 0, 0})
              .empty());
    const auto gapped = orderMeasuredCrossSourceFiberTraceConstraints(
        report, {0, 0, 2, 4});
    REQUIRE(gapped.size() == 2);
    for (const auto& link : gapped) {
        CHECK(link.ownerSource == 0);
        CHECK(link.targetSource == 2);
        CHECK(0.5 * static_cast<double>(
                        link.targetSource - link.ownerSource) == 1.0);
    }
    CHECK_THROWS_WITH_AS(
        orderMeasuredCrossSourceFiberTraceConstraints(report, {0, 1}),
        doctest::Contains("source IDs do not match input traces"),
        std::invalid_argument);

    auto reversedTie = report;
    auto& constraint = reversedTie.constraints[ordered.front().constraintIndex];
    std::swap(constraint.pieceA, constraint.pieceB);
    constraint.parallelScore = 0.5;
    constraint.perpendicularScore = 0.5;
    const auto reordered = orderMeasuredCrossSourceFiberTraceConstraints(
        reversedTie, sourceIds);
    REQUIRE(reordered.size() == ordered.size());
    CHECK(reordered.front().ownerSource == ordered.front().ownerSource);
    CHECK(reordered.front().targetSource == ordered.front().targetSource);
    CHECK(reordered.front().perpendicularDominant);
}

TEST_CASE("Trace constraint R-tree cube hits still require Euclidean radius")
{
    FiberletCropTraceLine first;
    first.pointsBaseXYZ = {{0, 0, 0}, {64, 0, 0}};
    FiberletCropTraceLine diagonal;
    diagonal.pointsBaseXYZ = {{0, 8, 8}, {64, 8, 8}};
    FiberTraceConstraintConfig config;
    config.resampleSpacingBaseVoxels = 32.0;
    config.maximumDistanceBaseVoxels = 10.0;
    config.parallelThreads = 1;
    const auto report = extractFiberTraceConstraints(
        {first, diagonal},
        config,
        [](const cv::Vec3d&, const cv::Vec3d&, double) { return 0.0; });
    CHECK(report.measuredCandidates == 0);
    CHECK(report.constraints.empty());
}

TEST_CASE("Trace constraint OBJ views apply strict disjoint thresholds")
{
    const TemporaryDirectory directory("constraint_objs");
    FiberTraceConstraintReport report;
    const auto constraint = [](
        std::size_t a,
        std::size_t b,
        double parallel,
        double perpendicular,
        double winding,
        bool hard = false) {
        FiberTraceConstraint result;
        result.pieceA = a;
        result.pieceB = b;
        result.pointABaseXYZ = {static_cast<double>(a), 0.0, 0.0};
        result.pointBBaseXYZ = {static_cast<double>(b), 1.0, 0.0};
        result.parallelScore = parallel;
        result.perpendicularScore = perpendicular;
        result.windingDistance = winding;
        result.hardContinuity = hard;
        return result;
    };
    report.constraints = {
        constraint(1, 2, 0.4, 0.6, 0.49),
        constraint(11, 12, 0.4, 0.6, 0.5),
        constraint(3, 4, 0.6, 0.4, 0.49),
        constraint(5, 6, 0.6, 0.4, 0.5),
        constraint(7, 8, 0.5, 0.5, 1.0),
        constraint(9, 10, 1.0, 0.0, 0.0, true),
    };

    const auto output = directory.path / "links.diagnostic";
    const auto written = writeFiberTraceConstraintObjs(report, output);
    CHECK(written.paths.perpendicularSameWinding ==
          directory.path / "links_perpendicular_same_winding.obj");
    CHECK(written.paths.perpendicularSeparateWinding ==
          directory.path / "links_perpendicular_separate_winding.obj");
    CHECK(written.paths.parallelSameWinding ==
          directory.path / "links_parallel_same_winding.obj");
    CHECK(written.paths.parallelSeparateWinding ==
          directory.path / "links_parallel_separate_winding.obj");
    CHECK(written.perpendicularSameWinding == 1);
    CHECK(written.perpendicularSeparateWinding == 1);
    CHECK(written.parallelSameWinding == 1);
    CHECK(written.parallelSeparateWinding == 1);
    const auto read = [](const std::filesystem::path& path) {
        std::ifstream input(path);
        std::ostringstream text;
        text << input.rdbuf();
        return text.str();
    };
    CHECK(read(written.paths.perpendicularSameWinding).find(
              "o constraint_piece_1_2\n") != std::string::npos);
    CHECK(read(written.paths.perpendicularSeparateWinding).find(
              "o constraint_piece_11_12\n") != std::string::npos);
    CHECK(read(written.paths.parallelSameWinding).find(
              "o constraint_piece_3_4\n") != std::string::npos);
    CHECK(read(written.paths.parallelSeparateWinding).find(
              "o constraint_piece_5_6\n") != std::string::npos);
    CHECK(read(written.paths.parallelSameWinding).find(
              "constraint_piece_9_10") == std::string::npos);

    const auto defaultPaths = fiberTraceConstraintObjPaths(
        directory.path / "crop_traces_constraints");
    CHECK(defaultPaths.perpendicularSameWinding.filename() ==
          "crop_traces_constraints_perpendicular_same_winding.obj");
}

TEST_CASE("Trace constraint strength pruning is mutual and deterministic")
{
    FiberTraceConstraintReport input;
    input.inputTraces = 5;
    for (const std::size_t trace : {0UL, 0UL, 1UL, 2UL, 3UL, 4UL}) {
        FiberTraceConstraintPiece piece;
        piece.traceIndex = trace;
        piece.pieceIndex = input.pieces.size();
        input.pieces.push_back(std::move(piece));
    }
    const auto add = [&](std::size_t a,
                         std::size_t b,
                         double distance,
                         double parallel,
                         bool hard = false) {
        FiberTraceConstraint constraint;
        constraint.pieceA = a;
        constraint.pieceB = b;
        constraint.closestDistanceBaseVoxels = distance;
        constraint.parallelScore = parallel;
        constraint.perpendicularScore = 1.0 - parallel;
        constraint.hardContinuity = hard;
        input.constraints.push_back(constraint);
    };
    add(0, 1, 0.0, 1.0, true);
    add(0, 2, 10.0, 1.0);
    add(2, 3, 1.0, 1.0);
    add(3, 4, 10.0, 0.0);
    add(0, 4, 50.0, 0.75);
    add(0, 5, 100.0, 1.0);
    input.hardConstraints = 1;

    const auto pruned = pruneFiberTraceConstraintsByStrength(input, 100.0, 1);
    REQUIRE(pruned.constraints.size() == 4);
    CHECK(pruned.constraints[0].hardContinuity);
    CHECK(pruned.constraints[0].pieceA == 0);
    CHECK(pruned.constraints[0].pieceB == 1);
    CHECK_FALSE(pruned.constraints[1].hardContinuity);
    CHECK(pruned.constraints[1].pieceA == 0);
    CHECK(pruned.constraints[1].pieceB == 2);
    CHECK(pruned.constraints[2].pieceA == 0);
    CHECK(pruned.constraints[2].pieceB == 4);
    CHECK(pruned.constraints[3].pieceA == 2);
    CHECK(pruned.constraints[3].pieceB == 3);
    CHECK(pruned.report.inputTotalConstraints == 6);
    CHECK(pruned.report.retainedTotalConstraints == 4);
    CHECK(pruned.report.hardConstraints == 1);
    CHECK(pruned.report.rejectedZeroStrength == 1);
    CHECK(pruned.report.rejectedNotMutual == 1);
    CHECK(pruned.report.recoveryCandidates == 3);
    CHECK(pruned.report.expectedRecoveryBridges == 2);
    CHECK(pruned.report.recoveryBridges == 2);
    CHECK(pruned.report.capRespectingRecoveryBridges == 1);
    CHECK(pruned.report.fallbackOverflowBridges == 1);
    CHECK(pruned.report.tracesAboveTargetDegree == 2);
    CHECK(pruned.report.before.traces == 5);
    CHECK(pruned.report.before.crossTraceConstraints == 4);
    CHECK(pruned.report.before.meanDegree == doctest::Approx(1.6));
    CHECK(pruned.report.before.isolatedTraces == 1);
    CHECK(pruned.report.before.connectedComponents == 2);
    CHECK(pruned.report.mutual.crossTraceConstraints == 1);
    CHECK(pruned.report.mutual.connectedComponents == 4);
    CHECK(pruned.report.after.crossTraceConstraints == 3);
    CHECK(pruned.report.after.minimumDegree == 0);
    CHECK(pruned.report.after.meanDegree == doctest::Approx(1.2));
    CHECK(pruned.report.after.medianDegree == doctest::Approx(1.0));
    CHECK(pruned.report.after.maximumDegree == 2);
    CHECK(pruned.report.after.isolatedTraces == 1);
    CHECK(pruned.report.after.connectedComponents == 2);

    auto reordered = input;
    std::reverse(reordered.constraints.begin(), reordered.constraints.end());
    const auto reorderedPruned =
        pruneFiberTraceConstraintsByStrength(reordered, 100.0, 1);
    REQUIRE(reorderedPruned.constraints.size() == pruned.constraints.size());
    for (std::size_t index = 0; index < pruned.constraints.size(); ++index) {
        CHECK(reorderedPruned.constraints[index].pieceA ==
              pruned.constraints[index].pieceA);
        CHECK(reorderedPruned.constraints[index].pieceB ==
              pruned.constraints[index].pieceB);
        CHECK(reorderedPruned.constraints[index].hardContinuity ==
              pruned.constraints[index].hardContinuity);
    }

    const auto capTwo =
        pruneFiberTraceConstraintsByStrength(input, 100.0, 2);
    CHECK(capTwo.report.after.maximumDegree == 2);
    CHECK(capTwo.report.after.crossTraceConstraints == 4);
    CHECK(capTwo.report.rejectedZeroStrength == 1);
    CHECK(capTwo.report.recoveryBridges == 0);

    auto ambiguous = input;
    ambiguous.constraints.back().closestDistanceBaseVoxels = 1.0;
    ambiguous.constraints.back().parallelScore = 0.5;
    ambiguous.constraints.back().perpendicularScore = 0.5;
    const auto ambiguousPruned =
        pruneFiberTraceConstraintsByStrength(ambiguous, 100.0, 1);
    CHECK(ambiguousPruned.report.rejectedZeroStrength == 1);
    CHECK(ambiguousPruned.report.after.connectedComponents == 2);

    auto solverInput = input;
    solverInput.constraints = pruned.constraints;
    FiberTraceLabelingConfig labelingConfig;
    labelingConfig.hvOnly = true;
    labelingConfig.relaxIntegrality = true;
    labelingConfig.parallelThreads = 1;
    labelingConfig.brokenCostPerConstraint = 100.0;
    const auto labeling = solveFiberTraceLabels(solverInput, labelingConfig);
    CHECK(labeling.retainedConstraints == pruned.constraints.size());
    std::vector<FiberletCropTraceLine> traces(input.inputTraces);
    for (std::size_t trace = 0; trace < traces.size(); ++trace) {
        traces[trace].pointsBaseXYZ = {
            {0.0, static_cast<double>(trace), 0.0},
            {10.0, static_cast<double>(trace), 0.0},
        };
    }
    FiberTraceConsensusConfig consensusConfig;
    consensusConfig.cropMinimumBaseXYZ = {0.0, 0.0, 0.0};
    consensusConfig.cropMaximumBaseXYZ = {10.0, 10.0, 10.0};
    const auto consensus =
        growFiberTraceConsensus(traces, solverInput, consensusConfig);
    CHECK(consensus.retainedCrossTraceConstraints ==
          pruned.report.after.crossTraceConstraints);

    auto duplicate = input;
    duplicate.constraints.push_back(input.constraints[1]);
    CHECK_THROWS_WITH_AS(
        pruneFiberTraceConstraintsByStrength(duplicate, 100.0, 1),
        doctest::Contains("duplicate piece pair"),
        std::invalid_argument);
    auto crossTraceHard = input;
    crossTraceHard.constraints[0].pieceB = 2;
    CHECK_THROWS_WITH_AS(
        pruneFiberTraceConstraintsByStrength(crossTraceHard, 100.0, 1),
        doctest::Contains("hard link crosses"),
        std::invalid_argument);
    CHECK_THROWS_WITH_AS(
        pruneFiberTraceConstraintsByStrength(input, 0.0, 1),
        doctest::Contains("maximum distance"),
        std::invalid_argument);
    CHECK_THROWS_WITH_AS(
        pruneFiberTraceConstraintsByStrength(input, 100.0, 0),
        doctest::Contains("limit must be positive"),
        std::invalid_argument);
}

TEST_CASE("Trace constraint strength pruning counts piece links per source fiber")
{
    FiberTraceConstraintReport input;
    input.inputTraces = 2;
    for (const std::size_t trace : {0UL, 0UL, 1UL}) {
        FiberTraceConstraintPiece piece;
        piece.traceIndex = trace;
        input.pieces.push_back(std::move(piece));
    }
    input.constraints = {
        {0, 2, 0.0, 0.0, {}, {}, 1.0, 1.0, 0.0, 0.0, false},
        {1, 2, 0.0, 0.0, {}, {}, 2.0, 1.0, 0.0, 0.0, false},
    };
    const auto one = pruneFiberTraceConstraintsByStrength(input, 100.0, 1);
    REQUIRE(one.constraints.size() == 1);
    CHECK(one.constraints.front().pieceA == 0);
    CHECK(one.constraints.front().pieceB == 2);
    CHECK(one.report.after.maximumDegree == 1);
}

TEST_CASE("Trace constraint recovery overflow is not an infeasibility proof")
{
    FiberTraceConstraintReport input;
    input.inputTraces = 4;
    for (std::size_t trace = 0; trace < input.inputTraces; ++trace) {
        FiberTraceConstraintPiece piece;
        piece.traceIndex = trace;
        input.pieces.push_back(std::move(piece));
    }
    const auto link = [](std::size_t a, std::size_t b, double distance) {
        FiberTraceConstraint constraint;
        constraint.pieceA = a;
        constraint.pieceB = b;
        constraint.closestDistanceBaseVoxels = distance;
        constraint.parallelScore = 1.0;
        return constraint;
    };
    input.constraints = {
        link(0, 1, 1.0),
        link(0, 2, 2.0),
        link(0, 3, 3.0),
        link(1, 2, 4.0),
    };

    const auto pruned = pruneFiberTraceConstraintsByStrength(input, 100.0, 2);
    CHECK(pruned.report.mutual.crossTraceConstraints == 3);
    CHECK(pruned.report.expectedRecoveryBridges == 1);
    CHECK(pruned.report.fallbackOverflowBridges == 1);
    CHECK(pruned.report.tracesAboveTargetDegree == 1);
    CHECK(pruned.report.after.maximumDegree == 3);
    CHECK(pruned.report.after.connectedComponents == 1);
    // AB, BC, and AD would connect the same input graph with maximum degree two.
    // The observed overflow is therefore a property of greedy mutual selection,
    // not a proof that a degree-bounded spanning subgraph is impossible.
}

TEST_CASE("Trace constraints discard winding distances at the exclusive cutoff")
{
    FiberletCropTraceLine first;
    first.pointsBaseXYZ = {{0, 0, 0}, {64, 0, 0}};
    FiberletCropTraceLine second;
    second.pointsBaseXYZ = {{0, 4, 0}, {64, 4, 0}};
    FiberTraceConstraintConfig config;
    config.maximumDistanceBaseVoxels = 8.0;
    config.parallelThreads = 1;
    const auto extract = [&](double winding) {
        return extractFiberTraceConstraints(
            {first, second},
            config,
            [winding](const cv::Vec3d&, const cv::Vec3d&, double) {
                return winding;
            });
    };

    const auto retained = extract(std::nextafter(1.5, 0.0));
    REQUIRE(retained.constraints.size() == 1);
    CHECK(retained.rejectedWinding == 0);
    CHECK(retained.rejectedWindingCutoff == 0);

    const auto cutoff = extract(1.5);
    CHECK(cutoff.constraints.empty());
    CHECK(cutoff.rejectedWinding == 0);
    CHECK(cutoff.rejectedWindingCutoff == 1);

    config.maximumWindingDistance = 4.0;
    const auto extended = extract(std::nextafter(4.0, 0.0));
    REQUIRE(extended.constraints.size() == 1);
    CHECK(extended.rejectedWindingCutoff == 0);
    const auto extendedCutoff = extract(4.0);
    CHECK(extendedCutoff.constraints.empty());
    CHECK(extendedCutoff.rejectedWindingCutoff == 1);

    config.enforceMaximumWindingDistance = false;
    const auto unbounded = extract(42.0);
    REQUIRE(unbounded.constraints.size() == 1);
    CHECK(unbounded.constraints.front().windingDistance == doctest::Approx(42.0));
    CHECK(unbounded.rejectedWinding == 0);
    CHECK(unbounded.rejectedWindingCutoff == 0);

    const auto invalid = extract(std::numeric_limits<double>::quiet_NaN());
    CHECK(invalid.constraints.empty());
    CHECK(invalid.rejectedWinding == 1);
    CHECK(invalid.rejectedWindingCutoff == 0);
}

TEST_CASE("Trace labeling minimizes orientation winding and broken costs")
{
    const auto makeReport = [](double parallel, double winding) {
        FiberTraceConstraintReport report;
        report.pieces.resize(2);
        report.constraints.push_back({
            0,
            1,
            0.0,
            0.0,
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            1.0,
            parallel,
            1.0 - parallel,
            winding,
            false,
        });
        return report;
    };
    FiberTraceLabelingConfig config;
    config.parallelThreads = 1;

    const auto parallel = solveFiberTraceLabels(makeReport(0.9, 0.0), config);
    REQUIRE(parallel.labels.size() == 2);
    CHECK(parallel.labels[0] == FiberTracePieceLabel::HEven);
    CHECK(parallel.labels[1] == FiberTracePieceLabel::HEven);
    CHECK(parallel.orientationCost == doctest::Approx(0.1));
    CHECK(parallel.windingCost == doctest::Approx(0.0));
    CHECK(parallel.brokenCost == doctest::Approx(0.0));
    CHECK(parallel.objective == doctest::Approx(0.1));

    const auto crossing = solveFiberTraceLabels(makeReport(0.1, 0.9), config);
    REQUIRE(crossing.labels.size() == 2);
    CHECK(crossing.labels[0] == FiberTracePieceLabel::HEven);
    CHECK(crossing.labels[1] == FiberTracePieceLabel::VOdd);
    CHECK(crossing.orientationCost == doctest::Approx(0.1));
    CHECK(crossing.windingCost == doctest::Approx(0.1));
    CHECK(crossing.objective == doctest::Approx(0.2));

    config.hvOnly = true;
    const auto hvOnly = solveFiberTraceLabels(makeReport(0.1, 0.9), config);
    REQUIRE(hvOnly.labels.size() == 2);
    CHECK(hvOnly.hvOnly);
    CHECK(hvOnly.labels[0] == FiberTracePieceLabel::HEven);
    CHECK(hvOnly.labels[1] == FiberTracePieceLabel::VEven);
    CHECK(hvOnly.orientationCost == doctest::Approx(0.1));
    CHECK(hvOnly.windingCost == doctest::Approx(0.0));
    CHECK(hvOnly.objective == doctest::Approx(0.1));
    CHECK(hvOnly.variables == 6);
    CHECK(hvOnly.integerVariables == 4);
    CHECK(hvOnly.rows == 10);
    for (const double odd : hvOnly.oddValues)
        CHECK(odd == 0.0);
    const auto unboundedWinding = solveFiberTraceLabels(
        makeReport(0.1, 42.0), config);
    CHECK(unboundedWinding.objective == doctest::Approx(0.1));
    CHECK(unboundedWinding.windingCost == doctest::Approx(0.0));
    config.hvOnly = false;
    CHECK_THROWS_AS(
        solveFiberTraceLabels(makeReport(0.1, 42.0), config),
        std::invalid_argument);

    const auto broken = solveFiberTraceLabels(makeReport(0.5, 0.5), config);
    REQUIRE(broken.labels.size() == 2);
    CHECK((broken.labels[0] == FiberTracePieceLabel::Broken) !=
          (broken.labels[1] == FiberTracePieceLabel::Broken));
    CHECK(broken.brokenCost == doctest::Approx(0.5));
    CHECK(broken.orientationCost == doctest::Approx(0.0));
    CHECK(broken.windingCost == doctest::Approx(0.0));
    CHECK(broken.objective == doctest::Approx(0.5));
    CHECK(broken.variables == 9);
    CHECK(broken.integerVariables == 6);
    CHECK(broken.rows == 17);

    auto isolated = makeReport(0.9, 0.0);
    isolated.pieces.emplace_back();
    const auto canonical = solveFiberTraceLabels(isolated, config);
    CHECK(canonical.labels[2] == FiberTracePieceLabel::Broken);
    CHECK(canonical.labels[0] == FiberTracePieceLabel::HEven);

    config.brokenCostPerConstraint = -0.01;
    CHECK_THROWS_AS(solveFiberTraceLabels(isolated, config), std::invalid_argument);
    config.brokenCostPerConstraint = 0.5;
    config.relativeMipGap = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(solveFiberTraceLabels(isolated, config), std::invalid_argument);
}

TEST_CASE("Trace labeling writes five stable piece OBJ classes")
{
    const TemporaryDirectory directory("trace_label_objs");
    FiberTraceConstraintReport constraints;
    constraints.pieces.resize(5);
    for (std::size_t index = 0; index < constraints.pieces.size(); ++index) {
        constraints.pieces[index].traceIndex = 10 + index;
        constraints.pieces[index].pieceIndex = index;
        constraints.pieces[index].samplePointsBaseXYZ = {
            {static_cast<double>(index), 0.0, 0.0},
            {static_cast<double>(index), 1.0, 0.0},
        };
    }
    FiberTraceLabelingReport labeling;
    labeling.labels = {
        FiberTracePieceLabel::HEven,
        FiberTracePieceLabel::HOdd,
        FiberTracePieceLabel::VEven,
        FiberTracePieceLabel::VOdd,
        FiberTracePieceLabel::Broken,
    };
    const auto result = writeFiberTraceLabelObjs(
        constraints, labeling, directory.path / "crop.labels");
    CHECK(result.paths.hEven == directory.path / "crop_h_even.obj");
    CHECK(result.paths.hOdd == directory.path / "crop_h_odd.obj");
    CHECK(result.paths.vEven == directory.path / "crop_v_even.obj");
    CHECK(result.paths.vOdd == directory.path / "crop_v_odd.obj");
    CHECK(result.paths.broken == directory.path / "crop_broken.obj");
    for (const auto count : result.pieceCounts)
        CHECK(count == 1);
    std::ifstream input(result.paths.vOdd);
    std::ostringstream text;
    text << input.rdbuf();
    CHECK(text.str().find("o piece_3_trace_13_part_3\n") != std::string::npos);
}

TEST_CASE("Trace labeling exposes a continuous LP relaxation")
{
    FiberTraceConstraintReport constraints;
    constraints.pieces.resize(2);
    constraints.pieces[0].traceIndex = 4;
    constraints.pieces[1].traceIndex = 9;
    constraints.pieces[0].samplePointsBaseXYZ = {
        {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    constraints.pieces[1].samplePointsBaseXYZ = {
        {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}};
    constraints.constraints.push_back({
        0, 1, 0.0, 0.0, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        1.0, 0.7, 0.3, 0.2, false});

    FiberTraceLabelingConfig config;
    config.parallelThreads = 1;
    config.relaxIntegrality = true;
    config.brokenCostPerConstraint = 10.0;
    const auto labeling = solveFiberTraceLabels(constraints, config);
    CHECK(labeling.labels.empty());
    CHECK(labeling.integerVariables == 0);
    REQUIRE(labeling.activeValues.size() == 2);
    REQUIRE(labeling.verticalValues.size() == 2);
    REQUIRE(labeling.oddValues.size() == 2);
    for (const auto* values : {
             &labeling.activeValues,
             &labeling.verticalValues,
             &labeling.oddValues}) {
        for (const double value : *values) {
            CHECK(value >= 0.0);
            CHECK(value <= 1.0);
        }
    }

    config.lpParallel = true;
    config.lpSolver = "simplex";
    const auto parallelSimplex = solveFiberTraceLabels(constraints, config);
    CHECK(parallelSimplex.modelStatus == "Optimal");
    CHECK(parallelSimplex.objective == doctest::Approx(labeling.objective));
    CHECK(parallelSimplex.rows == labeling.rows);
    CHECK(parallelSimplex.variables == labeling.variables);
    REQUIRE(parallelSimplex.activeValues.size() == labeling.activeValues.size());
    for (std::size_t piece = 0; piece < labeling.activeValues.size(); ++piece) {
        CHECK(parallelSimplex.activeValues[piece] ==
              doctest::Approx(labeling.activeValues[piece]));
        CHECK(parallelSimplex.verticalValues[piece] ==
              doctest::Approx(labeling.verticalValues[piece]));
        CHECK(parallelSimplex.oddValues[piece] ==
              doctest::Approx(labeling.oddValues[piece]));
    }

    const TemporaryDirectory directory("trace_label_relaxation");
    const auto path = writeFiberTraceLabelRelaxationCsv(
        constraints, labeling, directory.path / "crop.labels");
    CHECK(path.filename() == "crop_values.csv");
    std::ifstream input(path);
    std::ostringstream text;
    text << input.rdbuf();
    const std::string csv = text.str();
    CHECK(csv.find("piece_id,trace_index,piece_index,active,vertical,odd\n") == 0);
    CHECK(csv.find("0,4,0,") != std::string::npos);
    CHECK(csv.find("1,9,0,") != std::string::npos);

    config.hvOnly = true;
    const auto hvOnly = solveFiberTraceLabels(constraints, config);
    CHECK(hvOnly.hvOnly);
    CHECK(hvOnly.variables == 6);
    CHECK(hvOnly.rows == 10);
    CHECK(hvOnly.windingCost == doctest::Approx(0.0));
    for (const double odd : hvOnly.oddValues)
        CHECK(odd == 0.0);
    const auto hvVisual = writeFiberTraceLabelRelaxationObjs(
        constraints, hvOnly, directory.path / "crop.hv_only");
    CHECK(hvVisual.objects.pieceCounts[1] == 0);
    CHECK(hvVisual.objects.pieceCounts[3] == 0);
    for (const auto& path : {
             hvVisual.objects.paths.hOdd,
             hvVisual.objects.paths.vOdd}) {
        CHECK(std::filesystem::is_regular_file(path));
        std::ifstream oddInput(path);
        std::ostringstream oddText;
        oddText << oddInput.rdbuf();
        CHECK(oddText.str().find("\no ") == std::string::npos);
    }

    FiberTraceConstraintReport visualConstraints;
    visualConstraints.pieces.resize(5);
    for (std::size_t index = 0; index < 5; ++index) {
        visualConstraints.pieces[index].traceIndex = index;
        visualConstraints.pieces[index].samplePointsBaseXYZ = {
            {static_cast<double>(index), 0.0, 0.0},
            {static_cast<double>(index), 1.0, 0.0},
        };
    }
    FiberTraceLabelingReport visualLabels;
    visualLabels.activeValues = {0.9, 0.7, 0.8, 0.4, 0.6};
    visualLabels.verticalValues = {0.4, 0.5, 0.49, 1.0, 1.0};
    visualLabels.oddValues = {0.4, 0.49, 0.5, 1.0, 1.0};
    const auto visual = writeFiberTraceLabelRelaxationObjs(
        visualConstraints, visualLabels, directory.path / "crop.labels");
    CHECK(visual.activeThreshold == doctest::Approx(0.68));
    CHECK(visual.objects.paths.hEven.filename() ==
          "crop_h_even.obj");
    CHECK(visual.objects.paths.broken.filename() ==
          "crop_broken.obj");
    CHECK(visual.objects.pieceCounts[0] == 1);
    CHECK(visual.objects.pieceCounts[1] == 1);
    CHECK(visual.objects.pieceCounts[2] == 1);
    CHECK(visual.objects.pieceCounts[3] == 0);
    CHECK(visual.objects.pieceCounts[4] == 2);
}

TEST_CASE("Trace labeling rejects LP backend controls outside relaxation mode")
{
    FiberTraceConstraintReport constraints;
    FiberTraceLabelingConfig config;
    config.lpParallel = true;
    CHECK_THROWS_WITH_AS(
        solveFiberTraceLabels(constraints, config),
        doctest::Contains("LP solver options require relaxation mode"),
        std::invalid_argument);

    config.lpParallel = false;
    config.relaxIntegrality = true;
    config.lpSolver = "invalid";
    CHECK_THROWS_WITH_AS(
        solveFiberTraceLabels(constraints, config),
        doctest::Contains("LP solver must be choose, simplex, hipo, or ipm"),
        std::invalid_argument);
}

TEST_CASE("Direction label comparison aligns components and reports errors")
{
    FiberTraceConstraintReport constraints;
    constraints.inputTraces = 4;
    for (const auto [trace, local] :
         std::array<std::pair<std::size_t, std::size_t>, 6>{
             {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {2, 0}, {3, 0}}}) {
        FiberTraceConstraintPiece piece;
        piece.traceIndex = trace;
        piece.pieceIndex = local;
        piece.beginArcBaseVoxels = static_cast<double>(local * 10);
        piece.endArcBaseVoxels = piece.beginArcBaseVoxels + 10.0;
        constraints.pieces.push_back(std::move(piece));
    }
    const auto link = [](std::size_t a, std::size_t b, bool hard = false) {
        FiberTraceConstraint result;
        result.pieceA = a;
        result.pieceB = b;
        result.hardContinuity = hard;
        return result;
    };
    constraints.constraints = {
        link(0, 1, true),
        link(1, 2, true),
        link(3, 4),
        link(4, 5),
    };
    const std::array directions{
        FiberDirectionGroup::Direction1,
        FiberDirectionGroup::Direction2,
        FiberDirectionGroup::Direction1,
        FiberDirectionGroup::Direction2,
    };
    FiberTraceLabelingReport labeling;
    labeling.labels = {
        FiberTracePieceLabel::HEven,
        FiberTracePieceLabel::Broken,
        FiberTracePieceLabel::VEven,
        FiberTracePieceLabel::HEven,
        FiberTracePieceLabel::VEven,
        FiberTracePieceLabel::VEven,
    };

    const auto comparison =
        compareFiberDirectionLabels(constraints, directions, labeling);
    CHECK(comparison.rawH == 2);
    CHECK(comparison.rawV == 3);
    CHECK(comparison.rawBroken == 1);
    CHECK(comparison.activeComponents == 3);
    CHECK(comparison.flippedComponents == 2);
    CHECK(comparison.representedTraces == 4);
    CHECK(comparison.errorTraces == 2);
    CHECK(comparison.orientationErrors == 1);
    CHECK(comparison.brokenErrors == 1);
    CHECK(comparison.trusted.representedTraces == 4);
    CHECK(comparison.trusted.errorTraces == 2);
    CHECK(comparison.trusted.orientationErrors == 1);
    CHECK(comparison.trusted.brokenErrors == 1);
    CHECK(comparison.admitted.representedTraces == 0);
    CHECK(comparison.confusion[0].pieces == 4);
    CHECK(comparison.confusion[0].alignedDirection1 == 3);
    CHECK(comparison.confusion[0].alignedDirection2 == 0);
    CHECK(comparison.confusion[0].broken == 1);
    CHECK(comparison.confusion[0].errors == 1);
    CHECK(comparison.confusion[1].pieces == 2);
    CHECK(comparison.confusion[1].alignedDirection1 == 1);
    CHECK(comparison.confusion[1].alignedDirection2 == 1);
    CHECK(comparison.confusion[1].broken == 0);
    CHECK(comparison.confusion[1].errors == 1);
    REQUIRE(comparison.errors.size() == 2);
    CHECK(comparison.errors[0].pieceIndex == 1);
    CHECK(comparison.errors[0].filteredTraceIndex == 0);
    CHECK(comparison.errors[0].tracePieceIndex == 1);
    CHECK(comparison.errors[0].kind ==
          FiberDirectionLabelErrorKind::Broken);
    CHECK(comparison.errors[1].pieceIndex == 5);
    CHECK(comparison.errors[1].filteredTraceIndex == 3);
    CHECK(comparison.errors[1].componentFlipped);
    CHECK(comparison.errors[1].alignedDirection ==
          FiberDirectionGroup::Direction1);
    CHECK(comparison.errors[1].kind ==
          FiberDirectionLabelErrorKind::Orientation);
}

TEST_CASE("Direction label comparison validates discrete retained directions")
{
    FiberTraceConstraintReport constraints;
    constraints.inputTraces = 1;
    FiberTraceConstraintPiece piece;
    piece.traceIndex = 0;
    constraints.pieces.push_back(piece);
    FiberTraceLabelingReport labeling;
    labeling.labels = {FiberTracePieceLabel::Broken};
    const std::array retained{FiberDirectionGroup::Direction1};

    const auto comparison =
        compareFiberDirectionLabels(constraints, retained, labeling);
    CHECK(comparison.rawBroken == 1);
    CHECK(comparison.representedTraces == 1);
    CHECK(comparison.errorTraces == 1);
    CHECK(comparison.brokenErrors == 1);

    const std::array mixed{FiberDirectionGroup::Mixed};
    CHECK_THROWS_WITH_AS(
        compareFiberDirectionLabels(constraints, mixed, labeling),
        doctest::Contains("does not accept mixed"),
        std::invalid_argument);
    labeling.continuousPieceValues = true;
    CHECK_THROWS_WITH_AS(
        compareFiberDirectionLabels(constraints, retained, labeling),
        doctest::Contains("discrete label"),
        std::invalid_argument);
    labeling.continuousPieceValues = false;
    CHECK_THROWS_WITH_AS(
        compareFiberDirectionLabels(constraints, std::span<const FiberDirectionGroup>{}, labeling),
        doctest::Contains("trace count"),
        std::invalid_argument);
}

TEST_CASE("Direction label comparison uses trusted references for gauge")
{
    FiberTraceConstraintReport constraints;
    constraints.inputTraces = 3;
    constraints.pieces.resize(3);
    for (std::size_t piece = 0; piece < constraints.pieces.size(); ++piece)
        constraints.pieces[piece].traceIndex = piece;
    FiberTraceConstraint connected;
    connected.pieceA = 0;
    connected.pieceB = 1;
    constraints.constraints.push_back(connected);

    const std::array directions{
        FiberDirectionGroup::Direction1,
        FiberDirectionGroup::Mixed,
        FiberDirectionGroup::Mixed,
    };
    const std::array<std::uint8_t, 3> trusted{1, 0, 0};
    FiberTraceLabelingReport labeling;
    labeling.labels = {
        FiberTracePieceLabel::HEven,
        FiberTracePieceLabel::VEven,
        FiberTracePieceLabel::Broken,
    };

    const auto comparison = compareFiberDirectionLabels(
        constraints, directions, labeling, trusted);
    CHECK(comparison.activeComponents == 1);
    CHECK(comparison.flippedComponents == 0);
    CHECK(comparison.trusted.representedTraces == 1);
    CHECK(comparison.trusted.errorTraces == 0);
    CHECK(comparison.trusted.orientationErrors == 0);
    CHECK(comparison.admitted.representedTraces == 2);
    CHECK(comparison.admitted.errorTraces == 1);
    CHECK(comparison.admitted.orientationErrors == 0);
    CHECK(comparison.admitted.expectedDefectPieces == 2);
    CHECK(comparison.admitted.defectBrokenPieces == 1);
    CHECK(comparison.admitted.defectActiveErrors == 1);
    CHECK(comparison.errorTraces == 1);
    REQUIRE(comparison.errors.size() == 1);
    CHECK_FALSE(comparison.errors[0].trustedReference);
    CHECK(comparison.errors[0].kind ==
          FiberDirectionLabelErrorKind::DefectActive);

    const std::array<std::uint8_t, 2> wrongMask{1, 0};
    CHECK_THROWS_WITH_AS(
        compareFiberDirectionLabels(
            constraints, directions, labeling, wrongMask),
        doctest::Contains("trusted mask"),
        std::invalid_argument);
}

TEST_CASE("Continuous H/V labeling thresholds to discrete defects")
{
    FiberTraceLabelingReport continuous;
    continuous.continuousPieceValues = true;
    continuous.activeValues = {0.49, 0.5, 1.0};
    continuous.verticalValues = {0.0, 0.49, 0.5};
    const auto discrete = thresholdFiberTraceLabeling(continuous);
    CHECK_FALSE(discrete.continuousPieceValues);
    REQUIRE(discrete.labels.size() == 3);
    CHECK(discrete.labels[0] == FiberTracePieceLabel::Broken);
    CHECK(discrete.labels[1] == FiberTracePieceLabel::HEven);
    CHECK(discrete.labels[2] == FiberTracePieceLabel::VEven);
    CHECK(discrete.labelCounts ==
          std::array<std::size_t, 5>{1, 0, 1, 0, 1});
    CHECK_THROWS_WITH_AS(
        thresholdFiberTraceLabeling(continuous, 1.1),
        doctest::Contains("thresholds"),
        std::invalid_argument);
}

TEST_CASE("Trace labeling can exclude measured parallel separate winding links")
{
    FiberTraceConstraintReport constraints;
    constraints.pieces.resize(4);
    const auto add = [&](std::size_t a,
                         std::size_t b,
                         double parallel,
                         double winding,
                         bool hard) {
        constraints.constraints.push_back({
            a, b, 0.0, 0.0,
            {static_cast<double>(a), 0.0, 0.0},
            {static_cast<double>(b), 0.0, 0.0},
            1.0, parallel, 1.0 - parallel, winding, hard});
    };
    add(0, 1, 0.5, 0.5, false);
    add(1, 2, 0.5001, 0.5, false);
    add(2, 3, 0.9, 1.0, true);
    add(0, 3, 0.9, 0.4999, false);

    FiberTraceLabelingConfig config;
    config.parallelThreads = 1;
    config.relaxIntegrality = true;
    config.brokenCostPerConstraint = 10.0;
    const auto complete = solveFiberTraceLabels(constraints, config);
    CHECK(complete.retainedConstraints == 4);
    CHECK(complete.retainedConstraintIndices ==
          std::vector<std::size_t>{0, 1, 2, 3});
    CHECK(complete.excludedParallelSeparateWinding == 0);
    CHECK(complete.variables == 24);
    CHECK(complete.rows == 60);

    config.excludeParallelSeparateWinding = true;
    const auto filtered = solveFiberTraceLabels(constraints, config);
    CHECK(filtered.retainedConstraints == 3);
    CHECK(filtered.retainedConstraintIndices ==
          std::vector<std::size_t>{0, 2, 3});
    CHECK(filtered.excludedParallelSeparateWinding == 1);
    CHECK(filtered.variables == 21);
    CHECK(filtered.rows == 47);
    CHECK(filtered.gaugeRoots == 1);
    CHECK(filtered.triangles == 0);
}

TEST_CASE("Trace labeling can retain only measured perpendicular links")
{
    FiberTraceConstraintReport constraints;
    constraints.pieces.resize(5);
    const auto add = [&](std::size_t a,
                         std::size_t b,
                         double parallel,
                         bool hard) {
        constraints.constraints.push_back({
            a, b, 0.0, 0.0,
            {static_cast<double>(a), 0.0, 0.0},
            {static_cast<double>(b), 0.0, 0.0},
            1.0, parallel, 1.0 - parallel, 0.0, hard});
    };
    add(0, 1, 0.1, false);
    add(1, 2, 0.5, false);
    add(2, 3, 0.9, false);
    add(3, 4, 1.0, true);

    FiberTraceLabelingConfig config;
    config.parallelThreads = 1;
    config.relaxIntegrality = true;
    config.hvOnly = true;
    config.perpendicularOnly = true;
    config.brokenCostPerConstraint = 10.0;
    const auto labeling = solveFiberTraceLabels(constraints, config);
    CHECK(labeling.retainedConstraints == 2);
    CHECK(labeling.retainedConstraintIndices ==
          std::vector<std::size_t>{0, 3});
    CHECK(labeling.excludedNonPerpendicular == 2);
    CHECK(labeling.excludedParallelSeparateWinding == 0);
    CHECK(labeling.gaugeRoots == 3);
    CHECK(labeling.triangles == 0);

    config.excludeParallelSeparateWinding = true;
    CHECK_THROWS_WITH_AS(
        solveFiberTraceLabels(constraints, config),
        doctest::Contains("redundant"),
        std::invalid_argument);
}

TEST_CASE("Trace post-filter confidence uses the configured support width")
{
    CHECK(fiberTracePostFilterConfidence(0.5, 1.0) ==
          doctest::Approx(0.0));
    CHECK(fiberTracePostFilterConfidence(0.75, 1.0) ==
          doctest::Approx(0.5));
    CHECK(fiberTracePostFilterConfidence(1.0, 1.0) ==
          doctest::Approx(1.0));
    CHECK(fiberTracePostFilterConfidence(0.75, 0.5) ==
          doctest::Approx(0.0));
    CHECK(fiberTracePostFilterConfidence(0.875, 0.5) ==
          doctest::Approx(0.5));
    CHECK(fiberTracePostFilterConfidence(0.0, 0.5) ==
          doctest::Approx(1.0));
    CHECK_THROWS_AS(
        fiberTracePostFilterConfidence(0.5, 0.0),
        std::invalid_argument);
}

TEST_CASE("Trace post-filter averages unique inverted perpendicular neighbors")
{
    FiberTraceConstraintReport constraints;
    constraints.pieces.resize(3);
    for (std::size_t trace = 0; trace < 3; ++trace)
        constraints.pieces[trace].traceIndex = trace;
    const auto add = [&](std::size_t a, std::size_t b) {
        constraints.constraints.push_back({
            a, b, 0.0, 0.0,
            {static_cast<double>(a), 0.0, 0.0},
            {static_cast<double>(b), 0.0, 0.0},
            1.0, 0.0, 1.0, 0.0, false});
    };
    add(0, 1);
    add(0, 1);
    add(0, 2);

    FiberTraceLabelingReport labeling;
    labeling.labels = {
        FiberTracePieceLabel::Broken,
        FiberTracePieceLabel::VEven,
        FiberTracePieceLabel::HEven,
    };
    labeling.retainedConstraintIndices = {0, 1, 2};
    const auto once = postFilterPerpendicularFiberTraceLabels(
        constraints, labeling, 3, {1, 1.0});
    REQUIRE(once.size() == 3);
    CHECK(once[0] == doctest::Approx(0.5));
    CHECK(once[1] == doctest::Approx(0.0));
    CHECK(once[2] == doctest::Approx(1.0));

    FiberTraceConstraintReport propagation;
    propagation.pieces.resize(2);
    propagation.pieces[0].traceIndex = 0;
    propagation.pieces[1].traceIndex = 1;
    propagation.constraints.push_back({
        0, 1, 0.0, 0.0, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        1.0, 0.0, 1.0, 0.0, false});
    FiberTraceLabelingReport propagationLabels;
    propagationLabels.labels = {
        FiberTracePieceLabel::HEven,
        FiberTracePieceLabel::Broken,
    };
    propagationLabels.retainedConstraintIndices = {0};
    const auto twice = postFilterPerpendicularFiberTraceLabels(
        propagation, propagationLabels, 2, {2, 1.0});
    CHECK(twice[0] == doctest::Approx(1.0));
    CHECK(twice[1] == doctest::Approx(0.0));
}

TEST_CASE("Trace post-filter rejects split or missing represented fibers")
{
    FiberTraceConstraintReport constraints;
    constraints.pieces.resize(2);
    constraints.pieces[0].traceIndex = 0;
    constraints.pieces[1].traceIndex = 0;
    FiberTraceLabelingReport labeling;
    labeling.labels = {
        FiberTracePieceLabel::HEven,
        FiberTracePieceLabel::HEven,
    };
    CHECK_THROWS_WITH_AS(
        postFilterPerpendicularFiberTraceLabels(
            constraints, labeling, 2, {1, 1.0}),
        doctest::Contains("unique contiguous"),
        std::invalid_argument);
    CHECK_THROWS_WITH_AS(
        postFilterPerpendicularFiberTraceLabels(
            constraints, labeling, 3, {1, 1.0}),
        doctest::Contains("one piece"),
        std::invalid_argument);
}

TEST_CASE("Fiber value bands use fixed boundaries and short OBJ names")
{
    const std::vector<double> values{
        std::nextafter(0.0, -1.0),
        0.099,
        0.1,
        0.5,
        0.9,
        std::nextafter(1.0, 2.0),
    };
    const auto bands = classifyFiberValues(values);
    CHECK(bands.bands[0].lineIndices ==
          std::vector<std::size_t>{0, 1});
    CHECK(bands.bands[1].lineIndices ==
          std::vector<std::size_t>{2});
    CHECK(bands.bands[5].lineIndices ==
          std::vector<std::size_t>{3});
    CHECK(bands.bands[9].lineIndices ==
          std::vector<std::size_t>{4, 5});
    CHECK(bands.bands[0].minimumValue == 0.0);
    CHECK(bands.bands[9].maximumValue == 1.0);

    CHECK_THROWS_WITH_AS(
        classifyFiberValues(std::vector<double>{0.0, -1.0e-6}),
        doctest::Contains("input 1"),
        std::invalid_argument);
    CHECK_THROWS_WITH_AS(
        classifyFiberValues(std::vector<double>{1.0 + 1.0e-6}),
        doctest::Contains("value="),
        std::invalid_argument);
    CHECK_THROWS_WITH_AS(
        classifyFiberValues(std::vector<double>{
            std::numeric_limits<double>::quiet_NaN()}),
        doctest::Contains("value=nan"),
        std::invalid_argument);

    std::vector<FiberletCropTraceLine> lines(values.size());
    for (std::size_t index = 0; index < lines.size(); ++index) {
        lines[index].pointsBaseXYZ = {
            {static_cast<double>(index), 0.0, 0.0},
            {static_cast<double>(index), 1.0, 0.0},
        };
    }
    const TemporaryDirectory directory("trace_post_filter");
    const auto paths = writeFiberletCropValueBandObjs(
        lines, bands, directory.path / "384");
    for (std::size_t band = 0; band < paths.bands.size(); ++band) {
        CHECK(paths.bands[band] ==
              directory.path /
                  ("384_p" + std::to_string(band) + ".obj"));
        CHECK(std::filesystem::exists(paths.bands[band]));
    }

    const std::vector states{
        FiberTernaryState::Vertical,
        FiberTernaryState::Mixed,
        FiberTernaryState::Horizontal,
        FiberTernaryState::Tie,
        FiberTernaryState::Vertical,
        FiberTernaryState::Mixed,
    };
    const auto statePaths = writeFiberletCropTernaryStateObjs(
        lines, states, directory.path / "fibers_bp");
    CHECK(statePaths.vertical == directory.path / "fibers_bp_v.obj");
    CHECK(statePaths.mixed == directory.path / "fibers_bp_err.obj");
    CHECK(statePaths.horizontal == directory.path / "fibers_bp_h.obj");
    CHECK(statePaths.tie == directory.path / "fibers_bp_tie.obj");
    const auto read = [](const std::filesystem::path& path) {
        std::ifstream input(path);
        std::ostringstream text;
        text << input.rdbuf();
        return text.str();
    };
    CHECK(read(statePaths.vertical).find("fiber_000000") != std::string::npos);
    CHECK(read(statePaths.vertical).find("fiber_000004") != std::string::npos);
    CHECK(read(statePaths.mixed).find("fiber_000001") != std::string::npos);
    CHECK(read(statePaths.horizontal).find("fiber_000002") != std::string::npos);
    CHECK(read(statePaths.tie).find("fiber_000003") != std::string::npos);
}

TEST_CASE("Trace labeling LP enforces triangle-consistent differences")
{
    FiberTraceConstraintReport constraints;
    constraints.pieces.resize(3);
    const auto addDifferentPreferred = [&](std::size_t a, std::size_t b) {
        constraints.constraints.push_back({
            a, b, 0.0, 0.0,
            {static_cast<double>(a), 0.0, 0.0},
            {static_cast<double>(b), 0.0, 0.0},
            1.0, 0.0, 1.0, 1.0, false});
    };
    addDifferentPreferred(0, 1);
    addDifferentPreferred(0, 2);
    addDifferentPreferred(1, 2);

    FiberTraceLabelingConfig config;
    config.parallelThreads = 1;
    config.relaxIntegrality = true;
    config.brokenCostPerConstraint = 100.0;
    const auto labeling = solveFiberTraceLabels(constraints, config);
    CHECK_FALSE(labeling.hvOnly);
    CHECK(labeling.integerVariables == 0);
    CHECK(labeling.gaugeRoots == 1);
    CHECK(labeling.triangles == 1);
    CHECK(labeling.triangleRows == 8);
    CHECK(labeling.variables == 18);
    CHECK(labeling.rows == 53);
    CHECK(labeling.objective == doctest::Approx(2.0));
    CHECK(labeling.orientationCost == doctest::Approx(1.0));
    CHECK(labeling.windingCost == doctest::Approx(1.0));
    CHECK(labeling.brokenCost == doctest::Approx(0.0));
    CHECK(labeling.verticalValues[0] == doctest::Approx(0.0));
    CHECK(labeling.oddValues[0] == doctest::Approx(0.0));
    for (const double active : labeling.activeValues)
        CHECK(active == doctest::Approx(1.0));

    config.brokenCostPerConstraint = 0.1;
    const auto withBrokenVertex = solveFiberTraceLabels(constraints, config);
    CHECK(withBrokenVertex.objective == doctest::Approx(0.1));
    CHECK(withBrokenVertex.activeValues[0] == doctest::Approx(0.5));
    CHECK(withBrokenVertex.activeValues[1] == doctest::Approx(1.0));
    CHECK(withBrokenVertex.activeValues[2] == doctest::Approx(1.0));
    CHECK(withBrokenVertex.verticalValues[0] == doctest::Approx(0.0));
    CHECK(withBrokenVertex.oddValues[0] == doctest::Approx(0.0));

    config.hvOnly = true;
    const auto hvOnly = solveFiberTraceLabels(constraints, config);
    CHECK(hvOnly.hvOnly);
    CHECK(hvOnly.integerVariables == 0);
    CHECK(hvOnly.gaugeRoots == 1);
    CHECK(hvOnly.triangles == 1);
    CHECK(hvOnly.triangleRows == 4);
    CHECK(hvOnly.variables == 12);
    CHECK(hvOnly.rows == 31);
    CHECK(hvOnly.windingCost == doctest::Approx(0.0));
    for (const double odd : hvOnly.oddValues)
        CHECK(odd == 0.0);
}

TEST_CASE("Trace labeling H V only mode uses the retained triangle graph")
{
    FiberTraceConstraintReport constraints;
    constraints.pieces.resize(4);
    const auto add = [&](std::size_t a,
                         std::size_t b,
                         double parallel,
                         double winding) {
        constraints.constraints.push_back({
            a, b, 0.0, 0.0,
            {static_cast<double>(a), 0.0, 0.0},
            {static_cast<double>(b), 0.0, 0.0},
            1.0, parallel, 1.0 - parallel, winding, false});
    };
    add(0, 1, 0.1, 1.0);
    add(0, 2, 0.1, 1.0);
    add(1, 2, 0.1, 1.0);
    add(2, 3, 0.9, 1.0);

    FiberTraceLabelingConfig config;
    config.parallelThreads = 1;
    config.relaxIntegrality = true;
    config.brokenCostPerConstraint = 100.0;
    config.excludeParallelSeparateWinding = true;
    config.hvOnly = true;
    const auto labeling = solveFiberTraceLabels(constraints, config);
    CHECK(labeling.retainedConstraints == 3);
    CHECK(labeling.excludedParallelSeparateWinding == 1);
    CHECK(labeling.gaugeRoots == 2);
    CHECK(labeling.triangles == 1);
    CHECK(labeling.variables == 14);
    CHECK(labeling.rows == 32);
    CHECK(labeling.triangleRows == 4);
}

TEST_CASE("Trace labeling exact perpendicular MILP derives loss from continuous H V values")
{
    const auto triangle = [](double brokenCost) {
        FiberTraceConstraintReport constraints;
        constraints.pieces.resize(3);
        const auto addPerpendicular = [&](std::size_t a, std::size_t b) {
            constraints.constraints.push_back({
                a, b, 0.0, 0.0,
                {static_cast<double>(a), 0.0, 0.0},
                {static_cast<double>(b), 0.0, 0.0},
                1.0, 0.0, 1.0, 1.0, false});
        };
        addPerpendicular(0, 1);
        addPerpendicular(0, 2);
        addPerpendicular(1, 2);

        FiberTraceLabelingConfig config;
        config.parallelThreads = 1;
        config.brokenCostPerConstraint = brokenCost;
        config.hvOnly = true;
        config.exactPerpendicularMilp = true;
        return solveFiberTraceLabels(constraints, config);
    };

    const auto allActive = triangle(100.0);
    CHECK(allActive.exactPerpendicularMilp);
    CHECK(allActive.continuousPieceValues);
    CHECK(allActive.labels.empty());
    CHECK(allActive.variables == 15);
    CHECK(allActive.integerVariables == 6);
    CHECK(allActive.perpendicularBranchVariables == 3);
    CHECK(allActive.rows == 33);
    CHECK(allActive.triangles == 0);
    CHECK(allActive.triangleRows == 0);
    CHECK(allActive.orientationCost == doctest::Approx(1.0));
    CHECK(allActive.brokenCost == doctest::Approx(0.0));
    for (const double active : allActive.activeValues)
        CHECK(active == doctest::Approx(1.0));
    for (const double odd : allActive.oddValues)
        CHECK(odd == doctest::Approx(0.0));

    const auto oneBroken = triangle(0.1);
    CHECK(oneBroken.objective == doctest::Approx(0.2));
    CHECK(oneBroken.orientationCost == doctest::Approx(0.0));
    CHECK(oneBroken.brokenCost == doctest::Approx(0.2));
    CHECK(std::count(
              oneBroken.activeValues.begin(),
              oneBroken.activeValues.end(),
              0.0) == 1);
    CHECK(std::count(
              oneBroken.activeValues.begin(),
              oneBroken.activeValues.end(),
              1.0) == 2);
    bool brokenTouchesExtremeActive = false;
    for (std::size_t broken = 0; broken < 3; ++broken) {
        if (oneBroken.activeValues[broken] != 0.0)
            continue;
        for (std::size_t active = 0; active < 3; ++active) {
            if (oneBroken.activeValues[active] == 1.0 &&
                oneBroken.verticalValues[active] == doctest::Approx(1.0)) {
                brokenTouchesExtremeActive = true;
            }
        }
    }
    CHECK(brokenTouchesExtremeActive);

    FiberTraceConstraintReport boundary;
    boundary.pieces.resize(2);
    boundary.constraints.push_back({
        0, 1, 0.0, 0.0, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        1.0, 0.5, 0.5, 0.0, false});
    FiberTraceLabelingConfig boundaryConfig;
    boundaryConfig.parallelThreads = 1;
    boundaryConfig.brokenCostPerConstraint = 100.0;
    boundaryConfig.hvOnly = true;
    boundaryConfig.exactPerpendicularMilp = true;
    const auto neutral = solveFiberTraceLabels(boundary, boundaryConfig);
    CHECK(neutral.variables == 7);
    CHECK(neutral.integerVariables == 3);
    CHECK(neutral.perpendicularBranchVariables == 1);
    CHECK(neutral.rows == 12);
    CHECK(neutral.orientationCost == doctest::Approx(0.5));

    boundary.constraints.front().parallelScore = 1.0;
    boundary.constraints.front().perpendicularScore = 0.0;
    const auto parallel = solveFiberTraceLabels(boundary, boundaryConfig);
    CHECK(parallel.variables == 6);
    CHECK(parallel.integerVariables == 2);
    CHECK(parallel.perpendicularBranchVariables == 0);
    CHECK(parallel.rows == 10);
    CHECK(parallel.orientationCost == doctest::Approx(0.0));

    boundaryConfig.hvOnly = false;
    CHECK_THROWS_WITH_AS(
        solveFiberTraceLabels(boundary, boundaryConfig),
        doctest::Contains("requires H/V-only"),
        std::invalid_argument);
    boundaryConfig.hvOnly = true;
    boundaryConfig.relaxIntegrality = true;
    CHECK_THROWS_WITH_AS(
        solveFiberTraceLabels(boundary, boundaryConfig),
        doctest::Contains("mutually exclusive"),
        std::invalid_argument);
}

TEST_CASE("Trace consensus grows original fibers by stable active evidence")
{
    std::vector<FiberletCropTraceLine> traces(5);
    const std::array<double, 5> lengths{10.0, 8.0, 6.0, 5.0, 4.0};
    for (std::size_t trace = 0; trace < traces.size(); ++trace) {
        traces[trace].pointsBaseXYZ = {
            {0.0, static_cast<double>(trace), 0.0},
            {lengths[trace], static_cast<double>(trace), 0.0},
        };
    }

    FiberTraceConstraintReport constraints;
    for (const std::size_t trace : {0UL, 0UL, 1UL, 1UL, 2UL, 3UL, 4UL}) {
        FiberTraceConstraintPiece piece;
        piece.traceIndex = trace;
        piece.pieceIndex = constraints.pieces.size();
        constraints.pieces.push_back(std::move(piece));
    }
    const auto add = [&](std::size_t a,
                         std::size_t b,
                         double distance,
                         double parallel) {
        FiberTraceConstraint constraint;
        constraint.pieceA = a;
        constraint.pieceB = b;
        constraint.closestDistanceBaseVoxels = distance;
        constraint.parallelScore = parallel;
        constraint.perpendicularScore = 1.0 - parallel;
        constraints.constraints.push_back(constraint);
    };
    add(0, 2, 2.0, 1.0);
    add(1, 3, 2.0, 1.0);
    add(0, 4, 1.0, 0.0);
    add(4, 5, 1.0, 0.5);
    add(5, 6, 1.0, 1.0);

    FiberTraceConsensusConfig config;
    config.brokenCostPerConstraint = 0.1;
    config.cropMinimumBaseXYZ = {-5.0, -5.0, -5.0};
    config.cropMaximumBaseXYZ = {15.0, 5.0, 5.0};
    const auto consensus = growFiberTraceConsensus(traces, constraints, config);
    REQUIRE(consensus.steps.size() == 5);
    CHECK(consensus.steps[0].traceIndex == 0);
    CHECK(consensus.steps[0].componentSeed);
    CHECK(consensus.steps[0].label == FiberTraceConsensusLabel::H);
    CHECK(consensus.steps[1].traceIndex == 1);
    CHECK(consensus.steps[1].evidenceCount == 2);
    CHECK(consensus.steps[1].label == FiberTraceConsensusLabel::H);
    CHECK(consensus.steps[2].traceIndex == 2);
    CHECK(consensus.steps[2].label == FiberTraceConsensusLabel::V);
    CHECK(consensus.steps[3].traceIndex == 3);
    CHECK(consensus.steps[3].label == FiberTraceConsensusLabel::Broken);
    CHECK(consensus.steps[3].selectedCost == doctest::Approx(0.1));
    CHECK(consensus.steps[4].traceIndex == 4);
    CHECK(consensus.steps[4].componentSeed);
    CHECK(consensus.steps[4].label == FiberTraceConsensusLabel::H);
    CHECK(consensus.components == 2);
    CHECK(consensus.retainedCrossTraceConstraints == 5);
    CHECK(consensus.orientationCost == doctest::Approx(0.0));
    CHECK(consensus.brokenCost == doctest::Approx(0.1));
    CHECK(consensus.objective == doctest::Approx(0.1));

    std::reverse(
        constraints.constraints.begin(), constraints.constraints.end());
    const auto reordered = growFiberTraceConsensus(traces, constraints, config);
    REQUIRE(reordered.steps.size() == consensus.steps.size());
    for (std::size_t index = 0; index < consensus.steps.size(); ++index) {
        CHECK(reordered.steps[index].traceIndex ==
              consensus.steps[index].traceIndex);
        CHECK(reordered.steps[index].label == consensus.steps[index].label);
        CHECK(reordered.steps[index].selectedCost ==
              doctest::Approx(consensus.steps[index].selectedCost));
    }

    traces.push_back({});
    const auto withDegenerate =
        growFiberTraceConsensus(traces, constraints, config);
    CHECK(withDegenerate.degenerateTraces == 1);
    CHECK(withDegenerate.steps.size() == 5);
    CHECK(withDegenerate.labels.back() == FiberTraceConsensusLabel::Broken);
}

TEST_CASE("Trace consensus primary seed uses crop length and center")
{
    std::vector<FiberletCropTraceLine> traces(5);
    traces[0].pointsBaseXYZ = {{-2.5, 0.0, 0.0}, {2.5, 0.0, 0.0}};
    traces[1].pointsBaseXYZ = {{-3.0, 3.0, 0.0}, {3.0, 3.0, 0.0}};
    traces[2].pointsBaseXYZ = {{-3.0, 1.0, 0.0}, {3.0, 1.0, 0.0}};
    traces[3].pointsBaseXYZ = {{-2.0, 0.5, 0.0}, {2.0, 0.5, 0.0}};
    traces[4].pointsBaseXYZ = {
        {-3.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {3.0, 0.0, 0.0}};

    FiberTraceConsensusConfig config;
    config.cropMinimumBaseXYZ = {-5.0, -10.0, -20.0};
    config.cropMaximumBaseXYZ = {5.0, 10.0, 20.0};
    const auto consensus = growFiberTraceConsensus(
        traces, FiberTraceConstraintReport{}, config);
    REQUIRE(consensus.steps.size() == traces.size());
    CHECK(consensus.steps[0].traceIndex == 2);
    CHECK(consensus.steps[0].componentSeed);
    CHECK(consensus.steps[0].seedStraightness == doctest::Approx(1.0));
    CHECK(consensus.steps[0].seedCenterDistanceBaseVoxels ==
          doctest::Approx(1.0));
    CHECK(consensus.steps[0].seedArcLengthBaseVoxels ==
          doctest::Approx(6.0));
    CHECK(consensus.steps[1].traceIndex == 0);
    CHECK(consensus.steps[1].componentSeed);
    CHECK(consensus.steps[3].traceIndex == 1);
    CHECK(consensus.steps.back().traceIndex == 4);

    traces.resize(2);
    traces[1].pointsBaseXYZ = {{-2.0, 1.0, 0.0}, {2.0, 1.0, 0.0}};
    CHECK_THROWS_WITH_AS(
        growFiberTraceConsensus(
            traces, FiberTraceConstraintReport{}, config),
        doctest::Contains("longer than half"),
        std::invalid_argument);
}

namespace
{

std::vector<FiberletCropTraceLine> bpLines(std::size_t count)
{
    std::vector<FiberletCropTraceLine> lines(count);
    for (std::size_t index = 0; index < count; ++index) {
        lines[index].pointsBaseXYZ = {
            {-3.0, static_cast<double>(index), 0.0},
            {3.0, static_cast<double>(index), 0.0},
        };
    }
    return lines;
}

FiberTraceConstraintReport bpConstraints(std::size_t count)
{
    FiberTraceConstraintReport report;
    report.inputTraces = count;
    report.pieces.resize(count);
    for (std::size_t index = 0; index < count; ++index) {
        report.pieces[index].traceIndex = index;
        report.pieces[index].pieceIndex = 0;
        report.pieces[index].beginArcBaseVoxels = 0.0;
        report.pieces[index].endArcBaseVoxels = 6.0;
    }
    return report;
}

void addBpConstraint(
    FiberTraceConstraintReport& report,
    std::size_t a,
    std::size_t b,
    double parallel)
{
    FiberTraceConstraint constraint;
    constraint.pieceA = a;
    constraint.pieceB = b;
    constraint.parallelScore = parallel;
    constraint.perpendicularScore = 1.0 - parallel;
    report.constraints.push_back(constraint);
}

void addBpContinuity(
    FiberTraceConstraintReport& report,
    std::size_t a,
    std::size_t b,
    const cv::Vec3d& point)
{
    const auto& left = report.pieces[a].pieceIndex <
            report.pieces[b].pieceIndex
        ? report.pieces[a]
        : report.pieces[b];
    const auto& right = report.pieces[a].pieceIndex <
            report.pieces[b].pieceIndex
        ? report.pieces[b]
        : report.pieces[a];
    const double arc =
        0.5 * (left.endArcBaseVoxels + right.beginArcBaseVoxels);
    FiberTraceConstraint constraint;
    constraint.pieceA = a;
    constraint.pieceB = b;
    constraint.arcABaseVoxels = arc;
    constraint.arcBBaseVoxels = arc;
    constraint.pointABaseXYZ = point;
    constraint.pointBBaseXYZ = point;
    constraint.parallelScore = 1.0;
    constraint.perpendicularScore = 0.0;
    constraint.hardContinuity = true;
    report.constraints.push_back(constraint);
    ++report.hardConstraints;
}

FiberTraceConstraintReport bpSplitConstraints()
{
    FiberTraceConstraintReport report;
    report.inputTraces = 2;
    report.pieces.resize(3);
    report.pieces[0].traceIndex = 0;
    report.pieces[0].pieceIndex = 0;
    report.pieces[0].beginArcBaseVoxels = 0.0;
    report.pieces[0].endArcBaseVoxels = 4.0;
    report.pieces[1].traceIndex = 0;
    report.pieces[1].pieceIndex = 1;
    report.pieces[1].beginArcBaseVoxels = 2.0;
    report.pieces[1].endArcBaseVoxels = 6.0;
    report.pieces[2].traceIndex = 1;
    report.pieces[2].pieceIndex = 0;
    report.pieces[2].beginArcBaseVoxels = 0.0;
    report.pieces[2].endArcBaseVoxels = 6.0;
    addBpContinuity(report, 0, 1, {0.0, 0.0, 0.0});
    addBpConstraint(report, 1, 2, 0.0);
    return report;
}

FiberTraceBeliefPropagationConfig bpConfig()
{
    FiberTraceBeliefPropagationConfig config;
    config.cropMinimumBaseXYZ = {-5.0, -5.0, -5.0};
    config.cropMaximumBaseXYZ = {5.0, 5.0, 5.0};
    config.horizontalnessTemperature = 0.1;
    config.messageDamping = 0.5;
    config.messageResidualTolerance = 1.0e-12;
    config.maximumMessageIterations = 1000;
    return config;
}

std::vector<double> bruteForceBpAdvantages(
    std::size_t count,
    std::size_t seed,
    const FiberTraceConstraintReport& constraints)
{
    std::vector<double> minimumV(count, std::numeric_limits<double>::infinity());
    std::vector<double> minimumH(count, std::numeric_limits<double>::infinity());
    for (std::size_t bits = 0; bits < (std::size_t{1} << count); ++bits) {
        if (((bits >> seed) & 1U) == 0)
            continue;
        double energy = 0.0;
        for (const auto& constraint : constraints.constraints) {
            const bool a = ((bits >> constraint.pieceA) & 1U) != 0;
            const bool b = ((bits >> constraint.pieceB) & 1U) != 0;
            energy += a == b
                ? 1.0 - constraint.parallelScore
                : constraint.parallelScore;
        }
        for (std::size_t node = 0; node < count; ++node) {
            auto& minimum = ((bits >> node) & 1U) != 0
                ? minimumH[node]
                : minimumV[node];
            minimum = std::min(minimum, energy);
        }
    }
    std::vector<double> advantages(count);
    for (std::size_t node = 0; node < count; ++node)
        advantages[node] = minimumV[node] - minimumH[node];
    return advantages;
}

std::vector<double> bruteForceBpMarginals(
    std::size_t count,
    std::size_t seed,
    const FiberTraceConstraintReport& constraints,
    double temperature)
{
    std::vector<double> horizontalWeight(count, 0.0);
    double partition = 0.0;
    for (std::size_t bits = 0; bits < (std::size_t{1} << count); ++bits) {
        if (((bits >> seed) & 1U) == 0)
            continue;
        double energy = 0.0;
        for (const auto& constraint : constraints.constraints) {
            const bool a = ((bits >> constraint.pieceA) & 1U) != 0;
            const bool b = ((bits >> constraint.pieceB) & 1U) != 0;
            energy += a == b
                ? 1.0 - constraint.parallelScore
                : constraint.parallelScore;
        }
        const double weight = std::exp(-energy / temperature);
        partition += weight;
        for (std::size_t node = 0; node < count; ++node) {
            if (((bits >> node) & 1U) != 0)
                horizontalWeight[node] += weight;
        }
    }
    for (double& value : horizontalWeight)
        value /= partition;
    return horizontalWeight;
}

std::vector<std::array<double, 3>> bruteForceMixedBpMarginals(
    std::size_t count,
    std::size_t seed,
    const FiberTraceConstraintReport& constraints,
    double temperature,
    double mixedCost)
{
    struct PairCost {
        double same = 0.0;
        double different = 0.0;
    };
    std::map<std::pair<std::size_t, std::size_t>, PairCost> merged;
    for (const auto& constraint : constraints.constraints) {
        const auto key = std::minmax(
            constraint.pieceA, constraint.pieceB);
        auto& cost = merged[{key.first, key.second}];
        cost.same += 1.0 - constraint.parallelScore;
        cost.different += constraint.parallelScore;
    }
    for (auto& [key, cost] : merged) {
        (void)key;
        const double common = std::min(cost.same, cost.different);
        cost.same -= common;
        cost.different -= common;
    }
    std::vector<std::array<double, 3>> stateWeights(count);
    double partition = 0.0;
    std::size_t assignments = 1;
    for (std::size_t node = 0; node < count; ++node)
        assignments *= 3;
    for (std::size_t encoded = 0; encoded < assignments; ++encoded) {
        std::size_t remainder = encoded;
        std::vector<std::size_t> states(count);
        for (std::size_t node = 0; node < count; ++node) {
            states[node] = remainder % 3;
            remainder /= 3;
        }
        if (states[seed] != 2)
            continue;
        double energy = 0.0;
        for (const std::size_t state : states) {
            if (state == 1)
                energy += mixedCost;
        }
        for (const auto& [key, cost] : merged) {
            const std::size_t a = states[key.first];
            const std::size_t b = states[key.second];
            if (a != 1 && b != 1) {
                energy += a == b
                    ? cost.same
                    : cost.different;
            }
        }
        const double weight = std::exp(-energy / temperature);
        partition += weight;
        for (std::size_t node = 0; node < count; ++node)
            stateWeights[node][states[node]] += weight;
    }
    for (auto& weights : stateWeights) {
        for (double& weight : weights)
            weight /= partition;
    }
    return stateWeights;
}

}  // namespace

TEST_CASE("Binary min-sum BP matches exact seeded perpendicular tree")
{
    const auto lines = bpLines(3);
    auto constraints = bpConstraints(lines.size());
    addBpConstraint(constraints, 0, 1, 0.0);
    addBpConstraint(constraints, 1, 2, 0.0);
    addBpConstraint(constraints, 0, 1, 0.0);

    const auto report = solveFiberTraceBeliefPropagation(
        lines, constraints, bpConfig());
    const auto exact = bruteForceBpAdvantages(
        lines.size(), report.seedPieceIndex, constraints);
    CHECK(report.status == "converged");
    CHECK(report.seedPieceIndex == 0);
    CHECK(report.factors == 2);
    CHECK(report.mergedMeasurements == 3);
    CHECK(report.connectedComponents == 1);
    REQUIRE(report.horizontalness.size() == lines.size());
    CHECK(report.horizontalness[0] == doctest::Approx(1.0));
    for (std::size_t node = 1; node < lines.size(); ++node) {
        CHECK(report.minMarginalAdvantage[node] ==
              doctest::Approx(exact[node]).epsilon(1.0e-9));
    }
    CHECK(report.horizontalness[1] < 1.0e-8);
    CHECK(report.horizontalness[2] > 0.9999);

    std::reverse(
        constraints.constraints.begin(), constraints.constraints.end());
    const auto reordered = solveFiberTraceBeliefPropagation(
        lines, constraints, bpConfig());
    CHECK(reordered.horizontalness == report.horizontalness);
    CHECK(reordered.minMarginalAdvantage == report.minMarginalAdvantage);
}

TEST_CASE("Binary min-sum BP leaves unsupported component gauges uncertain")
{
    const auto lines = bpLines(4);
    auto constraints = bpConstraints(lines.size());
    addBpConstraint(constraints, 0, 1, 0.0);
    addBpConstraint(constraints, 2, 3, 0.0);
    const auto report = solveFiberTraceBeliefPropagation(
        lines, constraints, bpConfig());
    CHECK(report.seedPieceIndex == 0);
    CHECK(report.connectedComponents == 2);
    CHECK(report.horizontalness[0] == doctest::Approx(1.0));
    CHECK(report.horizontalness[1] < 1.0e-4);
    CHECK(report.horizontalness[2] == doctest::Approx(0.5));
    CHECK(report.horizontalness[3] == doctest::Approx(0.5));
}

TEST_CASE("Binary min-sum BP reports frustration as horizontalness uncertainty")
{
    const auto lines = bpLines(3);
    auto constraints = bpConstraints(lines.size());
    addBpConstraint(constraints, 0, 1, 0.0);
    addBpConstraint(constraints, 1, 2, 0.0);
    addBpConstraint(constraints, 2, 0, 0.0);
    auto config = bpConfig();
    config.messageDamping = 0.25;
    const auto report = solveFiberTraceBeliefPropagation(
        lines, constraints, config);
    const auto exact = bruteForceBpAdvantages(
        lines.size(), report.seedPieceIndex, constraints);
    CHECK(exact[1] == doctest::Approx(0.0));
    CHECK(exact[2] == doctest::Approx(0.0));
    CHECK(report.horizontalness[1] == doctest::Approx(0.5).epsilon(1.0e-6));
    CHECK(report.horizontalness[2] == doctest::Approx(0.5).epsilon(1.0e-6));
}

TEST_CASE("Binary min-sum BP balance modes move weighted H fraction")
{
    const auto lines = bpLines(3);
    auto constraints = bpConstraints(lines.size());
    addBpConstraint(constraints, 0, 1, 0.0);
    addBpConstraint(constraints, 1, 2, 0.0);

    auto config = bpConfig();
    config.horizontalnessTemperature = 0.5;
    const auto unbalanced = solveFiberTraceBeliefPropagation(
        lines, constraints, config);
    REQUIRE(unbalanced.achievedHorizontalFraction > 0.55);

    config.balanceMode = FiberTraceBalanceMode::Soft;
    config.softBalanceStrength = 2.0;
    config.balanceTolerance = 1.0e-8;
    config.maximumBalanceIterations = 256;
    const auto soft = solveFiberTraceBeliefPropagation(
        lines, constraints, config);
    CHECK(std::abs(soft.achievedHorizontalFraction - 0.5) <
          std::abs(unbalanced.achievedHorizontalFraction - 0.5));
    CHECK(soft.balanceConverged);

    config.softBalanceStrength = 0.0;
    const auto zeroSoft = solveFiberTraceBeliefPropagation(
        lines, constraints, config);
    CHECK(zeroSoft.horizontalness == unbalanced.horizontalness);

    config.balanceMode = FiberTraceBalanceMode::Tight;
    config.balanceTolerance = 1.0e-5;
    config.maximumBalanceIterations = 64;
    const auto tight = solveFiberTraceBeliefPropagation(
        lines, constraints, config);
    CHECK(tight.status == "converged");
    CHECK(tight.achievedHorizontalFraction ==
          doctest::Approx(0.5).epsilon(1.0e-5));

    config.targetHorizontalFraction = 0.0;
    const auto infeasible = solveFiberTraceBeliefPropagation(
        lines, constraints, config);
    CHECK(infeasible.status == "infeasible");
    CHECK_FALSE(infeasible.balanceConverged);
}

TEST_CASE("Binary min-sum BP accepts full orientation evidence")
{
    const auto lines = bpLines(2);
    auto constraints = bpConstraints(lines.size());
    addBpConstraint(constraints, 0, 1, 0.75);
    const auto parallel = solveFiberTraceBeliefPropagation(
        lines, constraints, bpConfig());
    CHECK(parallel.horizontalness[0] == doctest::Approx(1.0));
    CHECK(parallel.horizontalness[1] > 0.99);
    constraints.constraints.front().parallelScore = 0.0;
    constraints.constraints.front().perpendicularScore = 1.0;
    constraints.constraints.front().hardContinuity = true;
    CHECK_THROWS_WITH_AS(
        solveFiberTraceBeliefPropagation(lines, constraints, bpConfig()),
        doctest::Contains("continuity"),
        std::invalid_argument);

    constraints.constraints.front().hardContinuity = false;
    constraints.constraints.front().parallelScore = -0.1;
    constraints.constraints.front().perpendicularScore = 1.1;
    CHECK_THROWS_WITH_AS(
        solveFiberTraceBeliefPropagation(lines, constraints, bpConfig()),
        doctest::Contains("orientation scores"),
        std::invalid_argument);
    constraints.constraints.front().parallelScore = 0.25;
    constraints.constraints.front().perpendicularScore = 0.5;
    CHECK_THROWS_WITH_AS(
        solveFiberTraceBeliefPropagation(lines, constraints, bpConfig()),
        doctest::Contains("orientation scores"),
        std::invalid_argument);
}

TEST_CASE("All BP modes solve source fibers as split piece nodes")
{
    const auto lines = bpLines(2);
    const auto constraints = bpSplitConstraints();
    auto config = bpConfig();
    config.horizontalnessTemperature = 0.5;
    config.messageDamping = 1.0;
    config.mixedUnaryCost = 0.4;

    const auto minSum = solveFiberTraceBeliefPropagation(
        lines, constraints, config);
    const auto exactAdvantages = bruteForceBpAdvantages(
        constraints.pieces.size(), minSum.seedPieceIndex, constraints);
    REQUIRE(minSum.horizontalness.size() == constraints.pieces.size());
    CHECK(minSum.seedPieceIndex == 0);
    CHECK(minSum.factors == 2);
    CHECK(minSum.mergedMeasurements == 2);
    CHECK(minSum.isolatedPieces == 0);
    REQUIRE(minSum.normalizedArcWeights.size() == 3);
    CHECK(minSum.normalizedArcWeights[0] == doctest::Approx(6.0 / 7.0));
    CHECK(minSum.normalizedArcWeights[1] == doctest::Approx(6.0 / 7.0));
    CHECK(minSum.normalizedArcWeights[2] == doctest::Approx(9.0 / 7.0));
    for (std::size_t piece = 0; piece < constraints.pieces.size(); ++piece) {
        if (piece == minSum.seedPieceIndex)
            continue;
        CHECK(minSum.minMarginalAdvantage[piece] ==
              doctest::Approx(exactAdvantages[piece]).epsilon(1.0e-10));
    }

    const auto binary = solveFiberTraceSumProduct(lines, constraints, config);
    const auto exactBinary = bruteForceBpMarginals(
        constraints.pieces.size(), binary.seedPieceIndex, constraints,
        config.horizontalnessTemperature);
    for (std::size_t piece = 0; piece < constraints.pieces.size(); ++piece) {
        CHECK(binary.horizontalness[piece] ==
              doctest::Approx(exactBinary[piece]).epsilon(1.0e-10));
    }

    const auto mixed = solveFiberTraceMixedSumProduct(
        lines, constraints, config);
    const auto exactMixed = bruteForceMixedBpMarginals(
        constraints.pieces.size(), mixed.seedPieceIndex, constraints,
        config.horizontalnessTemperature, config.mixedUnaryCost);
    for (std::size_t piece = 0; piece < constraints.pieces.size(); ++piece) {
        CHECK(mixed.verticalProbability[piece] ==
              doctest::Approx(exactMixed[piece][0]).epsilon(1.0e-10));
        CHECK(mixed.mixedProbability[piece] ==
              doctest::Approx(exactMixed[piece][1]).epsilon(1.0e-10));
        CHECK(mixed.horizontalProbability[piece] ==
              doctest::Approx(exactMixed[piece][2]).epsilon(1.0e-10));
    }

    const auto pieceLines = makeFiberTraceConstraintPieceLines(
        lines, constraints);
    REQUIRE(pieceLines.size() == 3);
    CHECK(pieceLines[0].pointsBaseXYZ.front() == cv::Vec3d(-3.0, 0.0, 0.0));
    CHECK(cv::norm(pieceLines[0].pointsBaseXYZ.back() -
                   cv::Vec3d(1.0, 0.0, 0.0)) < 1.0e-12);
    CHECK(cv::norm(pieceLines[1].pointsBaseXYZ.front() -
                   cv::Vec3d(-1.0, 0.0, 0.0)) < 1.0e-12);
    CHECK(pieceLines[1].pointsBaseXYZ.back() == cv::Vec3d(3.0, 0.0, 0.0));
    CHECK(pieceLines[2].pointsBaseXYZ == lines[1].pointsBaseXYZ);
}

TEST_CASE("BP rejects malformed split continuity topology")
{
    const auto lines = bpLines(2);

    SUBCASE("missing continuity")
    {
        auto constraints = bpSplitConstraints();
        constraints.constraints.erase(constraints.constraints.begin());
        CHECK_THROWS_WITH_AS(
            solveFiberTraceSumProduct(lines, constraints, bpConfig()),
            doctest::Contains("missing canonical continuity"),
            std::invalid_argument);
    }

    SUBCASE("duplicate continuity")
    {
        auto constraints = bpSplitConstraints();
        constraints.constraints.push_back(constraints.constraints.front());
        CHECK_THROWS_WITH_AS(
            solveFiberTraceSumProduct(lines, constraints, bpConfig()),
            doctest::Contains("duplicate or inconsistent"),
            std::invalid_argument);
    }

    SUBCASE("malformed continuity arc")
    {
        auto constraints = bpSplitConstraints();
        constraints.constraints.front().arcABaseVoxels += 1.0;
        constraints.constraints.front().arcBBaseVoxels += 1.0;
        CHECK_THROWS_WITH_AS(
            solveFiberTraceSumProduct(lines, constraints, bpConfig()),
            doctest::Contains("continuity point"),
            std::invalid_argument);
    }

    SUBCASE("soft same-source link")
    {
        auto constraints = bpSplitConstraints();
        addBpConstraint(constraints, 0, 1, 0.5);
        CHECK_THROWS_WITH_AS(
            solveFiberTraceSumProduct(lines, constraints, bpConfig()),
            doctest::Contains("soft constraint"),
            std::invalid_argument);
    }

    SUBCASE("source count mismatch")
    {
        auto constraints = bpSplitConstraints();
        constraints.inputTraces = 3;
        CHECK_THROWS_WITH_AS(
            solveFiberTraceSumProduct(lines, constraints, bpConfig()),
            doctest::Contains("source count"),
            std::invalid_argument);
    }
}

TEST_CASE("BP continuity is finite same-label evidence")
{
    const auto lines = bpLines(1);
    FiberTraceConstraintReport constraints;
    constraints.inputTraces = 1;
    constraints.pieces.resize(2);
    constraints.pieces[0].traceIndex = 0;
    constraints.pieces[0].pieceIndex = 0;
    constraints.pieces[0].beginArcBaseVoxels = 0.0;
    constraints.pieces[0].endArcBaseVoxels = 4.0;
    constraints.pieces[1].traceIndex = 0;
    constraints.pieces[1].pieceIndex = 1;
    constraints.pieces[1].beginArcBaseVoxels = 2.0;
    constraints.pieces[1].endArcBaseVoxels = 6.0;
    addBpContinuity(constraints, 0, 1, {0.0, 0.0, 0.0});

    auto config = bpConfig();
    config.horizontalnessTemperature = 0.5;
    config.messageDamping = 1.0;
    const auto report = solveFiberTraceSumProduct(
        lines, constraints, config);
    REQUIRE(report.seedPieceIndex == 0);
    CHECK(report.horizontalness[0] == doctest::Approx(1.0));
    CHECK(report.horizontalness[1] ==
          doctest::Approx(1.0 / (1.0 + std::exp(-2.0))).epsilon(1.0e-12));
    CHECK(report.horizontalness[1] < 1.0);
}

TEST_CASE("Binary sum-product BP matches exact seeded tree marginals")
{
    const auto lines = bpLines(4);
    auto constraints = bpConstraints(lines.size());
    addBpConstraint(constraints, 0, 1, 0.1);
    addBpConstraint(constraints, 1, 2, 0.2);
    addBpConstraint(constraints, 1, 3, 0.8);
    addBpConstraint(constraints, 0, 1, 0.25);

    for (const double temperature : {0.25, 1.0}) {
        auto config = bpConfig();
        config.horizontalnessTemperature = temperature;
        config.messageDamping = 1.0;
        const auto report = solveFiberTraceSumProduct(
            lines, constraints, config);
        const auto exact = bruteForceBpMarginals(
            lines.size(), report.seedPieceIndex, constraints, temperature);
        CHECK(report.inference == FiberTraceBeliefInference::SumProduct);
        CHECK(report.inferenceTemperature == doctest::Approx(temperature));
        CHECK(report.status == "converged");
        CHECK(report.horizontalness[report.seedPieceIndex] ==
              doctest::Approx(1.0));
        REQUIRE(report.logOdds.size() == lines.size());
        for (std::size_t node = 0; node < lines.size(); ++node) {
            CHECK(report.horizontalness[node] ==
                  doctest::Approx(exact[node]).epsilon(1.0e-10));
        }

        auto reorderedConstraints = constraints;
        std::reverse(
            reorderedConstraints.constraints.begin(),
            reorderedConstraints.constraints.end());
        const auto reordered = solveFiberTraceSumProduct(
            lines, reorderedConstraints, config);
        CHECK(reordered.horizontalness == report.horizontalness);
        CHECK(reordered.logOdds == report.logOdds);

        config.messageDamping = 0.25;
        const auto damped = solveFiberTraceSumProduct(
            lines, constraints, config);
        CHECK(damped.status == "converged");
        for (std::size_t node = 0; node < lines.size(); ++node) {
            CHECK(damped.horizontalness[node] ==
                  doctest::Approx(exact[node]).epsilon(1.0e-10));
        }
    }
}

TEST_CASE("Binary sum-product BP uses the expected perpendicular sign")
{
    const auto lines = bpLines(2);
    auto constraints = bpConstraints(lines.size());
    addBpConstraint(constraints, 0, 1, 0.1);
    auto config = bpConfig();
    config.horizontalnessTemperature = 0.4;
    const auto report = solveFiberTraceSumProduct(
        lines, constraints, config);
    const double expected = 1.0 / (1.0 + std::exp(2.0));
    CHECK(report.seedPieceIndex == 0);
    CHECK(report.horizontalness[0] == doctest::Approx(1.0));
    CHECK(report.horizontalness[1] ==
          doctest::Approx(expected).epsilon(1.0e-10));
    CHECK(report.logOdds[1] == doctest::Approx(-2.0).epsilon(1.0e-10));
}

TEST_CASE("BP drops exactly neutral merged orientation factors")
{
    const auto lines = bpLines(3);
    auto constraints = bpConstraints(lines.size());
    addBpConstraint(constraints, 0, 1, 0.5);
    addBpConstraint(constraints, 1, 2, 0.0);
    addBpConstraint(constraints, 1, 2, 1.0);

    auto config = bpConfig();
    config.horizontalnessTemperature = 0.5;
    config.mixedUnaryCost = 0.7;
    const auto binary = solveFiberTraceSumProduct(
        lines, constraints, config);
    const auto mixed = solveFiberTraceMixedSumProduct(
        lines, constraints, config);
    for (const auto* report : {&binary, &mixed}) {
        CHECK(report->factors == 0);
        CHECK(report->mergedMeasurements == 0);
        CHECK(report->neutralFactors == 2);
        CHECK(report->neutralMeasurements == 3);
        CHECK(report->connectedComponents == 3);
        CHECK(report->isolatedPieces == 3);
    }

    const double mixedWeight = std::exp(
        -config.mixedUnaryCost / config.horizontalnessTemperature);
    const double normalization = 2.0 + mixedWeight;
    CHECK(mixed.verticalProbability[1] ==
          doctest::Approx(1.0 / normalization));
    CHECK(mixed.mixedProbability[1] ==
          doctest::Approx(mixedWeight / normalization));
    CHECK(mixed.horizontalProbability[1] ==
          doctest::Approx(1.0 / normalization));

    constraints = bpConstraints(lines.size());
    addBpConstraint(constraints, 0, 1, 0.500001);
    const auto nearTie = solveFiberTraceSumProduct(
        lines, constraints, config);
    CHECK(nearTie.factors == 1);
    CHECK(nearTie.neutralFactors == 0);
    CHECK(nearTie.connectedComponents == 2);
}

TEST_CASE("Binary sum-product BP preserves unsupported gauge uncertainty")
{
    const auto lines = bpLines(4);
    auto constraints = bpConstraints(lines.size());
    addBpConstraint(constraints, 0, 1, 0.0);
    addBpConstraint(constraints, 2, 3, 0.0);
    const auto report = solveFiberTraceSumProduct(
        lines, constraints, bpConfig());
    CHECK(report.seedPieceIndex == 0);
    CHECK(report.horizontalness[0] == doctest::Approx(1.0));
    CHECK(report.horizontalness[2] == doctest::Approx(0.5));
    CHECK(report.horizontalness[3] == doctest::Approx(0.5));
}

TEST_CASE("Binary sum-product BP remains stable at low temperature")
{
    const auto lines = bpLines(3);
    auto constraints = bpConstraints(lines.size());
    addBpConstraint(constraints, 0, 1, 0.1);
    addBpConstraint(constraints, 0, 1, 0.1);
    addBpConstraint(constraints, 1, 2, 0.1);
    auto config = bpConfig();
    config.horizontalnessTemperature = 1.0e-6;
    const auto report = solveFiberTraceSumProduct(
        lines, constraints, config);
    CHECK(report.messageConverged);
    for (const double value : report.horizontalness)
        CHECK(std::isfinite(value));
    for (std::size_t node = 0; node < report.logOdds.size(); ++node) {
        if (node != report.seedPieceIndex)
            CHECK(std::isfinite(report.logOdds[node]));
    }

    config.balanceMode = FiberTraceBalanceMode::Soft;
    CHECK_THROWS_WITH_AS(
        solveFiberTraceSumProduct(lines, constraints, config),
        doctest::Contains("does not support"),
        std::invalid_argument);
    config.balanceMode = FiberTraceBalanceMode::None;
    config.horizontalnessTemperature = 0.0;
    CHECK_THROWS_WITH_AS(
        solveFiberTraceSumProduct(lines, constraints, config),
        doctest::Contains("temperature"),
        std::invalid_argument);
}

TEST_CASE("Binary sum-product BP reports last-iterate message limits")
{
    const auto lines = bpLines(3);
    auto constraints = bpConstraints(lines.size());
    addBpConstraint(constraints, 0, 1, 0.1);
    addBpConstraint(constraints, 1, 2, 0.1);
    auto config = bpConfig();
    config.maximumMessageIterations = 1;
    config.messageResidualTolerance = 0.0;
    const auto limited = solveFiberTraceSumProduct(
        lines, constraints, config);
    CHECK(limited.status == "message_limit");
    CHECK_FALSE(limited.messageConverged);
    CHECK(limited.messageIterations == 1);
    for (const double value : limited.horizontalness)
        CHECK(std::isfinite(value));

    const auto seedOnly = solveFiberTraceSumProduct(
        bpLines(1), bpConstraints(1), bpConfig());
    CHECK(seedOnly.status == "converged");
    CHECK(seedOnly.factors == 0);
    CHECK(seedOnly.horizontalness == std::vector<double>{1.0});

    auto loopConstraints = bpConstraints(lines.size());
    addBpConstraint(loopConstraints, 0, 1, 0.1);
    addBpConstraint(loopConstraints, 1, 2, 0.2);
    addBpConstraint(loopConstraints, 2, 0, 0.3);
    config = bpConfig();
    const auto first = solveFiberTraceSumProduct(
        lines, loopConstraints, config);
    const auto second = solveFiberTraceSumProduct(
        lines, loopConstraints, config);
    CHECK(first.horizontalness == second.horizontalness);
    for (const double value : first.horizontalness)
        CHECK(std::isfinite(value));
}

TEST_CASE("Mixed-state sum-product BP matches exact seeded tree marginals")
{
    const auto lines = bpLines(4);
    auto constraints = bpConstraints(lines.size());
    addBpConstraint(constraints, 0, 1, 0.1);
    addBpConstraint(constraints, 0, 1, 0.25);
    addBpConstraint(constraints, 1, 2, 0.2);
    addBpConstraint(constraints, 1, 3, 0.8);

    for (const double temperature : {0.25, 1.0}) {
        auto config = bpConfig();
        config.horizontalnessTemperature = temperature;
        config.mixedUnaryCost = 0.4;
        config.messageDamping = 1.0;
        const auto report = solveFiberTraceMixedSumProduct(
            lines, constraints, config);
        const auto exact = bruteForceMixedBpMarginals(
            lines.size(), report.seedPieceIndex, constraints,
            temperature, config.mixedUnaryCost);
        CHECK(report.inference ==
              FiberTraceBeliefInference::SumProductMixed);
        CHECK(report.mixedUnaryCost == doctest::Approx(0.4));
        CHECK(report.status == "converged");
        for (std::size_t node = 0; node < lines.size(); ++node) {
            CHECK(report.verticalProbability[node] ==
                  doctest::Approx(exact[node][0]).epsilon(1.0e-10));
            CHECK(report.mixedProbability[node] ==
                  doctest::Approx(exact[node][1]).epsilon(1.0e-10));
            CHECK(report.horizontalProbability[node] ==
                  doctest::Approx(exact[node][2]).epsilon(1.0e-10));
            CHECK(report.verticalProbability[node] +
                      report.mixedProbability[node] +
                      report.horizontalProbability[node] ==
                  doctest::Approx(1.0).epsilon(1.0e-12));
            CHECK(report.horizontalness[node] == doctest::Approx(
                exact[node][2] + 0.5 * exact[node][1]));
        }

        auto reordered = constraints;
        std::reverse(
            reordered.constraints.begin(), reordered.constraints.end());
        const auto reorderedReport = solveFiberTraceMixedSumProduct(
            lines, reordered, config);
        CHECK(reorderedReport.verticalProbability ==
              report.verticalProbability);
        CHECK(reorderedReport.mixedProbability == report.mixedProbability);
        CHECK(reorderedReport.horizontalProbability ==
              report.horizontalProbability);

        config.messageDamping = 0.25;
        const auto damped = solveFiberTraceMixedSumProduct(
            lines, constraints, config);
        CHECK(damped.status == "converged");
        for (std::size_t node = 0; node < lines.size(); ++node) {
            CHECK(damped.mixedProbability[node] ==
                  doctest::Approx(exact[node][1]).epsilon(1.0e-10));
        }
    }
}

TEST_CASE("Mixed-state sum-product BP preserves gauge and isolate symmetry")
{
    const auto lines = bpLines(5);
    auto constraints = bpConstraints(lines.size());
    addBpConstraint(constraints, 0, 1, 0.0);
    addBpConstraint(constraints, 2, 3, 0.1);
    auto config = bpConfig();
    config.mixedUnaryCost = 0.4;
    const auto report = solveFiberTraceMixedSumProduct(
        lines, constraints, config);
    CHECK(report.verticalProbability[report.seedPieceIndex] ==
          doctest::Approx(0.0));
    CHECK(report.mixedProbability[report.seedPieceIndex] ==
          doctest::Approx(0.0));
    CHECK(report.horizontalProbability[report.seedPieceIndex] ==
          doctest::Approx(1.0));
    for (const std::size_t node : {2U, 3U}) {
        CHECK(report.verticalProbability[node] ==
              doctest::Approx(report.horizontalProbability[node])
                  .epsilon(1.0e-12));
    }
    const double mixedWeight = std::exp(
        -config.mixedUnaryCost / config.horizontalnessTemperature);
    const double isolateNormalization = 2.0 + mixedWeight;
    CHECK(report.verticalProbability[4] ==
          doctest::Approx(1.0 / isolateNormalization));
    CHECK(report.mixedProbability[4] ==
          doctest::Approx(mixedWeight / isolateNormalization));
    CHECK(report.horizontalProbability[4] ==
          doctest::Approx(1.0 / isolateNormalization));
}

TEST_CASE("Mixed-state sum-product BP charges one unary per piece")
{
    const auto lines = bpLines(2);
    auto oneConstraint = bpConstraints(lines.size());
    addBpConstraint(oneConstraint, 0, 1, 0.0);
    auto threeConstraints = oneConstraint;
    addBpConstraint(threeConstraints, 0, 1, 0.0);
    addBpConstraint(threeConstraints, 0, 1, 0.0);

    auto config = bpConfig();
    config.horizontalnessTemperature = 0.5;
    config.mixedUnaryCost = 0.7;
    config.messageDamping = 1.0;
    const auto one = solveFiberTraceMixedSumProduct(
        lines, oneConstraint, config);
    const auto three = solveFiberTraceMixedSumProduct(
        lines, threeConstraints, config);
    REQUIRE(three.seedPieceIndex == 0);

    const double verticalWeight = 1.0;
    const double mixedWeight = std::exp(
        -config.mixedUnaryCost / config.horizontalnessTemperature);
    const double horizontalWeight = std::exp(
        -3.0 / config.horizontalnessTemperature);
    const double normalization =
        verticalWeight + mixedWeight + horizontalWeight;
    CHECK(three.verticalProbability[1] ==
          doctest::Approx(verticalWeight / normalization).epsilon(1.0e-12));
    CHECK(three.mixedProbability[1] ==
          doctest::Approx(mixedWeight / normalization).epsilon(1.0e-12));
    CHECK(three.horizontalProbability[1] ==
          doctest::Approx(horizontalWeight / normalization).epsilon(1.0e-12));
    CHECK(three.verticalProbability[1] > one.verticalProbability[1]);
    CHECK(three.verticalProbability[three.seedPieceIndex] ==
          doctest::Approx(0.0));
    CHECK(three.mixedProbability[three.seedPieceIndex] ==
          doctest::Approx(0.0));
    CHECK(three.horizontalProbability[three.seedPieceIndex] ==
          doctest::Approx(1.0));
}

TEST_CASE("Mixed-state sum-product BP absorbs conflicting oriented evidence")
{
    const auto lines = bpLines(3);
    auto constraints = bpConstraints(lines.size());
    for (std::size_t repeat = 0; repeat < 5; ++repeat)
        addBpConstraint(constraints, 0, 1, 0.0);
    addBpConstraint(constraints, 0, 2, 0.0);
    addBpConstraint(constraints, 1, 2, 0.0);

    auto config = bpConfig();
    config.horizontalnessTemperature = 0.1;
    config.mixedUnaryCost = 0.2;
    config.messageDamping = 0.25;
    const auto report = solveFiberTraceMixedSumProduct(
        lines, constraints, config);
    CHECK(report.status == "converged");
    CHECK(report.mixedProbability[2] > report.verticalProbability[2]);
    CHECK(report.mixedProbability[2] > report.horizontalProbability[2]);
}

TEST_CASE("Mixed-state sum-product BP validates penalty and limits")
{
    const auto lines = bpLines(3);
    auto constraints = bpConstraints(lines.size());
    addBpConstraint(constraints, 0, 1, 0.1);
    addBpConstraint(constraints, 1, 2, 0.1);
    auto config = bpConfig();
    config.mixedUnaryCost = 0.0;
    const auto cheap = solveFiberTraceMixedSumProduct(
        lines, constraints, config);
    config.mixedUnaryCost = 100.0;
    const auto expensive = solveFiberTraceMixedSumProduct(
        lines, constraints, config);
    CHECK(cheap.mixedProbability[1] > expensive.mixedProbability[1]);
    CHECK(expensive.mixedProbability[1] < 1.0e-10);

    config.horizontalnessTemperature = 1.0e-6;
    const auto cold = solveFiberTraceMixedSumProduct(
        lines, constraints, config);
    for (std::size_t node = 0; node < lines.size(); ++node) {
        CHECK(std::isfinite(cold.verticalProbability[node]));
        CHECK(std::isfinite(cold.mixedProbability[node]));
        CHECK(std::isfinite(cold.horizontalProbability[node]));
    }

    config = bpConfig();
    config.maximumMessageIterations = 1;
    config.messageResidualTolerance = 0.0;
    const auto limited = solveFiberTraceMixedSumProduct(
        lines, constraints, config);
    CHECK(limited.status == "message_limit");
    CHECK_FALSE(limited.messageConverged);

    config = bpConfig();
    config.mixedUnaryCost = -0.1;
    CHECK_THROWS_WITH_AS(
        solveFiberTraceMixedSumProduct(lines, constraints, config),
        doctest::Contains("Mixed unary cost"),
        std::invalid_argument);
    config = bpConfig();
    config.mixedUnaryCost = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_WITH_AS(
        solveFiberTraceMixedSumProduct(lines, constraints, config),
        doctest::Contains("Mixed unary cost"),
        std::invalid_argument);
    config.mixedUnaryCost = std::numeric_limits<double>::infinity();
    CHECK_THROWS_WITH_AS(
        solveFiberTraceMixedSumProduct(lines, constraints, config),
        doctest::Contains("Mixed unary cost"),
        std::invalid_argument);
    config = bpConfig();
    config.balanceMode = FiberTraceBalanceMode::Soft;
    CHECK_THROWS_WITH_AS(
        solveFiberTraceMixedSumProduct(lines, constraints, config),
        doctest::Contains("does not support"),
        std::invalid_argument);
}

TEST_CASE("BP consistency separates resolved mismatches and uncertainty")
{
    auto constraints = bpConstraints(5);
    addBpConstraint(constraints, 0, 1, 0.0);
    addBpConstraint(constraints, 0, 2, 0.25);
    addBpConstraint(constraints, 0, 2, 0.25);
    addBpConstraint(constraints, 0, 3, 0.0);

    const std::vector<double> horizontalness{1.0, 0.0, 1.0, 0.5, 0.5};
    const auto report = analyzeFiberTraceConstraintConsistency(
        constraints, horizontalness);
    REQUIRE(report.pieces.size() == horizontalness.size());
    CHECK(report.verticalThreshold == doctest::Approx(0.25));
    CHECK(report.horizontalThreshold == doctest::Approx(0.75));

    const auto& center = report.pieces[0];
    CHECK(center.degree == 3);
    CHECK(center.totalStrength == doctest::Approx(3.0));
    CHECK(center.resolvedDegree == 2);
    CHECK(center.resolvedStrength == doctest::Approx(2.0));
    CHECK(center.unresolvedDegree == 1);
    CHECK(center.unresolvedStrength == doctest::Approx(1.0));
    CHECK(center.hardMismatches == 1);
    REQUIRE(center.hardMismatchRate);
    REQUIRE(center.weightedHardMismatchRate);
    REQUIRE(center.softMismatchProxy);
    REQUIRE(center.neighborSupportBalance);
    REQUIRE(center.neighborCertainty);
    CHECK(*center.hardMismatchRate == doctest::Approx(0.5));
    CHECK(*center.weightedHardMismatchRate == doctest::Approx(0.5));
    CHECK(*center.softMismatchProxy == doctest::Approx(0.5));
    CHECK(*center.neighborSupportBalance == doctest::Approx(1.0));
    CHECK(*center.neighborCertainty == doctest::Approx(2.0 / 3.0));
    CHECK(center.incidentMeasurements == 4);

    CHECK(report.pieces[1].hardMismatches == 0);
    CHECK(*report.pieces[1].softMismatchProxy == doctest::Approx(0.0));
    CHECK(*report.pieces[2].hardMismatchRate == doctest::Approx(1.0));
    CHECK(report.pieces[3].unresolvedDegree == 1);
    CHECK(*report.pieces[3].softMismatchProxy == doctest::Approx(0.5));
    CHECK(report.pieces[4].degree == 0);
    CHECK(report.pieces[4].totalStrength == doctest::Approx(0.0));
    CHECK_FALSE(report.pieces[4].hardMismatchRate);
    CHECK_FALSE(report.pieces[4].weightedHardMismatchRate);
    CHECK_FALSE(report.pieces[4].softMismatchProxy);
    CHECK_FALSE(report.pieces[4].neighborSupportBalance);
    CHECK_FALSE(report.pieces[4].neighborCertainty);

    CHECK_THROWS_WITH_AS(
        analyzeFiberTraceConstraintConsistency(
            constraints, horizontalness, 0.75, 0.25),
        doctest::Contains("thresholds"),
        std::invalid_argument);

    auto reversed = constraints;
    std::reverse(reversed.constraints.begin(), reversed.constraints.end());
    const auto reordered = analyzeFiberTraceConstraintConsistency(
        reversed, horizontalness);
    const std::vector<double> flipped{0.0, 1.0, 0.0, 0.5, 0.5};
    const auto gaugeFlipped = analyzeFiberTraceConstraintConsistency(
        constraints, flipped);
    for (std::size_t trace = 0; trace < report.pieces.size(); ++trace) {
        const auto& expected = report.pieces[trace];
        for (const auto* actual : {
                 &reordered.pieces[trace], &gaugeFlipped.pieces[trace]}) {
            CHECK(actual->degree == expected.degree);
            CHECK(actual->incidentMeasurements == expected.incidentMeasurements);
            CHECK(actual->hardMismatches == expected.hardMismatches);
            CHECK(actual->hardMismatchRate == expected.hardMismatchRate);
            CHECK(actual->weightedHardMismatchRate ==
                  expected.weightedHardMismatchRate);
            CHECK(actual->softMismatchProxy == expected.softMismatchProxy);
            CHECK(actual->neighborSupportBalance ==
                  expected.neighborSupportBalance);
            CHECK(actual->neighborCertainty == expected.neighborCertainty);
        }
        CHECK(expected.degree ==
              expected.resolvedDegree + expected.unresolvedDegree);
        CHECK(expected.totalStrength == doctest::Approx(
                  expected.resolvedStrength + expected.unresolvedStrength));
    }

    const std::vector<double> inclusiveThresholds{0.75, 0.25, 1.0, 0.5, 0.5};
    const auto inclusive = analyzeFiberTraceConstraintConsistency(
        constraints, inclusiveThresholds);
    CHECK(inclusive.pieces[0].resolvedDegree == 2);

    auto invalid = constraints;
    invalid.constraints.front().perpendicularScore = 0.75;
    CHECK_THROWS_WITH_AS(
        analyzeFiberTraceConstraintConsistency(invalid, horizontalness),
        doctest::Contains("complementary"),
        std::invalid_argument);
}

TEST_CASE("BP consistency respects parallel and perpendicular relations")
{
    auto constraints = bpConstraints(3);
    addBpConstraint(constraints, 0, 1, 1.0);
    addBpConstraint(constraints, 0, 2, 0.0);
    const std::vector<double> horizontalness{1.0, 1.0, 0.0};
    const auto report = analyzeFiberTraceConstraintConsistency(
        constraints, horizontalness);
    const auto& center = report.pieces[0];
    CHECK(center.degree == 2);
    CHECK(center.hardMismatches == 0);
    CHECK(center.hardMismatchRate == doctest::Approx(0.0));
    CHECK(center.weightedHardMismatchRate == doctest::Approx(0.0));
    CHECK(center.softMismatchProxy == doctest::Approx(0.0));
    CHECK(center.neighborSupportBalance == doctest::Approx(0.0));
    CHECK(center.neighborCertainty == doctest::Approx(1.0));

    const std::vector<double> wrong{1.0, 0.0, 1.0};
    const auto mismatched = analyzeFiberTraceConstraintConsistency(
        constraints, wrong);
    CHECK(mismatched.pieces[0].hardMismatches == 2);
    CHECK(mismatched.pieces[0].hardMismatchRate == doctest::Approx(1.0));
    CHECK(mismatched.pieces[0].softMismatchProxy == doctest::Approx(1.0));

    const std::vector<double> vertical{0.0, 0.0, 0.0};
    const std::vector<double> mixed{0.0, 1.0, 0.0};
    const std::vector<double> horizontal{1.0, 0.0, 1.0};
    auto parallelOnly = bpConstraints(2);
    addBpConstraint(parallelOnly, 0, 1, 1.0);
    const auto ternary = analyzeMixedFiberTraceConstraintConsistency(
        parallelOnly,
        std::span<const double>{vertical.data(), 2},
        std::span<const double>{mixed.data(), 2},
        std::span<const double>{horizontal.data(), 2});
    CHECK(ternary.pieces[0].softMismatchProxy == doctest::Approx(0.0));
    CHECK_FALSE(ternary.pieces[0].neighborSupportBalance);
    CHECK(ternary.pieces[0].neighborCertainty == doctest::Approx(0.0));
}

TEST_CASE("Label constraint selection is shared and preserves hard links")
{
    FiberTraceConstraintReport constraints;
    constraints.pieces.resize(4);
    addBpConstraint(constraints, 0, 1, 0.0);
    addBpConstraint(constraints, 1, 2, 0.75);
    FiberTraceConstraint hard;
    hard.pieceA = 2;
    hard.pieceB = 3;
    hard.parallelScore = 1.0;
    hard.perpendicularScore = 0.0;
    hard.hardContinuity = true;
    constraints.constraints.push_back(hard);

    FiberTraceLabelingConfig config;
    config.perpendicularOnly = true;
    const auto selection = selectFiberTraceLabelConstraints(constraints, config);
    CHECK(selection.retainedIndices == std::vector<std::size_t>{0, 2});
    CHECK(selection.excludedNonPerpendicular == 1);
    CHECK(selection.excludedParallelSeparateWinding == 0);

}

TEST_CASE("Trace consensus writes requested assignment milestones")
{
    std::vector<FiberletCropTraceLine> traces(205);
    for (std::size_t trace = 0; trace < traces.size(); ++trace) {
        const double length = static_cast<double>(trace + 1);
        traces[trace].pointsBaseXYZ = {{0.0, 0.0, 0.0}, {length, 0.0, 0.0}};
    }
    FiberTraceConstraintReport constraints;
    FiberTraceConsensusConfig config;
    config.cropMinimumBaseXYZ = {-100.0, -100.0, -100.0};
    config.cropMaximumBaseXYZ = {100.0, 100.0, 100.0};
    const auto consensus = growFiberTraceConsensus(
        traces, constraints, config);
    const std::vector<std::size_t> expected{
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200};
    CHECK(consensus.snapshotAddedCounts == expected);
    REQUIRE(consensus.steps.size() == traces.size());
    CHECK(consensus.steps.front().traceIndex == 204);
    CHECK(consensus.steps.back().traceIndex == 0);

    const TemporaryDirectory directory("trace_consensus_objs");
    const auto written = writeFiberTraceConsensusObjs(
        traces, consensus, directory.path / "crop.result");
    CHECK(written.finalPaths.h == directory.path / "crop_h.obj");
    CHECK(written.finalPaths.v == directory.path / "crop_v.obj");
    CHECK(written.finalPaths.broken == directory.path / "crop_broken.obj");
    CHECK(written.hCount == 205);
    CHECK(written.vCount == 0);
    CHECK(written.brokenCount == 0);
    CHECK(written.hCount + written.vCount + written.brokenCount == 205);
    REQUIRE(written.snapshots.size() == expected.size());
    for (const auto& snapshot : written.snapshots) {
        CHECK(snapshot.hCount + snapshot.vCount + snapshot.brokenCount ==
              snapshot.addedCount);
    }
    CHECK(written.snapshots.front().paths.h ==
          directory.path / "crop_step_10_h.obj");
    CHECK(written.snapshots.front().hCount == 10);
    CHECK(written.snapshots.front().brokenCount == 0);
    CHECK(written.snapshots.back().paths.v ==
          directory.path / "crop_step_200_v.obj");
    CHECK(written.snapshots.back().hCount == 200);
    CHECK(written.snapshots.back().paths.broken ==
          directory.path / "crop_step_200_broken.obj");

    const auto read = [](const std::filesystem::path& path) {
        std::ifstream input(path);
        std::ostringstream text;
        text << input.rdbuf();
        return text.str();
    };
    CHECK(read(written.finalPaths.h).find("o trace_204\n") !=
          std::string::npos);
    CHECK(read(written.finalPaths.v).find("\no ") == std::string::npos);
    CHECK(read(written.finalPaths.broken).find("\no ") == std::string::npos);

    auto withBroken = consensus;
    withBroken.labels[204] = FiberTraceConsensusLabel::Broken;
    withBroken.steps.front().label = FiberTraceConsensusLabel::Broken;
    const auto brokenWritten = writeFiberTraceConsensusObjs(
        traces, withBroken, directory.path / "with_broken.result");
    CHECK(brokenWritten.hCount == 204);
    CHECK(brokenWritten.vCount == 0);
    CHECK(brokenWritten.brokenCount == 1);
    CHECK(read(brokenWritten.finalPaths.broken).find("o trace_204\n") !=
          std::string::npos);
    CHECK(read(brokenWritten.finalPaths.h).find("o trace_204\n") ==
          std::string::npos);
    REQUIRE(!brokenWritten.snapshots.empty());
    CHECK(brokenWritten.snapshots.front().hCount == 9);
    CHECK(brokenWritten.snapshots.front().brokenCount == 1);
    CHECK(read(brokenWritten.snapshots.front().paths.broken)
              .find("o trace_204\n") != std::string::npos);

    auto tracesWithDegenerate = traces;
    tracesWithDegenerate.push_back({});
    withBroken.labels.push_back(FiberTraceConsensusLabel::Broken);
    const auto withoutDegenerate = writeFiberTraceConsensusObjs(
        tracesWithDegenerate,
        withBroken,
        directory.path / "with_degenerate.result");
    CHECK(withoutDegenerate.brokenCount == 1);
}
