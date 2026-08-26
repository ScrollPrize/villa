#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/io/PolylineObj.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include "vc/fiber_tracer/FiberletCropTrace.hpp"
#include "vc/fiber_tracer/FiberletCropTraceArtifact.hpp"
#include "vc/fiber_tracer/FiberletDataset.hpp"
#include "vc/fiber_tracer/FiberTraceConstraints.hpp"
#include "vc/lasagna/Dataset.hpp"

#include "utils/zarr.hpp"

#include <filesystem>
#include <fstream>
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
        return {id, source, target, positions.at(source), positions.at(target), cv::Vec3f(delta), cv::Vec3f(delta), edgeLength, {0, edgeLength, 0, 0, 0}, std::nullopt, std::nullopt};
    }
    [[nodiscard]] FiberletReplaySourceCostProfile costProfile(const DirectedFiberletStorageId& id) const override
    {
        const auto edge = arc(id);
        return {{edge.pathLengthPredictionVoxels}, {1.0F}};
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
    CHECK(std::abs(classified.direction1BaseXYZ.dot(cv::Vec3d{1, 0, 0})) == doctest::Approx(1.0));
    CHECK(std::abs(classified.direction2BaseXYZ.dot(diagonal)) == doctest::Approx(1.0));
    REQUIRE(classified.lines.size() == lines.size());
    CHECK(classified.lines[0].group == FiberDirectionGroup::Direction1);
    CHECK(classified.lines[1].group == FiberDirectionGroup::Direction1);
    CHECK(classified.lines[2].group == FiberDirectionGroup::Direction2);
    CHECK(classified.lines[3].group == FiberDirectionGroup::Mixed);
    CHECK(classified.lines[4].group == FiberDirectionGroup::Direction1);
    CHECK(classified.lines[4].direction1LengthBaseVoxels == doctest::Approx(3.0));
    CHECK(classified.lines[4].direction2LengthBaseVoxels == doctest::Approx(1.0));
    CHECK(classified.lines[5].group == FiberDirectionGroup::Mixed);
    CHECK(classified.groupCounts == std::array<std::size_t, 3>{3, 1, 2});
    CHECK(classified.analyzedSteps == 10);
    CHECK(classified.analyzedLengthBaseVoxels == doctest::Approx(42.0));
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

    REQUIRE(serial.constraints.size() == 3);
    REQUIRE(parallel.constraints.size() == serial.constraints.size());
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
    }
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
