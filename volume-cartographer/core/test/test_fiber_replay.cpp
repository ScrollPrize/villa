#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberReplay.hpp"

#include <filesystem>
#include <fstream>
#include <atomic>
#include <iterator>
#include <limits>
#include <random>
#include <thread>

namespace
{

std::filesystem::path temporaryDirectory()
{
    std::mt19937_64 generator(std::random_device{}());
    const auto path = std::filesystem::temp_directory_path() / ("vc_fiber_replay_" + std::to_string(generator()));
    std::filesystem::create_directories(path);
    return path;
}

std::string readText(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

float pointSegmentDistanceSquared(
    const cv::Vec3f& point,
    const cv::Vec3f& start,
    const cv::Vec3f& finish)
{
    const cv::Vec3f delta = finish - start;
    const float denominator = delta.dot(delta);
    const float fraction = denominator > 0.0f
        ? std::clamp(
              (point - start).dot(delta) / denominator, 0.0f, 1.0f)
        : 0.0f;
    const cv::Vec3f residual = point - (start + delta * fraction);
    return residual.dot(residual);
}

bool directFloatContains(
    const vc::fiber_tracer::FiberReplayTube& tube,
    const cv::Vec3d& pointPrediction,
    double predictionToBaseScale)
{
    const cv::Vec3f point{
        static_cast<float>(pointPrediction[0]),
        static_cast<float>(pointPrediction[1]),
        static_cast<float>(pointPrediction[2]),
    };
    const float radius = static_cast<float>(
        tube.radiusBaseVoxels / predictionToBaseScale);
    const float radiusSquared = radius * radius;
    for (const auto& source : vc::fiber_tracer::clippedPolylineArcSegments(
             tube.reference, tube.beginArcBase, tube.endArcBase)) {
        const cv::Vec3f start{
            static_cast<float>(source.start[0] / predictionToBaseScale),
            static_cast<float>(source.start[1] / predictionToBaseScale),
            static_cast<float>(source.start[2] / predictionToBaseScale),
        };
        const cv::Vec3f finish{
            static_cast<float>(source.finish[0] / predictionToBaseScale),
            static_cast<float>(source.finish[1] / predictionToBaseScale),
            static_cast<float>(source.finish[2] / predictionToBaseScale),
        };
        if (pointSegmentDistanceSquared(point, start, finish) <= radiusSquared)
            return true;
    }
    return false;
}

}  // namespace

TEST_CASE("fiber replay tube uses exact endpoint caps and sorted explicit cells")
{
    vc::fiber_tracer::FiberPredictionGridInfo grid;
    grid.shapeZYX = {8, 8, 8};
    grid.predictionToBaseScale = 2.0;
    const auto tube = vc::fiber_tracer::makeFiberReplayTube({{2.0, 4.0, 4.0}, {10.0, 4.0, 4.0}}, 4.0, 2.0, 2.0, grid, 2);

    CHECK(tube.beginArcBase == doctest::Approx(2.0));
    CHECK(tube.endArcBase == doctest::Approx(6.0));
    REQUIRE(tube.referenceIntervalBase.size() == 2);
    CHECK(tube.referenceIntervalBase.front()[0] == doctest::Approx(4.0));
    CHECK(tube.referenceIntervalBase.back()[0] == doctest::Approx(8.0));
    CHECK(tube.containsBasePoint({4.0, 6.0, 4.0}));
    CHECK_FALSE(tube.containsBasePoint({1.9, 6.1, 4.0}));
    CHECK(std::is_sorted(tube.cellsZYX.begin(), tube.cellsZYX.end()));
    REQUIRE_FALSE(tube.cellsZYX.empty());
    CHECK(tube.cellsZYX == vc::fiber_tracer::fiberAnchorCellsNearPolyline(tube.referenceIntervalBase, tube.radiusBaseVoxels, grid, 2));
    CHECK(
        tube.cellsZYX == std::vector<std::array<size_t, 3>>{
                             {0, 0, 0},
                             {0, 0, 1},
                             {0, 0, 2},
                             {0, 1, 0},
                             {0, 1, 1},
                             {0, 1, 2},
                             {1, 0, 0},
                             {1, 0, 1},
                             {1, 0, 2},
                             {1, 1, 0},
                             {1, 1, 1},
                             {1, 1, 2},
                         });
    CHECK(tube.volumeCropBaseXYZWHD == std::array<size_t, 6>{2, 2, 2, 8, 4, 4});
}

TEST_CASE("fiber replay tube float containment index matches direct segment scan")
{
    vc::fiber_tracer::FiberPredictionGridInfo grid;
    grid.shapeZYX = {128, 128, 128};
    grid.predictionToBaseScale = 2.0;
    const auto tube = vc::fiber_tracer::makeFiberReplayTube(
        {
            {4.0, 4.0, 4.0},
            {20.0, 4.0, 4.0},
            {20.0, 24.0, 8.0},
            {36.0, 24.0, 8.0},
        },
        18.0,
        17.0,
        5.0,
        grid,
        2);
    const auto query = tube.makePredictionContainmentQuery(
        grid.predictionToBaseScale);

    std::mt19937 generator(42);
    std::uniform_real_distribution<double> coordinate(-2.0, 24.0);
    for (size_t index = 0; index < 20000; ++index) {
        const cv::Vec3d point{
            coordinate(generator),
            coordinate(generator),
            coordinate(generator),
        };
        CHECK(
            query.containsPredictionPoint(point) ==
            directFloatContains(tube, point, grid.predictionToBaseScale));

        const double legacyDistancePrediction = tube.distanceToBasePoint(
            point * grid.predictionToBaseScale) / grid.predictionToBaseScale;
        const double radiusPrediction =
            tube.radiusBaseVoxels / grid.predictionToBaseScale;
        if (std::abs(legacyDistancePrediction - radiusPrediction) > 1.0e-4) {
            CHECK(
                query.containsPredictionPoint(point) ==
                (legacyDistancePrediction <= radiusPrediction + 5.0e-13));
        }
    }
}

TEST_CASE("fiber replay tube float containment owns its data and supports concurrent queries")
{
    vc::fiber_tracer::FiberPredictionGridInfo grid;
    grid.shapeZYX = {64, 64, 64};
    grid.predictionToBaseScale = 4.0;
    auto query = [&] {
        const auto tube = vc::fiber_tracer::makeFiberReplayTube(
            {{8.0, 12.0, 16.0}, {72.0, 12.0, 16.0}},
            32.0,
            24.0,
            8.0,
            grid,
            2);
        return tube.makePredictionContainmentQuery(
            grid.predictionToBaseScale);
    }();
    const auto copy = query;

    std::vector<std::thread> workers;
    std::atomic<bool> matched{true};
    for (size_t worker = 0; worker < 8; ++worker) {
        workers.emplace_back([&, worker] {
            for (size_t index = 0; index < 10000; ++index) {
                const cv::Vec3d point{
                    2.0 + static_cast<double>((index + worker) % 16),
                    1.0 + static_cast<double>(index % 5),
                    4.0,
                };
                if (query.containsPredictionPoint(point) !=
                    copy.containsPredictionPoint(point)) {
                    matched.store(false, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& worker : workers)
        worker.join();
    CHECK(matched.load(std::memory_order_relaxed));
    CHECK(query.containsPredictionPoint({8.0, 3.0, 4.0}));
    CHECK_FALSE(query.containsPredictionPoint({8.0, 6.1, 4.0}));
    CHECK_THROWS_AS(
        query.containsPredictionPoint(
            {std::numeric_limits<double>::infinity(), 0.0, 0.0}),
        std::invalid_argument);
}

TEST_CASE("forward replay matching uses caller supplied variable advance")
{
    const auto reference = vc::fiber_tracer::makePolylineArcGeometry({{0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}});
    const auto shortStep = vc::fiber_tracer::matchForwardPolylinePoint(reference, {3.0, 1.0, 0.0}, 0.0, 1.0, 0.0);
    CHECK(shortStep.predictedArc == doctest::Approx(1.0));
    CHECK(shortStep.projection.arc == doctest::Approx(1.0));
    CHECK(shortStep.projection.distance == doctest::Approx(std::sqrt(5.0)));

    const auto longStep = vc::fiber_tracer::matchForwardPolylinePoint(reference, {3.0, 1.0, 0.0}, 0.0, 3.0, 0.0);
    CHECK(longStep.predictedArc == doctest::Approx(3.0));
    CHECK(longStep.projection.arc == doctest::Approx(3.0));
    CHECK(longStep.projection.distance == doctest::Approx(1.0));
}

TEST_CASE("forward polyline interval defaults to the complete remaining reference")
{
    const auto reference = vc::fiber_tracer::makePolylineArcGeometry({
        {0.0, 0.0, 0.0},
        {3.0, 0.0, 0.0},
        {7.0, 0.0, 0.0},
        {12.0, 0.0, 0.0},
    });

    const auto complete =
        vc::fiber_tracer::selectForwardPolylineArcInterval(reference, 1);
    CHECK(complete.beginArc == doctest::Approx(3.0));
    CHECK(complete.endArc == doctest::Approx(12.0));

    const auto limited = vc::fiber_tracer::selectForwardPolylineArcInterval(
        reference, 1, 4.0);
    CHECK(limited.beginArc == doctest::Approx(3.0));
    CHECK(limited.endArc == doctest::Approx(7.0));

    const auto clamped = vc::fiber_tracer::selectForwardPolylineArcInterval(
        reference, 1, 100.0);
    CHECK(clamped.endArc == doctest::Approx(12.0));

    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::selectForwardPolylineArcInterval(reference, 3),
        doctest::Contains("no forward extent"), std::invalid_argument);
}

TEST_CASE("dual replay publication is deterministic and no-vis has only full traces")
{
    const auto directory = temporaryDirectory();
    vc::fiber_tracer::FiberReplayBundleInput input;
    input.request.fiber.linePointsXyzBase = {
        {0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}, {8.0, 0.0, 0.0}};
    input.request.fiber.controlPointsXyzBase = {
        input.request.fiber.linePointsXyzBase.front(),
        input.request.fiber.linePointsXyzBase.back()};
    input.request.fiber.controlPointLineIndices = {0, 2};
    input.request.referenceEndArcBase = 4.0;
    input.requestedLengthBaseVoxels = 4.0;
    input.greedyReplay.referenceEndArcBase = 4.0;
    input.greedyReplay.completedReferenceArcBase = 4.0;
    vc::fiber_tracer::FiberReplayTraceSegment greedyFirst;
    greedyFirst.endReferenceArcBase = 2.0;
    greedyFirst.terminationReason = "distance_above_threshold";
    greedyFirst.tracePointsBase = {{0.0, 0.0, 0.0}, {2.0, 1.0, 0.0}};
    greedyFirst.cumulativeLosses = {0.0, 1.0};
    vc::fiber_tracer::FiberReplayTraceSegment greedySecond;
    greedySecond.startReferenceArcBase = 2.0;
    greedySecond.endReferenceArcBase = 4.0;
    greedySecond.terminationReason = "reference_end";
    greedySecond.tracePointsBase = {{2.0, 0.0, 0.0}, {4.0, 0.0, 0.0}};
    greedySecond.cumulativeLosses = {0.0, 1.0};
    input.greedyReplay.segments = {greedyFirst, greedySecond};

    input.fiberletReplay.referenceEndArcBase = 4.0;
    input.fiberletReplay.completedReferenceArcBase = 4.0;
    vc::fiber_tracer::FiberletGraphReplaySegment fiberlet;
    fiberlet.endReferenceArcBase = 4.0;
    fiberlet.terminationReason = "reference_end";
    fiberlet.routePointsBaseXYZ = {{0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}};
    input.fiberletReplay.segments = {fiberlet};
    input.fiberletReplayConfig.referenceEndArcBase = 4.0;
    input.referenceGeometryBase = {{0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}};
    input.sources = nlohmann::json::object();
    input.traceBinding = nlohmann::json::object();
    input.predictionBinding = {
        {"mode", "canonical_stored_grid"},
        {"prediction_to_base_scale", 1.0},
        {"prediction_shape_zyx", {8, 8, 8}},
    };
    input.requestedTraceConfig = nlohmann::json::object();
    input.effectiveTraceConfig = nlohmann::json::object();

    const auto bundle = vc::fiber_tracer::writeFiberReplayBundle(directory, input);
    CHECK(bundle.at("version") == 2);
    CHECK(bundle.at("requested_length_base_voxels") == 4.0);
    CHECK(bundle.at("reference_begin_arc_base") == 0.0);
    CHECK(bundle.at("reference_end_arc_base") == 4.0);
    CHECK(bundle.at("reference_length_base_voxels") == 4.0);
    CHECK(bundle.at("reference_points_base_xyz").size() == 2);
    CHECK(bundle.at("visualizations").empty());
    REQUIRE(bundle.at("artifacts").size() == 5);
    CHECK(bundle.at("artifacts").contains("replay/reference.obj"));
    CHECK(bundle.at("artifacts").contains("replay/greedy.json"));
    CHECK(bundle.at("artifacts").contains("replay/greedy.obj"));
    CHECK(bundle.at("artifacts").contains("replay/fiberlet.json"));
    CHECK(bundle.at("artifacts").contains("replay/fiberlet.obj"));
    const auto greedyPath = directory / bundle.at("artifacts").at("replay/greedy.obj").at("path").get<std::string>();
    const std::string greedyObj = readText(greedyPath);
    CHECK(greedyObj.find("g segment_0") != std::string::npos);
    CHECK(greedyObj.find("g segment_1") != std::string::npos);
    CHECK(greedyObj.find("l 1 2\ng segment_1") != std::string::npos);
    CHECK(greedyObj.find("l 1 2 3") == std::string::npos);

    const std::string first = readText(directory / "fiber_replay.json");
    const auto repeated = vc::fiber_tracer::writeFiberReplayBundle(directory, input);
    CHECK(repeated == bundle);
    CHECK(readText(directory / "fiber_replay.json") == first);

    auto invalid = input;
    invalid.fiberletReplay.completedReferenceArcBase = 3.0;
    CHECK_THROWS_AS(vc::fiber_tracer::writeFiberReplayBundle(directory, invalid), std::invalid_argument);
    CHECK(readText(directory / "fiber_replay.json") == first);

    input.sources = {
        {"fiber_manifest", "fiber.json"},
        {"fiber_manifest_content_hash", "fnv1a64:1"},
        {"normal_manifest", "normal.json"},
        {"normal_manifest_content_hash", "fnv1a64:2"},
        {"fiber_json", "reference.json"},
        {"fiber_json_content_hash", "fnv1a64:3"},
    };
    input.greedyReplay.failures.push_back({
        0, 0, "distance_above_threshold", 2.0, 0.5,
        {2.0, 0.0, 0.0}, cv::Vec3d{2.0, 1.0, 0.0}, 1,
    });
    vc::fiber_tracer::FiberPredictionGridInfo grid;
    grid.shapeZYX = {8, 8, 8};
    grid.predictionToBaseScale = 1.0;
    vc::fiber_tracer::FiberReplayVisualizationInput visualization;
    visualization.tracer = vc::fiber_tracer::FiberReplayTracer::Greedy;
    visualization.tube = vc::fiber_tracer::makeFiberReplayTube(
        input.request.fiber.linePointsXyzBase, 2.0, 2.0, 2.0, grid, 2);
    visualization.anchors.grid = grid;
    visualization.anchorArtifact.sourceLocator = "fiber.json";
    visualization.anchorArtifact.manifestContentHash = "fnv1a64:1";
    visualization.paths.grid = grid;
    visualization.paths.anchorCellSizePredictionVoxels = 2;
    visualization.pathArtifact.fiberManifestLocator = "fiber.json";
    visualization.pathArtifact.fiberManifestContentHash = "fnv1a64:1";
    visualization.pathArtifact.normalManifestLocator = "normal.json";
    visualization.pathArtifact.normalManifestContentHash = "fnv1a64:2";
    visualization.pathArtifact.anchorArtifactLocator = "anchors/anchors.json";
    visualization.pathArtifact.anchorArtifactContentHash = "fnv1a64:3";
    input.visualizations.push_back(std::move(visualization));
    const auto visualBundle =
        vc::fiber_tracer::writeFiberReplayBundle(directory, input);
    REQUIRE(visualBundle.at("visualizations").size() == 1);
    const auto alias = visualBundle.at("visualizations").at(0).at("manifest").at("path").get<std::string>();
    CHECK(alias == "fiber_replay_visualization.greedy.000000.json");
    REQUIRE(std::filesystem::exists(directory / alias));
    const auto local = nlohmann::json::parse(readText(directory / alias));
    CHECK(local.at("format") == "vc_fiber_replay_visualization");
    CHECK(local.at("artifacts").at("replay/reference.obj").at("path").get<std::string>().starts_with("runs/"));

    input.visualizations.clear();
    input.greedyReplay.failures.clear();
    const auto noVisualBundle =
        vc::fiber_tracer::writeFiberReplayBundle(directory, input);
    CHECK(noVisualBundle.at("visualizations").empty());
    CHECK_FALSE(std::filesystem::exists(directory / alias));
    std::filesystem::remove_all(directory);
}
