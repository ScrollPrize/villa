#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberReplay.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>

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
