#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberReplay.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>

namespace
{

std::filesystem::path temporaryDirectory()
{
    std::mt19937_64 generator(std::random_device{}());
    const auto path = std::filesystem::temp_directory_path() /
        ("vc_fiber_replay_" + std::to_string(generator()));
    std::filesystem::create_directories(path);
    return path;
}

} // namespace

TEST_CASE("fiber replay tube uses exact endpoint caps and sorted explicit cells")
{
    vc::fiber_tracer::FiberPredictionGridInfo grid;
    grid.shapeZYX = {8, 8, 8};
    grid.predictionToBaseScale = 2.0;
    const auto tube = vc::fiber_tracer::makeFiberReplayTube(
        {{2.0, 4.0, 4.0}, {10.0, 4.0, 4.0}},
        4.0,
        2.0,
        2.0,
        grid,
        2);

    CHECK(tube.beginArcBase == doctest::Approx(2.0));
    CHECK(tube.endArcBase == doctest::Approx(6.0));
    CHECK(tube.containsBasePoint({4.0, 6.0, 4.0}));
    CHECK_FALSE(tube.containsBasePoint({1.9, 6.1, 4.0}));
    CHECK(std::is_sorted(tube.cellsZYX.begin(), tube.cellsZYX.end()));
    REQUIRE_FALSE(tube.cellsZYX.empty());
    CHECK(tube.cellsZYX == vc::fiber_tracer::fiberAnchorCellsNearPolyline(
        tube.referenceIntervalBase, tube.radiusBaseVoxels, grid, 2));
    CHECK(tube.cellsZYX == std::vector<std::array<size_t, 3>>{
        {0, 0, 0}, {0, 0, 1}, {0, 0, 2},
        {0, 1, 0}, {0, 1, 1}, {0, 1, 2},
        {1, 0, 0}, {1, 0, 1}, {1, 0, 2},
        {1, 1, 0}, {1, 1, 1}, {1, 1, 2},
    });
    CHECK(tube.volumeCropBaseXYZWHD ==
          std::array<size_t, 6>{2, 2, 2, 8, 4, 4});
}

TEST_CASE("forward replay matching uses caller supplied variable advance")
{
    const auto reference = vc::fiber_tracer::makePolylineArcGeometry(
        {{0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}});
    const auto shortStep = vc::fiber_tracer::matchForwardPolylinePoint(
        reference, {3.0, 1.0, 0.0}, 0.0, 1.0, 0.0);
    CHECK(shortStep.predictedArc == doctest::Approx(1.0));
    CHECK(shortStep.projection.arc == doctest::Approx(1.0));
    CHECK(shortStep.projection.distance == doctest::Approx(std::sqrt(5.0)));

    const auto longStep = vc::fiber_tracer::matchForwardPolylinePoint(
        reference, {3.0, 1.0, 0.0}, 0.0, 3.0, 0.0);
    CHECK(longStep.predictedArc == doctest::Approx(3.0));
    CHECK(longStep.projection.arc == doctest::Approx(3.0));
    CHECK(longStep.projection.distance == doctest::Approx(1.0));
}

TEST_CASE("fiber replay comparison uses one symmetric available extent")
{
    const auto reference = vc::fiber_tracer::makePolylineArcGeometry(
        {{0.0, 0.0, 0.0}, {30.0, 0.0, 0.0}});
    const auto trace = vc::fiber_tracer::makePolylineArcGeometry(
        {{0.0, 1.0, 0.0},
         {4.0, 1.0, 0.0},
         {10.0, 1.0, 0.0},
         {18.0, 1.0, 0.0}});

    const auto requested = vc::fiber_tracer::makeFiberReplayComparisonWindow(
        reference, 15.0, trace, 2, 7.5);
    CHECK(requested.effectiveHalfExtentBaseVoxels == doctest::Approx(7.5));
    CHECK(requested.referenceBeginArcBase == doctest::Approx(7.5));
    CHECK(requested.referenceEndArcBase == doctest::Approx(22.5));
    CHECK(requested.traceBeginArcBase == doctest::Approx(2.5));
    CHECK(requested.traceEndArcBase == doctest::Approx(17.5));

    const auto clipped = vc::fiber_tracer::makeFiberReplayComparisonWindow(
        reference, 3.0, trace, 1, 10.0);
    CHECK(clipped.effectiveHalfExtentBaseVoxels == doctest::Approx(3.0));
    CHECK(clipped.referenceBeginArcBase == doctest::Approx(0.0));
    CHECK(clipped.referenceEndArcBase == doctest::Approx(6.0));
    CHECK(clipped.traceBeginArcBase == doctest::Approx(1.0));
    CHECK(clipped.traceEndArcBase == doctest::Approx(7.0));
}

TEST_CASE("nonfailure replay publication retains only applicable artifacts")
{
    const auto directory = temporaryDirectory();
    vc::fiber_tracer::FiberReplayBundleInput input;
    input.replay.status = vc::fiber_tracer::FiberReplayStatus::NoFailure;
    input.replay.terminationReason = "reference_end";
    input.replay.tracePointsBase = {{0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}};
    input.replay.cumulativeLosses = {0.0, 1.0};
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
    CHECK(bundle.at("status") == "no_failure");
    CHECK(bundle.at("comparison_trace_points_base_xyz") ==
          bundle.at("trace_points_base_xyz"));
    CHECK(bundle.at("comparison").is_null());
    CHECK(bundle.at("tube").is_null());
    CHECK(bundle.at("volume_crop_base_xyzwhd").is_null());
    CHECK(bundle.at("fiberlet_replay").is_null());
    REQUIRE(bundle.at("artifacts").size() == 2);
    CHECK(bundle.at("artifacts").contains("replay/reference.obj"));
    CHECK(bundle.at("artifacts").contains("replay/trace.obj"));
    CHECK(std::filesystem::is_regular_file(directory / "fiber_replay.json"));
    std::ifstream firstInput(directory / "fiber_replay.json", std::ios::binary);
    const std::string first{
        std::istreambuf_iterator<char>(firstInput),
        std::istreambuf_iterator<char>()};
    const auto repeated = vc::fiber_tracer::writeFiberReplayBundle(directory, input);
    CHECK(repeated == bundle);
    std::ifstream repeatedInput(directory / "fiber_replay.json", std::ios::binary);
    CHECK(std::string(
              std::istreambuf_iterator<char>(repeatedInput),
              std::istreambuf_iterator<char>()) == first);

    auto invalid = input;
    invalid.replay.status = vc::fiber_tracer::FiberReplayStatus::FailureTruncated;
    invalid.replay.failureTracePointIndex = 1;
    invalid.replay.failureReferenceArcBase = 4.0;
    CHECK_THROWS_AS(
        vc::fiber_tracer::writeFiberReplayBundle(directory, invalid),
        std::invalid_argument);
    std::ifstream preservedInput(directory / "fiber_replay.json", std::ios::binary);
    CHECK(std::string(
              std::istreambuf_iterator<char>(preservedInput),
              std::istreambuf_iterator<char>()) == first);
    std::filesystem::remove_all(directory);
}
