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
    CHECK(tube.volumeCropBaseXYZWHD ==
          std::array<size_t, 6>{2, 2, 2, 8, 4, 4});
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
    CHECK(bundle.at("tube").is_null());
    CHECK(bundle.at("volume_crop_base_xyzwhd").is_null());
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
