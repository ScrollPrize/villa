#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberLocalScoring.hpp"
#include "vc/fiber_tracer/FiberPaths.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <tuple>

namespace
{

class ConstantNormalSampler final : public vc::lasagna::NormalSampler
{
public:
    explicit ConstantNormalSampler(cv::Vec3d normal = {0.0, 0.0, 1.0}, bool valid = true) : normal_(normal), valid_(valid) {}

    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override
    {
        return {normal_, valid_, valid_ ? std::string{} : "invalid"};
    }

private:
    cv::Vec3d normal_;
    bool valid_;
};

vc::fiber_tracer::LoadedFiberAnchorArtifact twoAnchorArtifact(
    cv::Vec3d firstAxis = {1.0, 0.0, 0.0},
    cv::Vec3d secondAxis = {1.0, 0.0, 0.0},
    cv::Vec3d firstPosition = {2.25, 4.25, 4.25},
    cv::Vec3d secondPosition = {10.25, 4.25, 4.25})
{
    vc::fiber_tracer::LoadedFiberAnchorArtifact artifact;
    artifact.artifact.sourceLocator = "/tmp/fiber.lasagna.json";
    artifact.artifact.manifestContentHash = "fnv1a64:0123456789abcdef";
    artifact.report.grid = {{16, 16, 16}, 2.0};
    artifact.report.config.cellSizePredictionVoxels = 2;
    artifact.report.config.gaussianSigmaPredictionVoxels = 1.0;
    artifact.report.config.observationPresenceFloor = 0.05;
    artifact.report.config.minimumAlignedSupport = 0.05;
    artifact.report.config.parallelThreads = 1;
    for (const auto& entry :
         std::array{std::tuple{std::array<size_t, 3>{2, 2, 1}, firstPosition, firstAxis}, std::tuple{std::array<size_t, 3>{2, 2, 5}, secondPosition, secondAxis}}) {
        vc::fiber_tracer::FiberCellAnchorResult cell;
        cell.cellZYX = std::get<0>(entry);
        cell.retainedAnchorCount = 1;
        cell.objective = 1.0;
        cell.components[0].retained = true;
        cell.components[0].assignedObservationCount = 8;
        cell.components[0].anchor.cellZYX = cell.cellZYX;
        cell.components[0].anchor.positionPredictionXYZ = std::get<1>(entry);
        cell.components[0].anchor.axisXYZ = std::get<2>(entry);
        cell.components[0].anchor.alignedSupport = 1.0;
        cell.components[0].anchor.directionalCoherence = 1.0;
        cell.components[1].rejectionReason = "empty";
        artifact.report.nonEmptyCells.push_back(cell);
    }
    artifact.report.diagnostics.totalCells = 2;
    artifact.report.diagnostics.oneAnchorCells = 2;
    artifact.report.selectedCrop = {{0, 0, 0}, {16, 16, 16}};
    artifact.report.selectedCellBeginZYX = {0, 0, 0};
    artifact.report.selectedCellEndZYX = {8, 8, 8};
    return artifact;
}

vc::fiber_tracer::FiberletPathConfig pathConfig()
{
    vc::fiber_tracer::FiberletPathConfig config;
    config.corridorRadiusPredictionVoxels = 2.0;
    return config;
}

vc::fiber_tracer::FiberStoredPredictionBatchSampler constantPredictions(cv::Vec3d direction = {1.0, 0.0, 0.0}, double presence = 1.0)
{
    return [=](const auto& indices, int, auto& samples) { samples.assign(indices.size(), {direction, presence, true}); };
}

std::filesystem::path temporaryPath(const std::string& tag)
{
    std::mt19937_64 generator(std::random_device{}());
    return std::filesystem::temp_directory_path() / ("vc_fiberlet_paths_" + tag + "_" + std::to_string(generator()) + ".json");
}

size_t occurrenceCount(const std::string& text, const std::string& needle)
{
    size_t count = 0;
    size_t offset = 0;
    while ((offset = text.find(needle, offset)) != std::string::npos) {
        ++count;
        offset += needle.size();
    }
    return count;
}

}  // namespace

TEST_CASE("fiber local smoothness preserves native split equations")
{
    const float pi = static_cast<float>(std::acos(-1.0));
    const vc::fiber_tracer::FiberLocalSmoothnessConfig config{2.0f, 0.1f, 10.0f, 0.0f};
    const auto tangent = vc::fiber_tracer::fiberLocalSmoothnessCost({1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, true, config);
    CHECK(tangent.mode == vc::fiber_tracer::FiberLocalSmoothnessMode::NormalAware);
    CHECK(tangent.tangent == doctest::Approx(10.0f * pi * pi / 4.0f));
    CHECK(tangent.normal == doctest::Approx(0.0f));
    CHECK(tangent.isotropic == doctest::Approx(0.0f));

    const float invSqrt2 = static_cast<float>(std::sqrt(0.5));
    const auto normal = vc::fiber_tracer::fiberLocalSmoothnessCost({1.0f, 0.0f, 0.0f}, {invSqrt2, 0.0f, invSqrt2}, {0.0f, 0.0f, 1.0f}, true, config);
    CHECK(normal.tangent == doctest::Approx(0.0f));
    CHECK(normal.normal == doctest::Approx(0.1f * pi * pi / 16.0f));

    const auto fallback = vc::fiber_tracer::fiberLocalSmoothnessCost({1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {}, false, config);
    CHECK(fallback.mode == vc::fiber_tracer::FiberLocalSmoothnessMode::IsotropicFallback);
    CHECK(fallback.isotropic == doctest::Approx(2.0f * pi * pi / 4.0f));
    CHECK(fallback.tangent == 0.0f);
    CHECK(fallback.normal == 0.0f);

    auto free = config;
    free.freeAngleRadians = pi * 0.5f;
    CHECK(vc::fiber_tracer::fiberLocalSmoothnessCost({1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {}, false, free).total() == doctest::Approx(0.0f));
}

TEST_CASE("fiberlet radius-four shell is symmetric and not axis-only")
{
    const auto offsets = vc::fiber_tracer::fiberletCellShellOffsets(4, 0.5);
    REQUIRE(offsets.size() > 6);
    const std::set<std::array<int, 3>> unique(offsets.begin(), offsets.end());
    CHECK(unique.size() == offsets.size());
    CHECK(unique.contains({0, 0, 4}));
    CHECK(unique.contains({0, 0, -4}));
    CHECK(unique.contains({1, 2, 3}));
    for (const auto& offset : offsets) {
        const double length = std::sqrt(static_cast<double>(offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]));
        CHECK(length >= 3.5);
        CHECK(length < 4.5);
        CHECK(unique.contains({-offset[0], -offset[1], -offset[2]}));
    }
}

TEST_CASE("fiberlet DP emits exact endpoints and integer monotone interior")
{
    const auto anchors = twoAnchorArtifact();
    const ConstantNormalSampler normals;
    const auto report = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, pathConfig(), constantPredictions(), normals);
    REQUIRE(report.diagnostics.generatedPairs == 1);
    REQUIRE_MESSAGE(report.diagnostics.successfulPaths == 1, report.candidates[0].reason);
    const auto& path = report.candidates[0];
    REQUIRE(path.pointsPredictionXYZ.size() >= 3);
    CHECK(path.pointsPredictionXYZ.front() == anchors.report.nonEmptyCells[0].components[0].anchor.positionPredictionXYZ);
    CHECK(path.pointsPredictionXYZ.back() == anchors.report.nonEmptyCells[1].components[0].anchor.positionPredictionXYZ);
    for (size_t index = 1; index + 1 < path.pointsPredictionXYZ.size(); ++index) {
        const auto& point = path.pointsPredictionXYZ[index];
        CHECK(point[0] == std::round(point[0]));
        CHECK(point[1] == std::round(point[1]));
        CHECK(point[2] == std::round(point[2]));
        CHECK(point[0] > path.pointsPredictionXYZ[index - 1][0]);
    }
    CHECK(path.cost.direction == doctest::Approx(0.0).epsilon(1.0e-10));
}

TEST_CASE("fiberlet pairing rejects an incompatible unoriented endpoint axis")
{
    const auto anchors = twoAnchorArtifact({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    const ConstantNormalSampler normals;
    bool sampled = false;
    const auto report =
        vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, pathConfig(), [&](const auto&, int, auto&) { sampled = true; }, normals);
    CHECK(report.diagnostics.axisRejectedPairs == 1);
    CHECK(report.diagnostics.searchedPairs == 0);
    CHECK_FALSE(sampled);
    CHECK(report.candidates[0].reason == "axis_mismatch");
}

TEST_CASE("fiberlet invalid prediction slabs remain finite bridge costs")
{
    const auto anchors = twoAnchorArtifact();
    const ConstantNormalSampler normals;
    const auto sampler = [](const auto& indices, int, auto& samples) {
        samples.clear();
        for (const auto& index : indices) {
            const bool invalid = index[2] == 6;
            samples.push_back({{1.0, 0.0, 0.0}, 1.0, !invalid});
        }
    };
    const auto report = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, pathConfig(), sampler, normals);
    REQUIRE_MESSAGE(report.diagnostics.successfulPaths == 1, report.candidates[0].reason);
    CHECK(report.candidates[0].cost.invalidPrediction > 0.0);
    CHECK(std::isfinite(report.candidates[0].cost.total()));
}

TEST_CASE("fiberlet narrow disconnected corridor reports no path")
{
    const auto anchors = twoAnchorArtifact({1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.2, 4.2, 4.2}, {10.2, 4.2, 4.2});
    auto config = pathConfig();
    config.maximumEndpointAngleDegrees = 60.0;
    config.corridorRadiusPredictionVoxels = 0.01;
    const ConstantNormalSampler normals;
    const auto report = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, config, constantPredictions(), normals);
    REQUIRE(report.diagnostics.searchedPairs == 1);
    CHECK(report.diagnostics.successfulPaths == 0);
    CHECK(report.diagnostics.noPathPairs == 1);
}

TEST_CASE("fiberlet path JSON and OBJ are deterministic and scaled")
{
    const auto anchors = twoAnchorArtifact();
    const ConstantNormalSampler normals;
    const auto first = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, pathConfig(), constantPredictions(), normals);
    const auto second = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, pathConfig(), constantPredictions(), normals);
    vc::fiber_tracer::FiberletArtifactInfo artifact;
    artifact.fiberManifestLocator = "/tmp/fiber.lasagna.json";
    artifact.fiberManifestContentHash = "fnv1a64:1111111111111111";
    artifact.normalManifestLocator = "/tmp/normal.lasagna.json";
    artifact.normalManifestContentHash = "fnv1a64:2222222222222222";
    artifact.anchorArtifactLocator = "/tmp/anchors.json";
    artifact.anchorArtifactContentHash = "fnv1a64:3333333333333333";
    const auto json = vc::fiber_tracer::fiberletPathReportJson(first, artifact);
    CHECK(json.dump() == vc::fiber_tracer::fiberletPathReportJson(second, artifact).dump());
    CHECK(json.at("coordinates").at("position_space") == "base_volume");
    CHECK(json.at("parameters").at("corridor_radius_base_voxels") == 4.0);
    CHECK_FALSE(json.at("parameters").contains("corridor_radius_prediction_voxels"));
    const auto& candidate = json.at("candidates").at(0);
    CHECK(candidate.at("score_valid") == true);
    CHECK(candidate.at("start_position_base_xyz") == nlohmann::json::array({4.5, 8.5, 8.5}));
    CHECK(candidate.contains("points_base_xyz"));
    CHECK_FALSE(candidate.contains("points_prediction_xyz"));
    const std::string obj = vc::fiber_tracer::fiberletPathReportObj(first);
    CHECK(obj.find("g fiberlet_2_2_1_0__2_2_5_0") != std::string::npos);
    CHECK(obj.find("v 4.5 8.5 8.5") != std::string::npos);
    CHECK(obj.find("\nl ") != std::string::npos);
    REQUIRE(first.candidates.size() == 1);
    CHECK(occurrenceCount(obj, "\nl ") == first.candidates[0].pointsPredictionXYZ.size() - 1);
}

TEST_CASE("fiberlet statistics separate unscored scored and accepted candidates")
{
    vc::fiber_tracer::FiberletPathReport report;
    report.diagnostics.occupiedAnchors = 7;
    report.candidates.resize(4);
    report.candidates[1].searched = true;
    report.candidates[2].searched = true;
    report.candidates[2].scoreValid = true;
    report.candidates[2].cost.presence = 3.0;
    report.candidates[3].searched = true;
    report.candidates[3].scoreValid = true;
    report.candidates[3].success = true;
    report.candidates[3].cost.presence = 1.0;

    const auto statistics = vc::fiber_tracer::fiberletPathStatistics(report);
    CHECK(statistics.anchors == 7);
    CHECK(statistics.candidates == 4);
    CHECK(statistics.preDpRejected == 1);
    CHECK(statistics.dpSearched == 3);
    CHECK(statistics.searchedUnscored == 1);
    CHECK(statistics.unscored == 2);
    CHECK(statistics.scored == 2);
    CHECK(statistics.accepted == 1);
    CHECK(statistics.allScores.minimum == 1.0);
    CHECK(statistics.allScores.mean == 2.0);
    CHECK(statistics.allScores.maximum == 3.0);
    CHECK(statistics.acceptedScores.minimum == 1.0);
    CHECK(statistics.acceptedScores.mean == 1.0);
    CHECK(statistics.acceptedScores.maximum == 1.0);

    const auto empty = vc::fiber_tracer::fiberletPathStatistics({});
    CHECK(empty.allScores.count == 0);
    CHECK_FALSE(empty.allScores.minimum.has_value());
    CHECK_FALSE(empty.allScores.mean.has_value());
    CHECK_FALSE(empty.allScores.maximum.has_value());
}

TEST_CASE("fiberlet anchor loader is strict and preserves component identity")
{
    const auto anchors = twoAnchorArtifact();
    const auto path = temporaryPath("strict");
    {
        std::ofstream output(path);
        output << vc::fiber_tracer::fiberAnchorReportJson(anchors.report, anchors.artifact).dump(2);
    }
    const auto loaded = vc::fiber_tracer::loadFiberAnchorArtifact(path);
    REQUIRE(loaded.report.nonEmptyCells.size() == 2);
    CHECK(loaded.report.nonEmptyCells[0].cellZYX == std::array<size_t, 3>{2, 2, 1});
    CHECK(loaded.report.nonEmptyCells[0].components[0].retained);
    CHECK_FALSE(loaded.report.nonEmptyCells[0].components[1].retained);

    auto oldCoordinates = vc::fiber_tracer::fiberAnchorReportJson(anchors.report, anchors.artifact);
    oldCoordinates["coordinates"]["position_space"] = "stored_prediction_grid";
    {
        std::ofstream output(path);
        output << oldCoordinates.dump(2);
    }
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::loadFiberAnchorArtifact(path), doctest::Contains("coordinate contract is unsupported"), std::runtime_error);

    auto json = vc::fiber_tracer::fiberAnchorReportJson(anchors.report, anchors.artifact);
    json["version"] = 2;
    {
        std::ofstream output(path);
        output << json.dump(2);
    }
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::loadFiberAnchorArtifact(path), doctest::Contains("version 1"), std::runtime_error);
    std::filesystem::remove(path);
}
