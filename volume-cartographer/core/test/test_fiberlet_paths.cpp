#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberLocalScoring.hpp"
#include "vc/fiber_tracer/FiberPaths.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
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

class CountingNormalSampler final : public vc::lasagna::NormalSampler
{
public:
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override { return {{0.0, 0.0, 1.0}, true, {}}; }

    vc::lasagna::NormalBatchReport sampleNormalBatch(
        const std::vector<cv::Vec3d>& points, bool withDerivative, int parallelThreads, std::vector<vc::lasagna::NormalSampleWithDerivative>& samples) const override
    {
        ++batchCalls;
        requestedThreads = parallelThreads;
        sampledPoints = points;
        samples.assign(
            points.size(),
            {
                {{0.0, 0.0, 1.0}, true, {}},
                cv::Matx33d::zeros(),
                withDerivative,
            });
        return {};
    }

    mutable std::atomic<size_t> batchCalls{0};
    mutable int requestedThreads = 0;
    mutable std::vector<cv::Vec3d> sampledPoints;
};

vc::fiber_tracer::LoadedFiberAnchorArtifact twoAnchorArtifact(
    cv::Vec3d firstAxis = {1.0, 0.0, 0.0},
    cv::Vec3d secondAxis = {1.0, 0.0, 0.0},
    cv::Vec3d firstPosition = {2.5, 4.5, 4.5},
    cv::Vec3d secondPosition = {10.5, 4.5, 4.5})
{
    vc::fiber_tracer::LoadedFiberAnchorArtifact artifact;
    artifact.artifact.sourceLocator = "/tmp/fiber.lasagna.json";
    artifact.artifact.manifestContentHash = "fnv1a64:0123456789abcdef";
    artifact.report.grid = {{16, 16, 16}, 2.0};
    artifact.report.config.cellSizePredictionVoxels = 2;
    artifact.report.config.gaussianSigmaPredictionVoxels = 1.0;
    artifact.report.config.localWindowRadiusPredictionVoxels = 2.0;
    artifact.report.config.axialSupportHalfWidthPredictionVoxels = 3.0;
    artifact.report.config.nmsLongitudinalRadiusPredictionVoxels = 1.0;
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
        cell.components[0].anchor.refinementScore = 1.0;
        cell.components[1].rejectionReason = "empty";
        artifact.report.nonEmptyCells.push_back(cell);
    }
    artifact.report.diagnostics.totalCells = 512;
    artifact.report.diagnostics.zeroAnchorCells = 510;
    artifact.report.diagnostics.oneAnchorCells = 2;
    artifact.report.selectedCrop = {{0, 0, 0}, {16, 16, 16}};
    artifact.report.selectedCellBeginZYX = {0, 0, 0};
    artifact.report.selectedCellEndZYX = {8, 8, 8};
    return artifact;
}

vc::fiber_tracer::LoadedFiberAnchorArtifact twoPathArtifact()
{
    auto artifact = twoAnchorArtifact();
    const size_t originalCount = artifact.report.nonEmptyCells.size();
    for (size_t index = 0; index < originalCount; ++index) {
        auto cell = artifact.report.nonEmptyCells[index];
        cell.cellZYX[1] += 4;
        cell.components[0].anchor.cellZYX = cell.cellZYX;
        cell.components[0].anchor.positionPredictionXYZ[1] += 8.0;
        artifact.report.nonEmptyCells.push_back(std::move(cell));
    }
    artifact.report.diagnostics.totalCells = 512;
    artifact.report.diagnostics.zeroAnchorCells = 508;
    artifact.report.diagnostics.oneAnchorCells = 4;
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

std::filesystem::path temporaryDirectory(const std::string& tag)
{
    auto path = temporaryPath(tag);
    path.replace_extension();
    std::filesystem::create_directories(path);
    return path;
}

std::string readText(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
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

TEST_CASE("fiber local alignment loss preserves native multiplicative scoring")
{
    using vc::fiber_tracer::fiberLocalAlignmentLoss;
    const cv::Vec3f x{1.0f, 0.0f, 0.0f};
    const cv::Vec3f y{0.0f, 1.0f, 0.0f};
    CHECK(fiberLocalAlignmentLoss(1.0f, x, x, x, x) == 0.0f);
    CHECK(fiberLocalAlignmentLoss(0.25f, x, x, x, x) == 0.75f);
    CHECK(fiberLocalAlignmentLoss(1.0f, x, x, x, y) == 1.0f);
    const float invSqrt2 = static_cast<float>(std::sqrt(0.5));
    const cv::Vec3f diagonal{invSqrt2, invSqrt2, 0.0f};
    float score = 0.5f;
    score *= x.dot(diagonal);
    score *= x.dot(x);
    score *= x.dot(diagonal);
    score *= x.dot(diagonal);
    score *= x.dot(diagonal);
    score *= diagonal.dot(diagonal);
    CHECK(fiberLocalAlignmentLoss(0.5f, x, diagonal, x, diagonal) ==
        1.0f - score);
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
    std::vector<vc::fiber_tracer::FiberletPathProgress> progress;
    const auto report =
        vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, pathConfig(), constantPredictions(), normals, [&](const auto& update) {
            progress.push_back(update);
        });
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
    CHECK(path.cost.alignment >= 0.0);
    REQUIRE(progress.size() == 1);
    CHECK(progress.back().completed == 1);
    CHECK(progress.back().total == 1);
    CHECK(progress.back().elapsedSeconds >= 0.0);
}

TEST_CASE("fiberlet DP preloads each scoring voxel once")
{
    const auto anchors = twoAnchorArtifact();
    auto options = pathConfig();
    options.parallelThreads = 4;
    size_t predictionCalls = 0;
    int predictionThreads = 0;
    std::vector<std::array<size_t, 3>> sampledIndices;
    const auto predictions = [&](const auto& indices, int threads, auto& samples) {
        ++predictionCalls;
        predictionThreads = threads;
        sampledIndices = indices;
        samples.assign(indices.size(), {{1.0, 0.0, 0.0}, 1.0, true});
    };
    const CountingNormalSampler normals;
    const auto report = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, options, predictions, normals);
    REQUIRE(report.diagnostics.successfulPaths == 1);
    CHECK(predictionCalls == 1);
    CHECK(predictionThreads == 4);
    CHECK(normals.batchCalls.load() == 1);
    CHECK(normals.requestedThreads == 4);
    REQUIRE(sampledIndices.size() == report.preloadedVoxels);
    REQUIRE(normals.sampledPoints.size() == sampledIndices.size());
    const std::set<std::array<size_t, 3>> unique(sampledIndices.begin(), sampledIndices.end());
    CHECK(unique.size() == sampledIndices.size());
    for (size_t index = 0; index < sampledIndices.size(); ++index) {
        const auto& zyx = sampledIndices[index];
        CHECK(zyx[0] < anchors.report.grid.shapeZYX[0]);
        CHECK(zyx[1] < anchors.report.grid.shapeZYX[1]);
        CHECK(zyx[2] < anchors.report.grid.shapeZYX[2]);
        CHECK(normals.sampledPoints[index] == cv::Vec3d{static_cast<double>(zyx[2]), static_cast<double>(zyx[1]), static_cast<double>(zyx[0])});
    }
}

TEST_CASE("fiberlet sparse replay preload preserves dense path bytes and costs")
{
    const auto anchors = twoAnchorArtifact();
    const ConstantNormalSampler normals;
    const auto dense = vc::fiber_tracer::traceFiberletPaths(
        anchors, anchors.report.grid, pathConfig(), constantPredictions(), normals);
    std::vector<std::array<size_t, 3>> sampled;
    const auto sparse = vc::fiber_tracer::traceFiberletPaths(
        anchors,
        anchors.report.grid,
        pathConfig(),
        [&](const auto& indices, int, auto& samples) {
            sampled = indices;
            samples.assign(indices.size(), {{1.0, 0.0, 0.0}, 1.0, true});
        },
        normals,
        {},
        [](const cv::Vec3d&) { return true; });

    REQUIRE(dense.diagnostics.successfulPaths == 1);
    REQUIRE(sparse.diagnostics.successfulPaths == 1);
    CHECK(sparse.preloadedVoxels <= dense.preloadedVoxels);
    CHECK(std::set<std::array<size_t, 3>>(sampled.begin(), sampled.end()).size() ==
          sampled.size());
    CHECK(sparse.candidates[0].pointsPredictionXYZ ==
          dense.candidates[0].pointsPredictionXYZ);
    CHECK(sparse.candidates[0].cost.total() == dense.candidates[0].cost.total());
    CHECK(vc::fiber_tracer::fiberletPathReportObj(sparse) ==
          vc::fiber_tracer::fiberletPathReportObj(dense));
}

TEST_CASE("fiberlet sparse replay domain rejects a disconnected corridor")
{
    const auto anchors = twoAnchorArtifact();
    const ConstantNormalSampler normals;
    const auto report = vc::fiber_tracer::traceFiberletPaths(
        anchors,
        anchors.report.grid,
        pathConfig(),
        constantPredictions(),
        normals,
        {},
        [](const cv::Vec3d& point) { return point[0] < 6.0 || point[0] > 6.0; });

    CHECK(report.diagnostics.successfulPaths == 0);
    REQUIRE(report.candidates.size() == 1);
    CHECK(report.candidates[0].reason == "no_path");
}

TEST_CASE("fiberlet candidate workers preserve deterministic results")
{
    const auto anchors = twoPathArtifact();
    const ConstantNormalSampler normals;
    auto serialConfig = pathConfig();
    serialConfig.parallelThreads = 1;
    auto parallelConfig = serialConfig;
    parallelConfig.parallelThreads = 8;
    std::vector<vc::fiber_tracer::FiberletPathProgress> progress;
    const auto serial = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, serialConfig, constantPredictions(), normals);
    const auto parallel =
        vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, parallelConfig, constantPredictions(), normals, [&](const auto& update) {
            progress.push_back(update);
        });
    REQUIRE(serial.diagnostics.searchedPairs == 2);
    REQUIRE(parallel.diagnostics.searchedPairs == 2);
    CHECK(serial.candidateWorkers == 1);
    CHECK(parallel.candidateWorkers == 2);
    REQUIRE_FALSE(progress.empty());
    for (size_t index = 1; index < progress.size(); ++index)
        CHECK(progress[index - 1].completed < progress[index].completed);
    CHECK(progress.back().completed == 2);
    CHECK(progress.back().total == 2);
    vc::fiber_tracer::FiberletArtifactInfo artifact;
    artifact.fiberManifestLocator = "/tmp/fiber.lasagna.json";
    artifact.fiberManifestContentHash = "fnv1a64:1111111111111111";
    artifact.normalManifestLocator = "/tmp/normal.lasagna.json";
    artifact.normalManifestContentHash = "fnv1a64:2222222222222222";
    artifact.anchorArtifactLocator = "/tmp/anchors.json";
    artifact.anchorArtifactContentHash = "fnv1a64:3333333333333333";
    CHECK(vc::fiber_tracer::fiberletPathReportJson(serial, artifact).dump() == vc::fiber_tracer::fiberletPathReportJson(parallel, artifact).dump());
    CHECK(vc::fiber_tracer::fiberletPathReportObj(serial) ==
        vc::fiber_tracer::fiberletPathReportObj(parallel));
}

TEST_CASE("fiberlet pairing rejects an incompatible unoriented endpoint axis")
{
    const auto anchors = twoAnchorArtifact({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    const CountingNormalSampler normals;
    bool sampled = false;
    std::vector<vc::fiber_tracer::FiberletPathProgress> progress;
    const auto report = vc::fiber_tracer::traceFiberletPaths(
        anchors,
        anchors.report.grid,
        pathConfig(),
        [&](const auto&, int, auto&) { sampled = true; },
        normals,
        [&](const auto& update) { progress.push_back(update); });
    CHECK(report.diagnostics.axisRejectedPairs == 1);
    CHECK(report.diagnostics.searchedPairs == 0);
    CHECK_FALSE(sampled);
    CHECK(normals.batchCalls.load() == 0);
    CHECK(report.preloadedVoxels == 0);
    CHECK(report.candidateWorkers == 0);
    CHECK(report.candidates[0].reason == "axis_mismatch");
    REQUIRE(progress.size() == 1);
    CHECK(progress.back().completed == 0);
    CHECK(progress.back().total == 0);
}

TEST_CASE("fiberlet progress callback failures are rethrown after search")
{
    const auto anchors = twoAnchorArtifact();
    const ConstantNormalSampler normals;
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::traceFiberletPaths(
            anchors,
            anchors.report.grid,
            pathConfig(),
            constantPredictions(),
            normals,
            [](const auto&) { throw std::runtime_error("progress failed"); }),
        doctest::Contains("progress failed"),
        std::runtime_error);
}

TEST_CASE("fiberlet invalid prediction slabs remain finite bridge costs")
{
    const auto anchors = twoAnchorArtifact(
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {2.0, 4.0, 4.0},
        {10.0, 4.0, 4.0});
    auto config = pathConfig();
    config.corridorRadiusPredictionVoxels = 0.01;
    const ConstantNormalSampler normals;
    const auto sampler = [](const auto& indices, int, auto& samples) {
        samples.clear();
        for (const auto& index : indices) {
            const bool invalid = index[2] == 6;
            samples.push_back({{1.0, 0.0, 0.0}, 1.0, !invalid});
        }
    };
    const auto report = vc::fiber_tracer::traceFiberletPaths(
        anchors, anchors.report.grid, config, sampler, normals);
    REQUIRE_MESSAGE(report.diagnostics.successfulPaths == 1, report.candidates[0].reason);
    CHECK(report.candidates[0].cost.invalidPrediction == 4.0);
    CHECK(report.candidates[0].cost.alignment == 0.0);
    CHECK(report.candidates[0].cost.isotropicSmoothness == 0.0);
    CHECK(report.candidates[0].cost.tangentSmoothness == 0.0);
    CHECK(report.candidates[0].cost.normalSmoothness == 0.0);
    CHECK(report.candidates[0].cost.total() == 4.0);
}

TEST_CASE("fiberlet DP uses multiplicative presence and unoriented predictions")
{
    const auto anchors = twoAnchorArtifact(
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {2.0, 4.0, 4.0},
        {10.0, 4.0, 4.0});
    auto config = pathConfig();
    config.corridorRadiusPredictionVoxels = 0.01;
    const ConstantNormalSampler normals;
    const auto report = vc::fiber_tracer::traceFiberletPaths(
        anchors,
        anchors.report.grid,
        config,
        constantPredictions({-1.0, 0.0, 0.0}, 0.5),
        normals);

    REQUIRE_MESSAGE(report.diagnostics.successfulPaths == 1, report.candidates[0].reason);
    CHECK(report.candidates[0].cost.invalidPrediction == 0.0);
    CHECK(report.candidates[0].cost.alignment == 3.5);
    CHECK(report.candidates[0].cost.total() == 3.5);
}

TEST_CASE("fiberlet multiplicative alignment changes the selected route")
{
    const auto anchors = twoAnchorArtifact(
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {2.0, 4.0, 4.0},
        {10.0, 4.0, 4.0});
    const float invSqrt2 = static_cast<float>(std::sqrt(0.5));
    const auto sampler = [=](const auto& indices, int, auto& samples) {
        samples.clear();
        for (const auto& zyx : indices) {
            const size_t y = zyx[1];
            const size_t x = zyx[2];
            cv::Vec3d direction{0.0, 1.0, 0.0};
            if (y == 5) {
                if (x == 3)
                    direction = {invSqrt2, invSqrt2, 0.0};
                else
                    direction = {1.0, 0.0, 0.0};
            }
            samples.push_back({direction, 1.0, true});
        }
    };
    const ConstantNormalSampler normals;
    auto narrowConfig = pathConfig();
    narrowConfig.corridorRadiusPredictionVoxels = 0.01;
    const auto narrow = vc::fiber_tracer::traceFiberletPaths(
        anchors, anchors.report.grid, narrowConfig, sampler, normals);
    const auto wide = vc::fiber_tracer::traceFiberletPaths(
        anchors, anchors.report.grid, pathConfig(), sampler, normals);

    REQUIRE(narrow.diagnostics.successfulPaths == 1);
    REQUIRE(wide.diagnostics.successfulPaths == 1);
    CHECK(wide.candidates[0].cost.total() < narrow.candidates[0].cost.total());
    CHECK(std::any_of(
        wide.candidates[0].pointsPredictionXYZ.begin(),
        wide.candidates[0].pointsPredictionXYZ.end(),
        [](const cv::Vec3d& point) { return point[1] == 5.0; }));
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
    CHECK(candidate.at("start_position_base_xyz") == nlohmann::json::array({5.0, 9.0, 9.0}));
    CHECK(candidate.contains("points_base_xyz"));
    CHECK_FALSE(candidate.contains("points_prediction_xyz"));
    CHECK(candidate.at("path_length_base_voxels").get<double>() > 0.0);
    CHECK(candidate.at("loss_per_prediction_voxel").get<double>() >= 0.0);
    CHECK(candidate.at("relative_visual_quality") == 1.0);
    CHECK_FALSE(candidate.contains("visual_color_rgb"));
    CHECK_FALSE(candidate.contains("visual_material"));
    CHECK(candidate.at("cost").contains("alignment"));
    CHECK_FALSE(candidate.at("cost").contains("presence"));
    CHECK_FALSE(candidate.at("cost").contains("direction"));
    CHECK(json.at("trace_quality_visualization").at("count") == 1);
    CHECK_FALSE(json.at("trace_quality_visualization").contains("color_ramp"));
    const std::string obj = vc::fiber_tracer::fiberletPathReportObj(first);
    CHECK(obj == vc::fiber_tracer::fiberletPathReportObj(second));
    CHECK(obj.find("mtllib") == std::string::npos);
    CHECK(obj.find("# trace_quality_count 1") != std::string::npos);
    CHECK(obj.find("# trace_loss_per_prediction_voxel ") != std::string::npos);
    CHECK(obj.find("usemtl") == std::string::npos);
    CHECK(obj.find("g fiberlet_2_2_1_0__2_2_5_0") != std::string::npos);
    CHECK(obj.find("v 5 9 9") != std::string::npos);
    CHECK(obj.find("\nl ") != std::string::npos);
    REQUIRE(first.candidates.size() == 1);
    CHECK(occurrenceCount(obj, "\nl ") == first.candidates[0].pointsPredictionXYZ.size() - 1);

    const auto directory = temporaryDirectory("quality_artifacts");
    {
        std::ofstream stale(directory / "fiberlets.mtl");
        stale << "obsolete material artifact\n";
    }
    vc::fiber_tracer::writeFiberletPathArtifacts(directory, first, artifact);
    CHECK(readText(directory / "fiberlets.json") == json.dump(2) + "\n");
    CHECK(readText(directory / "fiberlets.obj") == obj);
    CHECK_FALSE(std::filesystem::exists(directory / "fiberlets.mtl"));
    std::filesystem::remove_all(directory);
}

TEST_CASE("fiberlet relative visual quality normalizes loss density")
{
    vc::fiber_tracer::FiberletPathReport report;
    report.grid = {{16, 16, 16}, 2.0};
    for (size_t index = 0; index < 3; ++index) {
        vc::fiber_tracer::FiberletCandidateResult candidate;
        candidate.start.cellZYX = {0, 0, index};
        candidate.target.cellZYX = {0, 1, index};
        candidate.searched = true;
        candidate.scoreValid = true;
        candidate.success = true;
        candidate.reason = "success";
        candidate.pointsPredictionXYZ = {
            {0.0, static_cast<double>(index), 0.0},
            {2.0, static_cast<double>(index), 0.0},
        };
        candidate.cost.alignment = 2.0 * static_cast<double>(index + 1);
        report.candidates.push_back(candidate);
    }
    const auto visual = vc::fiber_tracer::fiberletPathVisualMetrics(report);
    REQUIRE(visual.paths.size() == 3);
    CHECK(visual.minimumLossPerPredictionVoxel == 1.0);
    CHECK(visual.maximumLossPerPredictionVoxel == 3.0);
    CHECK(visual.paths[0].relativeQuality == 1.0);
    CHECK(visual.paths[1].relativeQuality == 0.5);
    CHECK(visual.paths[2].relativeQuality == 0.0);
}

TEST_CASE("fiberlet equal-density and empty visual reports are deterministic")
{
    vc::fiber_tracer::FiberletPathReport equal;
    equal.grid = {{8, 8, 8}, 1.0};
    for (size_t index = 0; index < 2; ++index) {
        vc::fiber_tracer::FiberletCandidateResult candidate;
        candidate.start.cellZYX = {0, 0, index};
        candidate.target.cellZYX = {0, 1, index};
        candidate.scoreValid = true;
        candidate.success = true;
        candidate.pointsPredictionXYZ = {{0.0, 0.0, 0.0}, {1.0 + index, 0.0, 0.0}};
        candidate.cost.alignment = 1.0 + index;
        equal.candidates.push_back(candidate);
    }
    const auto equalVisual = vc::fiber_tracer::fiberletPathVisualMetrics(equal);
    REQUIRE(equalVisual.paths.size() == 2);
    CHECK(equalVisual.paths[0].relativeQuality == 1.0);
    CHECK(equalVisual.paths[1].relativeQuality == 1.0);

    vc::fiber_tracer::FiberletPathReport empty;
    empty.grid = {{1, 1, 1}, 1.0};
    const auto emptyVisual = vc::fiber_tracer::fiberletPathVisualMetrics(empty);
    CHECK(emptyVisual.paths.empty());
    CHECK_FALSE(emptyVisual.minimumLossPerPredictionVoxel.has_value());
    CHECK(vc::fiber_tracer::fiberletPathReportObj(empty).find(
              "# trace_loss_density_min none") != std::string::npos);
}

TEST_CASE("fiberlet visual metrics reject invalid geometry loss and identifiers")
{
    vc::fiber_tracer::FiberletPathReport report;
    report.grid = {{8, 8, 8}, 1.0};
    vc::fiber_tracer::FiberletCandidateResult candidate;
    candidate.start.cellZYX = {0, 0, 0};
    candidate.target.cellZYX = {0, 0, 1};
    candidate.scoreValid = true;
    candidate.success = true;
    candidate.pointsPredictionXYZ = {{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}};
    candidate.cost.alignment = 1.0;
    report.candidates.push_back(candidate);
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::fiberletPathVisualMetrics(report),
        doctest::Contains("non-positive path length"), std::runtime_error);

    report.candidates[0].pointsPredictionXYZ[1] = {2.0, 1.0, 1.0};
    report.candidates[0].cost.alignment = -1.0;
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::fiberletPathVisualMetrics(report),
        doctest::Contains("component loss"), std::runtime_error);

    report.candidates[0].cost.alignment = 1.0;
    report.candidates.push_back(report.candidates[0]);
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::fiberletPathVisualMetrics(report),
        doctest::Contains("duplicate path identifier"), std::runtime_error);
}

TEST_CASE("fiberlet statistics separate unscored scored and accepted candidates")
{
    vc::fiber_tracer::FiberletPathReport report;
    report.diagnostics.occupiedAnchors = 7;
    report.candidates.resize(4);
    report.candidates[1].searched = true;
    report.candidates[2].searched = true;
    report.candidates[2].scoreValid = true;
    report.candidates[2].cost.alignment = 3.0;
    report.candidates[3].searched = true;
    report.candidates[3].scoreValid = true;
    report.candidates[3].success = true;
    report.candidates[3].cost.alignment = 1.0;
    report.candidates[3].pointsPredictionXYZ = {
        {0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
    report.grid = {{4, 4, 4}, 1.0};

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
    CHECK(statistics.acceptedLossDensities.minimum == 0.5);
    CHECK(statistics.acceptedLossDensities.mean == 0.5);
    CHECK(statistics.acceptedLossDensities.maximum == 0.5);

    vc::fiber_tracer::FiberletPathReport emptyReport;
    emptyReport.grid = {{1, 1, 1}, 1.0};
    const auto empty = vc::fiber_tracer::fiberletPathStatistics(emptyReport);
    CHECK(empty.allScores.count == 0);
    CHECK_FALSE(empty.allScores.minimum.has_value());
    CHECK_FALSE(empty.allScores.mean.has_value());
    CHECK_FALSE(empty.allScores.maximum.has_value());
}

TEST_CASE("fiber presence slices sample deterministic central planes in base coordinates")
{
    const vc::fiber_tracer::FiberPredictionGridInfo grid{{6, 5, 4}, 2.0};
    const vc::fiber_tracer::FiberAnchorCrop crop{{1, 1, 1}, {3, 4, 5}};
    const auto report = vc::fiber_tracer::sampleFiberPresenceSlices(
        crop,
        grid,
        [](const auto& indices, int, auto& samples) {
            samples.clear();
            for (const auto& index : indices) {
                if (index == std::array<size_t, 3>{3, 2, 2}) {
                    samples.push_back({std::numeric_limits<double>::quiet_NaN(), false});
                } else {
                    const size_t encoded = index[0] * 25 + index[1] * 5 + index[2];
                    samples.push_back({static_cast<double>(encoded) / 255.0, true});
                }
            }
        },
        3);
    REQUIRE(report.planes.size() == 3);
    CHECK(report.planes[0].name == "xy");
    CHECK(report.planes[1].name == "xz");
    CHECK(report.planes[2].name == "yz");
    CHECK(report.planes[0].varyingAxesXYZ == std::array<size_t, 2>{0, 1});
    CHECK(report.planes[1].varyingAxesXYZ == std::array<size_t, 2>{0, 2});
    CHECK(report.planes[2].varyingAxesXYZ == std::array<size_t, 2>{1, 2});
    CHECK(report.planes[0].width == 3);
    CHECK(report.planes[0].height == 4);
    CHECK(report.planes[1].width == 3);
    CHECK(report.planes[1].height == 5);
    CHECK(report.planes[2].width == 4);
    CHECK(report.planes[2].height == 5);
    CHECK(report.pixelCount() == 47);
    CHECK(report.planes[0].pixels.front().indexZYX == std::array<size_t, 3>{3, 1, 1});
    CHECK(report.planes[0].pixels[4].presence == 0.0);

    const auto directory = temporaryDirectory("presence_slices");
    {
        std::ofstream stale(directory / "fiber_presence_slices.obj");
        stale << "obsolete point cloud\n";
    }
    vc::fiber_tracer::writeFiberPresenceSliceArtifacts(directory, report, grid);
    CHECK_FALSE(std::filesystem::exists(directory / "fiber_presence_slices.obj"));
    for (const std::string plane : {"xy", "xz", "yz"}) {
        const std::string stem = "fiber_presence_" + plane;
        CHECK(std::filesystem::exists(directory / (stem + ".obj")));
        CHECK(std::filesystem::exists(directory / (stem + ".mtl")));
        CHECK(std::filesystem::exists(directory / (stem + ".png")));
        const std::string obj = readText(directory / (stem + ".obj"));
        CHECK(obj.find("mtllib " + stem + ".mtl") != std::string::npos);
        CHECK(obj.find("o " + stem) != std::string::npos);
        CHECK(occurrenceCount(obj, "\nv ") == 4);
        CHECK(occurrenceCount(obj, "\nvt ") == 4);
        CHECK(obj.find("f 1/1 2/2 3/3 4/4") != std::string::npos);
        const std::string mtl = readText(directory / (stem + ".mtl"));
        CHECK(mtl.find("map_Kd " + stem + ".png") != std::string::npos);
    }
    const std::string xyObj = readText(directory / "fiber_presence_xy.obj");
    CHECK(xyObj.find("v 1 1 6") != std::string::npos);
    CHECK(xyObj.find("v 7 1 6") != std::string::npos);
    CHECK(xyObj.find("v 7 9 6") != std::string::npos);
    CHECK(xyObj.find("v 1 9 6") != std::string::npos);
    CHECK(xyObj.find("vt 0 1\nvt 1 1\nvt 1 0\nvt 0 0") != std::string::npos);

    const cv::Mat xy = cv::imread((directory / "fiber_presence_xy.png").string(), cv::IMREAD_GRAYSCALE);
    const cv::Mat xz = cv::imread((directory / "fiber_presence_xz.png").string(), cv::IMREAD_GRAYSCALE);
    const cv::Mat yz = cv::imread((directory / "fiber_presence_yz.png").string(), cv::IMREAD_GRAYSCALE);
    REQUIRE(xy.rows == 4);
    REQUIRE(xy.cols == 3);
    CHECK(xy.at<uint8_t>(0, 0) == 81);
    CHECK(xy.at<uint8_t>(0, 2) == 83);
    CHECK(xy.at<uint8_t>(3, 0) == 96);
    CHECK(xy.at<uint8_t>(3, 2) == 98);
    REQUIRE(xz.rows == 5);
    REQUIRE(xz.cols == 3);
    CHECK(xz.at<uint8_t>(0, 0) == 36);
    CHECK(xz.at<uint8_t>(0, 2) == 38);
    CHECK(xz.at<uint8_t>(4, 0) == 136);
    CHECK(xz.at<uint8_t>(4, 2) == 138);
    REQUIRE(yz.rows == 5);
    REQUIRE(yz.cols == 4);
    CHECK(yz.at<uint8_t>(0, 0) == 32);
    CHECK(yz.at<uint8_t>(0, 3) == 47);
    CHECK(yz.at<uint8_t>(4, 0) == 132);
    CHECK(yz.at<uint8_t>(4, 3) == 147);

    {
        std::ofstream stale(directory / "fiber_presence_slices.obj");
        stale << "obsolete point cloud\n";
    }
    vc::fiber_tracer::removeFiberPresenceSliceArtifacts(directory);
    CHECK_FALSE(std::filesystem::exists(directory / "fiber_presence_slices.obj"));
    for (const std::string plane : {"xy", "xz", "yz"}) {
        const std::string stem = "fiber_presence_" + plane;
        CHECK_FALSE(std::filesystem::exists(directory / (stem + ".obj")));
        CHECK_FALSE(std::filesystem::exists(directory / (stem + ".mtl")));
        CHECK_FALSE(std::filesystem::exists(directory / (stem + ".png")));
    }
    std::filesystem::remove_all(directory);
}

TEST_CASE("fiber presence slices reject unsafe output size before sampling")
{
    bool sampled = false;
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::sampleFiberPresenceSlices(
            {{0, 0, 0}, {1000, 1000, 1}}, {{1, 1000, 1000}, 1.0}, [&](const auto&, int, auto&) { sampled = true; }, 1),
        doctest::Contains("--no-slices"),
        std::invalid_argument);
    CHECK_FALSE(sampled);
}

TEST_CASE("fiber presence slice crop covers complete selected anchor cells")
{
    auto anchors = twoAnchorArtifact();
    anchors.report.selectedCellBeginZYX = {1, 2, 3};
    anchors.report.selectedCellEndZYX = {4, 6, 7};
    const auto crop = vc::fiber_tracer::fiberAnchorCellCoverageCrop(anchors);
    CHECK(crop.originXYZ == std::array<size_t, 3>{6, 4, 2});
    CHECK(crop.sizeXYZ == std::array<size_t, 3>{8, 8, 6});
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

    auto missingMergeParameter = vc::fiber_tracer::fiberAnchorReportJson(anchors.report, anchors.artifact);
    missingMergeParameter["parameters"].erase("merge_maximum_angle_degrees");
    {
        std::ofstream output(path);
        output << missingMergeParameter.dump(2);
    }
    CHECK_THROWS(vc::fiber_tracer::loadFiberAnchorArtifact(path));

    auto missingRefinementParameter = vc::fiber_tracer::fiberAnchorReportJson(anchors.report, anchors.artifact);
    missingRefinementParameter["parameters"].erase("peak_sigma_prediction_voxels");
    {
        std::ofstream output(path);
        output << missingRefinementParameter.dump(2);
    }
    CHECK_THROWS(vc::fiber_tracer::loadFiberAnchorArtifact(path));

    auto missingAxialPeakParameter = vc::fiber_tracer::fiberAnchorReportJson(anchors.report, anchors.artifact);
    missingAxialPeakParameter["parameters"].erase("peak_axial_sigma_prediction_voxels");
    {
        std::ofstream output(path);
        output << missingAxialPeakParameter.dump(2);
    }
    CHECK_THROWS(vc::fiber_tracer::loadFiberAnchorArtifact(path));

    auto extraParameter = vc::fiber_tracer::fiberAnchorReportJson(anchors.report, anchors.artifact);
    extraParameter["parameters"]["unknown"] = 1.0;
    {
        std::ofstream output(path);
        output << extraParameter.dump(2);
    }
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::loadFiberAnchorArtifact(path),
        doctest::Contains("version-1 schema"), std::runtime_error);

    auto missingRefinementValue = vc::fiber_tracer::fiberAnchorReportJson(anchors.report, anchors.artifact);
    missingRefinementValue["cells"][0]["components"][0].erase("refinement_score");
    {
        std::ofstream output(path);
        output << missingRefinementValue.dump(2);
    }
    CHECK_THROWS(vc::fiber_tracer::loadFiberAnchorArtifact(path));

    auto offPlane = vc::fiber_tracer::fiberAnchorReportJson(anchors.report, anchors.artifact);
    offPlane["cells"][0]["components"][0]["position_base_xyz"][0] = 5.5;
    {
        std::ofstream output(path);
        output << offPlane.dump(2);
    }
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::loadFiberAnchorArtifact(path), doctest::Contains("rotating-plane window"), std::runtime_error);

    auto outsideOwner = vc::fiber_tracer::fiberAnchorReportJson(anchors.report, anchors.artifact);
    outsideOwner["cells"][0]["components"][0]["position_base_xyz"][1] = 12.0;
    {
        std::ofstream output(path);
        output << outsideOwner.dump(2);
    }
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::loadFiberAnchorArtifact(path),
        doctest::Contains("owning cell"), std::runtime_error);

    auto missingNmsDiagnostic = vc::fiber_tracer::fiberAnchorReportJson(anchors.report, anchors.artifact);
    missingNmsDiagnostic["diagnostics"].erase("nms_suppressed_components");
    {
        std::ofstream output(path);
        output << missingNmsDiagnostic.dump(2);
    }
    CHECK_THROWS(vc::fiber_tracer::loadFiberAnchorArtifact(path));

    auto missingMergeDiagnostic = vc::fiber_tracer::fiberAnchorReportJson(anchors.report, anchors.artifact);
    missingMergeDiagnostic["diagnostics"].erase("merged_component_pairs");
    {
        std::ofstream output(path);
        output << missingMergeDiagnostic.dump(2);
    }
    CHECK_THROWS(vc::fiber_tracer::loadFiberAnchorArtifact(path));

    auto json = vc::fiber_tracer::fiberAnchorReportJson(anchors.report, anchors.artifact);
    json["version"] = 2;
    {
        std::ofstream output(path);
        output << json.dump(2);
    }
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::loadFiberAnchorArtifact(path), doctest::Contains("version 1"), std::runtime_error);
    std::filesystem::remove(path);
}
