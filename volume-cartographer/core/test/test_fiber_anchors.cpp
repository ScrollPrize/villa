#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "utils/zarr.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace
{

using vc::fiber_tracer::FiberAnchorConfig;
using vc::fiber_tracer::FiberAnchorObservation;

std::vector<FiberAnchorObservation> cellObservations(int size, const cv::Vec3d& first, const cv::Vec3d& second = {0.0, 0.0, 0.0}, double secondPresence = 1.0)
{
    std::vector<FiberAnchorObservation> observations;
    for (int z = 0; z < size; ++z) {
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                const bool useSecond = second.dot(second) > 0.0 && x >= size / 2;
                observations.push_back({
                    cv::Vec3d{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)},
                    useSecond ? second : first,
                    useSecond ? secondPresence : 1.0,
                    true,
                });
            }
        }
    }
    return observations;
}

double axialDot(const cv::Vec3d& left, const cv::Vec3d& right)
{
    return std::abs(left.dot(right) / std::sqrt(left.dot(left) * right.dot(right)));
}

std::vector<cv::Vec3d> retainedAxes(const vc::fiber_tracer::FiberCellAnchorResult& result)
{
    std::vector<cv::Vec3d> axes;
    for (const auto& component : result.components) {
        if (component.retained)
            axes.push_back(component.anchor.axisXYZ);
    }
    return axes;
}

cv::Vec3d directionAtDegrees(double degrees)
{
    const double radians = degrees * std::acos(-1.0) / 180.0;
    return {std::cos(radians), std::sin(radians), 0.0};
}

size_t occurrenceCount(const std::string& text, const std::string& needle)
{
    size_t count = 0;
    for (size_t position = 0; (position = text.find(needle, position)) != std::string::npos; position += needle.size()) {
        ++count;
    }
    return count;
}

FiberAnchorConfig config()
{
    FiberAnchorConfig value;
    value.cellSizePredictionVoxels = 4;
    value.gaussianSigmaPredictionVoxels = 2.0;
    value.observationPresenceFloor = 0.01;
    value.minimumAlignedSupport = 0.01;
    value.parallelThreads = 1;
    return value;
}

std::filesystem::path temporaryDirectory(const std::string& tag)
{
    std::mt19937_64 generator(std::random_device{}());
    const auto path = std::filesystem::temp_directory_path() / ("vc_fiber_anchors_" + tag + "_" + std::to_string(generator()));
    std::filesystem::create_directories(path);
    return path;
}

void createConstantZarr(const std::filesystem::path& path, const std::array<size_t, 3>& shape, const std::array<size_t, 3>& chunks, uint8_t value)
{
    utils::ZarrMetadata metadata;
    metadata.version = utils::ZarrVersion::v2;
    metadata.shape = {shape[0], shape[1], shape[2]};
    metadata.chunks = {chunks[0], chunks[1], chunks[2]};
    metadata.dtype = utils::ZarrDtype::uint8;
    metadata.compressor_id.clear();
    metadata.fill_value = 0.0;
    auto array = utils::ZarrArray::create(path, metadata);
    std::vector<std::byte> payload(chunks[0] * chunks[1] * chunks[2], static_cast<std::byte>(value));
    for (size_t z = 0; z < (shape[0] + chunks[0] - 1) / chunks[0]; ++z) {
        for (size_t y = 0; y < (shape[1] + chunks[1] - 1) / chunks[1]; ++y) {
            for (size_t x = 0; x < (shape[2] + chunks[2] - 1) / chunks[2]; ++x) {
                const std::array<size_t, 3> chunk{z, y, x};
                array.write_chunk(chunk, payload);
            }
        }
    }
}

void createEmptyFourDimensionalZarr(const std::filesystem::path& path)
{
    utils::ZarrMetadata metadata;
    metadata.version = utils::ZarrVersion::v2;
    metadata.shape = {3, 4, 4, 4};
    metadata.chunks = {3, 4, 4, 4};
    metadata.dtype = utils::ZarrDtype::uint8;
    metadata.compressor_id.clear();
    metadata.fill_value = 0.0;
    (void)utils::ZarrArray::create(path, metadata);
}

void writeText(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream output(path);
    output << text;
}

}  // namespace

TEST_CASE("fiber anchor extraction rejects an empty cell")
{
    auto observations = cellObservations(4, {1.0, 0.0, 0.0});
    for (auto& observation : observations)
        observation.valid = false;
    const auto result = vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, config());
    CHECK(result.retainedAnchorCount == 0);
    CHECK(result.components[0].rejectionReason == "empty");
    CHECK(result.components[1].rejectionReason == "empty");
}

TEST_CASE("fiber anchor extraction emits one unoriented straight component")
{
    const cv::Vec3d expected{1.0, 2.0, 3.0};
    auto observations = cellObservations(4, expected);
    for (size_t index = 0; index < observations.size(); ++index) {
        if (index % 2 != 0)
            observations[index].direction *= -1.0;
    }
    const auto result = vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, config());
    REQUIRE(result.retainedAnchorCount == 1);
    const auto axes = retainedAxes(result);
    REQUIRE(axes.size() == 1);
    CHECK(axialDot(axes[0], expected) > 1.0 - 1.0e-12);
    CHECK(result.components[0].anchor.alignedSupport == doctest::Approx(1.0));
    CHECK(result.components[0].anchor.directionalCoherence == doctest::Approx(1.0));
}

TEST_CASE("fiber anchor extraction fits two non-orthogonal direction modes")
{
    for (const double degrees : {15.0, 30.0, 45.0, 60.0, 90.0}) {
        const double radians = degrees * std::acos(-1.0) / 180.0;
        const cv::Vec3d first{1.0, 0.0, 0.0};
        const cv::Vec3d second{std::cos(radians), std::sin(radians), 0.0};
        const auto result = vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, cellObservations(4, first, second), config());
        REQUIRE_MESSAGE(result.retainedAnchorCount == 2, std::string("angle=") + std::to_string(degrees));
        const auto axes = retainedAxes(result);
        const double firstMatch = std::max(axialDot(axes[0], first), axialDot(axes[1], first));
        const double secondMatch = std::max(axialDot(axes[0], second), axialDot(axes[1], second));
        CHECK_MESSAGE(firstMatch > 1.0 - 1.0e-10, std::string("angle=") + std::to_string(degrees));
        CHECK_MESSAGE(secondMatch > 1.0 - 1.0e-10, std::string("angle=") + std::to_string(degrees));
    }
}

TEST_CASE("fiber anchor extraction merges duplicate nearby directions")
{
    const auto result =
        vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, cellObservations(4, {1.0, 0.0, 0.0}, directionAtDegrees(5.0)), config());
    REQUIRE(result.mergeEvaluation.has_value());
    CHECK(result.mergeEvaluation->merged);
    CHECK(result.retainedAnchorCount == 1);
    CHECK(result.components[0].retained);
    CHECK(result.components[0].assignedObservationCount == 64);
    CHECK(result.components[1].rejectionReason == "merged_same_direction");
    CHECK(result.objective == doctest::Approx(result.mergeEvaluation->jointObjective));
    CHECK(axialDot(result.components[0].anchor.axisXYZ, directionAtDegrees(2.5)) > 1.0 - 1.0e-10);
}

TEST_CASE("fiber anchor merge angle and loss thresholds are inclusive")
{
    const auto observations = cellObservations(4, {1.0, 0.0, 0.0}, directionAtDegrees(5.0));
    auto disabled = config();
    disabled.mergeMaximumAngleDegrees = 90.0;
    disabled.mergeMaximumAbsoluteObjectiveLoss = 0.0;
    disabled.mergeMaximumRelativeObjectiveLoss = 0.0;
    const auto measured = vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, disabled);
    REQUIRE(measured.mergeEvaluation.has_value());
    REQUIRE(measured.mergeEvaluation->objectiveLoss > 0.0);
    CHECK_FALSE(measured.mergeEvaluation->merged);

    auto absoluteBoundary = disabled;
    absoluteBoundary.mergeMaximumAbsoluteObjectiveLoss = measured.mergeEvaluation->objectiveLoss;
    CHECK(vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, absoluteBoundary).mergeEvaluation->merged);
    absoluteBoundary.mergeMaximumAbsoluteObjectiveLoss = std::nextafter(measured.mergeEvaluation->objectiveLoss, 0.0);
    CHECK_FALSE(vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, absoluteBoundary).mergeEvaluation->merged);

    auto relativeBoundary = disabled;
    relativeBoundary.mergeMaximumRelativeObjectiveLoss = measured.mergeEvaluation->objectiveLoss / measured.mergeEvaluation->jointObjective;
    CHECK(vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, relativeBoundary).mergeEvaluation->merged);
    relativeBoundary.mergeMaximumRelativeObjectiveLoss = std::nextafter(relativeBoundary.mergeMaximumRelativeObjectiveLoss, 0.0);
    CHECK_FALSE(vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, relativeBoundary).mergeEvaluation->merged);

    auto angleBoundary = config();
    angleBoundary.mergeMaximumAbsoluteObjectiveLoss = 1.0;
    angleBoundary.mergeMaximumRelativeObjectiveLoss = 1.0;
    angleBoundary.mergeMaximumAngleDegrees = measured.mergeEvaluation->angleDegrees;
    CHECK(vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, angleBoundary).mergeEvaluation->merged);
    angleBoundary.mergeMaximumAngleDegrees = std::nextafter(measured.mergeEvaluation->angleDegrees, 0.0);
    CHECK_FALSE(vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, angleBoundary).mergeEvaluation->merged);
}

TEST_CASE("fiber anchor merge applies support after joint refit")
{
    auto options = config();
    options.minimumAlignedSupport = 0.75;
    auto observations = cellObservations(4, {1.0, 0.0, 0.0}, directionAtDegrees(5.0));
    const auto merged = vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, options);
    REQUIRE(merged.mergeEvaluation.has_value());
    CHECK(merged.mergeEvaluation->merged);
    CHECK(merged.retainedAnchorCount == 1);

    options.mergeMaximumAngleDegrees = 0.0;
    const auto split = vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, options);
    CHECK(split.retainedAnchorCount == 0);
    CHECK(split.components[0].rejectionReason == "below_support");
    CHECK(split.components[1].rejectionReason == "below_support");

    for (auto& observation : observations)
        observation.presence = 0.05;
    options = config();
    options.minimumAlignedSupport = 0.1;
    const auto belowSupport = vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, options);
    REQUIRE(belowSupport.mergeEvaluation.has_value());
    CHECK(belowSupport.mergeEvaluation->merged);
    CHECK(belowSupport.retainedAnchorCount == 0);
    CHECK(belowSupport.components[0].rejectionReason == "below_support");
    CHECK(belowSupport.components[1].rejectionReason == "merged_same_direction");
}

TEST_CASE("fiber anchor exact single direction is not reported as a merge")
{
    const auto result = vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, cellObservations(4, {1.0, 0.0, 0.0}), config());
    CHECK_FALSE(result.mergeEvaluation.has_value());
    CHECK(result.retainedAnchorCount == 1);
    CHECK(result.components[1].rejectionReason == "empty");
}

TEST_CASE("fiber anchor merge configuration is bounded")
{
    auto options = config();
    options.mergeMaximumAngleDegrees = 90.0001;
    CHECK_THROWS_AS(vc::fiber_tracer::validateFiberAnchorConfig(options), std::invalid_argument);
    options = config();
    options.mergeMaximumAbsoluteObjectiveLoss = 1.0001;
    CHECK_THROWS_AS(vc::fiber_tracer::validateFiberAnchorConfig(options), std::invalid_argument);
    options = config();
    options.mergeMaximumRelativeObjectiveLoss = -0.0001;
    CHECK_THROWS_AS(vc::fiber_tracer::validateFiberAnchorConfig(options), std::invalid_argument);
}

TEST_CASE("fiber anchor extraction independently rejects weak second support")
{
    auto options = config();
    options.minimumAlignedSupport = 0.1;
    const auto result =
        vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, cellObservations(4, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, 0.05), options);
    CHECK(result.retainedAnchorCount == 1);
    CHECK((result.components[0].rejectionReason == "below_support" || result.components[1].rejectionReason == "below_support"));
}

TEST_CASE("fiber anchor extraction selects the two best-supported of three modes")
{
    auto options = config();
    options.cellSizePredictionVoxels = 6;
    options.gaussianSigmaPredictionVoxels = 3.0;
    std::vector<FiberAnchorObservation> observations;
    const cv::Vec3d first{1.0, 0.0, 0.0};
    const cv::Vec3d second{0.0, 1.0, 0.0};
    const cv::Vec3d weak{std::sqrt(0.5), std::sqrt(0.5), 0.0};
    for (int z = 0; z < 6; ++z) {
        for (int y = 0; y < 6; ++y) {
            for (int x = 0; x < 6; ++x) {
                const cv::Vec3d direction = x < 2 ? first : (x < 4 ? second : weak);
                observations.push_back({
                    cv::Vec3d{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)},
                    direction,
                    x < 4 ? 1.0 : 0.05,
                    true,
                });
            }
        }
    }
    const auto result = vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {6, 6, 6}, observations, options);
    REQUIRE(result.retainedAnchorCount == 2);
    const auto axes = retainedAxes(result);
    CHECK(std::max(axialDot(axes[0], first), axialDot(axes[1], first)) > 0.999);
    CHECK(std::max(axialDot(axes[0], second), axialDot(axes[1], second)) > 0.999);
}

TEST_CASE("fiber anchor support threshold is inclusive")
{
    auto observations = cellObservations(4, {1.0, 0.0, 0.0});
    const auto baseline = vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, config());
    REQUIRE(baseline.retainedAnchorCount == 1);
    auto options = config();
    options.minimumAlignedSupport = baseline.components[0].anchor.alignedSupport;
    const auto boundary = vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, options);
    CHECK(boundary.retainedAnchorCount == 1);
}

TEST_CASE("fiber anchor crops select complete globally anchored cells")
{
    const vc::fiber_tracer::FiberPredictionGridInfo grid{{9, 9, 9}, 2.5};
    const auto sampler = [](const auto& indices, int, auto& samples) {
        samples.clear();
        for (const auto& index : indices) {
            samples.push_back({
                cv::Vec3d{1.0, 0.0, 0.0},
                index[0] >= 4 && index[0] < 8 && index[1] >= 4 && index[1] < 8 && index[2] >= 4 && index[2] < 8 ? 1.0 : 0.0,
                true,
            });
        }
    };
    const auto full = vc::fiber_tracer::extractFiberAnchors(grid, config(), sampler);
    const auto cropped = vc::fiber_tracer::extractFiberAnchors(grid, config(), sampler, vc::fiber_tracer::FiberAnchorCrop{{5, 5, 5}, {1, 1, 1}});
    REQUIRE(cropped.diagnostics.totalCells == 1);
    REQUIRE(cropped.nonEmptyCells.size() == 1);
    CHECK(cropped.nonEmptyCells[0].cellZYX == std::array<size_t, 3>{1, 1, 1});
    const auto match = std::find_if(full.nonEmptyCells.begin(), full.nonEmptyCells.end(), [](const auto& cell) {
        return cell.cellZYX == std::array<size_t, 3>{1, 1, 1};
    });
    REQUIRE(match != full.nonEmptyCells.end());
    CHECK(cropped.nonEmptyCells[0].objective == match->objective);
    CHECK(cropped.nonEmptyCells[0].components[0].anchor.positionPredictionXYZ == match->components[0].anchor.positionPredictionXYZ);
}

TEST_CASE("fiber anchor extraction handles a clipped global edge cell")
{
    const vc::fiber_tracer::FiberPredictionGridInfo grid{{5, 5, 5}, 3.0};
    const auto report = vc::fiber_tracer::extractFiberAnchors(
        grid,
        config(),
        [](const auto& indices, int, auto& samples) {
            samples.clear();
            for (const auto& index : indices) {
                samples.push_back({
                    cv::Vec3d{0.0, 0.0, 1.0},
                    index == std::array<size_t, 3>{4, 4, 4} ? 1.0 : 0.0,
                    true,
                });
            }
        },
        vc::fiber_tracer::FiberAnchorCrop{{4, 4, 4}, {1, 1, 1}});
    REQUIRE(report.diagnostics.totalCells == 1);
    REQUIRE(report.nonEmptyCells.size() == 1);
    const auto& component = report.nonEmptyCells[0].components[0];
    REQUIRE(component.retained);
    CHECK(component.anchor.positionPredictionXYZ == cv::Vec3d{4.0, 4.0, 4.0});
}

TEST_CASE("fiber anchor artifacts expose only base-volume positions")
{
    const vc::fiber_tracer::FiberPredictionGridInfo grid{{4, 4, 4}, 2.0};
    const auto report = vc::fiber_tracer::extractFiberAnchors(grid, config(), [](const auto& indices, int, auto& samples) {
        samples.assign(indices.size(), {cv::Vec3d{1.0, 0.0, 0.0}, 1.0, true});
    });
    vc::fiber_tracer::FiberAnchorArtifactInfo artifact;
    artifact.sourceLocator = "https://example.test/fiber.lasagna.json";
    artifact.manifestContentHash = "fnv1a64:0123456789abcdef";
    artifact.glyphLengthBaseVoxels = 8.0;
    const auto json = vc::fiber_tracer::fiberAnchorReportJson(report, artifact);
    CHECK(json.at("version") == 1);
    CHECK(json.at("coordinates").at("position_space") == "base_volume");
    CHECK(json.at("coordinates").at("prediction_to_base_scale") == 2.0);
    CHECK(json.at("selection").contains("prediction_interval_origin_base_xyz"));
    CHECK(json.at("selection").contains("prediction_interval_size_base_xyz"));
    CHECK_FALSE(json.at("selection").contains("crop_origin_xyz"));
    REQUIRE(json.at("cells").size() == 1);
    const auto& anchor = json.at("cells").at(0).at("components").at(0);
    CHECK(anchor.contains("position_base_xyz"));
    CHECK_FALSE(anchor.contains("position_prediction_xyz"));
    const std::string obj = vc::fiber_tracer::fiberAnchorReportObj(report, artifact);
    CHECK(obj.find("g cell_0_0_0_anchor_0") != std::string::npos);
    CHECK(obj.find("\nl 1 2\n") != std::string::npos);

    auto parallelConfig = config();
    parallelConfig.parallelThreads = 7;
    const auto parallelReport = vc::fiber_tracer::extractFiberAnchors(grid, parallelConfig, [](const auto& indices, int, auto& samples) {
        samples.assign(indices.size(), {cv::Vec3d{1.0, 0.0, 0.0}, 1.0, true});
    });
    CHECK(vc::fiber_tracer::fiberAnchorReportJson(parallelReport, artifact).dump() == json.dump());
    CHECK(vc::fiber_tracer::fiberAnchorReportObj(parallelReport, artifact) == obj);

    auto layeredReport = report;
    auto secondCell = layeredReport.nonEmptyCells.front();
    secondCell.cellZYX = {0, 0, 1};
    secondCell.components[0].anchor.cellZYX = secondCell.cellZYX;
    secondCell.components[0].anchor.positionPredictionXYZ[0] += 4.0;
    secondCell.components[1] = secondCell.components[0];
    secondCell.components[1].anchor.axisXYZ = {0.0, 1.0, 0.0};
    secondCell.retainedAnchorCount = 2;
    layeredReport.nonEmptyCells.push_back(secondCell);
    const auto directory = temporaryDirectory("component_objs");
    vc::fiber_tracer::writeFiberAnchorArtifacts(directory, layeredReport, artifact);
    const auto read = [](const std::filesystem::path& path) {
        std::ifstream input(path);
        return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    };
    const std::string jointObj = read(directory / "anchors.obj");
    const std::string firstObj = read(directory / "anchors_0.obj");
    const std::string secondObj = read(directory / "anchors_1.obj");
    CHECK(occurrenceCount(jointObj, "\nl ") == 3);
    CHECK(occurrenceCount(firstObj, "\nl ") == 2);
    CHECK(occurrenceCount(secondObj, "\nl ") == 1);
    CHECK(secondObj.find("cell_0_0_1_anchor_1") != std::string::npos);
    std::filesystem::remove_all(directory);
}

TEST_CASE("base-volume crop maps half-open point coordinates to prediction samples")
{
    const vc::fiber_tracer::FiberAnchorCrop aligned{{12, 24, 36}, {12, 12, 12}};
    CHECK(vc::fiber_tracer::fiberAnchorCropFromBaseVoxels(aligned, 3.0).originXYZ == std::array<size_t, 3>{4, 8, 12});
    CHECK(vc::fiber_tracer::fiberAnchorCropFromBaseVoxels(aligned, 3.0).sizeXYZ == std::array<size_t, 3>{4, 4, 4});

    const vc::fiber_tracer::FiberAnchorCrop nonAligned{{13, 25, 37}, {10, 10, 10}};
    CHECK(vc::fiber_tracer::fiberAnchorCropFromBaseVoxels(nonAligned, 3.0).originXYZ == std::array<size_t, 3>{5, 9, 13});
    CHECK(vc::fiber_tracer::fiberAnchorCropFromBaseVoxels(nonAligned, 3.0).sizeXYZ == std::array<size_t, 3>{3, 3, 3});

    const vc::fiber_tracer::FiberAnchorCrop decimalScale{{9, 18, 27}, {9, 9, 9}};
    CHECK(vc::fiber_tracer::fiberAnchorCropFromBaseVoxels(decimalScale, 1.5).originXYZ == std::array<size_t, 3>{6, 12, 18});
    CHECK(vc::fiber_tracer::fiberAnchorCropFromBaseVoxels(decimalScale, 1.5).sizeXYZ == std::array<size_t, 3>{6, 6, 6});
}

TEST_CASE("fiber stored-grid sampling binds only canonical prediction channels")
{
    const auto directory = temporaryDirectory("canonical");
    createConstantZarr(directory / "presence.zarr", {4, 4, 4}, {4, 4, 4}, 255);
    createConstantZarr(directory / "nx.zarr", {4, 4, 4}, {2, 2, 2}, 255);
    createConstantZarr(directory / "ny.zarr", {4, 4, 4}, {1, 4, 2}, 128);
    createEmptyFourDimensionalZarr(directory / "legacy_extra.zarr");
    const auto manifestPath = directory / "fiber.lasagna.json";
    writeText(manifestPath, R"({
      "version": 2,
      "source_to_base": 2.0,
      "groups": {
        "presence": {"zarr":"presence.zarr","scaledown":1,"channels":["presence"]},
        "nx": {"zarr":"nx.zarr","scaledown":1,"channels":["nx"]},
        "ny": {"zarr":"ny.zarr","scaledown":1,"channels":["ny"]},
        "legacy_extra": {"zarr":"legacy_extra.zarr","scaledown":1,
                         "channels":["old_presence","old_nx","old_ny"]}
      }
    })");
    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    const vc::fiber_tracer::FiberPredictionField field(dataset, 16 * 1024 * 1024, vc::fiber_tracer::FiberPredictionFieldBindingMode::CanonicalStoredGrid);
    CHECK(field.optionCount() == 1);
    const auto grid = field.storedGridInfo();
    CHECK(grid.shapeZYX == std::array<size_t, 3>{4, 4, 4});
    CHECK(grid.predictionToBaseScale == doctest::Approx(4.0));
    std::vector<vc::fiber_tracer::FiberStoredPredictionSample> samples;
    field.sampleStoredGridBatch({{0, 0, 0}, {3, 3, 3}}, 2, samples);
    REQUIRE(samples.size() == 2);
    for (const auto& sample : samples) {
        REQUIRE(sample.valid);
        CHECK(sample.presence == doctest::Approx(1.0));
        CHECK(std::abs(sample.direction[0]) > 0.99);
    }
    std::vector<vc::fiber_tracer::FiberStoredPresenceSample> presence;
    field.sampleStoredPresenceBatch({{0, 0, 0}, {3, 3, 3}}, 2, presence);
    REQUIRE(presence.size() == 2);
    for (const auto& sample : presence) {
        REQUIRE(sample.valid);
        CHECK(sample.presence == doctest::Approx(1.0));
    }
    std::filesystem::remove_all(directory);
}

TEST_CASE("fiber stored-grid metadata rejects missing explicit source scale")
{
    const auto directory = temporaryDirectory("missing_scale");
    createConstantZarr(directory / "presence.zarr", {2, 2, 2}, {2, 2, 2}, 255);
    createConstantZarr(directory / "nx.zarr", {2, 2, 2}, {2, 2, 2}, 255);
    createConstantZarr(directory / "ny.zarr", {2, 2, 2}, {2, 2, 2}, 128);
    const auto manifestPath = directory / "fiber.lasagna.json";
    writeText(manifestPath, R"({
      "version": 2,
      "groups": {
        "presence": {"zarr":"presence.zarr","scaledown":0,"channels":["presence"]},
        "nx": {"zarr":"nx.zarr","scaledown":0,"channels":["nx"]},
        "ny": {"zarr":"ny.zarr","scaledown":0,"channels":["ny"]}
      }
    })");
    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    const vc::fiber_tracer::FiberPredictionField field(dataset, 1024 * 1024, vc::fiber_tracer::FiberPredictionFieldBindingMode::CanonicalStoredGrid);
    CHECK_THROWS_WITH_AS(field.storedGridInfo(), doctest::Contains("explicit numeric source_to_base"), std::runtime_error);
    std::filesystem::remove_all(directory);
}

TEST_CASE("fiber stored-grid metadata rejects mismatched canonical shapes")
{
    const auto directory = temporaryDirectory("shape_mismatch");
    createConstantZarr(directory / "presence.zarr", {2, 2, 2}, {2, 2, 2}, 255);
    createConstantZarr(directory / "nx.zarr", {2, 2, 3}, {2, 2, 3}, 255);
    createConstantZarr(directory / "ny.zarr", {2, 2, 2}, {2, 2, 2}, 128);
    const auto manifestPath = directory / "fiber.lasagna.json";
    writeText(manifestPath, R"({
      "version": 2,
      "source_to_base": 1.0,
      "groups": {
        "presence": {"zarr":"presence.zarr","scaledown":0,"channels":["presence"]},
        "nx": {"zarr":"nx.zarr","scaledown":0,"channels":["nx"]},
        "ny": {"zarr":"ny.zarr","scaledown":0,"channels":["ny"]}
      }
    })");
    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::FiberPredictionField(dataset, 1024 * 1024, vc::fiber_tracer::FiberPredictionFieldBindingMode::CanonicalStoredGrid), doctest::Contains("must share shape and spacing"), std::runtime_error);
    std::filesystem::remove_all(directory);
}

TEST_CASE("fiber stored-grid metadata does not substitute a prefixed triplet")
{
    const auto directory = temporaryDirectory("prefixed_only");
    createConstantZarr(directory / "presence.zarr", {2, 2, 2}, {2, 2, 2}, 255);
    createConstantZarr(directory / "nx.zarr", {2, 2, 2}, {2, 2, 2}, 255);
    createConstantZarr(directory / "ny.zarr", {2, 2, 2}, {2, 2, 2}, 128);
    const auto manifestPath = directory / "fiber.lasagna.json";
    writeText(manifestPath, R"({
      "version": 2,
      "source_to_base": 1.0,
      "groups": {
        "presence": {"zarr":"presence.zarr","scaledown":0,"channels":["old_presence"]},
        "nx": {"zarr":"nx.zarr","scaledown":0,"channels":["old_nx"]},
        "ny": {"zarr":"ny.zarr","scaledown":0,"channels":["old_ny"]}
      }
    })");
    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::FiberPredictionField(dataset, 1024 * 1024, vc::fiber_tracer::FiberPredictionFieldBindingMode::CanonicalStoredGrid), doctest::Contains("canonical presence/nx/ny"), std::runtime_error);
    std::filesystem::remove_all(directory);
}
