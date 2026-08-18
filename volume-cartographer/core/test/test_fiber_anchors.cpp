#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/fiber_tracer/detail/FiberAnchorSupportStencil.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "utils/zarr.hpp"

#include <algorithm>
#include <array>
#include <atomic>
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

template <typename Sample>
std::vector<FiberAnchorObservation> boxObservations(
    const std::array<int, 3>& beginZYX,
    const std::array<int, 3>& endZYX,
    Sample&& sample)
{
    std::vector<FiberAnchorObservation> observations;
    for (int z = beginZYX[0]; z < endZYX[0]; ++z) {
        for (int y = beginZYX[1]; y < endZYX[1]; ++y) {
            for (int x = beginZYX[2]; x < endZYX[2]; ++x) {
                observations.push_back(sample(x, y, z));
            }
        }
    }
    return observations;
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

TEST_CASE("fiber anchor peak kernel defaults integrate across neighboring cells")
{
    const FiberAnchorConfig value;
    CHECK(value.maximumIterations == 1);
    CHECK(value.peakSigmaPredictionVoxels == 1.5);
    CHECK(value.peakAxialSigmaPredictionVoxels == 6.0);
    CHECK(std::exp(-0.5 * std::pow(
              value.cellSizePredictionVoxels /
                  value.peakAxialSigmaPredictionVoxels,
              2.0)) == doctest::Approx(0.8007374029));
    CHECK(std::exp(-0.5 * std::pow(
              1.5 * value.cellSizePredictionVoxels /
                  value.peakAxialSigmaPredictionVoxels,
              2.0)) == doctest::Approx(std::exp(-0.5)));
}

TEST_CASE("fiber anchor support stencil preserves scalar order and tile strides")
{
    for (const size_t cellSize : {size_t{3}, size_t{4}}) {
        for (const bool gradients : {false, true}) {
            CAPTURE(cellSize);
            CAPTURE(gradients);
            const double radius = 3.25;
            const size_t halo = static_cast<size_t>(std::ceil(radius)) +
                (gradients ? 1 : 0);
            const size_t extent = cellSize + 2 * halo;
            const std::array<size_t, 3> cellSampleBegin{17, 19, 23};
            const std::array<size_t, 3> tileSampleBegin{11, 13, 17};
            const std::array<size_t, 3> tileShape{32, 34, 36};
            const double pivot = 0.5 * static_cast<double>(cellSize - 1);
            std::vector<uint32_t> expected;
            for (size_t z = 0; z < extent; ++z) {
                for (size_t y = 0; y < extent; ++y) {
                    for (size_t x = 0; x < extent; ++x) {
                        const std::array<double, 3> relative{
                            static_cast<double>(z) - halo,
                            static_cast<double>(y) - halo,
                            static_cast<double>(x) - halo,
                        };
                        const bool owned = std::all_of(
                            relative.begin(), relative.end(),
                            [cellSize](double value) {
                                return value >= 0.0 && value < cellSize;
                            });
                        const double dz = relative[0] - pivot;
                        const double dy = relative[1] - pivot;
                        const double dx = relative[2] - pivot;
                        if (!owned && dz * dz + dy * dy + dx * dx >
                                radius * radius + 1.0e-12) {
                            continue;
                        }
                        if (gradients) {
                            CHECK(z > 0);
                            CHECK(y > 0);
                            CHECK(x > 0);
                            CHECK(z + 1 < extent);
                            CHECK(y + 1 < extent);
                            CHECK(x + 1 < extent);
                        }
                        const size_t tileZ =
                            cellSampleBegin[0] + z - tileSampleBegin[0];
                        const size_t tileY =
                            cellSampleBegin[1] + y - tileSampleBegin[1];
                        const size_t tileX =
                            cellSampleBegin[2] + x - tileSampleBegin[2];
                        expected.push_back(static_cast<uint32_t>(
                            (tileZ * tileShape[1] + tileY) * tileShape[2] +
                            tileX));
                    }
                }
            }

            const auto stencil =
                vc::fiber_tracer::detail::buildFiberAnchorSupportStencil(
                    cellSize, halo, radius);
            CHECK(vc::fiber_tracer::detail::fiberAnchorSupportStencilSize(
                      stencil) == expected.size());
            std::vector<uint32_t> actual;
            vc::fiber_tracer::detail::visitFiberAnchorSupportStencilTileIndices(
                stencil, cellSampleBegin, tileSampleBegin, tileShape,
                [&](uint32_t index) { actual.push_back(index); });
            CHECK(actual == expected);
        }
    }
}

TEST_CASE("normalized float observations preserve anchor geometry")
{
    auto value = config();
    value.peakGradientWeight = 0.0;
    const auto original = cellObservations(
        4, directionAtDegrees(20.0), directionAtDegrees(70.0));
    auto compactEquivalent = original;
    for (auto& observation : compactEquivalent) {
        for (int axis = 0; axis < 3; ++axis) {
            observation.positionPredictionXYZ[axis] = static_cast<double>(
                static_cast<float>(observation.positionPredictionXYZ[axis]));
        }
        const double norm = std::sqrt(
            observation.direction.dot(observation.direction));
        REQUIRE(norm > 0.0);
        observation.direction /= norm;
        for (int axis = 0; axis < 3; ++axis) {
            observation.direction[axis] = static_cast<double>(
                static_cast<float>(observation.direction[axis]));
        }
        observation.presence = static_cast<double>(
            static_cast<float>(observation.presence));
    }

    const auto baseline = vc::fiber_tracer::fitFiberCellAnchors(
        {0, 0, 0}, {0, 0, 0}, {4, 4, 4}, original, value);
    const auto compact = vc::fiber_tracer::fitFiberCellAnchors(
        {0, 0, 0}, {0, 0, 0}, {4, 4, 4}, compactEquivalent, value);

    CHECK(compact.retainedAnchorCount == baseline.retainedAnchorCount);
    for (size_t component = 0; component < baseline.components.size();
         ++component) {
        const auto& expected = baseline.components[component];
        const auto& actual = compact.components[component];
        CHECK(actual.retained == expected.retained);
        CHECK(actual.rejectionReason == expected.rejectionReason);
        if (!expected.retained)
            continue;
        CHECK(axialDot(actual.anchor.axisXYZ, expected.anchor.axisXYZ) >
              1.0 - 1.0e-6);
        CHECK(cv::norm(
                  actual.anchor.positionPredictionXYZ -
                  expected.anchor.positionPredictionXYZ) < 1.0e-3);
    }
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
    vc::fiber_tracer::FiberAnchorFitProfile profile;
    const auto result = vc::fiber_tracer::fitFiberCellAnchors(
        {0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, config(),
        &profile);
    CHECK(result.retainedAnchorCount == 0);
    CHECK(result.components[0].rejectionReason == "empty");
    CHECK(result.components[1].rejectionReason == "empty");
    CHECK(profile.invocations == 1);
    CHECK(profile.nonemptyCells == 0);
    CHECK(profile.weightedObservations == 0);
    CHECK(profile.setupWorkSeconds >= 0.0);
    CHECK(profile.seedGenerationWorkSeconds == 0.0);
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

TEST_CASE("fiber anchor fit profile separates repeated fitting work")
{
    const auto observations = cellObservations(
        4, {1.0, 0.0, 0.0}, directionAtDegrees(45.0));
    vc::fiber_tracer::FiberAnchorFitProfile profile;
    const auto result = vc::fiber_tracer::fitFiberCellAnchors(
        {0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, config(),
        &profile);

    CHECK(result.retainedAnchorCount == 2);
    CHECK(profile.invocations == 1);
    CHECK(profile.nonemptyCells == 1);
    CHECK(profile.weightedObservations == observations.size());
    CHECK(profile.seeds > 0);
    CHECK(profile.seedPairs > 0);
    CHECK(profile.seedPairIterations >= profile.seedPairs);
    CHECK(profile.seedAssignmentObservationVisits > 0);
    CHECK(profile.seedTensorObservationVisits > 0);
    CHECK(profile.seedObjectiveObservationVisits > 0);
    CHECK(profile.localRefinementAttempts > 0);
    CHECK(profile.localRefinementAcceptedSteps <=
          profile.localRefinementAttempts);
    CHECK(profile.backtrackingEvaluations >=
          profile.localRefinementAttempts);
    CHECK(profile.localTensorObservationVisits > 0);
    CHECK(profile.localCentroidObservationVisits > 0);
    CHECK(profile.refinedEvaluationObservationVisits > 0);
    CHECK(profile.peakComponents == 2);
    CHECK(profile.peakGridResponseRequests >=
          profile.peakComputedGridResponses);
    CHECK(profile.peakComputedGridResponses > 0);
    CHECK(profile.peakAcceptanceResponses > 0);
    CHECK(profile.peakResponseObservationVisits > 0);
    CHECK(profile.finalEvaluationObservationVisits == observations.size());
    CHECK(profile.setupWorkSeconds >= 0.0);
    CHECK(profile.seedGenerationWorkSeconds >= 0.0);
    CHECK(profile.seedPairRefinementWorkSeconds >= 0.0);
    CHECK(profile.initializationWorkSeconds >= 0.0);
    CHECK(profile.localRefinementWorkSeconds >= 0.0);
    CHECK(profile.localTensorProposalWorkSeconds >= 0.0);
    CHECK(profile.localCentroidProposalWorkSeconds >= 0.0);
    CHECK(profile.localStateEvaluationWorkSeconds >= 0.0);
    const double localProfiledWorkSeconds =
        profile.localTensorProposalWorkSeconds +
        profile.localCentroidProposalWorkSeconds +
        profile.localStateEvaluationWorkSeconds;
    CHECK(localProfiledWorkSeconds <=
          profile.localRefinementWorkSeconds + 1.0e-3);
    CHECK(profile.peakSearchWorkSeconds >= 0.0);
    CHECK(profile.finalEvaluationWorkSeconds >= 0.0);
}

TEST_CASE("fiber anchor refinement preserves unsupported observation semantics")
{
    auto baseline = cellObservations(4, {1.0, 0.0, 0.0});
    auto observations = baseline;
    for (const size_t index : {size_t{0}, size_t{1}, size_t{2},
                              size_t{4}, size_t{5}, size_t{6}}) {
        baseline[index].valid = false;
    }
    observations[0].valid = false;
    observations[0].direction = {
        std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0};
    observations[1].direction = {0.0, 0.0, 0.0};
    observations[2].direction = {0.5e-15, 0.0, 0.0};
    observations[3].direction = {2.0e-15, 0.0, 0.0};
    observations[4].direction = {
        std::numeric_limits<double>::infinity(), 0.0, 0.0};
    observations[5].presence = std::numeric_limits<double>::quiet_NaN();
    observations[6].presence = 0.5 * config().observationPresenceFloor;
    observations[7].direction = {7.0, 0.0, 0.0};

    baseline.push_back({{1000.0, 1000.0, 1000.0}, {1.0, 0.0, 0.0}, 1.0, true});
    observations.push_back(
        {{1000.0, 1000.0, 1000.0}, {13.0, 0.0, 0.0}, 1.0, true});
    const double huge = std::numeric_limits<double>::max() * 0.5;
    baseline.push_back({{huge, huge, huge}, {1.0, 0.0, 0.0}, 1.0, true});
    observations.push_back(
        {{huge, huge, huge}, {17.0, 0.0, 0.0}, 1.0, true});

    vc::fiber_tracer::FiberAnchorFitProfile baselineProfile;
    vc::fiber_tracer::FiberAnchorFitProfile profile;
    const auto baselineResult = vc::fiber_tracer::fitFiberCellAnchors(
        {0, 0, 0}, {0, 0, 0}, {4, 4, 4}, baseline, config(),
        &baselineProfile);
    const auto result = vc::fiber_tracer::fitFiberCellAnchors(
        {0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, config(), &profile);

    CHECK(result.retainedAnchorCount == baselineResult.retainedAnchorCount);
    CHECK(result.objective == baselineResult.objective);
    CHECK(result.mergeEvaluation.has_value() ==
          baselineResult.mergeEvaluation.has_value());
    for (size_t component = 0; component < result.components.size(); ++component) {
        const auto& actual = result.components[component];
        const auto& expected = baselineResult.components[component];
        CHECK(actual.retained == expected.retained);
        CHECK(actual.rejectionReason == expected.rejectionReason);
        CHECK(actual.assignedObservationCount == expected.assignedObservationCount);
        CHECK(actual.anchor.refinementIterations ==
              expected.anchor.refinementIterations);
        CHECK(actual.anchor.alignedSupport == expected.anchor.alignedSupport);
        CHECK(actual.anchor.directionalCoherence ==
              expected.anchor.directionalCoherence);
        CHECK(actual.anchor.refinementScore == expected.anchor.refinementScore);
        for (int axis = 0; axis < 3; ++axis) {
            CHECK(actual.anchor.axisXYZ[axis] == expected.anchor.axisXYZ[axis]);
            CHECK(actual.anchor.positionPredictionXYZ[axis] ==
                  expected.anchor.positionPredictionXYZ[axis]);
        }
    }
    CHECK(profile.weightedObservations == baselineProfile.weightedObservations);
    CHECK(profile.localRefinementAttempts ==
          baselineProfile.localRefinementAttempts);
    CHECK(profile.localRefinementAcceptedSteps ==
          baselineProfile.localRefinementAcceptedSteps);
    CHECK(profile.backtrackingEvaluations ==
          baselineProfile.backtrackingEvaluations);
    CHECK(profile.refinedEvaluationObservationVisits ==
          baselineProfile.refinedEvaluationObservationVisits);
}

TEST_CASE("fiber anchor broad phase keeps combined support-boundary evidence")
{
    auto options = config();
    const cv::Vec3d pivot{1.5, 1.5, 1.5};
    const double cutoff = options.gaussianCutoffSigmas *
        options.gaussianSigmaPredictionVoxels;
    auto observations = cellObservations(4, {1.0, 0.0, 0.0});
    observations.push_back({
        pivot + cv::Vec3d{
            options.axialSupportHalfWidthPredictionVoxels,
            std::nextafter(cutoff, 0.0),
            0.0,
        },
        {1.0, 0.0, 0.0},
        1.0,
        true,
    });

    const auto result = vc::fiber_tracer::fitFiberCellAnchors(
        {0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, options);

    REQUIRE(result.retainedAnchorCount == 1);
    CHECK(result.components[0].assignedObservationCount == observations.size());
}

TEST_CASE("fiber anchor extraction fits two non-orthogonal direction modes")
{
    for (const double degrees : {15.0, 30.0, 45.0, 60.0, 90.0}) {
        const double radians = degrees * std::acos(-1.0) / 180.0;
        const cv::Vec3d first{1.0, 0.0, 0.0};
        const cv::Vec3d second{std::cos(radians), std::sin(radians), 0.0};
        const auto result = vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, cellObservations(4, first, second), config());
        REQUIRE_MESSAGE(result.retainedAnchorCount == 2, std::string("angle=") + std::to_string(degrees));
        REQUIRE(result.initializedDiagnostics[0].anchor.has_value());
        REQUIRE(result.initializedDiagnostics[1].anchor.has_value());
        const auto initializedFirst = result.initializedDiagnostics[0].anchor->axisXYZ;
        const auto initializedSecond = result.initializedDiagnostics[1].anchor->axisXYZ;
        CHECK_MESSAGE(
            std::max(axialDot(initializedFirst, first), axialDot(initializedSecond, first)) > 0.999,
            std::string("initialized first angle=") + std::to_string(degrees));
        CHECK_MESSAGE(
            std::max(axialDot(initializedFirst, second), axialDot(initializedSecond, second)) > 0.999,
            std::string("initialized second angle=") + std::to_string(degrees));
        const auto axes = retainedAxes(result);
        CAPTURE(degrees);
        CAPTURE(axes[0][0]);
        CAPTURE(axes[0][1]);
        CAPTURE(axes[0][2]);
        CAPTURE(axes[1][0]);
        CAPTURE(axes[1][1]);
        CAPTURE(axes[1][2]);
        const double firstMatch = std::max(axialDot(axes[0], first), axialDot(axes[1], first));
        const double secondMatch = std::max(axialDot(axes[0], second), axialDot(axes[1], second));
        const double toleranceDegrees = degrees == 15.0 ? 8.0 : 1.0e-3;
        const double minimumMatch = std::cos(
            toleranceDegrees * std::acos(-1.0) / 180.0);
        CHECK_MESSAGE(firstMatch >= minimumMatch, std::string("angle=") + std::to_string(degrees) + " match=" + std::to_string(firstMatch));
        CHECK_MESSAGE(secondMatch >= minimumMatch, std::string("angle=") + std::to_string(degrees) + " match=" + std::to_string(secondMatch));
    }
}

TEST_CASE("fiber anchor extraction preserves supported nearby directions")
{
    for (const double degrees : {5.0, 9.0, 10.0, 11.0}) {
        const cv::Vec3d first{1.0, 0.0, 0.0};
        const cv::Vec3d second = directionAtDegrees(degrees);
        auto options = config();
        options.gaussianSigmaPredictionVoxels = 100.0;
        const auto result = vc::fiber_tracer::fitFiberCellAnchors(
            {0, 0, 0}, {0, 0, 0}, {4, 4, 4},
            cellObservations(4, first, second), options);
        REQUIRE_MESSAGE(
            result.retainedAnchorCount == 2,
            std::string("angle=") + std::to_string(degrees));
        CHECK_FALSE(result.mergeEvaluation.has_value());
        const auto axes = retainedAxes(result);
        CAPTURE(degrees);
        CAPTURE(axes[0][0]);
        CAPTURE(axes[0][1]);
        CAPTURE(axes[0][2]);
        CAPTURE(axes[1][0]);
        CAPTURE(axes[1][1]);
        CAPTURE(axes[1][2]);
        const double firstMatch = std::max(
            axialDot(axes[0], first), axialDot(axes[1], first));
        const double secondMatch = std::max(
            axialDot(axes[0], second), axialDot(axes[1], second));
        CHECK_MESSAGE(
            firstMatch > std::cos(std::acos(-1.0) / 180.0),
            std::string("angle=") + std::to_string(degrees) +
                " first_match=" + std::to_string(firstMatch));
        CHECK_MESSAGE(
            secondMatch > std::cos(std::acos(-1.0) / 180.0),
            std::string("angle=") + std::to_string(degrees) +
                " second_match=" + std::to_string(secondMatch));
    }
}

TEST_CASE("fiber anchor robust cutoff retains coherent evidence")
{
    const double residual = std::pow(std::sin(2.0 * std::acos(-1.0) / 180.0), 2.0);
    std::vector<vc::fiber_tracer::FiberAnchorResidualSample> samples(32, {residual, 1.0});
    const auto cutoff = vc::fiber_tracer::selectFiberAnchorRobustCutoff(
        samples, 0.20, 3.0, 5.0);
    CHECK_FALSE(cutoff.detectedOutliers);
    CHECK(cutoff.trimmedMass == 0.0);
    CHECK(cutoff.retainedMass == doctest::Approx(32.0));
}

TEST_CASE("fiber anchor robust cutoff trims tails within its mass budget")
{
    const double tailResidual = std::pow(
        std::sin(25.0 * std::acos(-1.0) / 180.0), 2.0);
    std::vector<vc::fiber_tracer::FiberAnchorResidualSample> samples;
    samples.insert(samples.end(), 90, {0.0, 1.0});
    samples.insert(samples.end(), 10, {tailResidual, 1.0});
    for (const double maximumTrim : {0.0, 0.10, 0.20}) {
        const auto cutoff = vc::fiber_tracer::selectFiberAnchorRobustCutoff(
            samples, maximumTrim, 3.0, 5.0);
        CHECK(cutoff.retainedMass + 1.0e-12 >=
              (1.0 - maximumTrim) * cutoff.totalMass);
        if (maximumTrim == 0.0) {
            CHECK(cutoff.trimmedMass == 0.0);
        } else {
            CHECK(cutoff.detectedOutliers);
            CHECK(cutoff.trimmedMass == doctest::Approx(10.0));
        }
    }
}

TEST_CASE("fiber anchor robust direction fit rejects a minority angular tail")
{
    auto observations = cellObservations(4, {1.0, 0.0, 0.0});
    for (size_t index = 0; index < observations.size(); ++index) {
        if (index % 8 == 0)
            observations[index].direction = directionAtDegrees(30.0);
        else if (index % 2 == 0)
            observations[index].direction *= -1.0;
    }
    auto robust = config();
    robust.maximumSeedCount = 1;
    vc::fiber_tracer::FiberAnchorFitProfile robustProfile;
    const auto robustResult = vc::fiber_tracer::fitFiberCellAnchors(
        {0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, robust,
        &robustProfile);
    REQUIRE(robustResult.retainedAnchorCount == 1);
    CHECK(axialDot(
              robustResult.components[0].anchor.axisXYZ,
              cv::Vec3d{1.0, 0.0, 0.0}) > 0.9999);
    CHECK(robustProfile.robustTrimmedComponents > 0);
    CHECK(robustProfile.robustTrimmedMass > 0.0);

    auto untrimmed = robust;
    untrimmed.robustMaximumTrimMassFraction = 0.0;
    const auto untrimmedResult = vc::fiber_tracer::fitFiberCellAnchors(
        {0, 0, 0}, {0, 0, 0}, {4, 4, 4}, observations, untrimmed);
    REQUIRE(untrimmedResult.retainedAnchorCount == 1);
    CHECK(axialDot(
              robustResult.components[0].anchor.axisXYZ,
              cv::Vec3d{1.0, 0.0, 0.0}) >
          axialDot(
              untrimmedResult.components[0].anchor.axisXYZ,
              cv::Vec3d{1.0, 0.0, 0.0}));
}

TEST_CASE("fiber anchor exact single direction is not reported as a merge")
{
    const auto result = vc::fiber_tracer::fitFiberCellAnchors({0, 0, 0}, {0, 0, 0}, {4, 4, 4}, cellObservations(4, {1.0, 0.0, 0.0}), config());
    CHECK_FALSE(result.mergeEvaluation.has_value());
    CHECK(result.retainedAnchorCount == 1);
    CHECK(result.components[1].rejectionReason == "empty");
}

TEST_CASE("fiber anchor robust cutoff retains a complete boundary bin")
{
    std::vector<vc::fiber_tracer::FiberAnchorResidualSample> samples{
        {0.0, 8.0}, {0.5, 1.0}, {0.5, 1.0},
        {std::numeric_limits<double>::quiet_NaN(), 100.0},
        {1.0, -1.0},
    };
    const auto cutoff = vc::fiber_tracer::selectFiberAnchorRobustCutoff(
        samples, 0.10, 0.0, 0.0);
    CHECK(cutoff.totalMass == doctest::Approx(10.0));
    CHECK(cutoff.retainedMass == doctest::Approx(10.0));
    CHECK(cutoff.trimmedMass == 0.0);
}

TEST_CASE("fiber anchor spatial backtracking stops at the first half-voxel step")
{
    const auto fractions = vc::fiber_tracer::fiberAnchorSpatialBacktrackingFractions(
        3.0, 0.5);
    REQUIRE(fractions.size() == 4);
    CHECK(fractions == std::vector<double>{1.0, 0.5, 0.25, 0.125});
    CHECK(3.0 * fractions.back() <= 0.5);
    CHECK(3.0 * fractions[fractions.size() - 2] > 0.5);
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
    options = config();
    options.robustMaximumTrimMassFraction = 0.2001;
    CHECK_THROWS_AS(vc::fiber_tracer::validateFiberAnchorConfig(options), std::invalid_argument);
    options = config();
    options.robustMadMultiplier = -0.1;
    CHECK_THROWS_AS(vc::fiber_tracer::validateFiberAnchorConfig(options), std::invalid_argument);
    options = config();
    options.robustMinimumAngleDegrees = 90.0001;
    CHECK_THROWS_AS(vc::fiber_tracer::validateFiberAnchorConfig(options), std::invalid_argument);
    options = config();
    options.peakSigmaPredictionVoxels = 0.0;
    CHECK_THROWS_AS(vc::fiber_tracer::validateFiberAnchorConfig(options), std::invalid_argument);
    options = config();
    options.peakAxialSigmaPredictionVoxels = 0.0;
    CHECK_THROWS_AS(vc::fiber_tracer::validateFiberAnchorConfig(options), std::invalid_argument);
    options = config();
    options.peakGridStepPredictionVoxels = 0.0;
    CHECK_THROWS_AS(vc::fiber_tracer::validateFiberAnchorConfig(options), std::invalid_argument);
    options = config();
    options.peakGridStepPredictionVoxels =
        options.localWindowRadiusPredictionVoxels / 129.0;
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

TEST_CASE("fiber anchor refinement centers an off-center halo-supported fiber without axial motion")
{
    auto options = config();
    options.minimumAlignedSupport = 0.001;
    const auto observations = boxObservations(
        {0, 0, 0}, {12, 12, 12}, [](int x, int y, int z) {
            return FiberAnchorObservation{
                cv::Vec3d{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)},
                {1.0, 0.0, 0.0},
                y == 7 && z == 6 ? 1.0 : 0.0,
                true,
            };
        });
    const auto result = vc::fiber_tracer::fitFiberCellAnchors(
        {1, 1, 1}, {4, 4, 4}, {8, 8, 8}, observations, options);
    REQUIRE(result.retainedAnchorCount == 1);
    const auto& anchor = result.components[0].anchor;
    CHECK(anchor.positionPredictionXYZ[0] == doctest::Approx(5.5).epsilon(1.0e-12));
    CHECK(anchor.positionPredictionXYZ[1] == doctest::Approx(7.0).epsilon(1.0e-8));
    CHECK(anchor.positionPredictionXYZ[2] == doctest::Approx(6.0).epsilon(1.0e-8));
    CHECK(std::abs((anchor.positionPredictionXYZ - cv::Vec3d{5.5, 5.5, 5.5}).dot(anchor.axisXYZ)) < 1.0e-10);
    CHECK(anchor.refinementIterations > 0);
}

TEST_CASE("fiber anchor peak search stays on the local in-cell mode")
{
    auto options = config();
    options.gaussianSigmaPredictionVoxels = 4.0;
    options.peakSigmaPredictionVoxels = 0.75;
    options.localWindowRadiusPredictionVoxels = 2.0;
    options.minimumAlignedSupport = 0.001;
    const auto observations = boxObservations(
        {0, 0, 0}, {12, 12, 12}, [](int x, int y, int z) {
            return FiberAnchorObservation{
                cv::Vec3d{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)},
                {1.0, 0.0, 0.0},
                z == 6 && y == 9 ? 1.0 :
                    (z == 6 && y == 7 ? 0.2 : 0.0),
                true,
            };
        });
    const auto result = vc::fiber_tracer::fitFiberCellAnchors(
        {1, 1, 1}, {4, 4, 4}, {8, 8, 8}, observations, options);
    REQUIRE(result.retainedAnchorCount == 1);
    const auto& anchor = result.components[0].anchor;
    const cv::Vec3d pivot{5.5, 5.5, 5.5};
    CHECK(anchor.positionPredictionXYZ[1] == doctest::Approx(7.0).epsilon(1.0e-8));
    CHECK(anchor.positionPredictionXYZ[1] < 7.5);
    CHECK(std::abs((anchor.positionPredictionXYZ - pivot).dot(anchor.axisXYZ)) < 1.0e-10);
}

TEST_CASE("fiber anchor peak search leaves a symmetric parallel-ridge midpoint")
{
    auto options = config();
    options.peakSigmaPredictionVoxels = 0.75;
    options.minimumAlignedSupport = 0.001;
    const auto observations = boxObservations(
        {0, 0, 0}, {12, 12, 12}, [](int x, int y, int z) {
            return FiberAnchorObservation{
                cv::Vec3d{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)},
                {1.0, 0.0, 0.0},
                z == 6 && (y == 4 || y == 7) ? 1.0 : 0.0,
                true,
            };
        });
    const auto result = vc::fiber_tracer::fitFiberCellAnchors(
        {1, 1, 1}, {4, 4, 4}, {8, 8, 8}, observations, options);
    REQUIRE(result.retainedAnchorCount == 1);
    const auto& component = result.components[0];
    const auto& position = component.anchor.positionPredictionXYZ;
    CHECK(std::abs(position[1] - 5.5) > 1.0);
    CHECK((position[1] == doctest::Approx(4.0).epsilon(0.1) ||
           position[1] == doctest::Approx(7.0).epsilon(0.1)));
}

TEST_CASE("fiber anchor peak search applies a bounded subvoxel fit")
{
    auto options = config();
    options.minimumAlignedSupport = 0.001;
    const auto observations = boxObservations(
        {0, 0, 0}, {12, 12, 12}, [](int x, int y, int z) {
            const double presence = z == 6 && y == 5 ? 1.0 :
                (z == 6 && y == 6 ? 0.5 : 0.0);
            return FiberAnchorObservation{
                cv::Vec3d{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)},
                {1.0, 0.0, 0.0}, presence, true,
            };
        });
    const auto result = vc::fiber_tracer::fitFiberCellAnchors(
        {1, 1, 1}, {4, 4, 4}, {8, 8, 8}, observations, options);
    REQUIRE(result.retainedAnchorCount == 1);
    const auto& component = result.components[0];
    const auto& position = component.anchor.positionPredictionXYZ;
    CHECK(position[1] > 5.0);
    CHECK(position[1] < 5.5);
    CHECK(position[2] == doctest::Approx(6.0).epsilon(1.0e-8));
    REQUIRE(component.discretePeakPositionPredictionXYZ.has_value());
    REQUIRE(component.separablePeakPositionPredictionXYZ.has_value());
    const auto& separable = *component.separablePeakPositionPredictionXYZ;
    CHECK(separable[1] > 5.0);
    CHECK(separable[1] < 5.5);
    CHECK(separable[2] == doctest::Approx(6.0).epsilon(1.0e-8));
    const cv::Vec3d pivot{5.5, 5.5, 5.5};
    CHECK(std::abs((separable - pivot).dot(component.anchor.axisXYZ)) < 1.0e-10);
    CHECK(std::abs((position - pivot).dot(component.anchor.axisXYZ)) < 1.0e-10);
    const auto response = [&](const cv::Vec3d& candidate) {
        double numerator = 0.0;
        double denominator = 0.0;
        const double transverseCutoff = options.gaussianCutoffSigmas *
            options.peakSigmaPredictionVoxels;
        const double axialCutoff = options.gaussianCutoffSigmas *
            options.peakAxialSigmaPredictionVoxels;
        for (const auto& observation : observations) {
            const cv::Vec3d delta = observation.positionPredictionXYZ - candidate;
            const double axial = delta[0];
            const double transverseDistanceSquared =
                delta[1] * delta[1] + delta[2] * delta[2];
            if (std::abs(axial) > axialCutoff ||
                transverseDistanceSquared >
                    transverseCutoff * transverseCutoff) {
                continue;
            }
            const double weight = std::exp(
                -transverseDistanceSquared /
                    (2.0 * options.peakSigmaPredictionVoxels *
                     options.peakSigmaPredictionVoxels) -
                axial * axial /
                    (2.0 * options.peakAxialSigmaPredictionVoxels *
                     options.peakAxialSigmaPredictionVoxels));
            denominator += weight;
            numerator += weight * observation.presence;
        }
        return numerator / denominator;
    };
    CHECK(response(position) >= response({5.5, 5.0, 6.0}) - 1.0e-12);
}

TEST_CASE("fiber anchor quadratic peak recovers a cross-coupled maximum")
{
    constexpr double expectedFirst = 0.25;
    constexpr double expectedSecond = -0.2;
    std::array<std::array<double, 3>, 3> response{};
    for (int first = -1; first <= 1; ++first) {
        for (int second = -1; second <= 1; ++second) {
            const double x = static_cast<double>(first) - expectedFirst;
            const double y = static_cast<double>(second) - expectedSecond;
            response[static_cast<size_t>(first + 1)]
                    [static_cast<size_t>(second + 1)] =
                10.0 - 2.0 * x * x - 0.75 * x * y - 3.0 * y * y;
        }
    }

    const auto fitted = vc::fiber_tracer::fitFiberAnchorQuadraticPeak(response);
    REQUIRE(fitted.has_value());
    CHECK(fitted->firstGridSteps == doctest::Approx(expectedFirst).epsilon(1.0e-12));
    CHECK(fitted->secondGridSteps == doctest::Approx(expectedSecond).epsilon(1.0e-12));

    std::array<std::array<double, 3>, 3> transposed{};
    for (size_t first = 0; first < 3; ++first) {
        for (size_t second = 0; second < 3; ++second)
            transposed[first][second] = response[second][first];
    }
    const auto swapped = vc::fiber_tracer::fitFiberAnchorQuadraticPeak(transposed);
    REQUIRE(swapped.has_value());
    CHECK(swapped->firstGridSteps == doctest::Approx(expectedSecond).epsilon(1.0e-12));
    CHECK(swapped->secondGridSteps == doctest::Approx(expectedFirst).epsilon(1.0e-12));
}

TEST_CASE("fiber anchor quadratic peak least squares uses corner evidence")
{
    std::array<std::array<double, 3>, 3> response{};
    for (int first = -1; first <= 1; ++first) {
        for (int second = -1; second <= 1; ++second) {
            response[static_cast<size_t>(first + 1)]
                    [static_cast<size_t>(second + 1)] =
                5.0 - first * first - second * second;
        }
    }
    response[2][0] += 0.3;
    response[2][2] += 0.3;

    const auto fitted = vc::fiber_tracer::fitFiberAnchorQuadraticPeak(response);
    REQUIRE(fitted.has_value());
    CHECK(fitted->firstGridSteps > 0.04);
    CHECK(std::abs(fitted->secondGridSteps) < 1.0e-12);
}

TEST_CASE("fiber anchor quadratic peak rejects ill-defined curvature")
{
    const auto samples = [](const auto& function) {
        std::array<std::array<double, 3>, 3> response{};
        for (int first = -1; first <= 1; ++first) {
            for (int second = -1; second <= 1; ++second) {
                response[static_cast<size_t>(first + 1)]
                        [static_cast<size_t>(second + 1)] =
                    function(static_cast<double>(first),
                             static_cast<double>(second));
            }
        }
        return response;
    };

    CHECK_FALSE(vc::fiber_tracer::fitFiberAnchorQuadraticPeak(
        samples([](double, double) { return 1.0; })).has_value());
    CHECK_FALSE(vc::fiber_tracer::fitFiberAnchorQuadraticPeak(
        samples([](double x, double) { return 2.0 - x * x; })).has_value());
    CHECK_FALSE(vc::fiber_tracer::fitFiberAnchorQuadraticPeak(
        samples([](double x, double y) {
            return 2.0 - x * x - 1.0e-13 * y * y;
        })).has_value());

    auto nonFinite = samples([](double x, double y) {
        return 2.0 - x * x - y * y;
    });
    nonFinite[0][2] = std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(vc::fiber_tracer::fitFiberAnchorQuadraticPeak(nonFinite).has_value());
}

TEST_CASE("fiber anchor quadratic peak uses a closed half-step acceptance box")
{
    const auto peakAt = [](double expectedFirst) {
        std::array<std::array<double, 3>, 3> response{};
        for (int first = -1; first <= 1; ++first) {
            for (int second = -1; second <= 1; ++second) {
                const double x = static_cast<double>(first) - expectedFirst;
                response[static_cast<size_t>(first + 1)]
                        [static_cast<size_t>(second + 1)] =
                    4.0 - x * x - second * second;
            }
        }
        return vc::fiber_tracer::fitFiberAnchorQuadraticPeak(response);
    };

    const auto boundary = peakAt(0.5);
    REQUIRE(boundary.has_value());
    CHECK(boundary->firstGridSteps == doctest::Approx(0.5).epsilon(1.0e-12));
    CHECK_FALSE(peakAt(0.500001).has_value());
}

TEST_CASE("fiber anchor refinement rotates its cell-center plane with direction")
{
    auto options = config();
    options.minimumAlignedSupport = 0.001;
    const cv::Vec3d diagonal = directionAtDegrees(35.0);
    const auto observations = boxObservations(
        {0, 0, 0}, {12, 12, 12}, [&](int x, int y, int z) {
            const bool owned = x >= 4 && x < 8 && y >= 4 && y < 8 && z >= 4 && z < 8;
            const bool haloLine = z == 6 && std::abs(y - (6 + static_cast<int>(std::round((x - 6) * std::tan(35.0 * std::acos(-1.0) / 180.0))))) <= 0;
            return FiberAnchorObservation{
                cv::Vec3d{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)},
                owned ? cv::Vec3d{1.0, 0.0, 0.0} : diagonal,
                haloLine ? (owned ? 0.2 : 1.0) : 0.0,
                true,
            };
        });
    const auto result = vc::fiber_tracer::fitFiberCellAnchors(
        {1, 1, 1}, {4, 4, 4}, {8, 8, 8}, observations, options);
    REQUIRE(result.retainedAnchorCount == 1);
    const auto& anchor = result.components[0].anchor;
    CHECK(axialDot(anchor.axisXYZ, diagonal) > axialDot(cv::Vec3d{1.0, 0.0, 0.0}, diagonal));
    CHECK(std::abs((anchor.positionPredictionXYZ - cv::Vec3d{5.5, 5.5, 5.5}).dot(anchor.axisXYZ)) < 1.0e-8);
}

TEST_CASE("fiber anchor local refinement does not leave a weak mode for a stronger separated fiber")
{
    auto options = config();
    options.gaussianSigmaPredictionVoxels = 0.75;
    options.peakSigmaPredictionVoxels = 0.75;
    options.localWindowRadiusPredictionVoxels = 4.0;
    options.minimumAlignedSupport = 0.001;
    const auto observations = boxObservations(
        {0, 0, 0}, {12, 12, 12}, [](int x, int y, int z) {
            double presence = 0.0;
            if (y == 5 && z == 6)
                presence = 0.35;
            else if (y == 9 && z == 6)
                presence = 1.0;
            return FiberAnchorObservation{
                cv::Vec3d{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)},
                {1.0, 0.0, 0.0}, presence, true};
        });
    const auto result = vc::fiber_tracer::fitFiberCellAnchors(
        {1, 1, 1}, {4, 4, 4}, {8, 8, 8}, observations, options);
    REQUIRE(result.retainedAnchorCount == 1);
    CHECK(result.components[0].anchor.positionPredictionXYZ[1] < 6.0);
}

TEST_CASE("fiber anchor peak does not reassign evidence beyond robust axial membership")
{
    auto shortAxial = config();
    shortAxial.minimumAlignedSupport = 0.001;
    shortAxial.peakAxialSigmaPredictionVoxels = 0.5;
    const auto observations = boxObservations(
        {0, 0, 0}, {12, 12, 24}, [](int x, int y, int z) {
            double presence = 0.0;
            if (z == 6 && y == 5 && x >= 4 && x < 8)
                presence = 0.5;
            else if (z == 6 && y == 7 && x >= 13)
                presence = 1.0;
            return FiberAnchorObservation{
                cv::Vec3d{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)},
                {1.0, 0.0, 0.0}, presence, true};
        });
    const auto shortResult = vc::fiber_tracer::fitFiberCellAnchors(
        {1, 1, 1}, {4, 4, 4}, {8, 8, 8}, observations, shortAxial);
    auto longAxial = shortAxial;
    longAxial.peakAxialSigmaPredictionVoxels = 6.0;
    const auto longResult = vc::fiber_tracer::fitFiberCellAnchors(
        {1, 1, 1}, {4, 4, 4}, {8, 8, 8}, observations, longAxial);
    REQUIRE(shortResult.retainedAnchorCount == 1);
    REQUIRE(longResult.retainedAnchorCount == 1);
    CHECK(longResult.components[0].anchor.positionPredictionXYZ[1] ==
          doctest::Approx(shortResult.components[0].anchor.positionPredictionXYZ[1]));
}

TEST_CASE("fiber anchor extraction preserves robust membership during axial peak search")
{
    const vc::fiber_tracer::FiberPredictionGridInfo grid{{12, 12, 24}, 1.0};
    const auto sampler = [](const auto& indices, int, auto& samples) {
        samples.clear();
        for (const auto& index : indices) {
            double presence = 0.0;
            if (index[0] == 6 && index[1] == 5 &&
                index[2] >= 4 && index[2] < 8) {
                presence = 0.5;
            } else if (index[0] == 6 && index[1] == 7 && index[2] >= 13) {
                presence = 1.0;
            }
            samples.push_back({{1.0, 0.0, 0.0}, presence, true});
        }
    };
    auto shortAxial = config();
    shortAxial.minimumAlignedSupport = 0.001;
    shortAxial.peakAxialSigmaPredictionVoxels = 0.5;
    const auto shortReport = vc::fiber_tracer::extractFiberAnchorsForCells(
        grid, shortAxial, sampler, {{1, 1, 1}});
    auto longAxial = shortAxial;
    longAxial.peakAxialSigmaPredictionVoxels = 6.0;
    const auto longReport = vc::fiber_tracer::extractFiberAnchorsForCells(
        grid, longAxial, sampler, {{1, 1, 1}});
    REQUIRE(shortReport.nonEmptyCells.size() == 1);
    REQUIRE(longReport.nonEmptyCells.size() == 1);
    CHECK(longReport.nonEmptyCells[0].components[0]
              .anchor.positionPredictionXYZ[1] ==
          doctest::Approx(shortReport.nonEmptyCells[0].components[0]
                              .anchor.positionPredictionXYZ[1]));
}

TEST_CASE("fiber anchor extraction halo encloses every oblique peak kernel")
{
    auto options = config();
    std::array<size_t, 3> selectedMinimum{};
    std::array<size_t, 3> selectedMaximum{};
    size_t calls = 0;
    const auto report = vc::fiber_tracer::extractRefinedFiberAnchorsForCells(
        {{80, 80, 80}, 1.0}, options,
        [&](const auto& indices, int, auto& samples) {
            if (calls++ == 0) {
                selectedMinimum = indices.front();
                selectedMaximum = indices.back();
            }
            samples.assign(indices.size(), {
                cv::Vec3d{1.0 / std::sqrt(3.0),
                          1.0 / std::sqrt(3.0),
                          1.0 / std::sqrt(3.0)},
                0.0,
                true});
        },
        {{8, 8, 8}});
    CHECK(report.diagnostics.totalCells == 1);
    const double peakRadius = std::hypot(
        options.localWindowRadiusPredictionVoxels +
            options.gaussianCutoffSigmas *
                options.peakSigmaPredictionVoxels,
        options.gaussianCutoffSigmas *
            options.peakAxialSigmaPredictionVoxels);
    const size_t expectedHalo = static_cast<size_t>(std::ceil(peakRadius));
    REQUIRE(expectedHalo == 20);
    CHECK(selectedMinimum == std::array<size_t, 3>{11, 11, 11});
    CHECK(selectedMaximum == std::array<size_t, 3>{56, 56, 56});

    options.peakGradientWeight = 0.0;
    calls = 0;
    const auto noGradientReport = vc::fiber_tracer::extractRefinedFiberAnchorsForCells(
        {{80, 80, 80}, 1.0}, options,
        [&](const auto& indices, int, auto& samples) {
            if (calls++ == 0) {
                selectedMinimum = indices.front();
                selectedMaximum = indices.back();
            }
            samples.assign(indices.size(), {
                cv::Vec3d{1.0, 0.0, 0.0}, 0.0, true});
        },
        {{8, 8, 8}});
    CHECK(noGradientReport.diagnostics.totalCells == 1);
    CHECK(selectedMinimum == std::array<size_t, 3>{12, 12, 12});
    CHECK(selectedMaximum == std::array<size_t, 3>{55, 55, 55});
}

TEST_CASE("signed gradient evidence rejects a stationary midpoint between parallel ridges")
{
    const auto sampler = [](const auto& indices, int threads, auto& samples) {
        if (threads != 1)
            throw std::runtime_error("nested sampler threads used");
        samples.clear();
        for (const auto& index : indices) {
            const size_t y = index[1];
            const double presence = y == 2 || y == 5 ? 1.0 :
                (y == 3 || y == 4 ? 0.4 : 0.0);
            samples.push_back({
                cv::Vec3d{1.0, 0.0, 0.0}, presence, true, true});
        }
    };
    auto presenceOnly = config();
    presenceOnly.cellSizePredictionVoxels = 8;
    presenceOnly.minimumAlignedSupport = 0.001;
    presenceOnly.peakGradientWeight = 0.0;
    auto signedGradient = presenceOnly;
    signedGradient.peakGradientWeight = 1.0;
    signedGradient.peakGradientReliabilityScale = 0.01;

    const auto baseline = vc::fiber_tracer::extractRefinedFiberAnchorsForCells(
        {{16, 16, 16}, 1.0}, presenceOnly, sampler, {{0, 0, 0}});
    const auto centered = vc::fiber_tracer::extractRefinedFiberAnchorsForCells(
        {{16, 16, 16}, 1.0}, signedGradient, sampler, {{0, 0, 0}});
    const auto& baselineRefined = baseline.diagnosticStages[static_cast<size_t>(
        vc::fiber_tracer::FiberAnchorDiagnosticStage::Refined)];
    const auto& centeredRefined = centered.diagnosticStages[static_cast<size_t>(
        vc::fiber_tracer::FiberAnchorDiagnosticStage::Refined)];
    REQUIRE(baselineRefined.size() == 1);
    REQUIRE(centeredRefined.size() == 1);
    REQUIRE(baselineRefined[0].anchor.has_value());
    REQUIRE(centeredRefined[0].anchor.has_value());
    const double baselineOffset = std::abs(
        baselineRefined[0].anchor->positionPredictionXYZ[1] - 3.5);
    const double gradientOffset = std::abs(
        centeredRefined[0].anchor->positionPredictionXYZ[1] - 3.5);
    CHECK(baselineOffset < 0.1);
    CHECK(gradientOffset > 0.5);
}

TEST_CASE("fiber anchor truncated edge pivot remains feasible for an oblique direction")
{
    auto options = config();
    options.minimumAlignedSupport = 0.001;
    const cv::Vec3d direction{1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
    const auto observations = boxObservations(
        {0, 0, 0}, {5, 5, 5}, [&](int x, int y, int z) {
            return FiberAnchorObservation{
                cv::Vec3d{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)},
                direction,
                x == 4 && y == 4 && z == 4 ? 1.0 : 0.0,
                true};
        });
    const auto result = vc::fiber_tracer::fitFiberCellAnchors(
        {1, 1, 1}, {4, 4, 4}, {5, 5, 5}, observations, options);
    REQUIRE(result.retainedAnchorCount == 1);
    const auto& component = result.components[0];
    CHECK(component.anchor.positionPredictionXYZ == cv::Vec3d{4.0, 4.0, 4.0});
    REQUIRE(component.discretePeakPositionPredictionXYZ.has_value());
    CHECK(*component.discretePeakPositionPredictionXYZ ==
        component.anchor.positionPredictionXYZ);
}

TEST_CASE("fiber anchor NMS suppresses transverse duplicates but keeps longitudinal anchors")
{
    auto options = config();
    options.minimumAlignedSupport = 0.001;
    const vc::fiber_tracer::FiberPredictionGridInfo grid{{4, 8, 8}, 1.0};
    const auto report = vc::fiber_tracer::extractFiberAnchors(
        grid, options,
        [](const auto& indices, int, auto& samples) {
            samples.clear();
            for (const auto& index : indices) {
                const bool fiber = index[1] == 3 || index[1] == 4;
                samples.push_back({{1.0, 0.0, 0.0}, fiber ? 1.0 : 0.0, true});
            }
        });
    CHECK(report.diagnostics.nmsSuppressedComponents >= 1);
    CHECK(report.diagnostics.oneAnchorCells == 2);
    CHECK(report.nonEmptyCells.size() == 2);
    CHECK(report.nonEmptyCells[0].cellZYX[2] != report.nonEmptyCells[1].cellZYX[2]);
}

TEST_CASE("fiber anchor NMS preserves crossing directions")
{
    auto options = config();
    options.minimumAlignedSupport = 0.001;
    const vc::fiber_tracer::FiberPredictionGridInfo grid{{4, 4, 4}, 1.0};
    const auto report = vc::fiber_tracer::extractFiberAnchors(
        grid, options,
        [](const auto& indices, int, auto& samples) {
            samples.clear();
            for (const auto& index : indices) {
                samples.push_back({
                    index[2] < 2 ? cv::Vec3d{1.0, 0.0, 0.0} : cv::Vec3d{0.0, 1.0, 0.0},
                    1.0,
                    true});
            }
        });
    REQUIRE(report.nonEmptyCells.size() == 1);
    CHECK(report.nonEmptyCells[0].retainedAnchorCount == 2);
    CHECK(report.diagnostics.nmsSuppressedComponents == 0);
}

TEST_CASE("fiber anchor local-max NMS uses inclusive geometry and original candidates")
{
    auto options = config();
    options.localWindowRadiusPredictionVoxels = 4.0;
    options.nmsTransverseRadiusPredictionVoxels = 2.0;
    options.nmsLongitudinalRadiusPredictionVoxels = 1.0;
    const auto candidate = [](size_t cellX, cv::Vec3d position, cv::Vec3d axis, double support) {
        vc::fiber_tracer::FiberCellAnchorResult cell;
        cell.cellZYX = {0, 0, cellX};
        cell.retainedAnchorCount = 1;
        auto& component = cell.components[0];
        component.retained = true;
        component.anchor.cellZYX = cell.cellZYX;
        component.anchor.positionPredictionXYZ = position;
        component.anchor.axisXYZ = axis;
        component.anchor.alignedSupport = support;
        component.anchor.directionalCoherence = 1.0;
        component.anchor.refinementScore = support;
        component.rejectionReason.clear();
        cell.components[1].rejectionReason = "empty";
        return cell;
    };

    std::vector<vc::fiber_tracer::FiberCellAnchorResult> chain{
        candidate(0, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0.9),
        candidate(1, {0.0, 1.5, 0.0}, {1.0, 0.0, 0.0}, 0.8),
        candidate(2, {0.0, 3.0, 0.0}, {1.0, 0.0, 0.0}, 0.7),
    };
    vc::fiber_tracer::suppressFiberAnchorDuplicates(chain, options);
    CHECK(chain[0].retainedAnchorCount == 1);
    CHECK(chain[1].retainedAnchorCount == 0);
    CHECK(chain[2].retainedAnchorCount == 0);

    auto sameCell = candidate(
        0, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0.9);
    sameCell.components[1] = sameCell.components[0];
    sameCell.components[1].anchor.axisXYZ = directionAtDegrees(5.0);
    sameCell.components[1].anchor.alignedSupport = 0.8;
    sameCell.retainedAnchorCount = 2;
    std::vector<vc::fiber_tracer::FiberCellAnchorResult> sameCellModes{
        sameCell};
    vc::fiber_tracer::suppressFiberAnchorDuplicates(sameCellModes, options);
    CHECK(sameCellModes[0].retainedAnchorCount == 2);
    CHECK(sameCellModes[0].components[0].retained);
    CHECK(sameCellModes[0].components[1].retained);

    std::vector<vc::fiber_tracer::FiberCellAnchorResult> thresholds{
        candidate(0, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0.9),
        candidate(1, {1.0, 2.0, 0.0}, {1.0, 0.0, 0.0}, 0.8),
        candidate(2, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, 0.7),
        candidate(3, {3.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0.6),
    };
    vc::fiber_tracer::suppressFiberAnchorDuplicates(thresholds, options);
    CHECK(thresholds[0].retainedAnchorCount == 1);
    CHECK(thresholds[1].retainedAnchorCount == 0);
    CHECK(thresholds[2].retainedAnchorCount == 1);
    CHECK(thresholds[3].retainedAnchorCount == 1);

    std::vector<vc::fiber_tracer::FiberCellAnchorResult> transverseOutside{
        candidate(0, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0.9),
        candidate(1, {0.0, 2.1, 0.0}, {1.0, 0.0, 0.0}, 0.8),
    };
    vc::fiber_tracer::suppressFiberAnchorDuplicates(transverseOutside, options);
    CHECK(transverseOutside[0].retainedAnchorCount == 1);
    CHECK(transverseOutside[1].retainedAnchorCount == 1);

    std::vector<vc::fiber_tracer::FiberCellAnchorResult> longitudinalOutside{
        candidate(0, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0.9),
        candidate(1, {1.1, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0.8),
    };
    vc::fiber_tracer::suppressFiberAnchorDuplicates(longitudinalOutside, options);
    CHECK(longitudinalOutside[0].retainedAnchorCount == 1);
    CHECK(longitudinalOutside[1].retainedAnchorCount == 1);
}

TEST_CASE("fiber anchor NMS defaults are independent of refinement and cell size")
{
    for (const int cellSize : {2, 4, 8}) {
        FiberAnchorConfig options;
        options.cellSizePredictionVoxels = cellSize;
        options.localWindowRadiusPredictionVoxels =
            static_cast<double>(cellSize);
        CHECK(options.nmsTransverseRadiusPredictionVoxels == 2.0);
        CHECK(options.nmsLongitudinalRadiusPredictionVoxels == 1.0);
    }
    FiberAnchorConfig invalid;
    invalid.nmsTransverseRadiusPredictionVoxels = -1.0;
    std::vector<vc::fiber_tracer::FiberCellAnchorResult> cells;
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::suppressFiberAnchorDuplicates(cells, invalid),
        doctest::Contains("NMS transverse radius"), std::invalid_argument);
}

TEST_CASE("fiber anchor default falloff gives approximately uniform interior lattice coverage")
{
    const auto options = config();
    const double cellSize = options.cellSizePredictionVoxels;
    const double cutoff = options.gaussianCutoffSigmas *
        options.gaussianSigmaPredictionVoxels;
    for (const cv::Vec3d axis : std::array{
             cv::Vec3d{1.0, 0.0, 0.0},
             cv::Vec3d{1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0), 0.0},
             cv::Vec3d{1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)}}) {
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = 0.0;
        for (int z = 8; z < 12; ++z) {
            for (int y = 8; y < 12; ++y) {
                for (int x = 8; x < 12; ++x) {
                    double coverage = 0.0;
                    const cv::Vec3d point{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)};
                    for (int cz = -2; cz < 6; ++cz) {
                        for (int cy = -2; cy < 6; ++cy) {
                            for (int cx = -2; cx < 6; ++cx) {
                                const cv::Vec3d center{
                                    cx * cellSize + (cellSize - 1.0) * 0.5,
                                    cy * cellSize + (cellSize - 1.0) * 0.5,
                                    cz * cellSize + (cellSize - 1.0) * 0.5,
                                };
                                const cv::Vec3d delta = point - center;
                                const double axial = delta.dot(axis);
                                if (std::abs(axial) > options.axialSupportHalfWidthPredictionVoxels)
                                    continue;
                                const cv::Vec3d transverse = delta - axis * axial;
                                const double distanceSquared = transverse.dot(transverse);
                                if (distanceSquared > cutoff * cutoff)
                                    continue;
                                coverage += std::exp(-distanceSquared /
                                    (2.0 * options.gaussianSigmaPredictionVoxels *
                                     options.gaussianSigmaPredictionVoxels));
                            }
                        }
                    }
                    minimum = std::min(minimum, coverage);
                    maximum = std::max(maximum, coverage);
                }
            }
        }
        CHECK(maximum / minimum <= 1.35);
    }
}

TEST_CASE("fiber anchor cropped NMS includes suppressors outside the selected cells")
{
    auto options = config();
    options.minimumAlignedSupport = 0.001;
    options.nmsTransverseRadiusPredictionVoxels = 4.0;
    const vc::fiber_tracer::FiberPredictionGridInfo grid{{4, 8, 4}, 1.0};
    const auto sampler = [](const auto& indices, int, auto& samples) {
        samples.clear();
        for (const auto& index : indices) {
            const double presence = index[1] == 2 ? 1.0 :
                (index[1] == 5 ? 0.5 : 0.0);
            samples.push_back({{1.0, 0.0, 0.0}, presence, true});
        }
    };
    const auto full = vc::fiber_tracer::extractFiberAnchors(grid, options, sampler);
    const auto cropped = vc::fiber_tracer::extractFiberAnchors(
        grid, options, sampler,
        vc::fiber_tracer::FiberAnchorCrop{{0, 4, 0}, {4, 4, 4}});
    REQUIRE(full.diagnostics.nmsSuppressedComponents >= 1);
    CHECK(cropped.diagnostics.totalCells == 1);
    CHECK(cropped.diagnostics.zeroAnchorCells == 1);
    CHECK(cropped.diagnostics.nmsSuppressedComponents == 1);
    CHECK(cropped.nonEmptyCells.empty());
    const auto& selection = cropped.diagnosticStages[static_cast<size_t>(
        vc::fiber_tracer::FiberAnchorDiagnosticStage::Selection)];
    REQUIRE(selection.size() == 1);
    REQUIRE(selection[0].transition.suppressor.has_value());
    CHECK(selection[0].transition.reason == "nms_suppressed");
    CHECK(selection[0].transition.suppressor->externalContext);
    CHECK(selection[0].transition.suppressor->cellZYX !=
          selection[0].cellZYX);
}

TEST_CASE("fiber anchor artifacts are deterministic across outer worker counts")
{
    const vc::fiber_tracer::FiberPredictionGridInfo grid{{8, 8, 8}, 2.0};
    std::atomic<bool> allSamplerCallsSingleThreaded{true};
    std::atomic<size_t> samplerCalls{0};
    const auto sampler = [&](const auto& indices, int threads, auto& samples) {
        if (threads != 1)
            allSamplerCallsSingleThreaded.store(false);
        samplerCalls.fetch_add(1);
        samples.clear();
        for (const auto& index : indices) {
            const bool fiber = index[1] == 3 || index[1] == 4;
            samples.push_back({{1.0, 0.0, 0.0}, fiber ? 1.0 : 0.0, true});
        }
    };
    auto one = config();
    one.minimumAlignedSupport = 0.001;
    one.gaussianSigmaPredictionVoxels = 0.5;
    one.peakSigmaPredictionVoxels = 3.0;
    one.parallelThreads = 1;
    auto two = one;
    two.parallelThreads = 7;
    const auto first = vc::fiber_tracer::extractFiberAnchors(grid, one, sampler);
    const auto second = vc::fiber_tracer::extractFiberAnchors(grid, two, sampler);
    CHECK(allSamplerCallsSingleThreaded.load());
    CHECK(samplerCalls.load() == 2);
    vc::fiber_tracer::FiberAnchorArtifactInfo artifact;
    artifact.sourceLocator = "/tmp/fiber.lasagna.json";
    artifact.manifestContentHash = "fnv1a64:0123456789abcdef";
    const auto firstJson = vc::fiber_tracer::fiberAnchorReportJson(first, artifact);
    const auto secondJson = vc::fiber_tracer::fiberAnchorReportJson(second, artifact);
    CHECK(firstJson.dump() == secondJson.dump());
    CHECK(vc::fiber_tracer::fiberAnchorReportObj(first, artifact) ==
        vc::fiber_tracer::fiberAnchorReportObj(second, artifact));

    const auto firstDirectory = temporaryDirectory("block_one");
    const auto secondDirectory = temporaryDirectory("block_two");
    vc::fiber_tracer::writeFiberAnchorArtifacts(firstDirectory, first, artifact);
    vc::fiber_tracer::writeFiberAnchorArtifacts(secondDirectory, second, artifact);
    const auto read = [](const std::filesystem::path& path) {
        std::ifstream input(path);
        return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    };
    for (const auto* name : {
             "anchors.json",
             "anchors.obj",
             "anchors_0.obj",
             "anchors_1.obj",
             "anchor_cells.obj",
             "stages/initialized.json",
             "stages/refined.json",
             "stages/support.json",
             "stages/selection.json",
             "stages/nms.json",
         })
        CHECK(read(firstDirectory / name) == read(secondDirectory / name));
    std::filesystem::remove_all(firstDirectory);
    std::filesystem::remove_all(secondDirectory);
}

TEST_CASE("adjacent anchor cells share one dense tile sample")
{
    auto options = config();
    options.parallelThreads = 8;
    std::atomic<size_t> samplerCalls{0};
    std::atomic<size_t> submittedSamples{0};
    const auto report =
        vc::fiber_tracer::extractRefinedFiberAnchorsForCells(
            {{64, 64, 64}, 1.0}, options,
            [&](const auto& indices, int threads, auto& samples) {
                CHECK(threads == 1);
                ++samplerCalls;
                submittedSamples.fetch_add(indices.size());
                CHECK(std::is_sorted(indices.begin(), indices.end()));
                CHECK(std::adjacent_find(indices.begin(), indices.end()) ==
                    indices.end());
                samples.assign(indices.size(), {
                    cv::Vec3d{1.0, 0.0, 0.0}, 1.0, true});
            },
            {{8, 8, 8}, {8, 8, 9}, {8, 9, 8}, {9, 8, 8}});

    CHECK(report.diagnostics.totalCells == 4);
    CHECK(samplerCalls.load() == 1);
    CHECK(submittedSamples.load() < 4 * 46 * 46 * 46);
}

TEST_CASE("fiber anchor extraction enforces its concurrent sample budget")
{
    auto options = config();
    options.maximumConcurrentSampleBytes = 1;
    bool sampled = false;
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::extractFiberAnchors(
            {{8, 8, 8}, 1.0}, options,
            [&](const auto&, int, auto&) { sampled = true; }),
        doctest::Contains("byte limit"),
        std::runtime_error);
    CHECK_FALSE(sampled);
}

TEST_CASE("parallel anchor extraction reports the lowest canonical cell failure")
{
    auto options = config();
    options.parallelThreads = 8;
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::extractRefinedFiberAnchorsForCells(
            {{64, 64, 64}, 1.0},
            options,
            [](const auto& indices, int threads, auto&) {
                if (threads != 1)
                    throw std::runtime_error("nested sampler threads used");
                if (indices.front()[2] == 0)
                    throw std::runtime_error("canonical cell zero failed");
                throw std::runtime_error("later cell failed");
            },
            {{0, 0, 10}, {0, 0, 0}}),
        doctest::Contains("canonical cell zero failed"),
        std::runtime_error);
}

TEST_CASE("fiber anchor extraction keeps owned voxels with a narrower custom support kernel")
{
    auto options = config();
    options.gaussianSigmaPredictionVoxels = 0.1;
    options.localWindowRadiusPredictionVoxels = 0.1;
    options.axialSupportHalfWidthPredictionVoxels = 0.1;
    CHECK_NOTHROW(vc::fiber_tracer::extractFiberAnchors(
        {{4, 4, 4}, 1.0}, options,
        [](const auto& indices, int, auto& samples) {
            samples.assign(indices.size(), {
                cv::Vec3d{1.0, 0.0, 0.0}, 1.0, true});
        }));
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
    CHECK(json.at("version") == 2);
    CHECK(json.at("coordinates").at("position_space") == "base_volume");
    CHECK(json.at("coordinates").at("prediction_to_base_scale") == 2.0);
    CHECK(json.at("parameters").at("peak_sigma_prediction_voxels") == 1.5);
    CHECK(json.at("parameters").at("peak_axial_sigma_prediction_voxels") == 6.0);
    CHECK(json.at("parameters").at("peak_grid_step_prediction_voxels") == 0.5);
    CHECK(json.at("parameters").at("peak_gradient_weight") == 1.0);
    CHECK(json.at("parameters").at("peak_gradient_reliability_scale") == 0.05);
    CHECK(json.at("parameters").at("nms_transverse_radius_prediction_voxels") == 2.0);
    CHECK(json.at("parameters").at("nms_longitudinal_radius_prediction_voxels") == 1.0);
    CHECK(json.at("selection").contains("prediction_interval_origin_base_xyz"));
    CHECK(json.at("selection").contains("prediction_interval_size_base_xyz"));
    CHECK_FALSE(json.at("selection").contains("crop_origin_xyz"));
    REQUIRE(json.at("cells").size() == 1);
    const auto& anchor = json.at("cells").at(0).at("components").at(0);
    CHECK(anchor.contains("position_base_xyz"));
    CHECK_FALSE(anchor.contains("position_prediction_xyz"));
    CHECK_FALSE(anchor.contains("discrete_peak_position_prediction_xyz"));
    CHECK_FALSE(anchor.contains("separable_peak_position_prediction_xyz"));
    CHECK_FALSE(anchor.contains("joint_peak_position_prediction_xyz"));
    const std::string obj = vc::fiber_tracer::fiberAnchorReportObj(report, artifact);
    CHECK(obj.find("g cell_0_0_0_anchor_0") != std::string::npos);
    CHECK(obj.find("\nl 1 2\n") != std::string::npos);
    const std::string cellObj =
        vc::fiber_tracer::fiberAnchorCellReportObj(report);
    CHECK(cellObj.starts_with("# vc_fiberlet_anchor_cells version 1\n"));
    CHECK(occurrenceCount(cellObj, "\np ") == 1);
    CHECK(occurrenceCount(cellObj, "\nl ") == 1);

    auto parallelConfig = config();
    parallelConfig.parallelThreads = 7;
    const auto parallelReport = vc::fiber_tracer::extractFiberAnchors(grid, parallelConfig, [](const auto& indices, int, auto& samples) {
        samples.assign(indices.size(), {cv::Vec3d{1.0, 0.0, 0.0}, 1.0, true});
    });
    CHECK(vc::fiber_tracer::fiberAnchorReportJson(parallelReport, artifact).dump() == json.dump());
    CHECK(vc::fiber_tracer::fiberAnchorReportObj(parallelReport, artifact) == obj);
    for (size_t index = 0;
         index < vc::fiber_tracer::kFiberAnchorDiagnosticStageCount; ++index) {
        const auto stage = static_cast<
            vc::fiber_tracer::FiberAnchorDiagnosticStage>(index);
        CHECK(vc::fiber_tracer::fiberAnchorDiagnosticStageJson(
                  report, artifact, stage)
                  .at("parameters")
                  .at("peak_axial_sigma_prediction_voxels") == 6.0);
        CHECK(vc::fiber_tracer::fiberAnchorDiagnosticStageJson(
                  report, artifact, stage)
                  .at("parameters")
                  .at("nms_transverse_radius_prediction_voxels") == 2.0);
        const auto stageJson = vc::fiber_tracer::fiberAnchorDiagnosticStageJson(
            report, artifact, stage);
        if (stage == vc::fiber_tracer::FiberAnchorDiagnosticStage::Refined) {
            for (const auto& record : report.diagnosticStages[index]) {
                if (record.anchor.has_value()) {
                    CHECK(record.discretePeakPositionPredictionXYZ.has_value());
                    CHECK(record.separablePeakPositionPredictionXYZ.has_value());
                    CHECK(record.jointPeakPositionPredictionXYZ.has_value());
                }
            }
        }
        for (const auto& record : stageJson.at("records")) {
            CHECK_FALSE(record.contains("discrete_peak_position_prediction_xyz"));
            CHECK_FALSE(record.contains("separable_peak_position_prediction_xyz"));
            CHECK_FALSE(record.contains("joint_peak_position_prediction_xyz"));
            if (!record.at("geometry").is_null()) {
                CHECK_FALSE(record.at("geometry").contains(
                    "discrete_peak_position_prediction_xyz"));
                CHECK_FALSE(record.at("geometry").contains(
                    "separable_peak_position_prediction_xyz"));
                CHECK_FALSE(record.at("geometry").contains(
                    "joint_peak_position_prediction_xyz"));
            }
        }
        CHECK(
            vc::fiber_tracer::fiberAnchorDiagnosticStageJson(
                parallelReport, artifact, stage).dump() ==
            vc::fiber_tracer::fiberAnchorDiagnosticStageJson(
                report, artifact, stage).dump());
    }

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
        REQUIRE(sample.presenceValid);
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

TEST_CASE("explicit anchor cells remain sparse and filter refinement before NMS")
{
    vc::fiber_tracer::FiberPredictionGridInfo grid;
    grid.shapeZYX = {8, 8, 8};
    grid.predictionToBaseScale = 1.0;
    auto value = config();
    value.localWindowRadiusPredictionVoxels = 2.0;
    value.parallelThreads = 7;
    std::atomic<bool> allSamplerCallsSingleThreaded{true};
    std::vector<vc::fiber_tracer::FiberAnchorProgress> progress;
    const auto report = vc::fiber_tracer::extractFiberAnchorsForCells(
        grid,
        value,
        [&](const auto& indices, int threads, auto& samples) {
            if (threads != 1)
                allSamplerCallsSingleThreaded.store(false);
            samples.clear();
            for (size_t index = 0; index < indices.size(); ++index) {
                samples.push_back({{1.0, 0.0, 0.0}, 1.0, true});
            }
        },
        {{0, 0, 0}, {1, 1, 1}},
        [](const vc::fiber_tracer::FiberAnchor& anchor) {
            return vc::fiber_tracer::FiberAnchorRetainEvaluation{
                anchor.positionPredictionXYZ[0] < 4.0,
                anchor.positionPredictionXYZ[0],
                4.0,
            };
        },
        [&](const vc::fiber_tracer::FiberAnchorProgress& event) {
            progress.push_back(event);
        });

    CHECK(report.selectedCellsZYX ==
          std::vector<std::array<size_t, 3>>{{0, 0, 0}, {1, 1, 1}});
    CHECK(allSamplerCallsSingleThreaded.load());
    CHECK(report.diagnostics.totalCells == 2);
    CHECK(report.diagnostics.outsideSelectionComponents >= 1);
    CHECK(report.profile.selectedCells == 2);
    CHECK(report.profile.workCells ==
          report.profile.selectedCells + report.profile.contextCells);
    CHECK(report.profile.contextCells > 0);
    CHECK(report.profile.tiles > 0);
    CHECK(report.profile.samplingGroups > 0);
    CHECK(report.profile.samplingGroups <= report.profile.tiles);
    CHECK(report.profile.workers > 0);
    CHECK(report.profile.predictionSamplerCalls == report.profile.tiles);
    CHECK(report.profile.submittedPredictionVoxels > 0);
    CHECK(report.profile.uniqueTilePredictionVoxels > 0);
    CHECK(report.profile.uniqueTilePredictionVoxels <=
          report.profile.submittedPredictionVoxels);
    CHECK(report.profile.submittedPredictionVoxels +
              report.profile.reusedPredictionVoxels >=
          report.profile.uniqueTilePredictionVoxels);
    CHECK(report.profile.candidateObservations >=
          report.profile.retainedObservations);
    CHECK(report.profile.retainedObservations > 0);
    CHECK(report.profile.supportStencilCells +
              report.profile.clippedSupportCells ==
          report.profile.workCells);
    CHECK(report.profile.gradientAttempts ==
          report.profile.retainedObservations);
    CHECK(report.profile.validGradients <= report.profile.gradientAttempts);
    CHECK(report.profile.gradientComputations > 0);
    CHECK(report.profile.validGradientComputations <=
          report.profile.gradientComputations);
    CHECK(report.profile.gradientComputations <
          report.profile.gradientAttempts);
    CHECK(report.profile.retainPredicateCalls > 0);
    CHECK(report.profile.setupSeconds >= 0.0);
    CHECK(report.profile.tilePlanningSeconds >= 0.0);
    CHECK(report.profile.cellProcessingSeconds >= 0.0);
    CHECK(report.profile.coordinateConstructionWorkSeconds >= 0.0);
    CHECK(report.profile.predictionSamplingWorkSeconds >= 0.0);
    CHECK(report.profile.gradientConstructionWorkSeconds >= 0.0);
    CHECK(report.profile.observationConstructionWorkSeconds >= 0.0);
    CHECK(report.profile.fittingWorkSeconds >= 0.0);
    CHECK(report.profile.fit.invocations == report.profile.workCells);
    CHECK(report.profile.fit.nonemptyCells > 0);
    CHECK(report.profile.fit.nonemptyCells <= report.profile.fit.invocations);
    CHECK(report.profile.fit.weightedObservations > 0);
    CHECK(report.profile.fit.seedPairs > 0);
    CHECK(report.profile.fit.peakComputedGridResponses > 0);
    CHECK(report.profile.fit.localTensorProposalWorkSeconds > 0.0);
    CHECK(report.profile.fit.localCentroidProposalWorkSeconds > 0.0);
    CHECK(report.profile.fit.localStateEvaluationWorkSeconds > 0.0);
    CHECK(report.profile.duplicateSuppressionSeconds >= 0.0);
    CHECK(report.profile.elapsedCpuSeconds >= 0.0);
    const auto& initialized = report.diagnosticStages[static_cast<size_t>(
        vc::fiber_tracer::FiberAnchorDiagnosticStage::Initialized)];
    const auto& refined = report.diagnosticStages[static_cast<size_t>(
        vc::fiber_tracer::FiberAnchorDiagnosticStage::Refined)];
    const auto& support = report.diagnosticStages[static_cast<size_t>(
        vc::fiber_tracer::FiberAnchorDiagnosticStage::Support)];
    const auto& selection = report.diagnosticStages[static_cast<size_t>(
        vc::fiber_tracer::FiberAnchorDiagnosticStage::Selection)];
    const auto& nms = report.diagnosticStages[static_cast<size_t>(
        vc::fiber_tracer::FiberAnchorDiagnosticStage::Nms)];
    CHECK(initialized.size() == 4);
    CHECK(refined.size() == 2);
    CHECK(support.size() == 2);
    CHECK(selection.size() == 1);
    CHECK(nms.size() == 1);
    const auto outside = std::find_if(support.begin(), support.end(), [](const auto& record) {
        return record.transition.reason == "outside_selection";
    });
    REQUIRE(outside != support.end());
    CHECK(outside->transition.testedValue.has_value());
    CHECK(outside->transition.threshold == 4.0);
    for (const auto& cell : report.nonEmptyCells)
        CHECK(cell.cellZYX == std::array<size_t, 3>{0, 0, 0});
    REQUIRE(progress.size() >= 2);
    CHECK(progress.front().phase == "anchor_cells");
    CHECK(progress.front().completed == 0);
    CHECK(progress.front().total >= 2);
    CHECK(progress.back().phase == "anchor_cells");
    CHECK(progress.back().completed == progress.back().total);
    const std::string cellObj =
        vc::fiber_tracer::fiberAnchorCellReportObj(report);
    CHECK(occurrenceCount(cellObj, "\np ") == 2);
    CHECK(occurrenceCount(cellObj, "\nl ") == 1);
}

TEST_CASE("interior anchor cells reuse the canonical support stencil")
{
    const vc::fiber_tracer::FiberPredictionGridInfo grid{{64, 64, 64}, 1.0};
    auto serialConfig = config();
    serialConfig.parallelThreads = 1;
    auto parallelConfig = serialConfig;
    parallelConfig.parallelThreads = 7;
    const std::vector<std::array<size_t, 3>> cells{
        {6, 6, 6},
        {6, 6, 7},
    };
    const auto sampler = [](const auto& indices, int, auto& samples) {
        samples.assign(
            indices.size(),
            vc::fiber_tracer::FiberStoredPredictionSample{
                {1.0, 0.0, 0.0}, 1.0, true});
    };
    const auto serial =
        vc::fiber_tracer::extractRefinedFiberAnchorsForCells(
            grid, serialConfig, sampler, cells);
    const auto parallel =
        vc::fiber_tracer::extractRefinedFiberAnchorsForCells(
            grid, parallelConfig, sampler, cells);

    for (const auto* report : {&serial, &parallel}) {
        CHECK(report->profile.workCells == cells.size());
        CHECK(report->profile.supportStencilCells == cells.size());
        CHECK(report->profile.clippedSupportCells == 0);
        CHECK(report->profile.candidateObservations >=
              report->profile.retainedObservations);
        CHECK(report->profile.gradientAttempts ==
              report->profile.retainedObservations);
    }
    CHECK(serial.profile.candidateObservations ==
          parallel.profile.candidateObservations);
    CHECK(serial.profile.retainedObservations ==
          parallel.profile.retainedObservations);
    CHECK(serial.profile.validGradients == parallel.profile.validGradients);
    CHECK(serial.nonEmptyCells.size() == parallel.nonEmptyCells.size());
    for (size_t index = 0; index < serial.nonEmptyCells.size(); ++index) {
        CHECK(serial.nonEmptyCells[index].cellZYX ==
              parallel.nonEmptyCells[index].cellZYX);
        CHECK(serial.nonEmptyCells[index].retainedAnchorCount ==
              parallel.nonEmptyCells[index].retainedAnchorCount);
    }
}

TEST_CASE("anchor support stencil falls back only for clipped cells")
{
    vc::fiber_tracer::FiberPredictionGridInfo grid{{7, 7, 7}, 1.0};
    auto value = config();
    value.cellSizePredictionVoxels = 2;
    value.localWindowRadiusPredictionVoxels = 0.1;
    value.gaussianSigmaPredictionVoxels = 0.1;
    value.peakSigmaPredictionVoxels = 0.1;
    value.axialSupportHalfWidthPredictionVoxels = 0.2;
    value.peakAxialSigmaPredictionVoxels = 0.2;
    value.gaussianCutoffSigmas = 1.0;
    value.peakGridStepPredictionVoxels = 0.1;
    const auto report =
        vc::fiber_tracer::extractRefinedFiberAnchorsForCells(
            grid,
            value,
            [](const auto& indices, int, auto& samples) {
                samples.assign(
                    indices.size(),
                    vc::fiber_tracer::FiberStoredPredictionSample{
                        {1.0, 0.0, 0.0}, 1.0, true});
            },
            {{1, 1, 1}, {3, 1, 1}});

    CHECK(report.profile.workCells == 2);
    CHECK(report.profile.supportStencilCells == 1);
    CHECK(report.profile.clippedSupportCells == 1);
}

TEST_CASE("adjacent anchor tile groups reuse overlapping prediction halos")
{
    vc::fiber_tracer::FiberPredictionGridInfo grid;
    grid.shapeZYX = {32, 8, 8};
    grid.predictionToBaseScale = 1.0;
    auto value = config();
    value.localWindowRadiusPredictionVoxels = 2.0;
    value.parallelThreads = 2;

    const auto report = vc::fiber_tracer::extractFiberAnchorsForCells(
        grid,
        value,
        [](const auto& indices, int, auto& samples) {
            samples.assign(
                indices.size(),
                vc::fiber_tracer::FiberStoredPredictionSample{
                    {1.0, 0.0, 0.0}, 1.0, true});
        },
        {{5, 1, 1}, {6, 1, 1}});

    CHECK(report.profile.tiles >= 2);
    CHECK(report.profile.samplingGroups < report.profile.tiles);
    CHECK(report.profile.reusedPredictionVoxels > 0);
    CHECK(report.profile.uniqueTilePredictionVoxels <=
          report.profile.submittedPredictionVoxels);
}

TEST_CASE("anchor diagnostics retain unavailable attempts in zero-anchor cells")
{
    const auto report = vc::fiber_tracer::extractFiberAnchors(
        {{4, 4, 4}, 2.0},
        config(),
        [](const auto& indices, int, auto& samples) {
            samples.assign(indices.size(), {
                cv::Vec3d{1.0, 0.0, 0.0}, 0.0, true});
        });
    const auto& initialized = report.diagnosticStages[static_cast<size_t>(
        vc::fiber_tracer::FiberAnchorDiagnosticStage::Initialized)];
    REQUIRE(initialized.size() == 2);
    for (size_t index = 0; index < initialized.size(); ++index) {
        CHECK(initialized[index].candidateId == index);
        CHECK_FALSE(initialized[index].anchor.has_value());
        CHECK(initialized[index].transition.outcome == "rejected");
        CHECK(initialized[index].transition.reason == "empty");
    }
    for (size_t stage = 1;
         stage < vc::fiber_tracer::kFiberAnchorDiagnosticStageCount; ++stage) {
        CHECK(report.diagnosticStages[stage].empty());
    }
}

TEST_CASE("reference fiber cell selection uses exact closed ownership boxes")
{
    vc::fiber_tracer::FiberPredictionGridInfo grid{{8, 8, 8}, 2.0};
    const auto cells = vc::fiber_tracer::fiberAnchorCellsNearPolyline(
        {{7.0, 0.0, 2.0}, {7.0, 2.0, 2.0}}, 0.0, grid, 2);

    REQUIRE(cells.size() == 2);
    CHECK(cells[0] == std::array<size_t, 3>{0, 0, 1});
    CHECK(cells[1] == std::array<size_t, 3>{0, 0, 2});
}

TEST_CASE("refined anchor benchmark ignores all later filtering state")
{
    vc::fiber_tracer::FiberAnchorExtractionReport report;
    report.grid = {{8, 8, 8}, 2.0};
    report.selectedCellsZYX = {{0, 0, 0}, {0, 0, 1}, {0, 0, 2}};
    const auto record = [](std::array<size_t, 3> cell,
                           cv::Vec3d discrete,
                           cv::Vec3d separable,
                           cv::Vec3d joint) {
        vc::fiber_tracer::FiberAnchorDiagnosticRecord result;
        result.cellZYX = cell;
        result.anchor = vc::fiber_tracer::FiberAnchor{};
        result.anchor->cellZYX = cell;
        result.anchor->positionPredictionXYZ = separable;
        result.discretePeakPositionPredictionXYZ = discrete;
        result.separablePeakPositionPredictionXYZ = separable;
        result.jointPeakPositionPredictionXYZ = joint;
        result.transition.outcome = "rejected";
        result.transition.reason = "below_support";
        return result;
    };
    report.diagnosticStages[static_cast<size_t>(
        vc::fiber_tracer::FiberAnchorDiagnosticStage::Refined)] = {
        record({0, 0, 0}, {2.5, 2.5, 0.0}, {2.5, 1.5, 0.0}, {2.5, 2.0, 0.0}),
        record({0, 0, 0}, {2.5, 4.5, 0.0}, {2.5, 3.5, 0.0}, {2.5, 5.0, 0.0}),
        record({0, 0, 1}, {2.5, 4.5, 0.0}, {2.5, 5.0, 0.0}, {2.5, 4.0, 0.0}),
    };
    report.nonEmptyCells.clear();
    report.diagnostics.zeroAnchorCells = 3;

    const auto measured = vc::fiber_tracer::benchmarkRefinedFiberAnchors(
        report, {{0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}}, {4.0, 8.0});

    const auto checkPopulation = [](const auto& stage) {
        CHECK(stage.referenceCells == 3);
        CHECK(stage.cellsWithRefinedAnchors == 2);
        CHECK(stage.refinedAnchors == 3);
    };
    checkPopulation(measured.discrete);
    checkPopulation(measured.separable1d);
    checkPopulation(measured.joint2d);

    CHECK(measured.discrete.anchorDistancesBaseVoxels.minimum == 5.0);
    CHECK(measured.discrete.anchorDistancesBaseVoxels.mean == doctest::Approx(23.0 / 3.0));
    CHECK(measured.discrete.anchorDistancesBaseVoxels.median == 9.0);
    CHECK(measured.discrete.anchorDistancesBaseVoxels.percentile95 == 9.0);
    CHECK(measured.discrete.anchorDistancesBaseVoxels.maximum == 9.0);
    REQUIRE(measured.discrete.thresholds.size() == 2);
    CHECK(measured.discrete.thresholds[0].anchorHits == 0);
    CHECK(measured.discrete.thresholds[0].anchorHitRate == 0.0);
    CHECK(measured.discrete.thresholds[0].cellHits == 0);
    CHECK(measured.discrete.thresholds[0].cellHitRate == 0.0);
    CHECK(measured.discrete.thresholds[1].anchorHits == 1);
    CHECK(measured.discrete.thresholds[1].anchorHitRate == doctest::Approx(1.0 / 3.0));
    CHECK(measured.discrete.thresholds[1].cellHits == 1);
    CHECK(measured.discrete.thresholds[1].cellHitRate == doctest::Approx(1.0 / 3.0));

    CHECK(measured.separable1d.anchorDistancesBaseVoxels.minimum == 3.0);
    CHECK(measured.separable1d.anchorDistancesBaseVoxels.mean == doctest::Approx(20.0 / 3.0));
    CHECK(measured.separable1d.anchorDistancesBaseVoxels.median == 7.0);
    CHECK(measured.separable1d.anchorDistancesBaseVoxels.percentile95 == doctest::Approx(9.7));
    CHECK(measured.separable1d.anchorDistancesBaseVoxels.maximum == 10.0);
    REQUIRE(measured.separable1d.thresholds.size() == 2);
    CHECK(measured.separable1d.thresholds[0].anchorHits == 1);
    CHECK(measured.separable1d.thresholds[0].cellHits == 1);
    CHECK(measured.separable1d.thresholds[1].anchorHits == 2);
    CHECK(measured.separable1d.thresholds[1].cellHits == 1);

    CHECK(measured.joint2d.anchorDistancesBaseVoxels.minimum == 4.0);
    CHECK(measured.joint2d.anchorDistancesBaseVoxels.mean == doctest::Approx(22.0 / 3.0));
    CHECK(measured.joint2d.anchorDistancesBaseVoxels.median == 8.0);
    CHECK(measured.joint2d.anchorDistancesBaseVoxels.percentile95 == doctest::Approx(9.8));
    CHECK(measured.joint2d.anchorDistancesBaseVoxels.maximum == 10.0);
    REQUIRE(measured.joint2d.thresholds.size() == 2);
    CHECK(measured.joint2d.thresholds[0].anchorHits == 1);
    CHECK(measured.joint2d.thresholds[0].anchorHitRate == doctest::Approx(1.0 / 3.0));
    CHECK(measured.joint2d.thresholds[0].cellHits == 1);
    CHECK(measured.joint2d.thresholds[0].cellHitRate == doctest::Approx(1.0 / 3.0));
    CHECK(measured.joint2d.thresholds[1].anchorHits == 2);
    CHECK(measured.joint2d.thresholds[1].anchorHitRate == doctest::Approx(2.0 / 3.0));
    CHECK(measured.joint2d.thresholds[1].cellHits == 2);
    CHECK(measured.joint2d.thresholds[1].cellHitRate == doctest::Approx(2.0 / 3.0));
}

TEST_CASE("refined anchor benchmark reports empty anchor population explicitly")
{
    vc::fiber_tracer::FiberAnchorExtractionReport report;
    report.grid = {{4, 4, 4}, 1.0};
    report.selectedCellsZYX = {{0, 0, 0}};

    const auto measured = vc::fiber_tracer::benchmarkRefinedFiberAnchors(
        report, {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}, {4.0, 8.0});

    for (const auto* stage : {
             &measured.discrete, &measured.separable1d, &measured.joint2d}) {
        CHECK(stage->refinedAnchors == 0);
        CHECK_FALSE(stage->anchorDistancesBaseVoxels.mean.has_value());
        REQUIRE(stage->thresholds.size() == 2);
        for (const auto& threshold : stage->thresholds) {
            CHECK(threshold.anchorHits == 0);
            CHECK_FALSE(threshold.anchorHitRate.has_value());
            CHECK(threshold.cellHits == 0);
            CHECK(threshold.cellHitRate == 0.0);
        }
    }
}

TEST_CASE("refined-only extraction skips selection and NMS context")
{
    auto options = config();
    options.parallelThreads = 7;
    std::atomic<bool> allSamplerCallsSingleThreaded{true};
    std::vector<vc::fiber_tracer::FiberAnchorProgress> progress;
    const auto report = vc::fiber_tracer::extractRefinedFiberAnchorsForCells(
        {{8, 8, 8}, 1.0},
        options,
        [&](const auto& indices, int threads, auto& samples) {
            if (threads != 1)
                allSamplerCallsSingleThreaded.store(false);
            samples.assign(indices.size(), {
                cv::Vec3d{1.0, 0.0, 0.0}, 1.0, true});
        },
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}},
        [&](const auto& event) { progress.push_back(event); });

    CHECK(report.diagnostics.totalCells == 4);
    CHECK(allSamplerCallsSingleThreaded.load());
    CHECK_FALSE(report.diagnosticStages[static_cast<size_t>(
        vc::fiber_tracer::FiberAnchorDiagnosticStage::Refined)].empty());
    CHECK(report.diagnosticStages[static_cast<size_t>(
        vc::fiber_tracer::FiberAnchorDiagnosticStage::Support)].empty());
    CHECK(report.diagnosticStages[static_cast<size_t>(
        vc::fiber_tracer::FiberAnchorDiagnosticStage::Selection)].empty());
    CHECK(report.diagnosticStages[static_cast<size_t>(
        vc::fiber_tracer::FiberAnchorDiagnosticStage::Nms)].empty());
    REQUIRE_FALSE(progress.empty());
    CHECK(std::all_of(progress.begin(), progress.end(), [](const auto& event) {
        return event.phase == "selected_cells";
    }));
    CHECK(progress.front().completed == 0);
    CHECK(progress.back().completed == 4);
    CHECK(progress.back().total == 4);
    CHECK(std::is_sorted(
        progress.begin(), progress.end(),
        [](const auto& left, const auto& right) {
            return left.completed < right.completed;
        }));
}
