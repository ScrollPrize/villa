#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberAxisTensor.hpp"
#include "vc/fiber_tracer/FiberGraph.hpp"
#include "vc/fiber_tracer/FiberLocalScoring.hpp"
#include "vc/fiber_tracer/FiberPaths.hpp"
#include "vc/fiber_tracer/FiberletQuantization.hpp"
#include "vc/lasagna/ChannelSampler.hpp"

#include "../src/fiber_tracer/FiberLocalScoringInternal.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>

static_assert(std::is_same_v<
              decltype(vc::fiber_tracer::FiberletPathConfig::longitudinalStepPredictionVoxels),
              float>);
static_assert(std::is_same_v<
              decltype(vc::fiber_tracer::FiberletPathCost::alignment), float>);
static_assert(std::is_same_v<
              decltype(vc::fiber_tracer::FiberletPredictionSample::direction),
              cv::Vec3f>);
static_assert(std::is_same_v<
              decltype(vc::fiber_tracer::FiberletPredictionSample::presence),
              float>);
static_assert(std::is_same_v<
              decltype(vc::fiber_tracer::FiberletCandidateResult::pointsPredictionXYZ),
              std::vector<cv::Vec3f>>);
static_assert(std::is_same_v<
              decltype(vc::fiber_tracer::FiberletGraphEdge::pointsBaseXYZ),
              std::vector<cv::Vec3f>>);
static_assert(std::is_same_v<
              decltype(vc::fiber_tracer::FiberletGraphReplayCost::alignment),
              double>);

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

const ConstantNormalSampler& replayNormals()
{
    static const ConstantNormalSampler sampler;
    return sampler;
}

const ConstantNormalSampler& replayYNormals()
{
    static const ConstantNormalSampler sampler({0.0, 1.0, 0.0});
    return sampler;
}

class CountingNormalSampler final : public vc::lasagna::NormalSampler
{
public:
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override { return {{0.0, 0.0, 1.0}, true, {}}; }

    vc::lasagna::NormalBatchReport sampleNormalBatch(
        const std::vector<cv::Vec3d>& points, bool withDerivative, int parallelThreads, std::vector<vc::lasagna::NormalSampleWithDerivative>& samples) const override
    {
        ++batchCalls;
        requestedThreads = parallelThreads;
        sampledPoints.insert(sampledPoints.end(), points.begin(), points.end());
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
    artifact.report.config.nmsTransverseRadiusPredictionVoxels = 2.0;
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

vc::fiber_tracer::LoadedFiberAnchorArtifact chainAnchorArtifact()
{
    auto artifact = twoAnchorArtifact();
    artifact.report.grid.shapeZYX[2] = 32;
    artifact.report.nonEmptyCells.clear();
    for (size_t index = 0; index < 4; ++index) {
        vc::fiber_tracer::FiberCellAnchorResult cell;
        cell.cellZYX = {2, 2, 1 + 4 * index};
        cell.retainedAnchorCount = 1;
        cell.objective = 1.0;
        cell.components[0].retained = true;
        cell.components[0].assignedObservationCount = 8;
        cell.components[0].anchor.cellZYX = cell.cellZYX;
        cell.components[0].anchor.positionPredictionXYZ = {
            2.5F + 8.0F * static_cast<float>(index), 4.5F, 4.5F};
        cell.components[0].anchor.axisXYZ = {1.0, 0.0, 0.0};
        cell.components[0].anchor.alignedSupport = 1.0;
        cell.components[0].anchor.directionalCoherence = 1.0;
        cell.components[0].anchor.refinementScore = 1.0;
        cell.components[1].rejectionReason = "empty";
        artifact.report.nonEmptyCells.push_back(std::move(cell));
    }
    artifact.report.diagnostics.totalCells = 1024;
    artifact.report.diagnostics.zeroAnchorCells = 1020;
    artifact.report.diagnostics.oneAnchorCells = 4;
    artifact.report.selectedCrop = {{0, 0, 0}, {32, 16, 16}};
    artifact.report.selectedCellBeginZYX = {0, 0, 0};
    artifact.report.selectedCellEndZYX = {8, 8, 16};
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

vc::fiber_tracer::FiberletPathReport graphPathReport()
{
    vc::fiber_tracer::FiberletPathReport report;
    report.grid = {{16, 16, 16}, 1.0};
    report.anchorCellSizePredictionVoxels = 1;
    return report;
}

void addGraphPath(vc::fiber_tracer::FiberletPathReport& report, size_t startId, size_t targetId, std::vector<cv::Vec3f> points, float loss)
{
    REQUIRE(startId < targetId);
    REQUIRE(points.size() >= 2);
    vc::fiber_tracer::FiberletCandidateResult candidate;
    candidate.start = {{0, 0, startId}, 0};
    candidate.target = {{0, 0, targetId}, 0};
    candidate.startPositionPredictionXYZ = points.front();
    candidate.targetPositionPredictionXYZ = points.back();
    candidate.startAxisXYZ = points[1] - points[0];
    candidate.targetAxisXYZ = points.back() - points[points.size() - 2];
    candidate.startPrediction = {{1.0, 0.0, 0.0}, 1.0, true, true};
    candidate.targetPrediction = {{1.0, 0.0, 0.0}, 1.0, true, true};
    candidate.startNormalXYZ = {0.0, 0.0, 1.0};
    candidate.targetNormalXYZ = {0.0, 0.0, 1.0};
    candidate.startNormalValid = true;
    candidate.targetNormalValid = true;
    candidate.searched = true;
    candidate.scoreValid = true;
    candidate.success = true;
    candidate.reason = "success";
    candidate.cost.alignment = loss;
    candidate.pointsPredictionXYZ = std::move(points);
    report.candidates.push_back(std::move(candidate));
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

vc::fiber_tracer::FiberLocalSmoothnessCost legacyFiberLocalSmoothnessCost(
    const cv::Vec3f& previousStepDirection,
    const cv::Vec3f& candidateStepDirection,
    const cv::Vec3f& normal,
    bool normalValid,
    const vc::fiber_tracer::FiberLocalSmoothnessConfig& config)
{
    using vc::fiber_tracer::FiberLocalSmoothnessCost;
    using vc::fiber_tracer::FiberLocalSmoothnessMode;
    constexpr float epsilon = 1.0e-6f;
    constexpr float epsilon2 = epsilon * epsilon;
    const auto clampUnit = [](float value) {
        if (!std::isfinite(value))
            return 0.0f;
        return std::clamp(value, -1.0f, 1.0f);
    };
    const auto normalizeOrZero = [](const cv::Vec3f& value) {
        const float length = std::sqrt(value.dot(value));
        if (!(length > epsilon) || !std::isfinite(length))
            return cv::Vec3f{};
        return value / length;
    };
    const auto angleBetweenUnit = [&](const cv::Vec3f& left,
                                      const cv::Vec3f& right) {
        return std::acos(clampUnit(left.dot(right)));
    };
    const auto excessAngleSquared = [](float angle, float freeAngle) {
        const float excess = std::max(0.0f, angle - freeAngle);
        return excess * excess;
    };

    FiberLocalSmoothnessCost cost;
    if (previousStepDirection.dot(previousStepDirection) <= epsilon2 ||
        candidateStepDirection.dot(candidateStepDirection) <= epsilon2) {
        return cost;
    }

    const float isotropicAngle = angleBetweenUnit(
        previousStepDirection, candidateStepDirection);
    const float isotropic = config.isotropicWeight * excessAngleSquared(
        isotropicAngle, config.freeAngleRadians);
    if (!normalValid || normal.dot(normal) <= epsilon2) {
        cost.isotropic = isotropic;
        cost.mode = FiberLocalSmoothnessMode::IsotropicFallback;
        return cost;
    }

    const float previousNormal = clampUnit(previousStepDirection.dot(normal));
    const float candidateNormal = clampUnit(candidateStepDirection.dot(normal));
    const cv::Vec3f previousTangent = normalizeOrZero(
        previousStepDirection - normal * previousNormal);
    const cv::Vec3f candidateTangent = normalizeOrZero(
        candidateStepDirection - normal * candidateNormal);
    const bool tangentValid =
        previousTangent.dot(previousTangent) > epsilon2 &&
        candidateTangent.dot(candidateTangent) > epsilon2;
    const float tangentAngle = tangentValid
        ? angleBetweenUnit(previousTangent, candidateTangent)
        : isotropicAngle;
    const float normalAngle = std::abs(
        std::asin(candidateNormal) - std::asin(previousNormal));
    cost.tangent = config.tangentWeight * excessAngleSquared(
        tangentAngle, config.freeAngleRadians);
    cost.normal = config.normalWeight * excessAngleSquared(
        normalAngle, config.freeAngleRadians);
    cost.mode = FiberLocalSmoothnessMode::NormalAware;
    return cost;
}

vc::fiber_tracer::FiberLocalMetricCost legacyFiberLocalMetricCostPrepared(
    const vc::fiber_tracer::FiberLocalMetricSample* currentPrediction,
    const vc::fiber_tracer::FiberLocalMetricSample& candidatePrediction,
    const cv::Vec3f& previousStepDirection,
    float previousStepLength,
    const cv::Vec3f& candidateStepDirection,
    float candidateStepLength,
    const cv::Vec3f& normal,
    bool normalValid,
    const vc::fiber_tracer::FiberLocalMetricConfig& config)
{
    vc::fiber_tracer::FiberLocalMetricCost cost;
    if (!candidatePrediction.valid) {
        cost.invalidPrediction = config.invalidPredictionCostPerVoxel *
                                 std::max(0.0f, candidateStepLength);
        return cost;
    }
    const auto clampPositiveUnit = [](float value) {
        if (!std::isfinite(value))
            return 0.0f;
        return std::clamp(value, 0.0f, 1.0f);
    };
    cv::Vec3f currentAxis = previousStepDirection;
    if (currentPrediction != nullptr && currentPrediction->valid) {
        currentAxis = currentPrediction->direction;
        if (currentAxis.dot(previousStepDirection) < 0.0f)
            currentAxis *= -1.0f;
    }
    cv::Vec3f candidateAxis = candidatePrediction.direction;
    if (candidateAxis.dot(candidateStepDirection) < 0.0f)
        candidateAxis *= -1.0f;
    float score = clampPositiveUnit(candidatePrediction.presence);
    score *= clampPositiveUnit(
        previousStepDirection.dot(candidateStepDirection));
    score *= clampPositiveUnit(previousStepDirection.dot(currentAxis));
    score *= clampPositiveUnit(previousStepDirection.dot(candidateAxis));
    score *= clampPositiveUnit(currentAxis.dot(candidateStepDirection));
    score *= clampPositiveUnit(currentAxis.dot(candidateAxis));
    score *= clampPositiveUnit(candidateStepDirection.dot(candidateAxis));
    cost.alignment = (1.0f - score) *
                     std::max(0.0f, candidateStepLength);
    const auto smoothness = legacyFiberLocalSmoothnessCost(
        previousStepDirection, candidateStepDirection,
        normal, normalValid, config.smoothness);
    const float effectiveLength = std::max(
        1.0f,
        (std::max(0.0f, previousStepLength) +
         std::max(0.0f, candidateStepLength)) * 0.5f);
    cost.isotropicSmoothness = smoothness.isotropic / effectiveLength;
    cost.tangentSmoothness = smoothness.tangent / effectiveLength;
    cost.normalSmoothness = smoothness.normal / effectiveLength;
    return cost;
}

void checkMetricCostBits(
    const vc::fiber_tracer::FiberLocalMetricCost& actual,
    const vc::fiber_tracer::FiberLocalMetricCost& expected)
{
    const auto check = [](float actualValue, float expectedValue) {
        CHECK(std::bit_cast<uint32_t>(actualValue) ==
              std::bit_cast<uint32_t>(expectedValue));
    };
    check(actual.invalidPrediction, expected.invalidPrediction);
    check(actual.alignment, expected.alignment);
    check(actual.isotropicSmoothness, expected.isotropicSmoothness);
    check(actual.tangentSmoothness, expected.tangentSmoothness);
    check(actual.normalSmoothness, expected.normalSmoothness);
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

TEST_CASE("lazy fiber local isotropic evaluation preserves legacy branches")
{
    using vc::fiber_tracer::FiberLocalSmoothnessConfig;
    using vc::fiber_tracer::fiberLocalSmoothnessCost;
    const FiberLocalSmoothnessConfig config{2.0f, 0.1f, 10.0f, 0.05f};
    struct TestCase {
        cv::Vec3f previous;
        cv::Vec3f candidate;
        cv::Vec3f normal;
        bool normalValid;
    };
    const std::array cases{
        TestCase{{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
                 {0.0f, 0.0f, 1.0f}, true},
        TestCase{{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
                 {0.0f, 0.0f, 1.0f}, false},
        TestCase{{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
                 {}, true},
        TestCase{{0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f},
                 {0.0f, 0.0f, 1.0f}, true},
        TestCase{{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
                 {0.0f, 0.0f, 1.0f}, true},
        TestCase{{0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f},
                 {0.0f, 0.0f, 1.0f}, true},
    };

    for (const auto& test : cases) {
        const auto expected = legacyFiberLocalSmoothnessCost(
            test.previous, test.candidate, test.normal,
            test.normalValid, config);
        const auto actual = fiberLocalSmoothnessCost(
            test.previous, test.candidate, test.normal,
            test.normalValid, config);
        CHECK(actual.isotropic == expected.isotropic);
        CHECK(actual.tangent == expected.tangent);
        CHECK(actual.normal == expected.normal);
        CHECK(actual.mode == expected.mode);
    }
}

TEST_CASE("prepared fiber local scoring matches validating scoring")
{
    using namespace vc::fiber_tracer;
    const FiberLocalMetricConfig config{
        4.0f,
        FiberLocalSmoothnessConfig{2.0f, 0.1f, 10.0f, 0.05f},
    };
    const auto checkExact = [&](const FiberLocalMetricSample* current,
                                const FiberLocalMetricSample& candidate,
                                const cv::Vec3f& previousStep,
                                float previousLength,
                                const cv::Vec3f& candidateStep,
                                float candidateLength,
                                const cv::Vec3f& normal,
                                bool normalValid) {
        const auto validating = fiberLocalMetricCost(
            current, candidate, previousStep, previousLength,
            candidateStep, candidateLength, normal, normalValid, config);
        FiberLocalMetricSample preparedCurrent;
        const FiberLocalMetricSample* preparedCurrentPointer = nullptr;
        if (current != nullptr) {
            preparedCurrent = *current;
            preparedCurrent.direction =
                prepareFiberLocalUnitDirection(preparedCurrent.direction);
            preparedCurrentPointer = &preparedCurrent;
        }
        FiberLocalMetricSample preparedCandidate = candidate;
        preparedCandidate.direction =
            prepareFiberLocalUnitDirection(preparedCandidate.direction);
        const auto prepared = fiberLocalMetricCostPrepared(
            preparedCurrentPointer, preparedCandidate,
            prepareFiberLocalUnitDirection(previousStep), previousLength,
            prepareFiberLocalUnitDirection(candidateStep), candidateLength,
            normal, normalValid, config);

        CHECK(prepared.invalidPrediction == validating.invalidPrediction);
        CHECK(prepared.alignment == validating.alignment);
        CHECK(prepared.isotropicSmoothness == validating.isotropicSmoothness);
        CHECK(prepared.tangentSmoothness == validating.tangentSmoothness);
        CHECK(prepared.normalSmoothness == validating.normalSmoothness);
    };

    const FiberLocalMetricSample current{{2.0f, 1.0f, 0.0f}, 0.75f, true};
    const FiberLocalMetricSample invalidCurrent{{2.0f, 1.0f, 0.0f}, 0.75f, false};
    const FiberLocalMetricSample candidate{{3.0f, -1.0f, 1.0f}, 0.6f, true};
    const FiberLocalMetricSample flippedCandidate{{-3.0f, 1.0f, -1.0f}, 0.6f, true};
    const FiberLocalMetricSample invalidCandidate{{}, 0.6f, false};
    const FiberLocalMetricSample nonFinitePresence{
        {3.0f, -1.0f, 1.0f},
        std::numeric_limits<float>::quiet_NaN(),
        true};
    const cv::Vec3f previousStep{4.0f, 1.0f, 0.0f};
    const cv::Vec3f candidateStep{2.0f, -0.5f, 0.5f};
    const cv::Vec3f normal{0.0f, 0.0f, 1.0f};

    checkExact(&current, candidate, previousStep, 1.5f,
               candidateStep, 2.25f, normal, true);
    checkExact(nullptr, candidate, previousStep, 1.5f,
               candidateStep, 2.25f, normal, true);
    checkExact(&invalidCurrent, candidate, previousStep, 1.5f,
               candidateStep, 2.25f, {}, false);
    checkExact(&current, flippedCandidate, previousStep, 1.5f,
               candidateStep, 2.25f, {}, false);
    checkExact(&current, invalidCandidate, previousStep, -1.5f,
               candidateStep, -2.25f, normal, true);
    checkExact(&current, candidate, {}, 0.0f,
               {}, 0.0f, normal, true);
    checkExact(&current, nonFinitePresence, previousStep, 1.5f,
               candidateStep, 2.25f, normal, true);
}

TEST_CASE("candidate-prepared fiber local scoring preserves legacy metric branches")
{
    using namespace vc::fiber_tracer;
    const FiberLocalMetricConfig config{
        4.0f,
        FiberLocalSmoothnessConfig{2.0f, 0.1f, 10.0f, 0.05f},
    };
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    const FiberLocalMetricSample current{{0.8f, 0.6f, 0.0f}, 0.75f, true};
    const FiberLocalMetricSample candidate{{0.6f, -0.8f, 0.0f}, 0.6f, true};
    const FiberLocalMetricSample nonFiniteCurrent{
        {nan, 1.0f, 0.0f}, 0.75f, true};
    const FiberLocalMetricSample nonFiniteCandidate{
        {infinity, -1.0f, 0.0f}, 0.6f, true};
    const FiberLocalMetricSample invalidCandidate{{}, 0.6f, false};
    struct TestCase {
        const FiberLocalMetricSample* currentPrediction;
        FiberLocalMetricSample candidatePrediction;
        cv::Vec3f previous;
        float previousLength;
        cv::Vec3f candidate;
        float candidateLength;
        cv::Vec3f normal;
        bool normalValid;
    };
    const std::array cases{
        TestCase{&current, candidate, {1.0f, 0.0f, 0.0f}, 1.5f,
                 {0.0f, 1.0f, 0.0f}, 2.25f,
                 {0.0f, 0.0f, 1.0f}, true},
        TestCase{&current, candidate, {1.0f, 0.0f, 0.0f}, 1.5f,
                 {0.0f, 1.0f, 0.0f}, 2.25f, {}, false},
        TestCase{&current, candidate, {1.0f, 0.0f, 0.0f}, 1.5f,
                 {0.0f, 1.0f, 0.0f}, 2.25f, {}, true},
        TestCase{&current, candidate, {1.0f, 0.0f, 0.0f}, 1.5f,
                 {0.0f, 1.0f, 0.0f}, 2.25f,
                 {infinity, 0.0f, 0.0f}, true},
        TestCase{&current, candidate, {1.0f, 0.0f, 0.0f}, 1.5f,
                 {0.0f, 1.0f, 0.0f}, 2.25f,
                 {nan, 0.0f, 0.0f}, true},
        TestCase{&current, candidate, {0.0f, 0.0f, 1.0f}, 1.5f,
                 {1.0f, 0.0f, 0.0f}, 2.25f,
                 {0.0f, 0.0f, 1.0f}, true},
        TestCase{&current, candidate, {1.0f, 0.0f, 0.0f}, 1.5f,
                 {0.0f, 0.0f, 1.0f}, 2.25f,
                 {0.0f, 0.0f, 1.0f}, true},
        TestCase{&current, candidate, {0.0f, 0.0f, 1.0f}, 1.5f,
                 {0.0f, 0.0f, -1.0f}, 2.25f,
                 {0.0f, 0.0f, 1.0f}, true},
        TestCase{nullptr, candidate, {}, 0.0f,
                 {1.0f, 0.0f, 0.0f}, 2.25f,
                 {0.0f, 0.0f, 1.0f}, true},
        TestCase{nullptr, candidate, {1.0f, 0.0f, 0.0f}, 1.5f,
                 {}, 0.0f, {0.0f, 0.0f, 1.0f}, true},
        TestCase{nullptr, candidate, {nan, 0.0f, 0.0f}, 1.5f,
                 {1.0f, 0.0f, 0.0f}, 2.25f,
                 {0.0f, 0.0f, 1.0f}, true},
        TestCase{nullptr, candidate, {1.0f, 0.0f, 0.0f}, 1.5f,
                 {infinity, 0.0f, 0.0f}, 2.25f,
                 {0.0f, 0.0f, 1.0f}, true},
        TestCase{&current, invalidCandidate, {nan, 0.0f, 0.0f}, -1.5f,
                 {infinity, 0.0f, 0.0f}, -2.25f,
                 {infinity, nan, 0.0f}, true},
        TestCase{&current, candidate, {1.0f, 0.0f, 0.0f}, -1.5f,
                 {0.0f, 1.0f, 0.0f}, -2.25f,
                 {0.0f, 0.0f, 1.0f}, true},
        TestCase{&nonFiniteCurrent, candidate, {1.0f, 0.0f, 0.0f}, 1.5f,
                 {0.0f, 1.0f, 0.0f}, 2.25f,
                 {0.0f, 0.0f, 1.0f}, true},
        TestCase{&current, nonFiniteCandidate, {1.0f, 0.0f, 0.0f}, 1.5f,
                 {0.0f, 1.0f, 0.0f}, 2.25f,
                 {0.0f, 0.0f, 1.0f}, true},
        TestCase{&current, candidate, {-0.0f, 1.0f, 0.0f}, infinity,
                 {1.0f, -0.0f, 0.0f}, infinity,
                 {0.0f, 0.0f, 1.0f}, true},
        TestCase{&current, invalidCandidate, {nan, infinity, -0.0f}, nan,
                 {infinity, nan, -0.0f}, nan,
                 {infinity, nan, -0.0f}, true},
    };

    for (const auto& test : cases) {
        const auto expected = legacyFiberLocalMetricCostPrepared(
            test.currentPrediction, test.candidatePrediction,
            test.previous, test.previousLength,
            test.candidate, test.candidateLength,
            test.normal, test.normalValid, config);
        const auto candidateSmoothness = test.candidatePrediction.valid
            ? detail::prepareFiberLocalCandidateSmoothnessInline(
                  test.candidate, test.normal, test.normalValid)
            : detail::FiberLocalPreparedCandidateSmoothness{};
        const auto actual =
            detail::fiberLocalMetricCostCandidatePreparedInline(
                test.currentPrediction, test.candidatePrediction,
                test.previous, test.previousLength,
                test.candidate, test.candidateLength,
                candidateSmoothness, config);
        checkMetricCostBits(actual, expected);
        if (test.candidatePrediction.valid) {
            const auto candidateMetric =
                detail::prepareFiberLocalCandidateMetricInline(
                    test.candidatePrediction, test.candidate,
                    test.normal, test.normalValid);
            const auto incoming =
                detail::prepareFiberLocalIncomingAlignmentInline(
                    test.currentPrediction, test.previous);
            const auto fullyPrepared =
                detail::fiberLocalMetricCostFullyPreparedInline(
                    incoming, test.previousLength, test.candidateLength,
                    candidateMetric, config);
            checkMetricCostBits(fullyPrepared, expected);
        }
    }
}

TEST_CASE("fully prepared fiber local scoring matches randomized legacy metrics bitwise")
{
    using namespace vc::fiber_tracer;
    const FiberLocalMetricConfig config{
        4.0f,
        FiberLocalSmoothnessConfig{2.0f, 0.1f, 10.0f, 0.05f},
    };
    std::mt19937 generator(0x37a11U);
    const auto sampleValue = [&]() {
        return static_cast<float>(static_cast<int32_t>(generator() % 4001U) -
                                  2000) /
               1000.0f;
    };
    const auto sampleUnit = [&]() {
        cv::Vec3f value{sampleValue(), sampleValue(), sampleValue()};
        if (value.dot(value) < 0.01f)
            value[0] += 1.0f;
        return prepareFiberLocalUnitDirection(value);
    };

    for (size_t index = 0; index < 1024; ++index) {
        const cv::Vec3f previous = sampleUnit();
        const cv::Vec3f candidateDirection = sampleUnit();
        const cv::Vec3f normal = sampleUnit();
        FiberLocalMetricSample current{
            sampleUnit(), sampleValue(), (index % 5) != 0};
        FiberLocalMetricSample candidate{
            sampleUnit(), sampleValue(), true};
        const FiberLocalMetricSample* currentPointer =
            index % 7 == 0 ? nullptr : &current;
        const float previousLength = sampleValue() * 4.0f;
        const float candidateLength = sampleValue() * 4.0f;
        const bool normalValid = index % 6 != 0;

        const auto expected = legacyFiberLocalMetricCostPrepared(
            currentPointer, candidate, previous, previousLength,
            candidateDirection, candidateLength,
            normal, normalValid, config);
        const auto actual = detail::fiberLocalMetricCostFullyPreparedInline(
            detail::prepareFiberLocalIncomingAlignmentInline(
                currentPointer, previous),
            previousLength, candidateLength,
            detail::prepareFiberLocalCandidateMetricInline(
                candidate, candidateDirection, normal, normalValid),
            config);
        checkMetricCostBits(actual, expected);
    }
}

TEST_CASE("batched prepared alignment matches every scalar validity mask bitwise")
{
    using namespace vc::fiber_tracer;
    constexpr size_t capacity = 9;
    std::mt19937 generator(0x38ba7cU);
    const auto sampleValue = [&]() {
        return static_cast<float>(static_cast<int32_t>(generator() % 4001U) -
                                  2000) /
               1000.0f;
    };
    const auto sampleUnit = [&]() {
        cv::Vec3f value{sampleValue(), sampleValue(), sampleValue()};
        if (value.dot(value) < 0.01f)
            value[0] += 1.0f;
        return prepareFiberLocalUnitDirection(value);
    };

    for (uint32_t mask = 0; mask < (1U << capacity); ++mask) {
        const cv::Vec3f previous = sampleUnit();
        FiberLocalMetricSample current{sampleUnit(), sampleValue(), true};
        const auto incoming =
            detail::prepareFiberLocalIncomingAlignmentInline(
                &current, previous);
        std::array<detail::FiberLocalPreparedCandidateMetric, capacity>
            candidates;
        detail::FiberLocalPreparedCandidateAlignmentBatch<capacity> batch;
        for (size_t slot = 0; slot < capacity; ++slot) {
            if ((mask & (1U << slot)) == 0) {
                const float poison = std::numeric_limits<float>::quiet_NaN();
                candidates[slot].direction = {poison, poison, poison};
                candidates[slot].predictionDirection =
                    {poison, poison, poison};
                candidates[slot].presence = poison;
                candidates[slot].directionPredictionAlignment = poison;
                continue;
            }
            const cv::Vec3f candidateDirection = sampleUnit();
            const FiberLocalMetricSample candidate{
                sampleUnit(), sampleValue(), true};
            candidates[slot] = detail::prepareFiberLocalCandidateMetricInline(
                candidate, candidateDirection, {}, false);
            detail::appendFiberLocalCandidateAlignmentInline(
                batch, static_cast<uint8_t>(slot), candidates[slot]);
        }

        std::array<float, capacity> losses;
        losses.fill(std::numeric_limits<float>::quiet_NaN());
        detail::fiberLocalAlignmentLossPreparedBatchInline(
            incoming, batch, losses);
        CHECK(batch.count == static_cast<size_t>(std::popcount(mask)));
        for (size_t lane = 0; lane < batch.count; ++lane) {
            const size_t slot = batch.slotOfLane[lane];
            const float expected =
                detail::fiberLocalAlignmentLossPreparedInline(
                    incoming, candidates[slot]);
            const uint32_t actualBits =
                std::bit_cast<uint32_t>(losses[lane]);
            const uint32_t expectedBits = std::bit_cast<uint32_t>(expected);
            CHECK(actualBits == expectedBits);
            if (lane > 0)
                CHECK(batch.slotOfLane[lane - 1] < slot);
        }
    }

    detail::FiberLocalPreparedCandidateAlignmentBatch<capacity> batch;
    const FiberLocalMetricSample candidate{
        {1.0f, -0.0f, 0.0f},
        std::numeric_limits<float>::infinity(),
        true};
    const auto prepared = detail::prepareFiberLocalCandidateMetricInline(
        candidate, {-0.0f, 1.0f, 0.0f}, {}, false);
    detail::appendFiberLocalCandidateAlignmentInline(batch, 8, prepared);
    const FiberLocalMetricSample current{
        {1.0f, 0.0f, -0.0f}, 1.0f, true};
    const auto incoming = detail::prepareFiberLocalIncomingAlignmentInline(
        &current, {1.0f, -0.0f, 0.0f});
    std::array<float, capacity> losses;
    detail::fiberLocalAlignmentLossPreparedBatchInline(
        incoming, batch, losses);
    CHECK(std::bit_cast<uint32_t>(losses[0]) ==
          std::bit_cast<uint32_t>(
              detail::fiberLocalAlignmentLossPreparedInline(
                  incoming, prepared)));
}

TEST_CASE("batched prepared alignment preserves scalar relaxation choices")
{
    using namespace vc::fiber_tracer;
    constexpr size_t capacity = 9;
    constexpr size_t incomingCount = 5;
    constexpr std::array<uint32_t, 3> masks{
        0b101010101U,
        0b111111110U,
        0b011010011U,
    };
    const FiberLocalMetricConfig config{
        4.0f,
        FiberLocalSmoothnessConfig{2.0f, 0.1f, 10.0f, 0.05f},
    };
    std::mt19937 generator(0x38d9aU);
    const auto sampleValue = [&]() {
        return static_cast<float>(static_cast<int32_t>(generator() % 4001U) -
                                  2000) /
               1000.0f;
    };
    const auto sampleUnit = [&]() {
        cv::Vec3f value{sampleValue(), sampleValue(), sampleValue()};
        if (value.dot(value) < 0.01f)
            value[0] += 1.0f;
        return prepareFiberLocalUnitDirection(value);
    };

    for (const uint32_t mask : masks) {
        std::array<detail::FiberLocalPreparedCandidateMetric, capacity>
            candidates;
        std::array<float, capacity> candidateLengths;
        detail::FiberLocalPreparedCandidateAlignmentBatch<capacity> batch;
        for (size_t slot = 0; slot < capacity; ++slot) {
            if ((mask & (1U << slot)) == 0)
                continue;
            const FiberLocalMetricSample prediction{
                sampleUnit(), sampleValue(), true};
            candidates[slot] = detail::prepareFiberLocalCandidateMetricInline(
                prediction, sampleUnit(), sampleUnit(), slot % 3 != 0);
            candidateLengths[slot] = sampleValue() * 3.0f;
            detail::appendFiberLocalCandidateAlignmentInline(
                batch, static_cast<uint8_t>(slot), candidates[slot]);
        }

        std::array<float, capacity> scalarBest;
        std::array<float, capacity> batchedBest;
        std::array<uint8_t, capacity> scalarPrevious;
        std::array<uint8_t, capacity> batchedPrevious;
        scalarBest.fill(std::numeric_limits<float>::infinity());
        batchedBest.fill(std::numeric_limits<float>::infinity());
        scalarPrevious.fill(std::numeric_limits<uint8_t>::max());
        batchedPrevious.fill(std::numeric_limits<uint8_t>::max());
        size_t scalarRelaxations = 0;
        size_t batchedRelaxations = 0;

        for (size_t previousState = 0; previousState < incomingCount;
             ++previousState) {
            const FiberLocalMetricSample current{
                sampleUnit(), sampleValue(), previousState != 3};
            const auto incoming =
                detail::prepareFiberLocalIncomingAlignmentInline(
                    &current, sampleUnit());
            const float previousLength = sampleValue() * 3.0f;
            const float accumulated = sampleValue() * 5.0f;
            std::array<float, capacity> losses;
            detail::fiberLocalAlignmentLossPreparedBatchInline(
                incoming, batch, losses);

            for (size_t lane = 0; lane < batch.count; ++lane) {
                const size_t slot = batch.slotOfLane[lane];
                const auto scalar =
                    detail::fiberLocalMetricCostFullyPreparedInline(
                        incoming, previousLength, candidateLengths[slot],
                        candidates[slot], config);
                const auto batched =
                    detail::fiberLocalMetricCostFromPreparedAlignmentInline(
                        losses[lane], incoming, previousLength,
                        candidateLengths[slot], candidates[slot], config);
                checkMetricCostBits(batched, scalar);
                const float scalarTotal = accumulated + scalar.total();
                const float batchedTotal = accumulated + batched.total();
                CHECK(std::bit_cast<uint32_t>(batchedTotal) ==
                      std::bit_cast<uint32_t>(scalarTotal));
                if (scalarTotal < scalarBest[slot]) {
                    ++scalarRelaxations;
                    scalarBest[slot] = scalarTotal;
                    scalarPrevious[slot] =
                        static_cast<uint8_t>(previousState);
                }
                if (batchedTotal < batchedBest[slot]) {
                    ++batchedRelaxations;
                    batchedBest[slot] = batchedTotal;
                    batchedPrevious[slot] =
                        static_cast<uint8_t>(previousState);
                }
            }
        }

        CHECK(batchedRelaxations == scalarRelaxations);
        CHECK(batchedPrevious == scalarPrevious);
        for (size_t slot = 0; slot < capacity; ++slot) {
            CHECK(std::bit_cast<uint32_t>(batchedBest[slot]) ==
                  std::bit_cast<uint32_t>(scalarBest[slot]));
        }
    }
}

TEST_CASE("fiber local alignment loss preserves native multiplicative scoring")
{
    using vc::fiber_tracer::fiberLocalAlignmentLoss;
    const cv::Vec3f x{1.0f, 0.0f, 0.0f};
    const cv::Vec3f y{0.0f, 1.0f, 0.0f};
    CHECK(fiberLocalAlignmentLoss(1.0f, x, x, x, x) == 0.0f);
    CHECK(fiberLocalAlignmentLoss(0.25f, x, x, x, x) == 0.75f);
    CHECK(fiberLocalAlignmentLoss(1.0f, x, x, x, y) == 1.0f);
    CHECK(fiberLocalAlignmentLoss(1.0f, x, x, -x, x) == 1.0f);
    const float invSqrt2 = static_cast<float>(std::sqrt(0.5));
    const cv::Vec3f diagonal{invSqrt2, invSqrt2, 0.0f};
    float score = 0.5f;
    score *= x.dot(diagonal);
    score *= x.dot(x);
    score *= x.dot(diagonal);
    score *= x.dot(diagonal);
    score *= x.dot(diagonal);
    score *= diagonal.dot(diagonal);
    CHECK(fiberLocalAlignmentLoss(0.5f, x, diagonal, x, diagonal) == 1.0f - score);
}

TEST_CASE("compact fiber axes preserve unoriented directions")
{
    for (const cv::Vec3d axis : {
             cv::Vec3d{1.0, 0.0, 0.0},
             cv::normalize(cv::Vec3d{1.0, 2.0, 3.0}),
             cv::normalize(cv::Vec3d{-2.0, 1.0, -4.0})}) {
        const auto encoded = vc::lasagna::encodeCompactNormalToRaw(axis);
        REQUIRE(encoded.has_value());
        const cv::Vec3d decoded = vc::lasagna::decodeCompactNormalFromRaw(
            (*encoded)[0], (*encoded)[1]);
        CHECK(decoded[2] >= 0.0);
        CHECK(std::abs(axis.dot(decoded)) > 0.999);
    }
}

TEST_CASE("fiberlet radius-four neighborhood includes all shorter offsets")
{
    const auto offsets = vc::fiber_tracer::fiberletCellNeighborhoodOffsets(4, 0.5);
    REQUIRE(offsets.size() > 6);
    CHECK(offsets.size() == 388);
    CHECK(std::is_sorted(offsets.begin(), offsets.end()));
    const std::set<std::array<int, 3>> unique(offsets.begin(), offsets.end());
    CHECK(unique.size() == offsets.size());
    CHECK(unique.contains({0, 0, 4}));
    CHECK(unique.contains({0, 0, -4}));
    CHECK(unique.contains({0, 0, 1}));
    CHECK(unique.contains({1, 2, 3}));
    CHECK(unique.contains({2, 2, 3}));
    CHECK_FALSE(unique.contains({0, 0, 5}));
    for (const auto& offset : offsets) {
        const double length = std::sqrt(static_cast<double>(offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]));
        CHECK(length > 0.0);
        CHECK(length < 4.5);
        CHECK(unique.contains({-offset[0], -offset[1], -offset[2]}));
    }
}

TEST_CASE("fiberlet sampling batch size must be positive")
{
    auto config = pathConfig();
    config.samplingBatchCoordinates = 0;
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::validateFiberletPathConfig(config), doctest::Contains("sampling batch size must be positive"), std::invalid_argument);
}

TEST_CASE("fiberlet DP follows curved Hermite-normal planes with exact endpoints")
{
    const cv::Vec3d startAxis = cv::normalize(cv::Vec3d{1.0, 0.4, 0.0});
    const cv::Vec3d targetAxis = cv::normalize(cv::Vec3d{1.0, -0.4, 0.0});
    const auto anchors = twoAnchorArtifact(startAxis, targetAxis);
    const ConstantNormalSampler normals;
    auto config = pathConfig();
    config.corridorRadiusPredictionVoxels = 0.01;
    std::vector<vc::fiber_tracer::FiberletPathProgress> progress;
    const auto report =
        vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, config, constantPredictions(), normals, [&](const auto& update) {
            progress.push_back(update);
        });
    REQUIRE(report.diagnostics.generatedPairs == 1);
    REQUIRE_MESSAGE(report.diagnostics.successfulPaths == 1, report.candidates[0].reason);
    const auto& path = report.candidates[0];
    REQUIRE(path.pointsPredictionXYZ.size() >= 3);
    CHECK(path.pointsPredictionXYZ.front() == anchors.report.nonEmptyCells[0].components[0].anchor.positionPredictionXYZ);
    CHECK(path.pointsPredictionXYZ.back() == anchors.report.nonEmptyCells[1].components[0].anchor.positionPredictionXYZ);
    bool usedFloatingPoint = false;
    bool bowedAwayFromChord = false;
    for (size_t index = 1; index + 1 < path.pointsPredictionXYZ.size(); ++index) {
        const auto& point = path.pointsPredictionXYZ[index];
        usedFloatingPoint = usedFloatingPoint || std::abs(point[1] * 2.0 - std::round(point[1] * 2.0)) > 1.0e-6;
        bowedAwayFromChord = bowedAwayFromChord || point[1] > 4.5 + 0.1;
        CHECK(point[0] > path.pointsPredictionXYZ[index - 1][0]);
    }
    CHECK(usedFloatingPoint);
    CHECK(bowedAwayFromChord);
    CHECK(path.cost.alignment >= 0.0);
    REQUIRE(progress.size() >= 2);
    CHECK(progress.front().phase == "candidate_generation");
    CHECK(progress.front().completed == 0);
    CHECK(progress.front().total == 2);
    CHECK(progress.back().phase == "search");
    CHECK(progress.back().completed == 1);
    CHECK(progress.back().total == 1);
    CHECK(progress.back().elapsedSeconds >= 0.0);
}

TEST_CASE("fiberlet DP preserves a short pair as one exact transition")
{
    const cv::Vec3f start{2.25F, 4.25F, 4.25F};
    const cv::Vec3f target{3.75F, 4.25F, 4.25F};
    const auto anchors = twoAnchorArtifact({1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, start, target);
    const ConstantNormalSampler normals;
    auto config = pathConfig();
    config.corridorRadiusPredictionVoxels = 0.01;

    const auto report = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, config, constantPredictions(), normals);

    REQUIRE_MESSAGE(report.diagnostics.successfulPaths == 1, report.candidates[0].reason);
    const auto& path = report.candidates[0];
    REQUIRE(path.pointsPredictionXYZ.size() == 2);
    CHECK(path.pointsPredictionXYZ.front() == start);
    CHECK(path.pointsPredictionXYZ.back() == target);
    CHECK(cv::norm(path.pointsPredictionXYZ.back() - path.pointsPredictionXYZ.front()) == doctest::Approx(1.5));
    CHECK(report.endpointScoringInterpolations == 2);
    CHECK(report.lazyNodeScoringRequests == 0);
    CHECK(report.lazyNodeScoringMaterializations == 0);
    CHECK(report.lazyNodeScoringCacheHits == 0);
    CHECK(report.interpolatedScoringPoints == 2);
}

TEST_CASE("fiberlet quantization benchmark evaluates the complete scenario matrix")
{
    const auto anchors = twoAnchorArtifact(
        {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        {2.5, 4.0, 4.0}, {10.5, 4.0, 4.0});
    auto config = pathConfig();
    config.corridorRadiusPredictionVoxels = 0.01F;
    const ConstantNormalSampler normals;
    const auto paths = vc::fiber_tracer::traceFiberletPaths(
        anchors, anchors.report.grid, config, constantPredictions(), normals);
    vc::fiber_tracer::FiberletGraphReplayConfig replay;
    replay.errorThresholdBaseVoxels = 100.0;
    replay.referenceEndArcBase = 16.0;
    size_t extractionCalls = 0;
    const auto extractor = [&](const auto& quantizedAnchors) {
        ++extractionCalls;
        return vc::fiber_tracer::traceFiberletPaths(
            quantizedAnchors,
            quantizedAnchors.report.grid,
            config,
            constantPredictions(),
            normals);
    };

    const auto reports = vc::fiber_tracer::benchmarkFiberletQuantization(
        anchors, paths, {{5.0, 8.0, 8.0}, {21.0, 8.0, 8.0}},
        normals, 2.0, replay, extractor, 512);

    REQUIRE(reports.size() == 18);
    CHECK(extractionCalls == 7);
    CHECK(reports.front().scenario.name == "baseline");
    for (const auto& report : reports) {
        INFO(report.scenario.name << ": " << report.reason);
        CHECK(report.valid);
        CHECK(report.graphEdges == 1);
        CHECK(report.replayCompletedFraction == doctest::Approx(1.0));
        CHECK(report.baselineReplayFailures == 0);
        CHECK(report.replayFailureDelta == static_cast<int64_t>(report.replayFailures));
        CHECK(report.lineDistanceAvailable);
        CHECK(report.lineDistanceInvalidNormalSamples == 0);
    }
    CHECK(reports.front().replayFailures == 0);
    CHECK(reports.front().replayFailureDelta == 0);
    CHECK(reports.front().maximumLineDistanceBaseVoxels == 0.0);
    CHECK(reports.front().maximumLineNormalDistanceBaseVoxels == 0.0);
    CHECK(reports.front().maximumLineTangentialDistanceBaseVoxels == 0.0);
    const auto q1 = std::find_if(reports.begin(), reports.end(), [](const auto& report) {
        return report.scenario.name == "position_q1";
    });
    REQUIRE(q1 != reports.end());
    CHECK(q1->anchorPositionBits == 16);
    CHECK(q1->anchorDeltaBits == 8);
    const auto q2 = std::find_if(reports.begin(), reports.end(), [](const auto& report) {
        return report.scenario.name == "position_q2";
    });
    REQUIRE(q2 != reports.end());
    CHECK(q2->maximumLineDistanceBaseVoxels == doctest::Approx(1.0));
    CHECK(q2->maximumLineNormalDistanceBaseVoxels == doctest::Approx(0.0));
    CHECK(q2->maximumLineTangentialDistanceBaseVoxels == doctest::Approx(1.0));
}

TEST_CASE("fiberlet quantization benchmark selects one scenario and reports reference distances")
{
    const auto anchors = twoAnchorArtifact(
        {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        {2.5, 4.0, 4.0}, {10.5, 4.0, 4.0});
    auto config = pathConfig();
    config.corridorRadiusPredictionVoxels = 0.01F;
    const ConstantNormalSampler normals;
    const auto paths = vc::fiber_tracer::traceFiberletPaths(
        anchors, anchors.report.grid, config, constantPredictions(), normals);
    vc::fiber_tracer::FiberletGraphReplayConfig replay;
    replay.errorThresholdBaseVoxels = 100.0;
    replay.referenceEndArcBase = 16.0;
    size_t extractionCalls = 0;
    std::vector<vc::fiber_tracer::FiberletQuantizationProgress> progress;
    const auto reports = vc::fiber_tracer::benchmarkFiberletQuantization(
        anchors,
        paths,
        {{5.0, 8.0, 8.0}, {21.0, 8.0, 8.0}},
        normals,
        2.0,
        replay,
        [&](const auto& quantizedAnchors) {
            ++extractionCalls;
            return vc::fiber_tracer::traceFiberletPaths(
                quantizedAnchors,
                quantizedAnchors.report.grid,
                config,
                constantPredictions(),
                normals);
        },
        512,
        "combined_q4_axis_cost_u8",
        [&](const auto& update) { progress.push_back(update); });

    REQUIRE(reports.size() == 2);
    CHECK(extractionCalls == 1);
    CHECK(reports[0].scenario.name == "baseline");
    CHECK(reports[1].scenario.name == "combined_q4_axis_cost_u8");
    for (const auto& report : reports) {
        INFO(report.scenario.name << ": " << report.reason);
        REQUIRE(report.valid);
        CHECK(report.baselineReferenceDistanceBaseVoxels.count > 0);
        CHECK(report.scenarioReferenceDistanceBaseVoxels.count > 0);
        CHECK(report.baselineReferenceDistanceBaseVoxels.median >= 0.0);
        CHECK(report.scenarioReferenceDistanceBaseVoxels.median >= 0.0);
        CHECK(report.baselineReferenceInvalidNormalSamples == 0);
        CHECK(report.scenarioReferenceInvalidNormalSamples == 0);
    }
    REQUIRE_FALSE(progress.empty());
    CHECK(progress.front().phase == "reference_distance");
    CHECK(progress.front().scenario == "baseline");
    CHECK(progress.front().completed == 0);
    CHECK(progress.front().total > 0);
    CHECK(progress.back().completed == progress.back().total);
}

TEST_CASE("fiberlet quantization rejects more than two coordinate variants")
{
    auto anchors = chainAnchorArtifact();
    anchors.report.nonEmptyCells[0].components[0].anchor.positionPredictionXYZ[0] = 2.05F;
    anchors.report.nonEmptyCells[1].components[0].anchor.positionPredictionXYZ[0] = 2.10F;
    anchors.report.nonEmptyCells[2].components[0].anchor.positionPredictionXYZ[0] = 2.15F;
    const ConstantNormalSampler normals;
    auto config = pathConfig();
    config.cellRadius = 8;
    const auto paths = vc::fiber_tracer::traceFiberletPaths(
        anchors, anchors.report.grid, config, constantPredictions(), normals);
    vc::fiber_tracer::FiberletGraphReplayConfig replay;
    replay.errorThresholdBaseVoxels = 100.0;
    replay.referenceEndArcBase = 47.0;
    size_t extractionCalls = 0;
    const auto extractor = [&](const auto& quantizedAnchors) {
        ++extractionCalls;
        return vc::fiber_tracer::traceFiberletPaths(
            quantizedAnchors,
            quantizedAnchors.report.grid,
            config,
            constantPredictions(),
            normals);
    };
    const auto reports = vc::fiber_tracer::benchmarkFiberletQuantization(
        anchors, paths, {{4.1, 8.0, 8.0}, {52.0, 8.0, 8.0}},
        normals, 2.0, replay, extractor);
    const auto q4 = std::find_if(reports.begin(), reports.end(), [](const auto& report) {
        return report.scenario.name == "position_q4";
    });
    REQUIRE(q4 != reports.end());
    CHECK_FALSE(q4->valid);
    CHECK(q4->reason.find("more than two variants") != std::string::npos);
    CHECK(extractionCalls < 7);
}

TEST_CASE("fiberlet DP enforces a strict sampled fiber-direction bound")
{
    const auto anchors = twoAnchorArtifact({1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.5, 4.0, 4.0}, {10.5, 4.0, 4.0});
    auto config = pathConfig();
    config.corridorRadiusPredictionVoxels = 0.01;

    const ConstantNormalSampler tangentNormal({0.0, 0.0, 1.0});
    const auto accepted = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, config, constantPredictions(), tangentNormal);
    REQUIRE(accepted.diagnostics.successfulPaths == 1);

    const double radians = 25.0 * std::acos(-1.0) / 180.0;
    const cv::Vec3d boundaryDirection{std::cos(radians), std::sin(radians), 0.0};
    const auto boundary =
        vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, config, constantPredictions(boundaryDirection), tangentNormal);
    CHECK(boundary.diagnostics.successfulPaths == 0);

    const ConstantNormalSampler invalidNormal({}, false);
    const auto invalid = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, config, constantPredictions(), invalidNormal);
    CHECK(invalid.diagnostics.successfulPaths == 1);

    const auto invalidPredictions = [](const auto& indices, int, auto& samples) {
        samples.assign(indices.size(), {{0.0, 0.0, 0.0}, 0.0, false});
    };
    const auto bridge = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, config, invalidPredictions, tangentNormal);
    CHECK(bridge.diagnostics.successfulPaths == 0);
}

TEST_CASE("fiberlet floating interpolation preserves axes and invalid corners")
{
    const auto anchors = twoAnchorArtifact({1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.5, 4.0, 4.0}, {10.5, 4.0, 4.0});
    auto config = pathConfig();
    config.corridorRadiusPredictionVoxels = 0.01;
    const ConstantNormalSampler normals;

    const auto antipodal = [](const auto& indices, int, auto& samples) {
        samples.clear();
        samples.reserve(indices.size());
        for (const auto& zyx : indices) {
            const double sign = zyx[2] % 2 == 0 ? 1.0 : -1.0;
            samples.push_back({{sign, 0.0, 0.0}, 1.0, true, true});
        }
    };
    const auto signInvariant = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, config, antipodal, normals);
    CHECK(signInvariant.diagnostics.successfulPaths == 1);

    const auto zeroWeightInvalid = [](const auto& indices, int, auto& samples) {
        samples.clear();
        samples.reserve(indices.size());
        for (const auto& zyx : indices) {
            const bool valid = zyx[1] != 5;
            samples.push_back({{1.0, 0.0, 0.0}, 1.0, valid, valid});
        }
    };
    const auto zeroWeight = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, config, zeroWeightInvalid, normals);
    CHECK(zeroWeight.diagnostics.successfulPaths == 1);

    const auto requiredInvalid = [](const auto& indices, int, auto& samples) {
        samples.clear();
        samples.reserve(indices.size());
        for (const auto& zyx : indices) {
            const bool valid = zyx[2] != 6;
            samples.push_back({{1.0, 0.0, 0.0}, 1.0, valid, valid});
        }
    };
    const auto blocked = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, config, requiredInvalid, normals);
    CHECK(blocked.diagnostics.successfulPaths == 0);

    const auto degenerate = [](const auto& indices, int, auto& samples) {
        samples.clear();
        samples.reserve(indices.size());
        for (const auto& zyx : indices) {
            const cv::Vec3d direction = zyx[2] % 2 == 0 ? cv::Vec3d{1.0, 0.0, 0.0} : cv::Vec3d{0.0, 1.0, 0.0};
            samples.push_back({direction, 1.0, true, true});
        }
    };
    const auto ambiguous = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, config, degenerate, normals);
    CHECK(ambiguous.diagnostics.successfulPaths == 0);
}

TEST_CASE("fiber principal axis rejects an equal orthogonal tensor")
{
    const auto canonical =
        vc::fiber_tracer::canonicalFiberAxis({1.0, 0.0, 0.0});
    static_assert(std::is_same_v<decltype(canonical), const cv::Vec3d>);
    CHECK(canonical == cv::Vec3d{1.0, 0.0, 0.0});

    const auto tensor = vc::fiber_tracer::fiberAxisTensor(cv::Vec3d{1.0, 0.0, 0.0}, 0.5) + vc::fiber_tracer::fiberAxisTensor(cv::Vec3d{0.0, 1.0, 0.0}, 0.5);
    const auto principal = vc::fiber_tracer::principalFiberAxis(tensor);
    CHECK(principal.valid);
    CHECK_FALSE(principal.unique);
}

TEST_CASE("closed-form fiber principal axis matches the iterative resolver")
{
    const std::array<cv::Matx33d, 4> tensors{
        vc::fiber_tracer::fiberAxisTensor(
            cv::Vec3d{0.2672612419124244, 0.5345224838248488,
             0.8017837257372732}),
        vc::fiber_tracer::fiberAxisTensor(cv::Vec3d{1.0, 0.0, 0.0}, 0.7) +
            vc::fiber_tracer::fiberAxisTensor(cv::Vec3d{0.0, 1.0, 0.0}, 0.3),
        vc::fiber_tracer::fiberAxisTensor(cv::Vec3d{0.8, 0.6, 0.0}, 0.55) +
            vc::fiber_tracer::fiberAxisTensor(cv::Vec3d{0.2, -0.3, 0.9327379053}, 0.45),
        cv::Matx33d{0.2, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.4},
    };
    for (size_t tensorIndex = 0; tensorIndex < tensors.size(); ++tensorIndex) {
        const auto& tensor = tensors[tensorIndex];
        bool fallback = true;
        const auto iterative = vc::fiber_tracer::principalFiberAxis(tensor);
        const auto closedForm =
            vc::fiber_tracer::principalFiberAxisClosedForm(tensor, &fallback);
        REQUIRE(iterative.unique);
        REQUIRE(closedForm.unique);
        CHECK_FALSE(fallback);
        CHECK(std::abs(iterative.axis.dot(closedForm.axis)) ==
              doctest::Approx(1.0).epsilon(1.0e-10));
        CHECK(closedForm.largestEigenvalue ==
              doctest::Approx(iterative.largestEigenvalue).epsilon(1.0e-10));
        const double secondTolerance = 1.0e-8 * std::max(
            1.0, std::abs(iterative.largestEigenvalue));
        CHECK(std::abs(
                  closedForm.secondEigenvalue - iterative.secondEigenvalue) <=
              secondTolerance);
    }
}

TEST_CASE("closed-form fiber principal axis rejects ambiguous evidence")
{
    const auto tensor =
        vc::fiber_tracer::fiberAxisTensor(cv::Vec3d{1.0, 0.0, 0.0}, 0.5) +
        vc::fiber_tracer::fiberAxisTensor(cv::Vec3d{0.0, 1.0, 0.0}, 0.5);
    bool fallback = true;
    const auto principal =
        vc::fiber_tracer::principalFiberAxisClosedForm(tensor, &fallback);
    CHECK(principal.valid);
    CHECK_FALSE(principal.unique);
    CHECK_FALSE(fallback);
}

TEST_CASE("float fiber principal axis handles unique and ambiguous evidence")
{
    const auto ambiguousTensor =
        vc::fiber_tracer::fiberAxisTensorF(
            cv::Vec3f{1.0F, 0.0F, 0.0F}, 0.5F) +
        vc::fiber_tracer::fiberAxisTensorF(
            cv::Vec3f{0.0F, 1.0F, 0.0F}, 0.5F);
    const auto ambiguous =
        vc::fiber_tracer::principalFiberAxisF(ambiguousTensor);
    CHECK(ambiguous.valid);
    CHECK_FALSE(ambiguous.unique);

    cv::Vec3f expected{1.0F, 2.0F, 3.0F};
    expected /= std::sqrt(expected.dot(expected));
    const auto tensor = vc::fiber_tracer::fiberAxisTensorF(expected);
    bool fallback = true;
    const auto iterative = vc::fiber_tracer::principalFiberAxisF(tensor);
    const auto closed =
        vc::fiber_tracer::principalFiberAxisClosedFormF(tensor, &fallback);
    REQUIRE(iterative.unique);
    REQUIRE(closed.unique);
    CHECK_FALSE(fallback);
    CHECK(std::abs(iterative.axis.dot(closed.axis)) ==
          doctest::Approx(1.0).epsilon(1.0e-5));
}

TEST_CASE("fiberlet graph rejects float scale underflow and coordinate overflow")
{
    auto underflow = graphPathReport();
    underflow.grid.predictionToBaseScale =
        static_cast<double>(std::numeric_limits<float>::denorm_min()) * 0.5;
    addGraphPath(underflow, 0, 1, {{0, 0, 0}, {1, 0, 0}}, 1.0F);
    CHECK_THROWS_AS(
        vc::fiber_tracer::buildFiberletGraph(underflow),
        std::invalid_argument);

    auto overflow = graphPathReport();
    overflow.grid.predictionToBaseScale =
        static_cast<double>(std::numeric_limits<float>::max());
    addGraphPath(overflow, 0, 1, {{0, 0, 0}, {2, 0, 0}}, 1.0F);
    CHECK_THROWS_AS(
        vc::fiber_tracer::buildFiberletGraph(overflow),
        std::invalid_argument);
}

TEST_CASE("fiberlet tracing rejects grids beyond exact float coordinates")
{
    auto anchors = twoAnchorArtifact();
    anchors.report.grid.shapeZYX[2] = (size_t{1} << 24) + 1;
    const ConstantNormalSampler normals;
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::traceFiberletPaths(
            anchors, anchors.report.grid, pathConfig(),
            constantPredictions(), normals),
        doctest::Contains("exactly representable in float32"),
        std::invalid_argument);
}

TEST_CASE("fiberlet graph uses directed dense tangents and strict joins")
{
    auto report = graphPathReport();
    addGraphPath(report, 0, 1, {{0, 0, 0}, {1, 0, 0}}, 1.0);
    addGraphPath(report, 1, 2, {{1, 0, 0}, {2, 0, 0}}, 1.0);
    addGraphPath(report, 1, 3, {{1, 0, 0}, {2, 1, 0}}, 1.0);

    const auto graph = vc::fiber_tracer::buildFiberletGraph(report);
    REQUIRE(graph.edges.size() == 3);
    CHECK(graph.nodes.size() == 4);
    const auto hasTransition = [&](size_t incoming, size_t outgoing) {
        return std::any_of(graph.transitions.begin(), graph.transitions.end(), [&](const auto& transition) {
            return transition.incomingArc == incoming && transition.outgoingArc == outgoing;
        });
    };
    CHECK(hasTransition(0, 2));
    CHECK(hasTransition(3, 1));
    CHECK_FALSE(hasTransition(0, 4));
    CHECK_FALSE(hasTransition(5, 1));

    const auto json = vc::fiber_tracer::fiberletGraphJson(graph);
    CHECK(json.at("nodes").size() == 4);
    CHECK(json.at("edges").size() == 3);
    CHECK(json.at("maximum_join_angle_degrees") == 45.0);
}

TEST_CASE("fiberlet graph replay scores joins with the shared local metric")
{
    auto report = graphPathReport();
    addGraphPath(report, 0, 1, {{0, 0, 0}, {1, 0, 0}}, 0.0);
    addGraphPath(report, 1, 2, {{1, 0, 0}, {3, 0, 0}}, 1.0);
    const float cosine = std::cos(30.0F * std::acos(-1.0F) / 180.0F);
    const float sine = std::sin(30.0F * std::acos(-1.0F) / 180.0F);
    addGraphPath(report, 1, 3, {{1, 0, 0}, {1 + 2 * cosine, 2 * sine, 0}}, 0.0);

    const auto graph = vc::fiber_tracer::buildFiberletGraph(report);
    const auto transition = std::find_if(graph.transitions.begin(), graph.transitions.end(), [](const auto& item) {
        return item.incomingArc == 0 && item.outgoingArc == 4;
    });
    REQUIRE(transition != graph.transitions.end());
    CHECK(transition->angleDegrees == doctest::Approx(30.0));
    CHECK(transition->cost.alignment > 0.0);
    CHECK(transition->cost.tangentSmoothness > 0.0);
    CHECK(transition->cost.normalSmoothness == doctest::Approx(0.0));

    const vc::fiber_tracer::FiberLocalMetricSample sample{{1.0f, 0.0f, 0.0f}, 1.0f, true};
    const auto expected = vc::fiber_tracer::
        fiberLocalMetricCost(&sample, sample, {1.0f, 0.0f, 0.0f}, 1.0f, {static_cast<float>(cosine), static_cast<float>(sine), 0.0f}, 2.0f, {0.0f, 0.0f, 1.0f}, true, {4.0f, {2.0f, 0.1f, 10.0f, 0.0f}});
    CHECK(transition->cost.total() == doctest::Approx(expected.total()));

    vc::fiber_tracer::FiberletGraphReplayConfig config;
    config.beamWidth = 4;
    config.lookaheadEdges = 2;
    config.errorThresholdBaseVoxels = 2.0;
    std::vector<vc::fiber_tracer::FiberletGraphReplayProgress> replayProgress;
    const auto replay = vc::fiber_tracer::traceFiberletGraphReplay(
        graph, {{0, 0, 0}, {3, 0, 0}}, replayYNormals(), 1.0, config,
        {}, [&](const auto& update) { replayProgress.push_back(update); });
    REQUIRE(replay.segments.size() == 1);
    CHECK(replay.segments[0].candidateIndices == std::vector<size_t>{0, 1});
    REQUIRE(replay.segments[0].transitionIndices.size() == 1);
    CHECK(replay.segments[0].transitionCost.total() == doctest::Approx(0.0));
    CHECK(replay.completedReferenceArcBase == doctest::Approx(3.0));
    CHECK(replay.failures.empty());
    REQUIRE_FALSE(replayProgress.empty());
    CHECK(replayProgress.front().state == "segment_start");
    CHECK(replayProgress.front().referenceArcFraction == doctest::Approx(0.0));
    CHECK(replayProgress.back().state == "completed");
    CHECK(replayProgress.back().referenceArcFraction == doctest::Approx(1.0));
    for (size_t index = 1; index < replayProgress.size(); ++index) {
        CHECK(replayProgress[index].referenceArcBase >=
              replayProgress[index - 1].referenceArcBase);
        CHECK(replayProgress[index].referenceArcFraction >=
              replayProgress[index - 1].referenceArcFraction);
    }
    const auto json = vc::fiber_tracer::fiberletGraphJson(graph);
    CHECK(
        json.at("transitions").at(static_cast<size_t>(std::distance(graph.transitions.begin(), transition))).at("cost").at("total") ==
        doctest::Approx(expected.total()));
}

TEST_CASE("fiberlet graph replay lookahead avoids the greedy dead-end cost")
{
    auto report = graphPathReport();
    addGraphPath(report, 0, 1, {{0, 0, 0}, {1, 0, 0}}, 0.0);
    addGraphPath(report, 1, 2, {{1, 0, 0}, {2, 0, 0}}, 100.0);
    addGraphPath(report, 0, 3, {{0, 0, 0}, {1, 0.1, 0}}, 10.0);
    addGraphPath(report, 2, 3, {{2, 0, 0}, {1, 0.1, 0}}, 0.0);
    const auto graph = vc::fiber_tracer::buildFiberletGraph(report);
    vc::fiber_tracer::FiberletGraphReplayConfig config;
    config.beamWidth = 4;
    config.lookaheadEdges = 2;
    config.errorThresholdBaseVoxels = 0.5;
    const auto replay = vc::fiber_tracer::traceFiberletGraphReplay(
        graph, {{0, 0, 0}, {2, 0, 0}}, replayNormals(), 1.0, config);

    REQUIRE(replay.segments.size() == 1);
    CHECK(replay.segments[0].candidateIndices == std::vector<size_t>{2, 3});
    REQUIRE(replay.segments[0].routePointsBaseXYZ.size() == 3);
    CHECK(replay.segments[0].routePointsBaseXYZ[1][1] == doctest::Approx(0.1));
    CHECK(replay.failures.empty());
}

TEST_CASE("fiberlet graph distance lookahead compares a common physical arc")
{
    auto report = graphPathReport();
    report.grid.predictionToBaseScale = 2.0;
    addGraphPath(report, 0, 1, {{0, 0, 0}, {1, 0, 0}}, 0.0);
    addGraphPath(report, 1, 2, {{1, 0, 0}, {4, 0.5, 0}}, 12.0);
    report.candidates.back().cost.alignment = 3.0F;
    report.candidates.back().cost.tangentSmoothness = 6.0F;
    report.candidates.back().cost.normalSmoothness = 3.0F;
    addGraphPath(report, 0, 3, {{0, 0, 0}, {2, 0, 0}}, 3.0);
    addGraphPath(report, 3, 4, {{2, 0, 0}, {3, 0, 0}}, 3.0);
    const auto graph = vc::fiber_tracer::buildFiberletGraph(report);

    vc::fiber_tracer::FiberletGraphReplayConfig config;
    config.beamWidth = 8;
    config.lookaheadDistanceBaseVoxels = 6.0;
    config.errorThresholdBaseVoxels = 100.0;
    config.referenceEndArcBase = 6.0;
    config.recordDecisionDiagnostics = true;
    const auto replay = vc::fiber_tracer::traceFiberletGraphReplay(
        graph, {{0, 0, 0}, {8, 0, 0}}, replayYNormals(), 1.0, config);

    REQUIRE(replay.segments.size() == 1);
    REQUIRE_FALSE(replay.segments.front().decisions.empty());
    const auto& decision = replay.segments.front().decisions.front();
    REQUIRE(decision.routes.size() == 2);
    for (const auto& route : decision.routes) {
        CHECK(route.pathLengthPredictionVoxels == doctest::Approx(3.0));
        CHECK(route.lossPerPredictionVoxel ==
              doctest::Approx(route.totalLoss / 3.0));
    }
    const auto partial = std::find_if(
        decision.routes.begin(), decision.routes.end(), [](const auto& route) {
            return route.edgeCost.tangentSmoothness > 0.0 &&
                !route.includedArcFractions.empty() &&
                route.includedArcFractions.back() < 1.0 - 1.0e-9;
        });
    REQUIRE(partial != decision.routes.end());
    REQUIRE(partial->includedArcFractions.size() == 2);
    const double fraction = partial->includedArcFractions.back();
    CHECK(partial->edgeCost.alignment == doctest::Approx(3.0 * fraction));
    CHECK(partial->edgeCost.tangentSmoothness ==
          doctest::Approx(6.0 * fraction));
    CHECK(partial->edgeCost.normalSmoothness ==
          doctest::Approx(3.0 * fraction));
    CHECK(partial->edgeCost.total() ==
          doctest::Approx(12.0 * fraction));
    CHECK(partial->transitionCost.total() > 0.0);
    REQUIRE_FALSE(partial->routePointsBaseXYZ.empty());
    CHECK(partial->routePointsBaseXYZ.back()[0] < 8.0);

    REQUIRE(decision.selectedRouteIndex.has_value());
    const auto& selected =
        decision.routes.at(*decision.selectedRouteIndex);
    CHECK(selected.includedArcFractions == std::vector<double>{1.0, 1.0});
    const auto json =
        vc::fiber_tracer::fiberletGraphReplayJson(replay, config);
    CHECK(json.at("config").at("lookahead_mode") == "distance");
    CHECK(json.at("config").at("lookahead_edges").is_null());
    CHECK(json.at("config").at("lookahead_distance_base_voxels") ==
          doctest::Approx(6.0));
    CHECK(json.at("config").at("lookahead_distance_prediction_voxels") ==
          doctest::Approx(3.0));
}

TEST_CASE("fiberlet graph distance lookahead shortens at the selected end")
{
    auto report = graphPathReport();
    report.grid.predictionToBaseScale = 2.0;
    addGraphPath(report, 0, 1, {{0, 0, 0}, {4, 0, 0}}, 8.0);
    const auto graph = vc::fiber_tracer::buildFiberletGraph(report);
    vc::fiber_tracer::FiberletGraphReplayConfig config;
    config.lookaheadDistanceBaseVoxels = 6.0;
    config.errorThresholdBaseVoxels = 100.0;
    config.referenceEndArcBase = 2.0;
    config.recordDecisionDiagnostics = true;

    const auto replay = vc::fiber_tracer::traceFiberletGraphReplay(
        graph, {{0, 0, 0}, {8, 0, 0}}, replayYNormals(), 1.0, config);
    REQUIRE(replay.failures.empty());
    REQUIRE(replay.segments.size() == 1);
    CHECK(replay.segments.front().terminationReason == "reference_end");
    REQUIRE(replay.segments.front().decisions.size() == 1);
    const auto& route = replay.segments.front().decisions.front().routes.front();
    CHECK(route.pathLengthPredictionVoxels == doctest::Approx(1.0));
    CHECK(route.edgeCost.total() == doctest::Approx(2.0));
    REQUIRE(route.includedArcFractions.size() == 1);
    CHECK(route.includedArcFractions.front() == doctest::Approx(0.25));
}

TEST_CASE("fiberlet graph distance lookahead rejects an uncovered horizon")
{
    auto report = graphPathReport();
    addGraphPath(report, 0, 1, {{0, 0, 0}, {1, 0, 0}}, 1.0);
    const auto graph = vc::fiber_tracer::buildFiberletGraph(report);
    vc::fiber_tracer::FiberletGraphReplayConfig config;
    config.lookaheadDistanceBaseVoxels = 3.0;
    config.errorThresholdBaseVoxels = 100.0;
    config.referenceEndArcBase = 4.0;

    const auto replay = vc::fiber_tracer::traceFiberletGraphReplay(
        graph, {{0, 0, 0}, {4, 0, 0}}, replayYNormals(), 1.0, config);
    REQUIRE_FALSE(replay.failures.empty());
    CHECK(replay.failures.front().reason == "graph_exhausted");

    config.lookaheadDistanceBaseVoxels = 0.0;
    CHECK_THROWS_AS(
        vc::fiber_tracer::traceFiberletGraphReplay(
            graph, {{0, 0, 0}, {4, 0, 0}}, replayYNormals(), 1.0, config),
        std::invalid_argument);
}

TEST_CASE("fiberlet graph replay uses the Lasagna ellipsoid for seeds and route points")
{
    auto report = graphPathReport();
    addGraphPath(
        report, 0, 1,
        {{0.0, 1.5, 0.0}, {1.0, 1.5, 0.0}, {2.0, 1.5, 0.0}},
        0.0);
    const auto graph = vc::fiber_tracer::buildFiberletGraph(report);
    vc::fiber_tracer::FiberletGraphReplayConfig config;
    config.errorThresholdBaseVoxels = 0.5;

    const auto tangential = vc::fiber_tracer::traceFiberletGraphReplay(
        graph, {{0, 0, 0}, {2, 0, 0}}, replayNormals(), 1.0, config);
    CHECK(tangential.failures.empty());
    REQUIRE(tangential.segments.size() == 1);
    REQUIRE_FALSE(tangential.segments.front().matches.empty());
    const auto& seed =
        tangential.segments.front().matches.front().thresholdMeasurement;
    CHECK(seed.euclideanErrorBaseVoxels == doctest::Approx(1.5));
    CHECK(seed.thresholdErrorBaseVoxels == doctest::Approx(0.375));
    CHECK(seed.thresholdErrorRatio == doctest::Approx(0.75));
    CHECK(seed.localNormalValid);

    const auto normal = vc::fiber_tracer::traceFiberletGraphReplay(
        graph, {{0, 0, 0}, {2, 0, 0}}, replayYNormals(), 1.0, config);
    REQUIRE_FALSE(normal.failures.empty());
    CHECK(normal.failures.front().reason ==
          "no_usable_seed_for_remaining_reference");
}

TEST_CASE("fiberlet graph replay completes a failure edge then records an uncovered tail")
{
    auto report = graphPathReport();
    addGraphPath(report, 0, 1, {{0, 0, 0}, {1, 0, 0}, {2, 2, 0}}, 3.0);
    const auto graph = vc::fiber_tracer::buildFiberletGraph(report);
    vc::fiber_tracer::FiberletGraphReplayConfig config;
    config.errorThresholdBaseVoxels = 0.5;
    const auto replay = vc::fiber_tracer::traceFiberletGraphReplay(
        graph, {{0, 0, 0}, {3, 0, 0}}, replayYNormals(), 1.0, config);

    REQUIRE(replay.segments.size() == 2);
    CHECK(replay.segments[0].routePointsBaseXYZ.size() == 3);
    CHECK(replay.segments[0].candidateIndices == std::vector<size_t>{0});
    CHECK(replay.segments[0].arcIndices == std::vector<size_t>{0});
    CHECK(replay.segments[0].stopNodeIndex == 1);
    CHECK(replay.segments[0].totalLoss == doctest::Approx(3.0));
    CHECK(replay.segments[0].pathLengthPredictionVoxels > 0.0);
    REQUIRE(replay.segments[0].seedKey.has_value());
    CHECK(replay.segments[0].seedKey->coordinateZYX ==
          std::array<std::int64_t, 3>{0, 0, 0});
    CHECK(replay.segments[1].routePointsBaseXYZ.empty());
    REQUIRE(replay.failures.size() == 2);
    CHECK(replay.failures[0].reason == "distance_above_threshold");
    CHECK(replay.failures[0].candidateIndex == 0);
    CHECK(replay.failures[0].candidatePathPointIndex == 2);
    CHECK(replay.failures[0].arcIndex == 0);
    CHECK(replay.failures[0].segmentPointIndex == 2);
    CHECK(replay.failures[1].reason == "no_usable_seed_for_remaining_reference");
    CHECK(replay.completedReferenceArcBase == doctest::Approx(3.0));
    const auto json = vc::fiber_tracer::fiberletGraphReplayJson(replay, config);
    CHECK(json.at("version") == 2);
    CHECK(json.at("config").at("threshold").at("shape") ==
          "lasagna_normal_ellipsoid");
    CHECK(json.at("config").at("threshold").at(
              "normal_radius_base_voxels") == doctest::Approx(0.5));
    CHECK(json.at("config").at("threshold").at(
              "tangential_radius_base_voxels") == doctest::Approx(2.0));
    CHECK_FALSE(json.at("config").contains(
        "error_threshold_base_voxels"));
    CHECK(json.at("segments").size() == 2);
    CHECK(json.at("segments").at(0).at("seed_key") ==
          nlohmann::json::array({0, 0, 0, 0}));
    CHECK(json.at("failures").size() == 2);
    const auto& failedMatch =
        json.at("segments").at(0).at("matches").back();
    CHECK(failedMatch.at("euclidean_error_base_voxels") ==
          doctest::Approx(2.0));
    CHECK(failedMatch.at("normal_error_base_voxels") ==
          doctest::Approx(2.0));
    CHECK(failedMatch.at("tangential_error_base_voxels") ==
          doctest::Approx(0.0));
    CHECK(failedMatch.at("threshold_error_base_voxels") ==
          doctest::Approx(2.0));
    CHECK(failedMatch.at("threshold_error_ratio") ==
          doctest::Approx(4.0));
    CHECK(failedMatch.at("local_normal_valid") == true);
    CHECK_FALSE(failedMatch.contains("error_base_voxels"));
    CHECK(json.at("failures").at(0).at("threshold_error_ratio") ==
          doctest::Approx(4.0));
    CHECK(json.at("failures").at(1).at(
              "euclidean_error_base_voxels").is_null());
    const auto windows =
        vc::fiber_tracer::fiberletGraphReplayFailureWindows(replay);
    REQUIRE(windows.size() == 2);
    CHECK(windows[0].failureIndex == 0);
    CHECK(windows[0].segmentIndex == 0);
    CHECK(windows[0].reason == "distance_above_threshold");
    CHECK(windows[0].replayBeginArcBase == doctest::Approx(
        replay.segments[0].matches.front().searchBeginArcBase));
    CHECK(windows[0].failureReferenceArcBase == doctest::Approx(
        replay.failures[0].referenceArcBase));
    CHECK(windows[0].replayEndArcBase == doctest::Approx(
        replay.segments[0].endReferenceArcBase));
    CHECK(windows[0].seedKey == replay.segments[0].seedKey);
    CHECK(windows[1].replayBeginArcBase == doctest::Approx(
        replay.segments[1].startReferenceArcBase));
    CHECK(windows[1].replayEndArcBase == doctest::Approx(
        replay.segments[1].endReferenceArcBase));
    const auto obj = vc::fiber_tracer::fiberletGraphReplayObj(replay);
    CHECK(occurrenceCount(obj, "\nv ") == 3);
    CHECK(obj.find("\nl 1 2 3\n") != std::string::npos);
}

TEST_CASE("fiberlet graph replay can force the original focused seed")
{
    auto report = graphPathReport();
    addGraphPath(report, 0, 1, {{0, 0, 0}, {1, 0, 0}}, 1.0);
    addGraphPath(report, 1, 2, {{1, 0, 0}, {2, 0, 0}}, 1.0);
    const auto graph = vc::fiber_tracer::buildFiberletGraph(report);
    vc::fiber_tracer::FiberletGraphReplayConfig config;
    config.errorThresholdBaseVoxels = 10.0;
    config.referenceEndArcBase = 2.0;
    config.initialSeedKey = {{0, 0, 1}, 0};
    config.recordDecisionDiagnostics = true;

    const auto replay = vc::fiber_tracer::traceFiberletGraphReplay(
        graph, {{0, 0, 0}, {2, 0, 0}}, replayYNormals(), 1.0, config);

    REQUIRE(replay.segments.size() == 1);
    REQUIRE(replay.segments.front().seedKey.has_value());
    CHECK(*replay.segments.front().seedKey == *config.initialSeedKey);
    REQUIRE_FALSE(replay.segments.front().routePointsBaseXYZ.empty());
    CHECK(replay.segments.front().routePointsBaseXYZ.front()[0] ==
          doctest::Approx(1.0));
    REQUIRE(replay.segments.front().decisions.size() == 1);
    const auto& decision = replay.segments.front().decisions.front();
    CHECK(decision.sourceKey == *config.initialSeedKey);
    REQUIRE(decision.selectedRouteIndex.has_value());
    REQUIRE(*decision.selectedRouteIndex < decision.routes.size());
    const auto& selected = decision.routes[*decision.selectedRouteIndex];
    REQUIRE(selected.logicalArcs.size() == 1);
    CHECK(selected.edgeCost.alignment == doctest::Approx(1.0));
    CHECK(selected.transitionCost.total() == doctest::Approx(0.0));
    CHECK(selected.committedEdgeCost.alignment == doctest::Approx(1.0));
    CHECK(selected.committedTransitionCost.total() == doctest::Approx(0.0));
    CHECK(selected.committedPathLengthPredictionVoxels == doctest::Approx(1.0));
    CHECK(selected.totalLoss == doctest::Approx(1.0));
    CHECK(selected.lossPerPredictionVoxel == doctest::Approx(1.0));
    CHECK(selected.routePointsBaseXYZ.size() == 2);
    REQUIRE(replay.segments.front().committedSteps.size() == 1);
    const auto& committed = replay.segments.front().committedSteps.front();
    CHECK(committed.referenceBeginArcBase == doctest::Approx(1.0));
    CHECK(committed.referenceEndArcBase == doctest::Approx(2.0));
    CHECK(committed.edgeCost.alignment == doctest::Approx(1.0));
    CHECK(committed.transitionCost.total() == doctest::Approx(0.0));
    CHECK(committed.pathLengthPredictionVoxels == doctest::Approx(1.0));
    const auto json = vc::fiber_tracer::fiberletGraphReplayJson(replay, config);
    CHECK(json.at("config").at("initial_seed_key") ==
          nlohmann::json::array({0, 0, 1, 0}));
    CHECK(json.at("config").at("record_decision_diagnostics") == true);
    CHECK(json.at("segments").at(0).at("decisions").size() == 1);
    CHECK(json.at("segments").at(0).at("decisions").at(0)
              .at("routes").at(0).at("edge_cost").at("alignment") ==
          doctest::Approx(1.0));
}

TEST_CASE("fiberlet replay failure windows reject inconsistent results")
{
    vc::fiber_tracer::FiberletGraphReplayResult replay;
    replay.referenceBeginArcBase = 2.0;
    replay.referenceEndArcBase = 12.0;
    replay.segments.push_back({});
    replay.segments.front().startReferenceArcBase = 4.0;
    replay.segments.front().endReferenceArcBase = 8.0;
    replay.failures.push_back({0, 1, "broken", 6.0});
    CHECK_THROWS_AS(
        vc::fiber_tracer::fiberletGraphReplayFailureWindows(replay),
        std::invalid_argument);

    replay.failures.front().segmentIndex = 0;
    replay.failures.front().index = 1;
    CHECK_THROWS_AS(
        vc::fiber_tracer::fiberletGraphReplayFailureWindows(replay),
        std::invalid_argument);

    replay.failures.front().index = 0;
    replay.failures.front().referenceArcBase = 3.0;
    CHECK_THROWS_AS(
        vc::fiber_tracer::fiberletGraphReplayFailureWindows(replay),
        std::invalid_argument);
}

TEST_CASE("fiberlet graph replay completes on a partial terminal edge")
{
    auto report = graphPathReport();
    addGraphPath(
        report, 0, 1,
        {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}}, 3.0);
    const auto graph = vc::fiber_tracer::buildFiberletGraph(report);
    vc::fiber_tracer::FiberletGraphReplayConfig config;
    config.errorThresholdBaseVoxels = 10.0;
    config.referenceEndArcBase = 1.5;
    const auto replay = vc::fiber_tracer::traceFiberletGraphReplay(
        graph, {{0, 0, 0}, {4, 0, 0}}, replayYNormals(), 1.0, config);

    REQUIRE(replay.segments.size() == 1);
    const auto& segment = replay.segments.front();
    CHECK(segment.terminationReason == "reference_end");
    CHECK(segment.terminalPartialEdge);
    CHECK_FALSE(segment.stopNodeIndex.has_value());
    CHECK(segment.routePointsBaseXYZ.size() == 3);
    CHECK(segment.candidateIndices == std::vector<size_t>{0});
    CHECK(segment.endReferenceArcBase == doctest::Approx(1.5));
    CHECK(replay.completedReferenceArcBase == doctest::Approx(1.5));
    CHECK(replay.failures.empty());
    const auto json = vc::fiber_tracer::fiberletGraphReplayJson(replay, config);
    CHECK(json.at("segments").at(0).at("terminal_partial_edge") == true);
}

TEST_CASE("fiberlet graph replay reports a boundary failure before completion")
{
    auto report = graphPathReport();
    addGraphPath(
        report, 0, 1,
        {{0, 0, 0}, {1, 0, 0}, {2, 2, 0}, {3, 2, 0}}, 3.0);
    const auto graph = vc::fiber_tracer::buildFiberletGraph(report);
    vc::fiber_tracer::FiberletGraphReplayConfig config;
    config.errorThresholdBaseVoxels = 0.5;
    config.referenceEndArcBase = 1.5;
    const auto replay = vc::fiber_tracer::traceFiberletGraphReplay(
        graph, {{0, 0, 0}, {4, 0, 0}}, replayYNormals(), 1.0, config);

    REQUIRE_FALSE(replay.failures.empty());
    CHECK(replay.failures.front().reason == "distance_above_threshold");
    CHECK(replay.failures.front().referenceArcBase == doctest::Approx(1.5));
    CHECK(replay.failures.front().referenceArcFraction == doctest::Approx(1.0));
    REQUIRE_FALSE(replay.segments.empty());
    CHECK(replay.segments.front().routePointsBaseXYZ.size() == 3);
    CHECK(replay.segments.front().terminalPartialEdge);
}

TEST_CASE("fiberlet graph replay reseeds independently after multiple failures")
{
    auto report = graphPathReport();
    addGraphPath(report, 0, 1, {{0, 0, 0}, {1, 0.6, 0}, {2, 0.6, 0}}, 1.0);
    addGraphPath(report, 2, 3, {{3, 0, 0}, {4, 0.6, 0}, {5, 0.6, 0}}, 2.0);
    const auto graph = vc::fiber_tracer::buildFiberletGraph(report);
    vc::fiber_tracer::FiberletGraphReplayConfig config;
    config.errorThresholdBaseVoxels = 0.5;
    const auto replay = vc::fiber_tracer::traceFiberletGraphReplay(
        graph, {{0, 0, 0}, {6, 0, 0}}, replayYNormals(), 1.0, config);

    REQUIRE(replay.failures.size() >= 3);
    CHECK(replay.failures[0].reason == "distance_above_threshold");
    CHECK(
        std::any_of(replay.failures.begin(), replay.failures.end(), [](const auto& failure) { return failure.reason == "missing_seed_gap"; }));
    CHECK(std::count_if(replay.failures.begin(), replay.failures.end(), [](const auto& failure) {
              return failure.reason == "distance_above_threshold";
          }) == 2);
    CHECK(replay.completedReferenceArcBase == doctest::Approx(6.0));
    for (const auto& segment : replay.segments)
        CHECK(segment.transitionIndices.empty());
}

TEST_CASE("empty fiberlet graph reports one uncovered-tail reset")
{
    const auto graph = vc::fiber_tracer::buildFiberletGraph(graphPathReport());
    vc::fiber_tracer::FiberletGraphReplayConfig config;
    const auto replay = vc::fiber_tracer::traceFiberletGraphReplay(
        graph, {{0, 0, 0}, {5, 0, 0}}, replayNormals(), 1.0, config);

    REQUIRE(replay.failures.size() == 1);
    CHECK(replay.failures[0].reason == "no_usable_seed_for_remaining_reference");
    CHECK(replay.failures[0].evaluatorPointBase.has_value() == false);
    CHECK(replay.completedReferenceArcBase == doctest::Approx(5.0));
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
    REQUIRE(sampledIndices.size() == report.sampledVoxels);
    CHECK(report.samplingCoordinateBatches == 1);
    CHECK(report.peakCoordinateBatchVoxels == report.sampledVoxels);
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

TEST_CASE("fiberlet curve sampling is sparse with and without a selection predicate")
{
    const auto anchors = twoAnchorArtifact();
    const ConstantNormalSampler normals;
    std::vector<std::array<size_t, 3>> unselectedSamples;
    const auto unselected = vc::fiber_tracer::traceFiberletPaths(
        anchors,
        anchors.report.grid,
        pathConfig(),
        [&](const auto& indices, int, auto& samples) {
            unselectedSamples = indices;
            samples.assign(indices.size(), {{1.0, 0.0, 0.0}, 1.0, true});
        },
        normals);
    std::vector<std::array<size_t, 3>> selectedSamples;
    const auto selected = vc::fiber_tracer::traceFiberletPaths(
        anchors,
        anchors.report.grid,
        pathConfig(),
        [&](const auto& indices, int, auto& samples) {
            selectedSamples = indices;
            samples.assign(indices.size(), {{1.0, 0.0, 0.0}, 1.0, true});
        },
        normals,
        {},
        [](const cv::Vec3f&) { return true; });

    REQUIRE(unselected.diagnostics.successfulPaths == 1);
    REQUIRE(selected.diagnostics.successfulPaths == 1);
    CHECK(unselectedSamples == selectedSamples);
    CHECK(std::set<std::array<size_t, 3>>(unselectedSamples.begin(), unselectedSamples.end()).size() == unselectedSamples.size());
    CHECK(selected.candidates[0].pointsPredictionXYZ == unselected.candidates[0].pointsPredictionXYZ);
    CHECK(selected.candidates[0].cost.total() == unselected.candidates[0].cost.total());
    CHECK(vc::fiber_tracer::fiberletPathReportObj(selected) == vc::fiber_tracer::fiberletPathReportObj(unselected));
}

TEST_CASE("fiberlet global sampling coordinates are invariant under batching and workers")
{
    const auto anchors = twoPathArtifact();
    std::vector<std::array<size_t, 3>> baseline;
    size_t baselineCount = 0;
    for (const int batchSize : {1, 7, 100000}) {
        for (const int threads : {1, 3}) {
            auto config = pathConfig();
            config.samplingBatchCoordinates = batchSize;
            config.parallelThreads = threads;
            std::vector<std::array<size_t, 3>> predictionCoordinates;
            size_t predictionCalls = 0;
            const auto predictions = [&](const auto& indices, int requestedThreads, auto& samples) {
                CHECK(requestedThreads == threads);
                ++predictionCalls;
                predictionCoordinates.insert(predictionCoordinates.end(), indices.begin(), indices.end());
                samples.assign(indices.size(), {{1.0, 0.0, 0.0}, 1.0, true});
            };
            const CountingNormalSampler normals;
            const auto report =
                vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, config, predictions, normals, {}, [](const cv::Vec3f&) {
                    return true;
                });
            CAPTURE(batchSize);
            CAPTURE(threads);
            REQUIRE(report.diagnostics.searchedPairs == 2);
            REQUIRE(predictionCoordinates.size() == report.sampledVoxels);
            REQUIRE(normals.sampledPoints.size() == report.sampledVoxels);
            std::vector<std::array<size_t, 3>> normalCoordinates;
            normalCoordinates.reserve(normals.sampledPoints.size());
            for (const auto& point : normals.sampledPoints) {
                normalCoordinates.push_back({static_cast<size_t>(point[2]), static_cast<size_t>(point[1]), static_cast<size_t>(point[0])});
            }
            CHECK(predictionCoordinates == normalCoordinates);
            CHECK(std::is_sorted(predictionCoordinates.begin(), predictionCoordinates.end()));
            CHECK(std::adjacent_find(predictionCoordinates.begin(), predictionCoordinates.end()) == predictionCoordinates.end());
            const size_t expectedCalls = report.sampledVoxels == 0 ? 0 : (report.sampledVoxels - 1) / static_cast<size_t>(batchSize) + 1;
            CHECK(predictionCalls == expectedCalls);
            CHECK(normals.batchCalls.load() == expectedCalls);
            CHECK(report.predictionSamplingCalls == expectedCalls);
            CHECK(report.normalSamplingCalls == expectedCalls);
            CHECK(report.samplingCoordinateBatches == expectedCalls);
            CHECK(report.preparedCandidates == report.diagnostics.searchedPairs);
            if (baseline.empty()) {
                baseline = predictionCoordinates;
                baselineCount = report.sampledVoxels;
            } else {
                CHECK(predictionCoordinates == baseline);
                CHECK(report.sampledVoxels == baselineCount);
            }
        }
    }
}

TEST_CASE("sparse bitmap corner finalization matches a serial reference")
{
    using Voxel = std::array<int64_t, 3>;
    const auto check = [](const std::vector<std::vector<Voxel>>& sets) {
        std::vector<Voxel> expected;
        for (const auto& set : sets)
            expected.insert(expected.end(), set.begin(), set.end());
        const auto storedLess = [](const Voxel& left, const Voxel& right) {
            return std::array{
                       static_cast<size_t>(left[2]),
                       static_cast<size_t>(left[1]),
                       static_cast<size_t>(left[0])} <
                std::array{
                       static_cast<size_t>(right[2]),
                       static_cast<size_t>(right[1]),
                       static_cast<size_t>(right[0])};
        };
        std::sort(expected.begin(), expected.end(), storedLess);
        expected.erase(std::unique(expected.begin(), expected.end()), expected.end());

        CHECK(vc::fiber_tracer::testing::debugFinalizeFiberletCornerSets(sets) ==
              expected);
    };

    SUBCASE("empty")
    {
        check({});
        check({{}, {}, {}});
    }
    SUBCASE("overlapping")
    {
        check({
            {{3, 2, 1}, {0, 0, 0}, {4, 2, 1}},
            {{4, 2, 1}, {3, 2, 1}, {9, 0, 0}},
            {{3, 2, 1}, {1, 5, 2}},
        });
    }
    SUBCASE("duplicate heavy")
    {
        check({
            std::vector<Voxel>(200, {7, 8, 9}),
            std::vector<Voxel>(300, {7, 8, 9}),
            {{7, 8, 8}, {7, 8, 9}, {7, 8, 10}},
        });
    }
    SUBCASE("uneven")
    {
        std::vector<std::vector<Voxel>> sets(17);
        for (int64_t index = 0; index < 1000; ++index)
            sets[3].push_back({index % 13, index % 17, index % 19});
        sets[0] = {{100, 1, 0}};
        sets[16] = {{-1, 2, 3}, {100, 1, 0}};
        check(sets);
    }
}

TEST_CASE("fiberlet sparse replay domain rejects a disconnected corridor")
{
    const auto anchors = twoAnchorArtifact();
    const ConstantNormalSampler normals;
    const auto report =
        vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, pathConfig(), constantPredictions(), normals, {}, [](const cv::Vec3f& point) {
            return point[0] < 6.4 || point[0] > 6.6;
        });

    CHECK(report.diagnostics.successfulPaths == 0);
    REQUIRE(report.candidates.size() == 1);
    CHECK(report.candidates[0].reason == "no_path");
    CHECK(report.lazyNodeScoringMaterializations < report.retainedSearchNodes);
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
    CHECK(serial.candidateGenerationWorkers == 1);
    CHECK(parallel.candidateGenerationWorkers == 4);
    const auto checkProfile = [](const auto& report) {
        CHECK(report.latticeNodePositions >= report.corridorAcceptedNodes);
        CHECK(report.corridorAcceptedNodes >= report.retainedSearchNodes);
        CHECK(report.interpolationCornerInsertions >= report.sampledVoxels);
        CHECK(report.endpointScoringInterpolations ==
              2 * report.preparedCandidates);
        CHECK(report.interpolatedScoringPoints ==
              report.endpointScoringInterpolations +
                  report.lazyNodeScoringMaterializations);
        CHECK(report.lazyNodeScoringMaterializations <=
              report.retainedSearchNodes);
        CHECK(report.lazyNodeScoringRequests ==
              report.lazyNodeScoringMaterializations +
                  report.lazyNodeScoringCacheHits);
        CHECK(report.lazyNodeScoringCacheHits > 0);
        CHECK(report.dpPreparedNodes ==
              report.lazyNodeScoringMaterializations);
        CHECK(report.retainedSearchNodes + 2 * report.preparedCandidates ==
              report.evaluatedDpNodes);
        CHECK(report.scoringPageCount <= report.sampledVoxels);
        CHECK(report.scoringPageSlots >= report.sampledVoxels);
        CHECK(report.scoringPageDirectoryProbes >=
              report.interpolatedScoringPoints);
        CHECK(report.scoringPageDirectoryProbes <=
              report.interpolationCornerInsertions);
        CHECK(report.interpolationProfiledPoints > 0);
        CHECK(report.interpolationProfiledPoints <=
              report.interpolatedScoringPoints);
        CHECK(report.interpolationProfiledCorners >=
              report.interpolationProfiledPoints);
        CHECK(report.interpolationProfiledPredictionPrincipalSolves <=
              report.interpolationProfiledPoints);
        CHECK(report.interpolationProfiledNormalPrincipalSolves <=
              report.interpolationProfiledPoints);
        CHECK(report.interpolationPredictionClosedFormResolutions <=
              report.interpolatedScoringPoints);
        CHECK(report.interpolationNormalClosedFormResolutions <=
              report.interpolatedScoringPoints);
        CHECK(report.interpolationPredictionIterativeFallbacks <=
              report.interpolationPredictionClosedFormResolutions);
        CHECK(report.interpolationNormalIterativeFallbacks <=
              report.interpolationNormalClosedFormResolutions);
        CHECK(report.dpNodeIndexEntries <= report.retainedSearchNodes);
        CHECK(report.dpNodeIndexSlots >= report.dpNodeIndexEntries);
        CHECK(report.dpRelaxations <= report.dpTransitionLookups);
        CHECK(report.dpValidEdges > 0);
        CHECK(report.dpReusedEdges > 0);
        CHECK(report.preparationGeometryWorkSeconds >= 0.0);
        CHECK(report.preparationNodeEnumerationWorkSeconds >= 0.0);
        CHECK(report.preparationCornerCollectionWorkSeconds >= 0.0);
        CHECK(report.scoringIndexSeconds >= 0.0);
        CHECK(report.scoringPreparationSeconds >= 0.0);
        CHECK(report.interpolationMaterializationSeconds >= 0.0);
        CHECK(report.interpolationProfiledLookupSeconds >= 0.0);
        CHECK(report.interpolationProfiledPredictionCornerSeconds >= 0.0);
        CHECK(report.interpolationProfiledNormalCornerSeconds >= 0.0);
        CHECK(report.interpolationProfiledPredictionResolveSeconds >= 0.0);
        CHECK(report.interpolationProfiledNormalResolveSeconds >= 0.0);
        CHECK(report.searchNodeIndexWorkSeconds >= 0.0);
        CHECK(report.searchDpWorkSeconds >= 0.0);
    };
    checkProfile(serial);
    checkProfile(parallel);
    CHECK(serial.candidatePointPredicateCalls ==
          parallel.candidatePointPredicateCalls);
    CHECK(serial.latticeNodePositions == parallel.latticeNodePositions);
    CHECK(serial.corridorSegmentTests == parallel.corridorSegmentTests);
    CHECK(serial.corridorAcceptedNodes == parallel.corridorAcceptedNodes);
    CHECK(serial.nodePointPredicateCalls == parallel.nodePointPredicateCalls);
    CHECK(serial.retainedSearchNodes == parallel.retainedSearchNodes);
    CHECK(serial.interpolationCornerInsertions ==
          parallel.interpolationCornerInsertions);
    CHECK(serial.interpolatedScoringPoints ==
          parallel.interpolatedScoringPoints);
    CHECK(serial.endpointScoringInterpolations ==
          parallel.endpointScoringInterpolations);
    CHECK(serial.lazyNodeScoringRequests ==
          parallel.lazyNodeScoringRequests);
    CHECK(serial.lazyNodeScoringMaterializations ==
          parallel.lazyNodeScoringMaterializations);
    CHECK(serial.lazyNodeScoringCacheHits ==
          parallel.lazyNodeScoringCacheHits);
    CHECK(serial.scoringPageCount == parallel.scoringPageCount);
    CHECK(serial.scoringPageSlots == parallel.scoringPageSlots);
    CHECK(serial.scoringPageDirectoryProbes ==
          parallel.scoringPageDirectoryProbes);
    CHECK(serial.dpNodeIndexEntries == parallel.dpNodeIndexEntries);
    CHECK(serial.dpNodeIndexSlots == parallel.dpNodeIndexSlots);
    CHECK(serial.dpTransitionLookups == parallel.dpTransitionLookups);
    CHECK(serial.dpValidEdges == parallel.dpValidEdges);
    CHECK(serial.dpReusedEdges == parallel.dpReusedEdges);
    CHECK(serial.dpReachedStateVisits == parallel.dpReachedStateVisits);
    CHECK(serial.dpRelaxations == parallel.dpRelaxations);
    REQUIRE_FALSE(progress.empty());
    for (size_t index = 1; index < progress.size(); ++index) {
        if (progress[index - 1].phase == progress[index].phase)
            CHECK(progress[index - 1].completed < progress[index].completed);
    }
    CHECK(progress.back().phase == "search");
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
    CHECK(vc::fiber_tracer::fiberletPathReportObj(serial) == vc::fiber_tracer::fiberletPathReportObj(parallel));
}

TEST_CASE("fiberlet batches preserve global graph nodes and cross-batch transitions")
{
    const auto anchors = chainAnchorArtifact();
    const ConstantNormalSampler normals;
    vc::fiber_tracer::FiberletArtifactInfo artifact;
    artifact.fiberManifestLocator = "/tmp/fiber.lasagna.json";
    artifact.fiberManifestContentHash = "fnv1a64:1111111111111111";
    artifact.normalManifestLocator = "/tmp/normal.lasagna.json";
    artifact.normalManifestContentHash = "fnv1a64:2222222222222222";
    artifact.anchorArtifactLocator = "/tmp/anchors.json";
    artifact.anchorArtifactContentHash = "fnv1a64:3333333333333333";

    auto baselineConfig = pathConfig();
    baselineConfig.samplingBatchCoordinates = 32;
    baselineConfig.parallelThreads = 1;
    const auto baseline = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, baselineConfig, constantPredictions(), normals);
    REQUIRE(baseline.diagnostics.successfulPaths == 3);
    const auto baselineJson = vc::fiber_tracer::fiberletPathReportJson(baseline, artifact).dump();
    const auto baselineObj = vc::fiber_tracer::fiberletPathReportObj(baseline);
    const auto baselineGraphJson = vc::fiber_tracer::fiberletGraphJson(vc::fiber_tracer::buildFiberletGraph(baseline)).dump();

    for (const int batchSize : {1, 2, 32}) {
        for (const int threads : {1, 8}) {
            auto config = baselineConfig;
            config.samplingBatchCoordinates = batchSize;
            config.parallelThreads = threads;
            const auto candidate = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, config, constantPredictions(), normals);
            CAPTURE(batchSize);
            CAPTURE(threads);
            CHECK(vc::fiber_tracer::fiberletPathReportJson(candidate, artifact).dump() == baselineJson);
            CHECK(vc::fiber_tracer::fiberletPathReportObj(candidate) == baselineObj);
            CHECK(vc::fiber_tracer::fiberletGraphJson(vc::fiber_tracer::buildFiberletGraph(candidate)).dump() == baselineGraphJson);
        }
    }
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
    CHECK(report.sampledVoxels == 0);
    CHECK(report.samplingCoordinateBatches == 0);
    CHECK(report.candidateWorkers == 0);
    CHECK(report.candidates[0].reason == "axis_mismatch");
    REQUIRE_FALSE(progress.empty());
    CHECK(progress.front().phase == "candidate_generation");
    CHECK(progress.front().completed == 0);
    CHECK(progress.front().total == 2);
    CHECK(progress.back().phase == "search");
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

TEST_CASE("fiberlet invalid prediction slabs reject the DP path")
{
    const auto anchors = twoAnchorArtifact({1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 4.0, 4.0}, {10.0, 4.0, 4.0});
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
    const auto report = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, config, sampler, normals);
    CHECK(report.diagnostics.successfulPaths == 0);
    REQUIRE(report.candidates.size() == 1);
    CHECK(report.candidates[0].reason == "no_path");
    CHECK_FALSE(report.candidates[0].scoreValid);
}

TEST_CASE("fiberlet DP uses multiplicative presence and unoriented predictions")
{
    const auto anchors = twoAnchorArtifact({1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 4.0, 4.0}, {10.0, 4.0, 4.0});
    auto config = pathConfig();
    config.corridorRadiusPredictionVoxels = 0.01;
    const ConstantNormalSampler normals;
    const auto report = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, config, constantPredictions({-1.0, 0.0, 0.0}, 0.5), normals);

    REQUIRE_MESSAGE(report.diagnostics.successfulPaths == 1, report.candidates[0].reason);
    CHECK(report.candidates[0].cost.invalidPrediction == 0.0);
    const float quantizedCost = 6.0F * (1.0F - 128.0F / 255.0F);
    CHECK(report.candidates[0].cost.alignment ==
          doctest::Approx(quantizedCost).epsilon(1.0e-5));
    CHECK(report.candidates[0].cost.total() ==
          doctest::Approx(quantizedCost).epsilon(1.0e-5));
}

TEST_CASE("fiberlet multiplicative alignment changes the selected route")
{
    const auto anchors = twoAnchorArtifact({1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 4.0, 4.0}, {10.0, 4.0, 4.0});
    const auto sampler = [=](const auto& indices, int, auto& samples) {
        samples.clear();
        for (const auto& zyx : indices) {
            const size_t y = zyx[1];
            const double angle = 30.0 * std::acos(-1.0) / 180.0;
            cv::Vec3d direction{std::cos(angle), std::sin(angle), 0.0};
            if (y == 5) {
                direction = {1.0, 0.0, 0.0};
            }
            samples.push_back({direction, 1.0, true});
        }
    };
    const ConstantNormalSampler normals;
    auto narrowConfig = pathConfig();
    narrowConfig.corridorRadiusPredictionVoxels = 0.01;
    const auto narrow = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, narrowConfig, sampler, normals);
    const auto wide = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, pathConfig(), sampler, normals);

    CHECK(narrow.diagnostics.successfulPaths == 0);
    REQUIRE(wide.diagnostics.successfulPaths == 1);
    CHECK(std::any_of(wide.candidates[0].pointsPredictionXYZ.begin(), wide.candidates[0].pointsPredictionXYZ.end(), [](const cv::Vec3f& point) {
        return point[1] > 4.25;
    }));
}

TEST_CASE("fiberlet local grid follows a narrow subvoxel corridor")
{
    const auto anchors = twoAnchorArtifact({1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.2, 4.2, 4.2}, {10.2, 4.2, 4.2});
    auto config = pathConfig();
    config.maximumEndpointAngleDegrees = 60.0;
    config.corridorRadiusPredictionVoxels = 0.01;
    const ConstantNormalSampler normals;
    const auto report = vc::fiber_tracer::traceFiberletPaths(anchors, anchors.report.grid, config, constantPredictions(), normals);
    REQUIRE(report.diagnostics.searchedPairs == 1);
    CHECK(report.diagnostics.successfulPaths == 1);
    CHECK(report.diagnostics.noPathPairs == 0);
}

TEST_CASE("fiberlet packed node keys reject an unrepresentable transverse grid")
{
    const auto anchors = twoAnchorArtifact();
    auto config = pathConfig();
    config.transverseStepPredictionVoxels = 1.0e-20;
    const ConstantNormalSampler normals;
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::traceFiberletPaths(
            anchors, anchors.report.grid, config,
            constantPredictions(), normals),
        doctest::Contains("packed-key limits"), std::overflow_error);
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
    CHECK(json.at("parameters").at("dp_longitudinal_step_prediction_voxels") == 2.0);
    CHECK(json.at("parameters").at("dp_longitudinal_step_base_voxels") == 4.0);
    CHECK(json.at("parameters").at("dp_transverse_step_prediction_voxels") == 0.5);
    CHECK(json.at("parameters").at("dp_transverse_step_base_voxels") == 1.0);
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
    const auto graphJson = nlohmann::json::parse(readText(directory / "fiberlet_graph.json"));
    CHECK(graphJson.at("format") == "vc_fiberlet_graph");
    CHECK(graphJson.at("source").at("anchor_artifact_content_hash") == artifact.anchorArtifactContentHash);
    CHECK(graphJson.at("edges").size() == 1);
    CHECK(graphJson.at("edges").at(0).at("forward_arc") == 0);
    CHECK(graphJson.at("edges").at(0).at("reverse_arc") == 1);
    CHECK_FALSE(std::filesystem::exists(directory / "fiberlets.mtl"));
    std::filesystem::remove_all(directory);
}

TEST_CASE("fiberlet path artifacts reject scaled float overflow")
{
    auto report = graphPathReport();
    report.grid.predictionToBaseScale =
        static_cast<double>(std::numeric_limits<float>::max());
    addGraphPath(report, 0, 1, {{0, 0, 0}, {2, 0, 0}}, 1.0F);

    vc::fiber_tracer::FiberletArtifactInfo artifact;
    artifact.fiberManifestLocator = "/tmp/fiber.lasagna.json";
    artifact.fiberManifestContentHash = "fnv1a64:1111111111111111";
    artifact.normalManifestLocator = "/tmp/normal.lasagna.json";
    artifact.normalManifestContentHash = "fnv1a64:2222222222222222";
    artifact.anchorArtifactLocator = "/tmp/anchors.json";
    artifact.anchorArtifactContentHash = "fnv1a64:3333333333333333";

    CHECK_THROWS_AS(
        vc::fiber_tracer::fiberletPathReportJson(report, artifact),
        std::overflow_error);
    CHECK_THROWS_AS(
        vc::fiber_tracer::fiberletPathReportObj(report),
        std::overflow_error);
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
            {0.0F, static_cast<float>(index), 0.0F},
            {2.0F, static_cast<float>(index), 0.0F},
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
        candidate.pointsPredictionXYZ = {
            {0.0F, 0.0F, 0.0F},
            {1.0F + static_cast<float>(index), 0.0F, 0.0F}};
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
    CHECK(vc::fiber_tracer::fiberletPathReportObj(empty).find("# trace_loss_density_min none") != std::string::npos);
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
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::fiberletPathVisualMetrics(report), doctest::Contains("non-positive path length"), std::runtime_error);

    report.candidates[0].pointsPredictionXYZ[1] = {2.0, 1.0, 1.0};
    report.candidates[0].cost.alignment = -1.0;
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::fiberletPathVisualMetrics(report), doctest::Contains("component loss"), std::runtime_error);

    report.candidates[0].cost.alignment = 1.0;
    report.candidates.push_back(report.candidates[0]);
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::fiberletPathVisualMetrics(report), doctest::Contains("duplicate path identifier"), std::runtime_error);
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
    report.candidates[3].pointsPredictionXYZ = {{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
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
    CHECK(loaded.report.config.nmsTransverseRadiusPredictionVoxels == 2.0);
    CHECK(loaded.report.config.nmsLongitudinalRadiusPredictionVoxels == 1.0);
    CHECK(loaded.report.config.robustMaximumTrimMassFraction ==
          anchors.report.config.robustMaximumTrimMassFraction);
    CHECK(loaded.report.config.robustMadMultiplier ==
          anchors.report.config.robustMadMultiplier);
    CHECK(loaded.report.config.robustMinimumAngleDegrees ==
          anchors.report.config.robustMinimumAngleDegrees);

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

    auto missingRobustParameter =
        vc::fiber_tracer::fiberAnchorReportJson(anchors.report, anchors.artifact);
    missingRobustParameter["parameters"].erase("robust_mad_multiplier");
    {
        std::ofstream output(path);
        output << missingRobustParameter.dump(2);
    }
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::loadFiberAnchorArtifact(path),
        doctest::Contains("version-2 schema"), std::runtime_error);

    auto legacyRobustParameters =
        vc::fiber_tracer::fiberAnchorReportJson(anchors.report, anchors.artifact);
    legacyRobustParameters["parameters"].erase(
        "robust_maximum_trim_mass_fraction");
    legacyRobustParameters["parameters"].erase("robust_mad_multiplier");
    legacyRobustParameters["parameters"].erase(
        "robust_minimum_angle_degrees");
    legacyRobustParameters["version"] = 1;
    {
        std::ofstream output(path);
        output << legacyRobustParameters.dump(2);
    }
    const auto legacyLoaded = vc::fiber_tracer::loadFiberAnchorArtifact(path);
    CHECK(legacyLoaded.report.config.robustMaximumTrimMassFraction ==
          vc::fiber_tracer::FiberAnchorConfig{}.robustMaximumTrimMassFraction);

    auto oversizedGrid =
        vc::fiber_tracer::fiberAnchorReportJson(anchors.report,
                                                anchors.artifact);
    oversizedGrid["coordinates"]["prediction_shape_zyx"][2] =
        (size_t{1} << 24) + 1;
    {
        std::ofstream output(path);
        output << oversizedGrid.dump(2);
    }
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::loadFiberAnchorArtifact(path),
        doctest::Contains("exactly representable in float32"),
        std::runtime_error);

    auto unrepresentablePosition =
        vc::fiber_tracer::fiberAnchorReportJson(anchors.report,
                                                anchors.artifact);
    for (auto& cell : unrepresentablePosition["cells"]) {
        for (auto& component : cell["components"]) {
            if (component.value("retained", false))
                component["position_base_xyz"][0] = 1.0e100;
        }
    }
    REQUIRE(unrepresentablePosition["cells"][0]["components"][0]
                                   ["position_base_xyz"][0]
                                       .get<double>() >
            static_cast<double>(std::numeric_limits<float>::max()));
    {
        std::ofstream output(path);
        output << unrepresentablePosition.dump(2);
    }
    CHECK_THROWS_AS(vc::fiber_tracer::loadFiberAnchorArtifact(path),
                    std::runtime_error);

    auto missingTransverseNmsParameter = vc::fiber_tracer::fiberAnchorReportJson(anchors.report, anchors.artifact);
    missingTransverseNmsParameter["parameters"].erase("nms_transverse_radius_prediction_voxels");
    {
        std::ofstream output(path);
        output << missingTransverseNmsParameter.dump(2);
    }
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::loadFiberAnchorArtifact(path), doctest::Contains("version-2 schema"), std::runtime_error);

    auto missingRefinementParameter = vc::fiber_tracer::fiberAnchorReportJson(anchors.report, anchors.artifact);
    missingRefinementParameter["parameters"].erase("peak_sigma_prediction_voxels");
    {
        std::ofstream output(path);
        output << missingRefinementParameter.dump(2);
    }
    CHECK_THROWS(vc::fiber_tracer::loadFiberAnchorArtifact(path));

    auto missingGradientParameter = vc::fiber_tracer::fiberAnchorReportJson(anchors.report, anchors.artifact);
    missingGradientParameter["parameters"].erase("peak_gradient_weight");
    {
        std::ofstream output(path);
        output << missingGradientParameter.dump(2);
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
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::loadFiberAnchorArtifact(path), doctest::Contains("version-2 schema"), std::runtime_error);

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
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::loadFiberAnchorArtifact(path), doctest::Contains("owning cell"), std::runtime_error);

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
    json["version"] = 3;
    {
        std::ofstream output(path);
        output << json.dump(2);
    }
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::loadFiberAnchorArtifact(path),
        doctest::Contains("version 1 or 2"), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("fiberlet adjacent corridor segment fast path matches complete float scan")
{
    const std::vector<cv::Vec3f> reference{
        {0.0f, 0.0f, 0.0f},
        {2.0f, 0.0f, 0.0f},
        {4.0f, 1.0f, 0.0f},
        {6.0f, 1.0f, 1.0f},
        {8.0f, 2.0f, 1.0f},
    };
    constexpr float radius = 1.25f;
    std::mt19937 generator(12345);
    std::uniform_real_distribution<float> x(-1.0f, 9.0f);
    std::uniform_real_distribution<float> yz(-2.0f, 3.0f);
    for (size_t index = 0; index < 20000; ++index) {
        const cv::Vec3f point{x(generator), yz(generator), yz(generator)};
        const auto complete =
            vc::fiber_tracer::testing::debugFiberletCorridorContains(
                point, reference, radius);
        for (size_t adjacent = 0; adjacent + 1 < reference.size(); ++adjacent) {
            const auto accelerated =
                vc::fiber_tracer::testing::debugFiberletCorridorContains(
                    point, reference, radius, adjacent);
            CHECK(accelerated.inside == complete.inside);
        }
    }

    const auto immediate =
        vc::fiber_tracer::testing::debugFiberletCorridorContains(
            {3.0f, 1.0f, 0.0f}, reference, radius, 1);
    CHECK(immediate.inside);
    CHECK(immediate.segmentTests == 1);
}
