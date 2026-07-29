#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberTrace.hpp"

#include <string>
#include <utility>
#include <vector>

namespace {

class StraightPrediction final : public vc::fiber_tracer::FiberPredictionSource {
public:
    vc::fiber_tracer::FiberPredictionSample sample(
        const cv::Vec3d&,
        const cv::Vec3d& referenceDirection) const override
    {
        vc::fiber_tracer::FiberPredictionSample out;
        out.options.push_back({
            referenceDirection[0] < 0.0
                ? cv::Vec3d{-1.0, 0.0, 0.0}
                : cv::Vec3d{1.0, 0.0, 0.0},
            1.0,
            true,
        });
        return out;
    }
};

class ConstantNormalSampler final : public vc::lasagna::NormalSampler {
public:
    explicit ConstantNormalSampler(cv::Vec3d normal = {0.0, 1.0, 0.0})
        : normal_(normal)
    {
    }

    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override
    {
        return {normal_, true, {}};
    }

private:
    cv::Vec3d normal_;
};

vc::lasagna::LasagnaChannelGroup makeGroup(
    std::string name,
    int scaledown,
    std::vector<std::string> channels)
{
    vc::lasagna::LasagnaChannelGroup group;
    group.name = std::move(name);
    group.scaledown = scaledown;
    group.channels = std::move(channels);
    return group;
}

} // namespace

TEST_CASE("native fiber tracer defaults match regular Trace2CP command")
{
    const vc::fiber_tracer::FiberTraceConfig config;

    CHECK(config.stepVoxels == doctest::Approx(4.0));
    CHECK(config.beamWidth == 8);
    CHECK(config.beamLookaheadSteps == 2);
    CHECK(config.smoothnessNormalWeight == doctest::Approx(0.1));
    CHECK(config.smoothnessTangentWeight == doctest::Approx(10.0));
}

TEST_CASE("fiber prediction working scale is inferred from single-output manifest")
{
    vc::lasagna::LasagnaDatasetManifest manifest;
    manifest.sourceToBase = 2.0;
    manifest.groups.push_back(makeGroup("fiber", 3, {"presence", "nx", "ny"}));

    const double scale =
        vc::fiber_tracer::inferFiberPredictionWorkingToBaseScale(manifest);

    CHECK(scale == doctest::Approx(16.0));
}

TEST_CASE("fiber prediction working scale is inferred from prefixed multi-output manifest")
{
    vc::lasagna::LasagnaDatasetManifest manifest;
    manifest.sourceToBase = 1.0;
    manifest.groups.push_back(makeGroup(
        "fiber_option_000",
        2,
        {"option_000_presence", "option_000_nx", "option_000_ny"}));
    manifest.groups.push_back(makeGroup(
        "fiber_option_001",
        2,
        {"option_001_presence", "option_001_nx", "option_001_ny"}));

    const double scale =
        vc::fiber_tracer::inferFiberPredictionWorkingToBaseScale(manifest);

    CHECK(scale == doctest::Approx(4.0));
}

TEST_CASE("fiber prediction working scale rejects missing prediction channels")
{
    vc::lasagna::LasagnaDatasetManifest manifest;
    manifest.sourceToBase = 1.0;
    manifest.groups.push_back(makeGroup("fiber", 2, {"presence", "nx"}));

    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::inferFiberPredictionWorkingToBaseScale(manifest),
        doctest::Contains("presence/nx/ny"),
        std::runtime_error);
}

TEST_CASE("fiber prediction working scale rejects mixed prediction channel scales")
{
    vc::lasagna::LasagnaDatasetManifest manifest;
    manifest.sourceToBase = 1.0;
    manifest.groups.push_back(makeGroup("presence", 2, {"presence"}));
    manifest.groups.push_back(makeGroup("directions", 3, {"nx", "ny"}));

    CHECK_THROWS_AS(
        vc::fiber_tracer::inferFiberPredictionWorkingToBaseScale(manifest),
        std::runtime_error);
}

TEST_CASE("native fiber tracer requires normals for normal-aware smoothness")
{
    StraightPrediction predictions;
    vc::fiber_tracer::FiberTraceOneWayRequest request;
    request.startPoint = {0.0, 0.0, 0.0};
    request.targetPoint = {16.0, 0.0, 0.0};
    request.initialDirection = {1.0, 0.0, 0.0};
    request.targetPlaneNormal = {1.0, 0.0, 0.0};
    request.budgetSpanVoxels = 16.0;
    request.config.stepVoxels = 4.0;
    request.config.coneAngleDegrees = 0.0;
    request.config.beamWidth = 1;

    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::traceFiberOneWay(predictions, request, nullptr),
        doctest::Contains("Lasagna normal sampler"),
        std::invalid_argument);

    request.config.smoothnessNormalWeight = 0.0;
    request.config.smoothnessTangentWeight = 0.0;
    CHECK_NOTHROW(
        vc::fiber_tracer::traceFiberOneWay(predictions, request, nullptr));
}

TEST_CASE("native fiber tracer fuses a straight cp-to-cp segment")
{
    StraightPrediction predictions;
    ConstantNormalSampler normals;
    vc::fiber_tracer::FiberTraceSegmentRequest request;
    request.referenceLine = {
        {0.0, 0.0, 0.0},
        {64.0, 0.0, 0.0},
    };
    request.startIndex = 0;
    request.targetIndex = 1;
    request.targetPlaneNormal = cv::Vec3d{1.0, 0.0, 0.0};
    request.config.stepVoxels = 4.0;
    request.config.coneAngleDegrees = 0.0;
    request.config.coneAngleStepDegrees = 5.0;
    request.config.beamWidth = 1;
    request.config.maxStepFactor = 2.0;
    request.config.endpointAcceptThresholdUm = 50.0;
    request.config.voxelSizeUm = 1.0;

    const auto result =
        vc::fiber_tracer::traceFiberSegment(predictions, request, &normals);

    CHECK(result.forward.reachedTargetPlane);
    CHECK(result.reverse.reachedTargetPlane);
    CHECK(result.accepted);
    REQUIRE(result.fusedLine.size() >= 3);
    CHECK(result.fusedLine.front()[0] == doctest::Approx(0.0));
    CHECK(result.fusedLine.back()[0] == doctest::Approx(64.0));
    CHECK(result.maxEndpointErrorVoxels == doctest::Approx(0.0));
}

TEST_CASE("native fiber tracer computes whole-fiber one-way restart metric")
{
    StraightPrediction predictions;
    ConstantNormalSampler normals;
    vc::fiber_tracer::FiberInput fiber;
    fiber.path = "synthetic_fiber.json";
    fiber.linePointsXyzBase = {
        {0.0, 0.0, 0.0},
        {64.0, 0.0, 0.0},
        {128.0, 0.0, 0.0},
    };
    fiber.controlPointsXyzBase = fiber.linePointsXyzBase;
    fiber.controlPointLineIndices = {0, 1, 2};

    vc::fiber_tracer::FiberTraceWholeFiberMetricRequest request;
    request.fiber = fiber;
    request.workingToBaseScale = 1.0;
    request.errorThresholdVoxels = 10.0;
    request.voxelSizeUm = 2.0;
    request.config.stepVoxels = 4.0;
    request.config.coneAngleDegrees = 0.0;
    request.config.coneAngleStepDegrees = 5.0;
    request.config.beamWidth = 1;
    request.config.maxStepFactor = 2.0;

    const auto result =
        vc::fiber_tracer::traceWholeFiberMetric(predictions, request, &normals);

    CHECK(result.segmentCount == 2);
    CHECK(result.restartCount == 0);
    CHECK(result.restartsPerKvx == doctest::Approx(0.0));
    REQUIRE(result.referenceLengthMeters.has_value());
    CHECK(*result.referenceLengthMeters == doctest::Approx(0.000256));
    REQUIRE(result.restartsPerMeter.has_value());
    CHECK(*result.restartsPerMeter == doctest::Approx(0.0));
    REQUIRE(result.segments.size() == 2);
    CHECK(result.segments[0].success);
    CHECK(result.segments[1].success);
}
