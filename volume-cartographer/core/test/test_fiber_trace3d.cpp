#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberTrace.hpp"

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

} // namespace

TEST_CASE("native fiber tracer fuses a straight cp-to-cp segment")
{
    StraightPrediction predictions;
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

    const auto result = vc::fiber_tracer::traceFiberSegment(predictions, request);

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

    const auto result = vc::fiber_tracer::traceWholeFiberMetric(predictions, request);

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
