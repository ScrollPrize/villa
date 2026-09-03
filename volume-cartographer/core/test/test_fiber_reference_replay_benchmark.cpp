#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberReferenceReplayBenchmark.hpp"
#include "vc/lasagna/LineOptimizer.hpp"

#include <cmath>

namespace
{

class ConstantNormalSampler final : public vc::lasagna::NormalSampler
{
public:
    explicit ConstantNormalSampler(cv::Vec3d normal, bool valid = true)
        : normal_(normal), valid_(valid)
    {
    }

    vc::lasagna::NormalSample sampleNormal(
        const cv::Vec3d&) const override
    {
        return {normal_, valid_, valid_ ? "" : "missing"};
    }

private:
    cv::Vec3d normal_;
    bool valid_ = true;
};

}  // namespace

TEST_CASE("reference replay preparation retains crop re-entry runs in both directions")
{
    const std::vector<std::vector<cv::Vec3d>> fibers{{{-1, 1, 1}, {1, 1, 1}, {3, 1, 1}, {1, 1, 1}, {-1, 1, 1}}};
    const auto cases = vc::fiber_tracer::makeFiberReferenceReplayCases(fibers, {0, 0, 0}, {2, 2, 2});

    REQUIRE(cases.size() == 4);
    CHECK(cases[0].runIndex == 0);
    CHECK_FALSE(cases[0].reverse);
    CHECK(cases[1].reverse);
    CHECK(cases[2].runIndex == 1);
    CHECK(cases[0].pointsBaseXYZ.front()[0] == doctest::Approx(0.0));
    CHECK(cases[0].pointsBaseXYZ.back()[0] == doctest::Approx(2.0));
    CHECK(cases[1].pointsBaseXYZ.front() == cases[0].pointsBaseXYZ.back());
}

TEST_CASE("reference replay outcome retains every failure and credits only seeded spans")
{
    vc::fiber_tracer::FiberReferenceReplayCase replayCase;
    replayCase.sourceIndex = 2;
    replayCase.runIndex = 3;
    replayCase.referenceLengthBaseVoxels = 10.0;
    vc::fiber_tracer::FiberletGraphReplayResult replay;
    replay.referenceBeginArcBase = 0.0;
    replay.referenceEndArcBase = 10.0;
    replay.completedReferenceArcBase = 10.0;
    replay.segments.resize(3);
    replay.segments[0].seedKey = vc::fiber_tracer::FiberletStorageKey{};
    replay.segments[0].endReferenceArcBase = 4.25;
    replay.segments[0].matches.push_back({0, 0.0, 0.0, {}, 0.0, 1.0, {}});
    replay.segments[1].startReferenceArcBase = 4.25;
    replay.segments[1].endReferenceArcBase = 6.0;
    replay.segments[1].terminationReason = "missing_seed_gap";
    replay.segments[2].seedKey = vc::fiber_tracer::FiberletStorageKey{};
    replay.segments[2].endReferenceArcBase = 10.0;
    replay.segments[2].matches.push_back({0, 6.0, 6.0, {}, 6.0, 7.0, {}});
    replay.failures.push_back({0, 0, "distance_above_threshold", 4.25, 0.425, {4.25, 0, 0}});
    replay.failures.push_back({1, 1, "missing_seed_gap", 4.25, 0.425, {4.25, 0, 0}});

    const auto forward = vc::fiber_tracer::measureFiberReferenceReplayOutcome(replayCase, replay);
    CHECK(forward.tracedLengthBaseVoxels == doctest::Approx(8.25));
    CHECK(forward.evaluationComplete);
    CHECK_FALSE(forward.failureFree);
    REQUIRE(forward.failures.size() == 2);
    CHECK(forward.failures[0].tracedSpanBaseVoxels == doctest::Approx(4.25));
    CHECK(forward.failures[1].tracedSpanBaseVoxels == doctest::Approx(0.0));
    CHECK(forward.failures[0].sourceReferenceArcBaseVoxels == doctest::Approx(4.25));
    replayCase.reverse = true;
    const auto reverse = vc::fiber_tracer::measureFiberReferenceReplayOutcome(replayCase, replay);
    CHECK(reverse.reverse);
    CHECK(reverse.failures[0].sourceReferenceArcBaseVoxels == doctest::Approx(5.75));
}

TEST_CASE("reference replay summary includes successes and explicit voxel scale")
{
    std::vector<vc::fiber_tracer::FiberReferenceReplayOutcome> outcomes(2);
    outcomes[0].referenceLengthBaseVoxels = 100.0;
    outcomes[0].tracedLengthBaseVoxels = 100.0;
    outcomes[0].evaluatedThroughBaseVoxels = 100.0;
    outcomes[0].seededSpans = 1;
    outcomes[0].evaluationComplete = true;
    outcomes[0].failureFree = true;
    outcomes[1].reverse = true;
    outcomes[1].referenceLengthBaseVoxels = 100.0;
    outcomes[1].tracedLengthBaseVoxels = 75.0;
    outcomes[1].evaluatedThroughBaseVoxels = 100.0;
    outcomes[1].seededSpans = 2;
    outcomes[1].evaluationComplete = true;
    outcomes[1].failures.push_back({0, 0, "route_state_limit", 25.0, 0.25, 75.0, 0.75, 25.0});

    const auto summary = vc::fiber_tracer::summarizeFiberReferenceReplay(1, outcomes, 5.0);
    CHECK(summary.directedCases == 2);
    CHECK(summary.evaluatedCases == 2);
    CHECK(summary.failureFreeCases == 1);
    CHECK(summary.totalFailures == 1);
    CHECK(summary.meanCreditedLengthMillimeters == doctest::Approx(0.4375));
    CHECK(summary.meanSeededSpanLengthMillimeters == doctest::Approx(175.0 / 3.0 * 0.005));
    CHECK(summary.meanFailedSpanLengthMillimeters == doctest::Approx(0.125));
    CHECK(summary.lengthWeightedSuccessPercent == doctest::Approx(87.5));
    CHECK(summary.failuresPerDirectedMillimeter == doctest::Approx(1.0));
    CHECK(summary.meanDistancePerFailureBaseVoxels == doctest::Approx(200.0));
    CHECK(summary.meanDistancePerFailureMillimeters == doctest::Approx(1.0));
    CHECK(summary.meanDistancePerFailurePercent == doctest::Approx(100.0));
    REQUIRE(summary.failureReasons.size() == 1);
    CHECK(summary.failureReasons.front().first == "route_state_limit");
}

TEST_CASE("reference replay distance per failure handles zero and multiple failures")
{
    vc::fiber_tracer::FiberReferenceReplayOutcome outcome;
    outcome.referenceLengthBaseVoxels = 200.0;
    outcome.tracedLengthBaseVoxels = 200.0;
    outcome.evaluatedThroughBaseVoxels = 200.0;
    outcome.evaluationComplete = true;
    outcome.failureFree = true;

    const std::array zeroFailureOutcomes{outcome};
    const auto zero = vc::fiber_tracer::summarizeFiberReferenceReplay(1, zeroFailureOutcomes, 2.4);
    CHECK(zero.meanDistancePerFailureBaseVoxels == doctest::Approx(200.0));
    CHECK(zero.meanDistancePerFailureMillimeters == doctest::Approx(0.48));
    CHECK(zero.meanDistancePerFailurePercent == doctest::Approx(100.0));

    outcome.failureFree = false;
    outcome.failures.resize(4);
    for (std::size_t index = 0; index < outcome.failures.size(); ++index) {
        outcome.failures[index].index = index;
        outcome.failures[index].reason = "distance_above_threshold";
    }
    const std::array fourFailureOutcomes{outcome};
    const auto four = vc::fiber_tracer::summarizeFiberReferenceReplay(1, fourFailureOutcomes, 2.4);
    CHECK(four.meanDistancePerFailureBaseVoxels == doctest::Approx(50.0));
    CHECK(four.meanDistancePerFailureMillimeters == doctest::Approx(0.12));
    CHECK(four.meanDistancePerFailurePercent == doctest::Approx(25.0));

    outcome.evaluationComplete = false;
    outcome.evaluatedThroughBaseVoxels = 100.0;
    const std::array incompleteOutcomes{outcome};
    CHECK_THROWS_AS(vc::fiber_tracer::summarizeFiberReferenceReplay(1, incompleteOutcomes, 2.4), std::invalid_argument);
}

TEST_CASE("reference replay JSON version three preserves failure diagnostics")
{
    vc::fiber_tracer::FiberReferenceReplayOutcome outcome;
    outcome.referenceLengthBaseVoxels = 10.0;
    outcome.tracedLengthBaseVoxels = 8.0;
    outcome.evaluatedThroughBaseVoxels = 10.0;
    outcome.seededSpans = 2;
    outcome.evaluationComplete = true;
    vc::fiber_tracer::FiberReferenceReplayFailure failure;
    failure.reason = "distance_above_threshold";
    failure.directionalReferenceArcBaseVoxels = 3.0;
    failure.directionalReferenceArcFraction = 0.3;
    failure.sourceReferenceArcBaseVoxels = 7.0;
    failure.sourceReferenceArcFraction = 0.7;
    failure.referencePointBaseXYZ = {1.0, 2.0, 3.0};
    failure.evaluatorPointBaseXYZ = cv::Vec3d{4.0, 5.0, 6.0};
    failure.thresholdMeasurement = vc::fiber_tracer::FiberReplayThresholdMeasurement{5.0, 3.0, 4.0, std::sqrt(10.0), std::sqrt(10.0) / 2.0, true};
    outcome.failures.push_back(failure);
    const std::array outcomes{outcome};
    const auto summary = vc::fiber_tracer::summarizeFiberReferenceReplay(1, outcomes, 2.4);
    const nlohmann::json commonConfig = {
        {"error_threshold_base_voxels", 2.0},
        {"match_refine_steps", 1.0},
    };
    const auto json = vc::fiber_tracer::fiberReferenceReplayBenchmarkJson(
        summary, outcomes, "fiberlet", commonConfig,
        {{"beam_width", 16}}, {0, 0, 0}, {10, 10, 10});

    CHECK(json.at("version") == 3);
    CHECK(json.at("tracer") == "fiberlet");
    CHECK(json.at("backend_config").at("beam_width") == 16);
    CHECK(json.at("summary").at("mean_distance_per_failure_mm") == doctest::Approx(0.024));
    CHECK_FALSE(json.at("summary").at("zero_failure_convention_applied").get<bool>());
    const auto& failureJson = json.at("cases").at(0).at("failures").at(0);
    CHECK(failureJson.at("source_reference_arc_base_voxels") == doctest::Approx(7.0));
    CHECK(failureJson.at("threshold_measurement").at("normal_error_base_voxels") == doctest::Approx(3.0));
}

TEST_CASE("direct replay outcomes use the common benchmark accounting")
{
    vc::fiber_tracer::FiberReferenceReplayCase replayCase;
    replayCase.referenceLengthBaseVoxels = 10.0;
    vc::fiber_tracer::FiberReplayTraceResult replay;
    replay.referenceEndArcBase = 10.0;
    replay.completedReferenceArcBase = 10.0;
    replay.segments.resize(1);
    replay.segments[0].endReferenceArcBase = 10.0;
    replay.segments[0].matches.push_back(
        {0, 0.0, 0.0, {}, 0.0, 1.0, {}});

    const auto outcome =
        vc::fiber_tracer::measureFiberReferenceReplayOutcome(
            replayCase, replay);
    CHECK(outcome.evaluationComplete);
    CHECK(outcome.failureFree);
    CHECK(outcome.seededSpans == 1);
    CHECK(outcome.tracedLengthBaseVoxels == doctest::Approx(10.0));
}

TEST_CASE("Lasagna replay transports a reference tangent on a planar normal field")
{
    ConstantNormalSampler normals({0.0, 0.0, 1.0});
    vc::fiber_tracer::LasagnaReplayTraceRequest request;
    request.referencePointsBase = {{0, 0, 0}, {40, 0, 0}};
    request.stepBaseVoxels = 4.0;
    request.errorThresholdBaseVoxels = 1.0;
    const auto replay = vc::fiber_tracer::traceLasagnaReplay(normals, request);

    CHECK(replay.completedReferenceArcBase == doctest::Approx(40.0));
    CHECK(replay.failures.empty());
    REQUIRE(replay.segments.size() == 1);
    CHECK(replay.segments.front().tracePointsBase.back()[0] ==
          doctest::Approx(40.0));
}

TEST_CASE("Lasagna replay retains direction across invalid normal samples")
{
    ConstantNormalSampler normals({0.0, 0.0, 0.0}, false);
    vc::fiber_tracer::LasagnaReplayTraceRequest request;
    request.referencePointsBase = {{0, 0, 0}, {20, 0, 0}};
    request.stepBaseVoxels = 4.0;
    request.errorThresholdBaseVoxels = 1.0;
    const auto replay = vc::fiber_tracer::traceLasagnaReplay(normals, request);

    CHECK(replay.completedReferenceArcBase == doctest::Approx(20.0));
    CHECK(replay.failures.empty());
}

TEST_CASE("Lasagna tangent transport resolves antipodal normal encodings")
{
    const cv::Vec3d direction{1.0, 0.0, 0.0};
    const auto same = vc::lasagna::transportDirectionToNormalPlane(
        direction, {0.0, 0.0, 1.0}, {0.0, 0.0, -1.0});
    CHECK(same[0] == doctest::Approx(1.0));
    CHECK(same[1] == doctest::Approx(0.0));
    CHECK(same[2] == doctest::Approx(0.0));
}
