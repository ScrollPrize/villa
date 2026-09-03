#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberReferenceReplayBenchmark.hpp"

#include <cmath>

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
    REQUIRE(summary.failureReasons.size() == 1);
    CHECK(summary.failureReasons.front().first == "route_state_limit");
}

TEST_CASE("reference replay JSON version two preserves failure diagnostics")
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
    vc::fiber_tracer::FiberletGraphReplayConfig config;
    config.errorThresholdBaseVoxels = 2.0;
    const auto json = vc::fiber_tracer::fiberReferenceReplayBenchmarkJson(summary, outcomes, config, 8.0, {0, 0, 0}, {10, 10, 10});

    CHECK(json.at("version") == 2);
    CHECK(json.at("config").at("seed_window_base_voxels") == doctest::Approx(8.0));
    const auto& failureJson = json.at("cases").at(0).at("failures").at(0);
    CHECK(failureJson.at("source_reference_arc_base_voxels") == doctest::Approx(7.0));
    CHECK(failureJson.at("threshold_measurement").at("normal_error_base_voxels") == doctest::Approx(3.0));
}
