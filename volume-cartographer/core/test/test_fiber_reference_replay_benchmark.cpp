#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberReferenceReplayBenchmark.hpp"

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

TEST_CASE("reference replay outcome credits first failure arc in either direction")
{
    vc::fiber_tracer::FiberReferenceReplayCase replayCase;
    replayCase.sourceIndex = 2;
    replayCase.runIndex = 3;
    replayCase.referenceLengthBaseVoxels = 10.0;
    vc::fiber_tracer::FiberletGraphReplayResult replay;
    replay.referenceBeginArcBase = 0.0;
    replay.referenceEndArcBase = 10.0;
    replay.failures.push_back({0, 0, "distance_above_threshold", 4.25});

    const auto forward = vc::fiber_tracer::measureFiberReferenceReplayOutcome(replayCase, replay);
    CHECK(forward.tracedLengthBaseVoxels == doctest::Approx(4.25));
    CHECK_FALSE(forward.completed);
    replayCase.reverse = true;
    replay.failures.front().referenceArcBase = 6.5;
    const auto reverse = vc::fiber_tracer::measureFiberReferenceReplayOutcome(replayCase, replay);
    CHECK(reverse.tracedLengthBaseVoxels == doctest::Approx(6.5));
    CHECK(reverse.reverse);
}

TEST_CASE("reference replay summary includes successes and explicit voxel scale")
{
    std::vector<vc::fiber_tracer::FiberReferenceReplayOutcome> outcomes(2);
    outcomes[0].referenceLengthBaseVoxels = 100.0;
    outcomes[0].tracedLengthBaseVoxels = 100.0;
    outcomes[0].completed = true;
    outcomes[1].reverse = true;
    outcomes[1].referenceLengthBaseVoxels = 100.0;
    outcomes[1].tracedLengthBaseVoxels = 25.0;
    outcomes[1].failureReason = "route_state_limit";

    const auto summary = vc::fiber_tracer::summarizeFiberReferenceReplay(1, outcomes, 5.0);
    CHECK(summary.directedCases == 2);
    CHECK(summary.completedCases == 1);
    CHECK(summary.meanCreditedLengthMillimeters == doctest::Approx(0.3125));
    CHECK(summary.meanFailureLengthMillimeters == doctest::Approx(0.125));
    CHECK(summary.lengthWeightedSuccessPercent == doctest::Approx(62.5));
    REQUIRE(summary.failureReasons.size() == 1);
    CHECK(summary.failureReasons.front().first == "route_state_limit");
}
