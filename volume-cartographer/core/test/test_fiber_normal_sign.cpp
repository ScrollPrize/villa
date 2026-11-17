#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberNormalSign.hpp"

#include <cmath>
#include <limits>
#include <vector>

using namespace vc::fiber_tracer;

namespace
{

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

FiberSignOptions signOnly(const FiberSignBackend backend)
{
    FiberSignOptions options;
    options.backend = backend;
    options.fixedKeep = true;  // the sign problem alone, no rejection
    options.history = true;
    return options;
}

std::vector<int> signsOf(const FiberSignResult& result)
{
    return {result.sign.begin(), result.sign.end()};
}

}  // namespace

TEST_CASE("wrapped residual, Huber cost and winding-distance factor")
{
    CHECK(fiberSignResidual(0.0) == doctest::Approx(0.0));
    CHECK(fiberSignResidual(0.25) == doctest::Approx(0.25));
    CHECK(fiberSignResidual(1.0) == doctest::Approx(-1.0));      // the wrap lands on the lower end
    CHECK(fiberSignResidual(1.75) == doctest::Approx(-0.25));
    CHECK(fiberSignResidual(2.25) == doctest::Approx(0.25));      // period two
    CHECK(fiberSignResidual(-0.25) == doctest::Approx(-0.25));
    CHECK(fiberSignResidual(1.25, 1.0) == doctest::Approx(0.25));  // a target of one is the odd class

    CHECK(fiberSignCost(0.02, 0.05) == doctest::Approx(0.5 * 0.02 * 0.02 / 0.05));
    CHECK(fiberSignCost(0.5, 0.05) == doctest::Approx(0.5 - 0.025));
    CHECK(fiberSignCost(-0.5, 0.0) == doctest::Approx(0.5));

    CHECK(fiberSignWindingDistanceFactor(0.4, FiberSignDistanceWeight::InverseSquare) == doctest::Approx(1.0));
    CHECK(fiberSignWindingDistanceFactor(2.0, FiberSignDistanceWeight::InverseSquare) == doctest::Approx(0.25));
    CHECK(fiberSignWindingDistanceFactor(-3.0, FiberSignDistanceWeight::InverseSquare) == doctest::Approx(1.0 / 9.0));
    CHECK(fiberSignWindingDistanceFactor(kNaN, FiberSignDistanceWeight::InverseSquare) == doctest::Approx(1.0));
    CHECK(fiberSignWindingDistanceFactor(5.0, FiberSignDistanceWeight::None) == doctest::Approx(1.0));
}

TEST_CASE("a chain of crossings propagates one radial prior along the fibers")
{
    // Fiber 0 is held outward by its radial prior; every crossing opposes the
    // two axes it meets, so the signs must alternate down the chain.
    const std::vector<double> agreement{0.5, 0.0, 0.0, 0.0};
    const std::vector<FiberSignEvent> events{{0, 1, -1.0, 0.2}, {1, 2, -1.0, 0.2}, {2, 3, -1.0, 0.2}};
    for (const auto backend : {FiberSignBackend::Irls, FiberSignBackend::Ceres}) {
        const FiberSignResult result = solveFiberNormalSigns(agreement, events, signOnly(backend));
        CHECK(signsOf(result) == std::vector<int>{-1, 1, -1, 1});
        CHECK(result.metrics.weightedDisagreementPercent == doctest::Approx(0.0));
        CHECK(result.metrics.strictEvents == 3);
        CHECK(result.metrics.strictDisagreementPercent == doctest::Approx(0.0));
        CHECK(result.events == 3);
        CHECK(result.isolatedNodes == 0);
        CHECK(result.eventsWithWindingDistance == 3);
        CHECK(!result.history.empty());
    }
}

TEST_CASE("an agreeing crossing keeps two fibers on the same sign")
{
    const std::vector<double> agreement{0.9, 0.0, 0.0};
    const std::vector<FiberSignEvent> events{{0, 1, -0.95, 0.1}, {1, 2, 0.95, 0.1}};
    for (const auto backend : {FiberSignBackend::Irls, FiberSignBackend::Ceres}) {
        const FiberSignResult result = solveFiberNormalSigns(agreement, events, signOnly(backend));
        CHECK(signsOf(result) == std::vector<int>{-1, 1, 1});
        CHECK(result.metrics.weightedDisagreementPercent == doctest::Approx(0.0));
    }
}

TEST_CASE("a frustrated triangle pays for exactly one crossing")
{
    // Three fibers that pairwise oppose cannot all be satisfied: one crossing
    // of the three must disagree, and the metric must say so.
    const std::vector<double> agreement{0.5, 0.0, 0.0};
    const std::vector<FiberSignEvent> events{{0, 1, -1.0, 0.1}, {1, 2, -1.0, 0.1}, {0, 2, -1.0, 0.1}};
    for (const auto backend : {FiberSignBackend::Irls, FiberSignBackend::Ceres}) {
        const FiberSignResult result = solveFiberNormalSigns(agreement, events, signOnly(backend));
        CHECK(result.metrics.strictEvents == 3);
        CHECK(result.metrics.strictDisagreementPercent == doctest::Approx(100.0 / 3.0));
        CHECK(result.metrics.weightedDisagreementPercent == doctest::Approx(100.0 / 3.0));
    }
}

TEST_CASE("fixed signs hold against the crossing evidence")
{
    // Both fibers are pinned to opposite inward signs while their crossing
    // asks for the same sign: the prior outweighs the normalized evidence.
    const std::vector<double> agreement{0.0, 0.0};
    const std::vector<FiberSignEvent> events{{0, 1, 1.0, 0.1}};
    const std::vector<int> fixed{1, -1};
    for (const auto backend : {FiberSignBackend::Irls, FiberSignBackend::Ceres}) {
        const FiberSignResult result = solveFiberNormalSigns(agreement, events, signOnly(backend), fixed);
        CHECK(signsOf(result) == std::vector<int>{1, -1});
        CHECK(result.fixedNodes == 2);
        CHECK(result.metrics.weightedDisagreementPercent == doctest::Approx(100.0));
    }
    // Pinning them to the same sign satisfies the crossing instead.
    const FiberSignResult agreeing =
        solveFiberNormalSigns(agreement, events, signOnly(FiberSignBackend::Irls), {1, 1});
    CHECK(signsOf(agreeing) == std::vector<int>{1, 1});
    CHECK(agreeing.metrics.weightedDisagreementPercent == doctest::Approx(0.0));
}

TEST_CASE("the winding-distance factor lets a close crossing outweigh a far one")
{
    // Fiber 1 is pulled one way by a crossing within a winding and the other
    // way by one three windings out: 1/9 of the weight loses.
    const std::vector<double> agreement{0.5, 0.0, -0.5};
    const std::vector<FiberSignEvent> close{{0, 1, -1.0, 0.1}, {1, 2, 1.0, 3.0}};
    FiberSignOptions options = signOnly(FiberSignBackend::Irls);
    const FiberSignResult weighted = solveFiberNormalSigns(agreement, close, options);
    CHECK(weighted.sign[1] == 1);  // the close crossing decided it
    CHECK(weighted.metrics.strictEvents == 1);
    options.distanceWeight = FiberSignDistanceWeight::None;
    const FiberSignResult unweighted = solveFiberNormalSigns(agreement, close, options);
    CHECK(unweighted.metrics.strictEvents == 1);
}

TEST_CASE("both backends agree on a ring of crossings and report the same objective")
{
    // A ring with one frustrating crossing: the two backends must reach the
    // same signs and, being the same objective, the same value.
    const std::size_t count = 12;
    std::vector<double> agreement(count, 0.0);
    agreement[0] = 0.8;
    std::vector<FiberSignEvent> events;
    for (std::size_t v = 0; v < count; ++v) {
        const double dot = v % 3 == 0 ? 0.9 : -0.85;
        events.push_back({std::int64_t(v), std::int64_t((v + 1) % count), dot, 0.3 + 0.1 * double(v % 4)});
    }
    const FiberSignResult irls = solveFiberNormalSigns(agreement, events, signOnly(FiberSignBackend::Irls));
    const FiberSignResult ceres = solveFiberNormalSigns(agreement, events, signOnly(FiberSignBackend::Ceres));
    CHECK(signsOf(irls) == signsOf(ceres));
    CHECK(ceres.objective == doctest::Approx(irls.objective).epsilon(0.02));
    CHECK(ceres.metrics.weightedDisagreementPercent
          == doctest::Approx(irls.metrics.weightedDisagreementPercent).epsilon(1e-6));
}

TEST_CASE("an isolated fiber keeps the sign of its radial prior")
{
    const std::vector<double> agreement{0.7, -0.7, 0.0};
    const std::vector<FiberSignEvent> events{{0, 1, 1.0, 0.1}};
    const FiberSignResult result = solveFiberNormalSigns(agreement, events, signOnly(FiberSignBackend::Irls));
    CHECK(result.isolatedNodes == 1);
    CHECK(result.sign[2] == -1);  // outward +1 from the default initialization
}

TEST_CASE("rejection can drop a fiber whose crossings cannot be satisfied")
{
    // Fiber 2 contradicts a well-supported pair from both sides; with a small
    // rejection penalty the solve is allowed to stop paying for it.
    const std::vector<double> agreement{0.9, 0.9, 0.0, 0.0};
    const std::vector<FiberSignEvent> events{
        {0, 1, 1.0, 0.1}, {0, 2, -1.0, 0.1}, {1, 2, -1.0, 0.1}, {2, 3, 1.0, 0.1}};
    FiberSignOptions options;
    options.history = true;
    options.brokenPenalty = 1e-4;
    const FiberSignResult result = solveFiberNormalSigns(agreement, events, options);
    CHECK(result.keep.size() == 4);
    CHECK(result.metrics.expectedBrokenPercent >= 0.0);
    CHECK(result.objective <= result.history.front().objective);
}

TEST_CASE("invalid input is rejected")
{
    const std::vector<double> agreement{0.5, 0.0};
    const std::vector<FiberSignEvent> events{{0, 1, -1.0, 0.1}};
    CHECK_THROWS_AS(solveFiberNormalSigns({}, events), std::invalid_argument);
    CHECK_THROWS_AS(solveFiberNormalSigns(agreement, {{0, 2, -1.0, 0.1}}), std::invalid_argument);
    CHECK_THROWS_AS(solveFiberNormalSigns(agreement, {{0, 1, -2.0, 0.1}}), std::invalid_argument);
    CHECK_THROWS_AS(solveFiberNormalSigns({0.5, kNaN}, events), std::invalid_argument);
    CHECK_THROWS_AS(solveFiberNormalSigns(agreement, events, {}, {1, 2}), std::invalid_argument);
    CHECK_THROWS_AS(solveFiberNormalSigns(agreement, events, {}, {1}), std::invalid_argument);
    FiberSignOptions options;
    options.epsilon = 0.0;
    CHECK_THROWS_AS(solveFiberNormalSigns(agreement, events, options), std::invalid_argument);
    options = {};
    options.dotWeightPower = 0.5;
    CHECK_THROWS_AS(solveFiberNormalSigns(agreement, events, options), std::invalid_argument);
}

TEST_CASE("the thread count does not change the solution")
{
    // Large enough that the node and event loops span several work units:
    // below that every backend runs serially and the check is vacuous.
    std::vector<double> agreement(20000, 0.0);
    std::vector<FiberSignEvent> events;
    for (std::size_t v = 0; v + 1 < agreement.size(); ++v) {
        agreement[v] = v % 7 == 0 ? 0.6 : 0.0;
        events.push_back({std::int64_t(v), std::int64_t(v + 1), v % 5 == 0 ? 0.8 : -0.9, 0.5});
        if (v + 7 < agreement.size()) events.push_back({std::int64_t(v), std::int64_t(v + 7), -0.7, 1.5});
    }
    FiberSignOptions serial = signOnly(FiberSignBackend::Irls);
    FiberSignOptions threaded = serial;
    threaded.threads = 8;
    serial.iterations = threaded.iterations = 20;
    const FiberSignResult one = solveFiberNormalSigns(agreement, events, serial);
    const FiberSignResult many = solveFiberNormalSigns(agreement, events, threaded);
    CHECK(one.nodeChunks > 1);
    CHECK(one.eventChunks > 1);
    CHECK(signsOf(one) == signsOf(many));
    CHECK(one.objective == doctest::Approx(many.objective).epsilon(1e-12));
    CHECK(one.linearIterations == many.linearIterations);
}
