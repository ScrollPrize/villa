#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberSignSupport.hpp"

#include <cmath>
#include <limits>
#include <vector>

using namespace vc::fiber_tracer;

namespace
{

// Fibers A, B, C (0-2) cross pairwise: a loop, the core. D (3) hangs off A by
// one crossing, E (4) off D; F (5) crosses nothing; G, H (6, 7) only each other.
std::vector<FiberSupportCrossing> deadEnds()
{
    return {{0, 1, 100, 100}, {0, 2, 900, 100}, {1, 2, 900, 900},  // core loop
            {0, 3, 500, 0},                                       // D at its arc 0
            {3, 4, 500, 0},                                       // E at its arc 0
            {6, 7, 50, 50}};
}

const std::vector<double> kLengths{1000, 1000, 1000, 1000, 300, 400, 100, 100};

FiberSupportPoint at(const FiberSupportResult& r, const int fiber, const double arc)
{
    for (const auto& p : r.points[fiber])
        if (p.arc == arc) return p;
    FAIL("no sample at arc " << arc);
    return {};
}

}  // namespace

TEST_CASE("missed-flip length grows along dead ends and vanishes on loops")
{
    const auto r = fiberSignSupport(kLengths, deadEnds(), 4.0);
    // A inside the loop: supported both ways; its tail before the first crossing counts.
    CHECK(at(r, 0, 500).missedLength == 0.0);
    CHECK(at(r, 0, 0).missedLength == doctest::Approx(100));
    CHECK(at(r, 0, 0).missedCrossingCost == 0.0);
    // D: one bridging crossing, then its chain is a single path.
    CHECK(at(r, 3, 0).missedLength == 0.0);
    CHECK(at(r, 3, 0).missedCrossingCost == 1.0);
    CHECK(at(r, 3, 252).missedLength == doctest::Approx(252));
    CHECK(at(r, 3, 1000).missedLength == doctest::Approx(1000));  // twice as far, twice the length
    // E: behind D's 500 voxels and a second bridging crossing.
    CHECK(at(r, 4, 256).missedLength == doctest::Approx(756));
    CHECK(at(r, 4, 256).missedCrossingCost == 2.0);
    // Not linked to the core at all.
    CHECK(std::isinf(at(r, 5, 0).missedLength));
    CHECK(std::isinf(at(r, 6, 52).missedLength));
    CHECK(r.unlinkedEnds == 2);
    CHECK(r.bridgeCrossings == 3);
}

TEST_CASE("closing a dead end into a loop removes its first-order risk")
{
    auto crossings = deadEnds();
    crossings.push_back({3, 2, 1000, 500});  // D's far end now crosses C
    const auto r = fiberSignSupport(kLengths, crossings, 4.0);
    CHECK(at(r, 3, 252).missedLength == 0.0);
    CHECK(at(r, 3, 252).missedCrossingCost == 0.0);
    CHECK(at(r, 4, 256).missedLength == doctest::Approx(256));  // only E's own tail
    CHECK(at(r, 4, 256).missedCrossingCost == 1.0);
}

TEST_CASE("a weak bridging crossing costs one over its weight")
{
    auto crossings = deadEnds();
    crossings[4].weight = 0.25;  // D-E crossing far apart or oblique
    const auto r = fiberSignSupport(kLengths, crossings, 4.0);
    CHECK(at(r, 4, 256).missedCrossingCost == doctest::Approx(5.0));  // A-D 1 + D-E 4
    CHECK_THROWS(fiberSignSupport(kLengths, {{0, 1, 0, 0, 0.0}}, 4.0));
}

TEST_CASE("parallel crossings between the same ends are no bridge")
{
    // D crosses A twice at the same arcs: two independent crossing measurements.
    auto crossings = deadEnds();
    crossings.push_back({0, 3, 500, 0});
    const auto r = fiberSignSupport(kLengths, crossings, 4.0);
    CHECK(at(r, 3, 0).missedCrossingCost == 0.0);
}

TEST_CASE("calibration separates wrong crossings from flipped stretches in the core")
{
    // Fibers 0-4 cross fibers 5-9: a 5 x 5 grid, every crossing in the core.
    // Crossing (h, v) lies at arc 100 + 200 (v - 5) on h and 100 + 200 h on v.
    const std::vector<double> lengths(10, 1000.0);
    std::vector<FiberSupportCrossing> agreeing, disagreeing;
    for (int h = 0; h < 5; ++h)
        for (int v = 5; v < 10; ++v) {
            const FiberSupportCrossing c{h, v, 100.0 + 200.0 * (v - 5), 100.0 + 200.0 * h, 1.0};
            // (3, 6) alone is a wrong crossing; (1, 7) and (1, 8) flip a stretch of fiber 1.
            const bool wrong = (h == 3 && v == 6) || (h == 1 && (v == 7 || v == 8));
            (wrong ? disagreeing : agreeing).push_back(c);
        }
    const auto c = fiberSignCalibration(lengths, agreeing, disagreeing);
    CHECK(c.coreCrossings == 25);
    CHECK(c.coreDisagreeing == 3);
    CHECK(c.wrongCrossings == 1);
    CHECK(c.chainSwitches == 2);
    CHECK(c.coreChainLength == doctest::Approx(10 * 800.0));
    CHECK(c.lambda == doctest::Approx(2.0 / 8000.0));
    CHECK(c.q == doctest::Approx(1.0 / 25.0));
    // Too few flips per bin for a bound by default; with one flip enough, the bins give it.
    CHECK(c.lambdaBound == 0.0);
    const auto loose = fiberSignCalibration(lengths, agreeing, disagreeing, 1);
    CHECK(loose.lambdaBound == doctest::Approx(2.0 / 8000.0));  // every gap is 200 voxels: one bin
    CHECK(loose.qBound == doctest::Approx(1.0 / 25.0));
}

double cutAt(const FiberCutResult& r, const int fiber, const double arc)
{
    for (const auto& p : r.points[fiber])
        if (p.arc == arc) return p.cut;
    FAIL("no sample at arc " << arc);
    return 0.0;
}

TEST_CASE("local cut: a pendant fiber, a loosely tied pair and a well-connected fiber")
{
    // Fibers 0-4 cross fibers 5-9 (K5,5). P (10) crosses fiber 0 once; Q (11)
    // and R (12) cross each other and hang on fibers 1 and 2 by half-weight
    // crossings; S (13) crosses nothing.
    const std::vector<double> lengths(14, 100.0);
    std::vector<FiberSupportCrossing> crossings;
    for (int h = 0; h < 5; ++h)
        for (int v = 5; v < 10; ++v) crossings.push_back({h, v, 0, 0, 1.0});
    crossings.push_back({10, 0, 0, 0, 1.0});
    crossings.push_back({11, 12, 0, 0, 1.0});
    crossings.push_back({11, 1, 0, 0, 0.5});
    crossings.push_back({12, 2, 0, 0, 0.5});
    FiberCutOptions options;
    options.maxHops = 1;
    options.threads = 2;
    const auto r = fiberSignCut(lengths, crossings, 50.0, options);
    CHECK(cutAt(r, 3, 0) == doctest::Approx(4.0));    // five ways out, capped
    CHECK(cutAt(r, 10, 50) == doctest::Approx(1.0));  // one crossing; the tail takes it too
    CHECK(cutAt(r, 11, 0) == doctest::Approx(1.0));   // the pair together: 0.5 + 0.5
    CHECK(cutAt(r, 13, 0) == 0.0);
    // A second crossing between P and fiber 0 doubles its cut.
    crossings.push_back({10, 0, 5, 5, 1.0});
    CHECK(cutAt(fiberSignCut(lengths, crossings, 50.0, options), 10, 0) == doctest::Approx(2.0));
}

TEST_CASE("local cut: a chain bridge splits a fiber")
{
    // Fiber 14 crosses fibers 0 and 1 near its start, then 15 at arc 500; 15
    // crosses nothing else, so 14's chain from 10 to 500 is a bridge.
    std::vector<double> lengths(16, 100.0);
    lengths[14] = 600.0;
    std::vector<FiberSupportCrossing> crossings;
    for (int h = 0; h < 5; ++h)
        for (int v = 5; v < 10; ++v) crossings.push_back({h, v, 0, 0, 1.0});
    crossings.push_back({14, 0, 0, 0, 1.0});
    crossings.push_back({14, 1, 10, 0, 1.0});
    crossings.push_back({14, 15, 500, 0, 1.0});
    FiberCutOptions options;
    options.chainWeightLength = 49.0;  // 490 voxels of bridge weigh 0.1
    const auto r = fiberSignCut(lengths, crossings, 50.0, options);
    CHECK(cutAt(r, 14, 0) == doctest::Approx(2.0));    // two crossings into the core
    CHECK(cutAt(r, 14, 550) == doctest::Approx(0.1));  // beyond the bridge
    CHECK(cutAt(r, 14, 250) == doctest::Approx(0.1));  // on it
    CHECK(cutAt(r, 15, 0) == doctest::Approx(0.1));
}
