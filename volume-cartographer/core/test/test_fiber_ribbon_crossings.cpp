#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberRibbonCrossings.hpp"

#include <cmath>
#include <vector>

using namespace vc::fiber_tracer;

namespace
{

// A straight fiber along x at y = z = 0 with its normal along +z.
std::vector<FiberStripSample> surfaceFiber()
{
    std::vector<FiberStripSample> samples;
    for (int i = 0; i <= 40; ++i) samples.push_back({16.0 * i, {16.0 * i, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}});
    return samples;
}

// A fiber through the given points (x fixed at 320), resampled every 8 vx; `normal` is its normal throughout.
std::vector<FiberStripSample> path(const std::vector<cv::Vec3d>& corners, const cv::Vec3d& normal)
{
    std::vector<FiberStripSample> samples;
    double arc = 0.0;
    for (std::size_t c = 0; c + 1 < corners.size(); ++c) {
        const double length = cv::norm(corners[c + 1] - corners[c]);
        const int steps = int(std::ceil(length / 8.0));
        for (int s = c == 0 ? 0 : 1; s <= steps; ++s) {
            const double t = double(s) / steps;
            samples.push_back({arc + t * length, corners[c] + t * (corners[c + 1] - corners[c]), normal, {0.0, 0.0, 0.0}});
        }
        arc += length;
    }
    return samples;
}

struct Result {
    std::vector<FiberRibbonCrossing> crossings;
    std::vector<FiberRibbonEncounter> encounters;
};

// The crossings of fiber 1 with the ribbon of fiber 0 only.
Result run(const std::vector<FiberStripSample>& partner)
{
    const std::vector<std::vector<FiberStripSample>> fibers{surfaceFiber(), partner};
    FiberRibbonOptions options;
    options.halfHeight = 200.0;
    options.maxTangentCosine = 1.0;
    options.threads = 2;
    Result result;
    for (const auto& c : fiberRibbonCrossings(fibers, options))
        if (c.ribbon == 0) result.crossings.push_back(c);
    result.encounters = fiberRibbonEncounters(result.crossings);
    return result;
}

}  // namespace

TEST_CASE("a straight crossing is one passage on its side")
{
    const auto r = run(path({{320.5, -100.0, 50.0}, {320.5, 100.0, 50.0}}, {0.0, 0.0, 1.0}));
    REQUIRE(r.crossings.size() == 1);
    CHECK(r.crossings[0].height == doctest::Approx(50.0));
    CHECK(r.crossings[0].ribbonArc == doctest::Approx(320.5));
    CHECK(r.crossings[0].normalDot == doctest::Approx(1.0));
    REQUIRE(r.encounters.size() == 1);
    CHECK(r.encounters[0].left == 1);
    CHECK(r.encounters[0].remaining == r.encounters[0].closest);
}

TEST_CASE("an S curve on one side leaves its nearest passage")
{
    // Across at z = 120, back at z = 80, across again at z = 40.
    const auto r = run(path({{320.5, -100.0, 120.0}, {320.5, 100.0, 120.0}, {320.5, 100.0, 80.0}, {320.5, -100.0, 80.0},
                             {320.5, -100.0, 40.0}, {320.5, 100.0, 40.0}}, {0.0, 0.0, 1.0}));
    REQUIRE(r.crossings.size() == 3);
    REQUIRE(r.encounters.size() == 1);
    CHECK(r.encounters[0].left == 1);
    REQUIRE(r.encounters[0].remaining >= 0);
    CHECK(r.crossings[r.encounters[0].remaining].height == doctest::Approx(40.0));
}

TEST_CASE("a passage and its return leave nothing, and never cancel across the fiber")
{
    const auto dip = run(path({{320.5, -100.0, 60.0}, {320.5, 100.0, 60.0}, {320.5, 100.0, 90.0}, {320.5, -100.0, 90.0}},
                              {0.0, 0.0, 1.0}));
    REQUIRE(dip.encounters.size() == 1);
    CHECK(dip.encounters[0].count == 2);
    CHECK(dip.encounters[0].left == 0);
    CHECK(dip.encounters[0].remaining == -1);

    // Across above the fiber, back below it: two passages, one per side, both stay.
    const auto around = run(path({{320.5, -100.0, 60.0}, {320.5, 100.0, 60.0}, {320.5, 100.0, -60.0}, {320.5, -100.0, -60.0}},
                                 {0.0, 0.0, 1.0}));
    REQUIRE(around.encounters.size() == 1);
    CHECK(around.encounters[0].left == 2);
}

TEST_CASE("verdicts: single, unmatched, nothing left, mixed")
{
    const FiberCrossingEventView view{0, 1, 320.5, 100.0};
    const auto single = run(path({{320.5, -100.0, 50.0}, {320.5, 100.0, 50.0}}, {0.0, 0.0, 1.0}));
    const auto one = fiberCrossingVerdicts(single.crossings, single.encounters, std::vector{view})[0];
    CHECK(one.matched);
    CHECK_FALSE(one.multiple);
    const FiberCrossingEventView elsewhere{0, 1, 100.0, 100.0};
    CHECK_FALSE(fiberCrossingVerdicts(single.crossings, single.encounters, std::vector{elsewhere})[0].matched);

    const auto dip = run(path({{320.5, -100.0, 60.0}, {320.5, 100.0, 60.0}, {320.5, 100.0, 90.0}, {320.5, -100.0, 90.0}},
                              {0.0, 0.0, 1.0}));
    const auto none = fiberCrossingVerdicts(dip.crossings, dip.encounters, std::vector{view})[0];
    CHECK(none.nothingLeft);
    CHECK_FALSE(none.mixed);

    // An S whose middle limb carries the opposite normal: agreeing and opposing strong crossings.
    auto fold = path({{320.5, -100.0, 120.0}, {320.5, 100.0, 120.0}, {320.5, 100.0, 80.0}, {320.5, -100.0, 80.0},
                      {320.5, -100.0, 40.0}, {320.5, 100.0, 40.0}}, {0.0, 0.0, 1.0});
    for (auto& sample : fold)
        if (std::abs(sample.point[2] - 80.0) < 1.0) sample.normal = {0.0, 0.0, -1.0};
    const auto folded = run(fold);
    FiberCrossingFilterOptions options;
    options.nearHeight = 60.0;
    const auto both = fiberCrossingVerdicts(folded.crossings, folded.encounters, std::vector{view}, options)[0];
    CHECK(both.mixed);
    CHECK_FALSE(both.mixedNear);  // only the passage at 40 lies within 60
    CHECK_FALSE(both.nothingLeft);
    CHECK_FALSE(both.severalLeft);
    // The event sits on the passage at 120 (agreeing); the nearest crossing of the other relation is the limb at 80.
    CHECK(both.alternativeDot == doctest::Approx(1.0));
    CHECK(both.alternativeHeight == doctest::Approx(80.0));
    CHECK(one.alternativeDot == 0.0);
}

TEST_CASE("findings of one measured crossing, from both fibers' surfaces")
{
    namespace finding = vc::geometry::crossing_finding;
    FiberRibbonOptions surface;
    surface.halfHeight = 200.0;
    surface.maxTangentCosine = 1.0;
    const auto a = surfaceFiber();

    // The pair pass finds what the whole-volume pass finds.
    const auto fold = [] {
        auto samples = path({{320.5, -100.0, 120.0}, {320.5, 100.0, 120.0}, {320.5, 100.0, 80.0}, {320.5, -100.0, 80.0},
                             {320.5, -100.0, 40.0}, {320.5, 100.0, 40.0}}, {0.0, 0.0, 1.0});
        for (auto& sample : samples)
            if (std::abs(sample.point[2] - 80.0) < 1.0) sample.normal = {0.0, 0.0, -1.0};
        return samples;
    }();
    CHECK(fiberPairRibbonCrossings(a, fold, surface).size() == run(fold).crossings.size());

    const auto single = path({{320.5, -100.0, 50.0}, {320.5, 100.0, 50.0}}, {0.0, 0.0, 1.0});
    const auto one = fiberCrossingFindings(a, single, 320.5, 100.0, surface);
    CHECK((one.findings & finding::Matched) != 0);
    CHECK((one.findings & (finding::Multiple | finding::NothingLeft | finding::Mixed | finding::SeveralLeft)) == 0);
    CHECK(one.alternative == 0.0);
    // The rule keeps it with or without signs.
    CHECK(vc::geometry::fiberCrossingUse(one.findings, one.alternative, 1.0, 0).used);

    const auto dip = path({{320.5, -100.0, 60.0}, {320.5, 100.0, 60.0}, {320.5, 100.0, 90.0}, {320.5, -100.0, 90.0}},
                          {0.0, 0.0, 1.0});
    const auto none = fiberCrossingFindings(a, dip, 320.5, 100.0, surface);
    CHECK((none.findings & finding::NothingLeft) != 0);
    CHECK_FALSE(vc::geometry::fiberCrossingUse(none.findings, none.alternative, 1.0, 0).used);
    CHECK_FALSE(vc::geometry::fiberCrossingUse(none.findings, none.alternative, 1.0, 1).used);

    // The measured crossing is the passage at 120 (agreeing); the limb at 80 opposes.
    const auto mixed = fiberCrossingFindings(a, fold, 320.5, 100.0, surface);
    CHECK((mixed.findings & finding::Mixed) != 0);
    CHECK(mixed.alternative == doctest::Approx(1.0));
    // Without signs a mixed encounter decides nothing; with signs it is kept, and
    // where the measured crossing disagrees it takes the agreeing alternative.
    CHECK_FALSE(vc::geometry::fiberCrossingUse(mixed.findings, mixed.alternative, 1.0, 0).used);
    const auto agreeing = vc::geometry::fiberCrossingUse(mixed.findings, mixed.alternative, 1.0, 1);
    CHECK(agreeing.used);
    CHECK_FALSE(agreeing.replaced);
    const auto disagreeing = vc::geometry::fiberCrossingUse(mixed.findings, mixed.alternative, 1.0, -1);
    CHECK(disagreeing.used);
    CHECK(disagreeing.replaced);
    CHECK(disagreeing.dot == doctest::Approx(-1.0));
}
