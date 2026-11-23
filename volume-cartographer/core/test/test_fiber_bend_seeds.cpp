#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberBendSeeds.hpp"

using namespace vc::fiber_tracer;

TEST_CASE("bend seeds pair close same-family fibers that run in different directions")
{
    // Fibers 0 and 1 run along x, 30 voxels apart; fiber 2 passes them at 60 degrees;
    // fiber 3 does the same but belongs to the other family; fiber 4 is far away.
    const double c = 0.5, s = std::sqrt(0.75);
    std::vector<FiberBendNode> nodes{{{0, 0, 0}, {1, 0, 0}, 0, 0},   {{0, 30, 0}, {1, 0, 0}, 1, 0},
                                     {{10, 10, 0}, {c, s, 0}, 2, 0}, {{10, 12, 0}, {c, s, 0}, 3, 1},
                                     {{500, 0, 0}, {0, 1, 0}, 4, 0}, {{0, 5, 0}, {-1, 0, 0}, 0, 0}};
    const auto seeds = fiberBendSeeds(nodes, {100.0, 45.0, 2});
    // Fiber 2 against the three nodes of fibers 0 and 1; parallel fibers, the other family and the far one give none.
    REQUIRE(seeds.size() == 3);
    for (const auto& seed : seeds) {
        CHECK(seed.angleDegrees == doctest::Approx(60.0));
        CHECK(seed.family == 0);
        CHECK((seed.fiberA == 2 || seed.fiberB == 2));
    }
    CHECK(seeds[0].point[0] == doctest::Approx(5.0));
    CHECK(seeds[0].distance == doctest::Approx(std::sqrt(200.0)));
    CHECK(fiberBendSeeds(nodes, {100.0, 70.0, 1}).empty());
    CHECK(fiberBendSeeds(nodes, {12.0, 45.0, 1}).size() == 1);  // only the node 11 voxels away
    CHECK_THROWS(fiberBendSeeds(nodes, {0.0, 45.0, 1}));
}
