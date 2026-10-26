#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"
#include "vc/core/types/FiberView.hpp"

#include <cmath>
using namespace vc::fibers;
constexpr Bounds kUnit{{-1,-1,-1},{1,1,1}};
constexpr double kDiagonal = 0.7071067811865475244;

TEST_CASE("clipSegment clips exactly and handles degenerate segments")
{
    double lo, hi;
    CHECK((clipSegment({-2, 1, 0}, {2, 1, 0}, kUnit, lo, hi) && lo == .25 && hi == .75));
    CHECK(clipSegment({0, 0, 0}, {0, 0, 0}, kUnit, lo, hi));
    CHECK_FALSE(clipSegment({0, 2, 0}, {0, 2, 0}, kUnit, lo, hi));
}

TEST_CASE("ViewRegion::contains accepts only slices inside the loaded prism")
{
    ViewRegion loaded;
    loaded.local = {{0, 0, -1}, {10, 10, 1}};
    auto inside = loaded;
    inside.origin = {1, 2, 0};
    inside.local.upper = {8, 7, 1};
    CHECK(loaded.contains(inside));
    inside.origin[2] = .01;
    CHECK_FALSE(loaded.contains(inside));
    inside = loaded;
    inside.origin[0] = 1;
    CHECK_FALSE(loaded.contains(inside));
    loaded.u = {kDiagonal, kDiagonal, 0};
    loaded.v = {-kDiagonal, kDiagonal, 0};
    inside = loaded;
    inside.local.upper = {5, 5, 1};
    inside.origin = {0, 2, 0};
    CHECK(loaded.contains(inside));
    inside.origin = {0, 8, 0};
    CHECK_FALSE(loaded.contains(inside));
    inside = loaded;
    inside.n = {0, .1, 1};
    CHECK_FALSE(loaded.contains(inside));
}

TEST_CASE("ViewRegion::contains tests a rotated slice by its corners")
{
    ViewRegion loaded;
    loaded.local = {{-10, -10, -2}, {10, 10, 2}};
    // A slice tilted about the u axis: its v edge rises out of the plane.
    const double c = std::cos(.05), s = std::sin(.05);
    ViewRegion tilted;
    tilted.v = {0, c, s};
    tilted.n = {0, -s, c};
    tilted.local = {{-5, -5, -1}, {5, 5, 1}};
    // Corners reach |z| = 5 sin(.05) + cos(.05) ~ 1.25 inside the 2-voxel slab.
    CHECK(loaded.contains(tilted));
    tilted.local = {{-5, -30, -1}, {5, 30, 1}};
    CHECK_FALSE(loaded.contains(tilted));
    tilted.local = {{-5, -5, -1}, {5, 5, 1}};
    tilted.origin = {0, 0, 1};
    CHECK_FALSE(loaded.contains(tilted));
}

