#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "FiberSliceGeometry.hpp"

#include <cmath>
#include <limits>
#include <vector>

TEST_CASE("fiber slice line positions map through physical polyline arclength")
{
    const std::vector<cv::Vec3d> linePoints{
        {0.0, 0.0, 0.0},
        {3.0, 0.0, 0.0},
        {3.0, 40.0, 0.0},
        {43.0, 40.0, 0.0},
    };
    const auto cumulative =
        vc3d::fiber_slice::cumulativePolylineArclengths(linePoints);
    REQUIRE(cumulative.size() == 4);
    CHECK(cumulative[0] == doctest::Approx(0.0));
    CHECK(cumulative[1] == doctest::Approx(3.0));
    CHECK(cumulative[2] == doctest::Approx(43.0));
    CHECK(cumulative[3] == doctest::Approx(83.0));
    CHECK(vc3d::fiber_slice::arclengthAtLinePosition(cumulative, 1.5) ==
          doctest::Approx(23.0));
    CHECK(vc3d::fiber_slice::linePositionAtArclength(cumulative, 23.0) ==
          doctest::Approx(1.5));

    const std::vector<double> candidates{0.0, 1.0, 1.8, 2.0, 3.0};
    const auto matches =
        vc3d::fiber_slice::linePositionIndicesWithinArclengthRadius(
            cumulative, 1.8, candidates, 32.0);
    CHECK(matches == std::vector<size_t>{1, 2, 3});
    const auto justOutside =
        vc3d::fiber_slice::linePositionIndicesWithinArclengthRadius(
            cumulative, 2.0, std::vector<double>{2.8, 2.800001}, 32.0);
    REQUIRE(justOutside.size() == 1);
    CHECK(justOutside.front() == 0);
}

TEST_CASE("fiber slice max extrapolation distance leaves control interiors unrestricted")
{
    const auto cumulative = vc3d::fiber_slice::cumulativePolylineArclengths(
        std::vector<cv::Vec3d>{{0.0, 0.0, 0.0},
                               {4.0, 0.0, 0.0},
                               {36.0, 0.0, 0.0},
                               {68.0, 0.0, 0.0}});
    const std::vector<double> controls{1.0, 2.0};

    CHECK(vc3d::fiber_slice::linePositionWithinControlExtrapolationDistance(
        cumulative, 1.5, controls, 1.0));
    CHECK(vc3d::fiber_slice::linePositionWithinControlExtrapolationDistance(
        cumulative, 0.0, controls, 4.0));
    CHECK_FALSE(vc3d::fiber_slice::linePositionWithinControlExtrapolationDistance(
        cumulative, 0.0, controls, 3.9));
    CHECK(vc3d::fiber_slice::linePositionWithinControlExtrapolationDistance(
        cumulative, 3.0, controls, 32.0));
    CHECK_FALSE(vc3d::fiber_slice::linePositionWithinControlExtrapolationDistance(
        cumulative, 3.0, controls, 31.9));
    CHECK(vc3d::fiber_slice::linePositionWithinControlExtrapolationDistance(
        cumulative, 3.0, controls, 0.0));
}

TEST_CASE("fiber slice extrapolation boundary converts base arclength to line position")
{
    const auto cumulative = vc3d::fiber_slice::cumulativePolylineArclengths(
        std::vector<cv::Vec3d>{{0.0, 0.0, 0.0},
                               {4.0, 0.0, 0.0},
                               {36.0, 0.0, 0.0},
                               {68.0, 0.0, 0.0},
                               {100.0, 0.0, 0.0}});
    const std::vector<double> controls{1.0, 2.0};

    const auto right =
        vc3d::fiber_slice::controlExtrapolationBoundaryLinePosition(
            cumulative, controls, 1, 4.0, 40.0);
    REQUIRE(right.has_value());
    CHECK(*right == doctest::Approx(3.25));

    const auto left =
        vc3d::fiber_slice::controlExtrapolationBoundaryLinePosition(
            cumulative, controls, -1, 0.0, 3.0);
    REQUIRE(left.has_value());
    CHECK(*left == doctest::Approx(0.25));

    const auto unlimited =
        vc3d::fiber_slice::controlExtrapolationBoundaryLinePosition(
            cumulative, controls, 1, 4.0, 0.0);
    REQUIRE(unlimited.has_value());
    CHECK(*unlimited == doctest::Approx(4.0));

    CHECK_FALSE(vc3d::fiber_slice::controlExtrapolationBoundaryLinePosition(
                    cumulative, std::vector<double>{2.0, 4.0}, 1, 4.0, 20.0)
                    .has_value());
}

namespace
{

// A straight line along x, one point per `step` voxels, from 0 to (count-1)*step.
std::vector<cv::Vec3d> lineAlongX(int count, double step)
{
    std::vector<cv::Vec3d> points;
    points.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        points.emplace_back(step * i, 0.0, 0.0);
    }
    return points;
}

}  // namespace

TEST_CASE("fiber slice range length sums the segments between the indices")
{
    const auto line = lineAlongX(11, 8.0);
    CHECK(vc3d::fiber_slice::polylineLengthRange(line, 0, 10) == doctest::Approx(80.0));
    CHECK(vc3d::fiber_slice::polylineLengthRange(line, 3, 5) == doctest::Approx(16.0));
    CHECK(vc3d::fiber_slice::polylineLengthRange(line, 5, 5) == doctest::Approx(0.0));
    CHECK(vc3d::fiber_slice::polylineLengthRange(line, 5, 3) == doctest::Approx(0.0));
    CHECK(vc3d::fiber_slice::polylineLengthRange(line, 0, 11) == doctest::Approx(0.0));

    auto broken = lineAlongX(5, 10.0);
    broken[2] = cv::Vec3d(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0);
    // Both segments touching the bad point drop out; the other two remain.
    CHECK(vc3d::fiber_slice::polylineLengthRange(broken, 0, 4) == doctest::Approx(20.0));
}

TEST_CASE("fiber slice annotated length stops at the outermost control points")
{
    // 1200 vx of open tail on each side of a 2400 vx annotated span:
    // 0..1200 tail, 1200..3600 fiber, 3600..4800 tail (8 vx per point).
    const auto line = lineAlongX(601, 8.0);
    const std::vector<cv::Vec3d> controls{
        {1200.0, 0.0, 0.0}, {2000.0, 0.0, 0.0}, {2400.0, 0.0, 0.0},
        {2600.0, 0.0, 0.0}, {3600.0, 0.0, 0.0}};
    CHECK(vc3d::fiber_slice::annotatedLineLengthVx(line, controls) == doctest::Approx(2400.0));
    // The tails alone would have doubled it.
    CHECK(vc3d::fiber_slice::polylineLengthRange(line, 0, 600) == doctest::Approx(4800.0));

    // Control point order does not matter, and a control point beside the
    // line snaps to its nearest line point.
    const std::vector<cv::Vec3d> shuffled{
        {3600.0, 0.0, 0.0}, {1201.0, 3.0, 0.0}, {2400.0, -3.0, 0.0}};
    CHECK(vc3d::fiber_slice::annotatedLineLengthVx(line, shuffled) == doctest::Approx(2400.0));
}

TEST_CASE("fiber slice annotated length is nothing when the controls share one line point")
{
    // Two finite controls that both snap to line point 1: there is a control
    // pair, but no line between them - not the whole line.
    const std::vector<cv::Vec3d> line{
        {0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}, {10.0, 0.0, 0.0}, {20.0, 0.0, 0.0}};
    const std::vector<cv::Vec3d> controls{{10.0, 0.0, 0.0}, {10.0, 0.0, 0.0}};
    CHECK(vc3d::fiber_slice::annotatedLineLengthVx(line, controls) == doctest::Approx(0.0));
}

TEST_CASE("fiber slice annotated length is the whole line without two control points")
{
    const auto line = lineAlongX(11, 8.0);
    CHECK(vc3d::fiber_slice::annotatedLineLengthVx(line, {}) == doctest::Approx(80.0));
    CHECK(vc3d::fiber_slice::annotatedLineLengthVx(line, {{40.0, 0.0, 0.0}}) ==
          doctest::Approx(80.0));
    const std::vector<cv::Vec3d> oneBad{
        {40.0, 0.0, 0.0}, {std::numeric_limits<double>::infinity(), 0.0, 0.0}};
    CHECK(vc3d::fiber_slice::annotatedLineLengthVx(line, oneBad) == doctest::Approx(80.0));
    CHECK(vc3d::fiber_slice::annotatedLineLengthVx({}, {{0.0, 0.0, 0.0}, {8.0, 0.0, 0.0}}) ==
          doctest::Approx(0.0));
}
