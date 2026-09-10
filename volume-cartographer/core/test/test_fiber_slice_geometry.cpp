#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "FiberSliceGeometry.hpp"

#include <cmath>
#include <vector>

TEST_CASE("fiber slice arclength sampling interpolates point and tangent")
{
    const std::vector<cv::Vec3d> linePoints{
        {0.0, 0.0, 0.0},
        {3.0, 0.0, 0.0},
        {3.0, 4.0, 0.0},
    };

    const auto sample = vc3d::fiber_slice::samplePolylineAtArclength(linePoints, 5.0);
    CHECK(sample.valid);
    CHECK(sample.point[0] == doctest::Approx(3.0));
    CHECK(sample.point[1] == doctest::Approx(2.0));
    CHECK(sample.point[2] == doctest::Approx(0.0));
    CHECK(sample.tangent[0] == doctest::Approx(0.0));
    CHECK(sample.tangent[1] == doctest::Approx(1.0));
    CHECK(sample.tangent[2] == doctest::Approx(0.0));
    CHECK(sample.arclength == doctest::Approx(5.0));
    CHECK(sample.linePosition == doctest::Approx(1.5));
    CHECK(vc3d::fiber_slice::linePositionAtArclength(linePoints, 5.0) ==
          doctest::Approx(1.5));
}

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

TEST_CASE("fiber slice control triplet selects previous current and next positions")
{
    const std::vector<cv::Vec3d> linePoints{
        {0.0, 0.0, 0.0},
        {10.0, 0.0, 0.0},
        {20.0, 0.0, 0.0},
        {30.0, 0.0, 0.0},
        {40.0, 0.0, 0.0},
    };
    const std::vector<cv::Vec3d> controls{
        {0.0, 0.0, 0.0},
        {10.1, 0.0, 0.0},
        {30.1, 0.0, 0.0},
        {40.0, 0.0, 0.0},
    };

    const auto triplet = vc3d::fiber_slice::selectControlTriplet(
        linePoints,
        controls,
        2.0,
        {20.0, 0.0, 0.0});
    CHECK(triplet.valid);
    CHECK(triplet.previousLinePosition == doctest::Approx(1.0));
    CHECK(triplet.currentLinePosition == doctest::Approx(2.0));
    CHECK(triplet.nextLinePosition == doctest::Approx(3.0));
    CHECK(triplet.currentPoint[0] == doctest::Approx(20.0));
}

TEST_CASE("fiber slice control triplet falls back to endpoint when neighbor is missing")
{
    const std::vector<cv::Vec3d> linePoints{
        {0.0, 0.0, 0.0},
        {10.0, 0.0, 0.0},
        {20.0, 0.0, 0.0},
    };
    const std::vector<cv::Vec3d> controls{
        {0.0, 0.0, 0.0},
    };

    const auto triplet = vc3d::fiber_slice::selectControlTriplet(
        linePoints,
        controls,
        0.0,
        {0.0, 0.0, 0.0});
    CHECK(triplet.valid);
    CHECK(triplet.previousLinePosition == doctest::Approx(0.0));
    CHECK(triplet.nextLinePosition == doctest::Approx(2.0));
}

TEST_CASE("fiber slice plane construction falls back when tangent is parallel to normal")
{
    const auto fit = vc3d::fiber_slice::planeFromNormalAndTangent(
        {1.0, 2.0, 3.0},
        {0.0, 0.0, 1.0},
        {0.0, 0.0, 4.0});

    CHECK(fit.valid);
    CHECK(fit.origin[0] == doctest::Approx(1.0));
    CHECK(fit.origin[1] == doctest::Approx(2.0));
    CHECK(fit.origin[2] == doctest::Approx(3.0));
    CHECK(fit.normal[0] == doctest::Approx(0.0));
    CHECK(fit.normal[1] == doctest::Approx(0.0));
    CHECK(fit.normal[2] == doctest::Approx(1.0));
    CHECK(std::abs(fit.upHint.dot(fit.normal)) < 1.0e-9);
    CHECK(cv::norm(fit.upHint) == doctest::Approx(1.0));
}

TEST_CASE("fiber slice plane construction falls back for zero connector normal")
{
    const auto fit = vc3d::fiber_slice::planeFromNormalAndTangent(
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0});

    CHECK(fit.valid);
    CHECK(fit.normal[0] == doctest::Approx(0.0));
    CHECK(fit.normal[1] == doctest::Approx(0.0));
    CHECK(fit.normal[2] == doctest::Approx(1.0));
    CHECK(cv::norm(fit.upHint) == doctest::Approx(1.0));
}

TEST_CASE("fiber slice connection plane contains connector and fiber tangent")
{
    const cv::Vec3d connector{1.0, 2.0, 0.0};
    const cv::Vec3d tangent{0.0, 0.0, 1.0};
    const auto fit = vc3d::fiber_slice::planeFromDirections(
        {0.0, 0.0, 0.0},
        connector,
        tangent);

    CHECK(fit.valid);
    CHECK(std::abs(fit.normal.dot(connector)) < 1.0e-9);
    CHECK(std::abs(fit.normal.dot(tangent)) < 1.0e-9);
    CHECK(cv::norm(fit.upHint) == doctest::Approx(1.0));
}

TEST_CASE("fiber slice connection plane keeps connector in plane when tangent is parallel")
{
    const cv::Vec3d connector{3.0, 0.0, 0.0};
    const cv::Vec3d tangent{9.0, 0.0, 0.0};
    const auto fit = vc3d::fiber_slice::planeFromDirections(
        {0.0, 0.0, 0.0},
        connector,
        tangent);

    CHECK(fit.valid);
    CHECK(std::abs(fit.normal.dot(connector)) < 1.0e-9);
    CHECK(cv::norm(fit.upHint) == doctest::Approx(1.0));
}

TEST_CASE("fiber slice connector thickness handles zero-length connectors")
{
    using vc3d::fiber_slice::connectorNormalizedThickness;

    CHECK(connectorNormalizedThickness(0.0, 0.0, 5.0, 1.0) == doctest::Approx(5.0));
    CHECK(connectorNormalizedThickness(1.0e-12, 0.0, 5.0, 1.0) > 4.0);
    CHECK(connectorNormalizedThickness(5.0, 10.0, 5.0, 1.0) == doctest::Approx(3.0));
    CHECK(connectorNormalizedThickness(20.0, 10.0, 5.0, 1.0) == doctest::Approx(1.0));
}
