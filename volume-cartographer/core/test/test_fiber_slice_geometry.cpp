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
