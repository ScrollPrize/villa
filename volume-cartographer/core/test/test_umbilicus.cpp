// Coverage for core/src/Umbilicus.cpp: the dense axis VC3D and
// vc_fiber_constraints build from a stored or bound umbilicus.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Umbilicus.hpp"

#include <opencv2/core.hpp>

#include <stdexcept>
#include <vector>

using vc::core::util::Umbilicus;

TEST_CASE("FromPoints: requires positive volume shape")
{
    std::vector<cv::Vec3f> pts{{0, 0, 0}};
    CHECK_THROWS_AS(Umbilicus::FromPoints(pts, cv::Vec3i(0, 100, 100)), std::invalid_argument);
    CHECK_THROWS_AS(Umbilicus::FromPoints(pts, cv::Vec3i(100, 0, 100)), std::invalid_argument);
    CHECK_THROWS_AS(Umbilicus::FromPoints(pts, cv::Vec3i(100, 100, 0)), std::invalid_argument);
}

TEST_CASE("FromPoints: requires at least one control point")
{
    std::vector<cv::Vec3f> empty;
    CHECK_THROWS_AS(Umbilicus::FromPoints(empty, cv::Vec3i(10, 10, 10)), std::invalid_argument);
}

TEST_CASE("Single-point umbilicus: one center per slice, volume_shape preserved")
{
    std::vector<cv::Vec3f> pts{{5.f, 7.f, 3.f}};
    auto u = Umbilicus::FromPoints(pts, cv::Vec3i(10, 20, 30));
    CHECK(u.volume_shape() == cv::Vec3i(10, 20, 30));
    REQUIRE(u.centers().size() == 10);
    CHECK(u.centers()[0] == cv::Vec3f(5.f, 7.f, 0.f));
}

TEST_CASE("centers interpolate between control points")
{
    std::vector<cv::Vec3f> pts{{10.f, 10.f, 10.f}, {0.f, 0.f, 0.f}};
    auto u = Umbilicus::FromPoints(pts, cv::Vec3i(20, 20, 20));
    REQUIRE(u.centers().size() == 20);
    CHECK(u.centers()[5][0] == doctest::Approx(5.0f).epsilon(1e-3));
    CHECK(u.centers()[5][1] == doctest::Approx(5.0f).epsilon(1e-3));
}

TEST_CASE("vector_to_umbilicus points from a position to its slice's center")
{
    std::vector<cv::Vec3f> pts{{10.f, 10.f, 5.f}};
    auto u = Umbilicus::FromPoints(pts, cv::Vec3i(10, 20, 20));
    auto v = u.vector_to_umbilicus(cv::Vec3f(0, 0, 5));
    CHECK(v[0] == doctest::Approx(10.0f));
    CHECK(v[1] == doctest::Approx(10.0f));
    CHECK(v[2] == doctest::Approx(0.0f));
}
