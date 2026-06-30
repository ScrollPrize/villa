// Unit tests for flip_surface_normals (issue #1044 core helper).
//
// grid_normal uses the right-handed convention N = dP/dU x dP/dV, which points
// away from the scroll centre when text is readable. flip_surface_normals
// negates each normal so renders can point the offset (w) direction the other
// way, while leaving NaN-sentinel (invalid) normals untouched.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Geometry.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <limits>

namespace {
const float QNAN = std::numeric_limits<float>::quiet_NaN();
}

TEST_CASE("flip_surface_normals negates every finite normal")
{
    cv::Mat_<cv::Vec3f> n(2, 2);
    n(0, 0) = cv::Vec3f(1.f, 0.f, 0.f);
    n(0, 1) = cv::Vec3f(0.f, -2.f, 3.f);
    n(1, 0) = cv::Vec3f(-4.f, 5.f, -6.f);
    n(1, 1) = cv::Vec3f(0.f, 0.f, 1.f);

    flip_surface_normals(n);

    CHECK(n(0, 0) == cv::Vec3f(-1.f, 0.f, 0.f));
    CHECK(n(0, 1) == cv::Vec3f(0.f, 2.f, -3.f));
    CHECK(n(1, 0) == cv::Vec3f(4.f, -5.f, 6.f));
    CHECK(n(1, 1) == cv::Vec3f(0.f, 0.f, -1.f));
}

TEST_CASE("flip_surface_normals is its own inverse (applied twice = identity)")
{
    cv::Mat_<cv::Vec3f> n(1, 3);
    n(0, 0) = cv::Vec3f(0.37f, -1.5f, 2.25f);
    n(0, 1) = cv::Vec3f(-9.f, 8.f, -7.f);
    n(0, 2) = cv::Vec3f(0.f, 0.f, 0.f);
    const cv::Mat_<cv::Vec3f> orig = n.clone();

    flip_surface_normals(n);
    flip_surface_normals(n);

    for (int x = 0; x < n.cols; ++x)
        CHECK(n(0, x) == orig(0, x));
}

TEST_CASE("flip_surface_normals leaves NaN-sentinel normals untouched")
{
    cv::Mat_<cv::Vec3f> n(1, 2);
    n(0, 0) = cv::Vec3f(QNAN, QNAN, QNAN);
    n(0, 1) = cv::Vec3f(1.f, 2.f, 3.f);

    flip_surface_normals(n);

    CHECK(std::isnan(n(0, 0)[0]));
    CHECK(std::isnan(n(0, 0)[1]));
    CHECK(std::isnan(n(0, 0)[2]));
    CHECK(n(0, 1) == cv::Vec3f(-1.f, -2.f, -3.f));
}

TEST_CASE("flip_surface_normals on an empty matrix is a no-op")
{
    cv::Mat_<cv::Vec3f> n;
    flip_surface_normals(n); // must not crash
    CHECK(n.empty());
}
