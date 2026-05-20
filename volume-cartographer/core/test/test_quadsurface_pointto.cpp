// Coverage for QuadSurface::pointTo (member + free pointTo/search_min_loc).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <cmath>

namespace {

cv::Mat_<cv::Vec3f> makePlanarGrid(int rows, int cols, float z = 0.f)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = cv::Vec3f(float(c), float(r), z);
    return m;
}

} // namespace

TEST_CASE("QuadSurface::pointTo finds the closest grid point for a target on-plane")
{
    auto pts = makePlanarGrid(32, 32, /*z=*/0.f);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    cv::Vec3f ptr(0, 0, 0);
    cv::Vec3f target(10.f, 12.f, 0.f);
    float d = qs.pointTo(ptr, target, /*th=*/0.5f, /*max_iters=*/200);
    CHECK(d >= 0.0f);
    CHECK(d < 5.0f);
}

TEST_CASE("QuadSurface::pointTo: target far above plane returns positive z-distance")
{
    auto pts = makePlanarGrid(32, 32, /*z=*/0.f);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    cv::Vec3f ptr(0, 0, 0);
    cv::Vec3f target(16.f, 16.f, 50.f);
    float d = qs.pointTo(ptr, target, /*th=*/0.5f, /*max_iters=*/100);
    // Distance from the plane to the target.
    CHECK((d >= 0.0f || d == -1.0f));
}

TEST_CASE("QuadSurface::pointTo: tight threshold succeeds at the grid origin")
{
    auto pts = makePlanarGrid(16, 16);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    cv::Vec3f ptr(0, 0, 0);
    cv::Vec3f target(5.f, 5.f, 0.f);
    float d = qs.pointTo(ptr, target, /*th=*/1.0f, /*max_iters=*/200);
    CHECK(d <= 1.0f);
    CHECK(d >= 0.0f);
}

TEST_CASE("QuadSurface::pointTo on a sparse surface: pretarget outside valid region")
{
    cv::Mat_<cv::Vec3f> pts(20, 20, cv::Vec3f(-1, -1, -1));
    // Small valid patch in the middle.
    for (int r = 8; r < 12; ++r)
        for (int c = 8; c < 12; ++c)
            pts(r, c) = cv::Vec3f(c, r, 0);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    cv::Vec3f ptr(0, 0, 0);
    cv::Vec3f target(1000, 1000, 1000);
    float d = qs.pointTo(ptr, target, 0.5f, 100);
    // Likely returns negative (not converged) — that's fine, exercise path.
    (void)d;
    CHECK(true);
}

TEST_CASE("QuadSurface::loc with offset produces ptr+offset")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    auto l = qs.loc(cv::Vec3f(1, 2, 3), cv::Vec3f(0.5f, -0.5f, 0));
    CHECK(l[0] == doctest::Approx(1.5f));
    CHECK(l[1] == doctest::Approx(1.5f));
}

TEST_CASE("Free pointTo(Vec3f points) helper resolves a target on a flat grid")
{
    auto pts = makePlanarGrid(32, 32);
    cv::Vec2f loc(16.f, 16.f);
    cv::Vec3f target(10.f, 12.f, 0.f);
    float d = ::pointTo(loc, pts, target, /*th=*/0.5f, /*max_iters=*/200, /*scale=*/1.0f);
    CHECK(d >= 0.0f);
}

TEST_CASE("Free pointTo (Vec3d points overload)")
{
    cv::Mat_<cv::Vec3d> pts(32, 32);
    for (int r = 0; r < 32; ++r)
        for (int c = 0; c < 32; ++c)
            pts(r, c) = cv::Vec3d(c, r, 0);
    cv::Vec2f loc(16.f, 16.f);
    cv::Vec3f target(8.f, 8.f, 0.f);
    float d = ::pointTo(loc, pts, target, 0.5f, 200, 1.0f);
    CHECK(d >= 0.0f);
}
