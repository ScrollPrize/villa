#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Geometry.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <vector>

namespace {

cv::Mat_<cv::Vec3f> makePlanarGrid(int rows, int cols, float dx = 1.0f, float dy = 1.0f, float z = 0.0f)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            m(r, c) = cv::Vec3f(static_cast<float>(c) * dx, static_cast<float>(r) * dy, z);
        }
    }
    return m;
}

} // namespace

TEST_CASE("at_int Vec3f bilinear at integer corner returns same value")
{
    auto m = makePlanarGrid(4, 4);
    cv::Vec3f v = at_int(m, cv::Vec2f(1.0f, 2.0f));
    CHECK(v[0] == doctest::Approx(1.0f));
    CHECK(v[1] == doctest::Approx(2.0f));
    CHECK(v[2] == doctest::Approx(0.0f));
}

TEST_CASE("at_int Vec3f bilinear at midpoint averages neighbors")
{
    auto m = makePlanarGrid(4, 4);
    cv::Vec3f v = at_int(m, cv::Vec2f(1.5f, 1.5f));
    CHECK(v[0] == doctest::Approx(1.5f));
    CHECK(v[1] == doctest::Approx(1.5f));
}

TEST_CASE("at_int float overload")
{
    cv::Mat_<float> m(3, 3);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            m(r, c) = static_cast<float>(r * 10 + c);
    CHECK(at_int(m, cv::Vec2f(0.0f, 0.0f)) == doctest::Approx(0.0f));
    CHECK(at_int(m, cv::Vec2f(1.0f, 1.0f)) == doctest::Approx(11.0f));
    CHECK(at_int(m, cv::Vec2f(0.5f, 0.0f)) == doctest::Approx(0.5f));
}

TEST_CASE("at_int Vec3d overload")
{
    cv::Mat_<cv::Vec3d> m(3, 3);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            m(r, c) = cv::Vec3d(c, r, 0);
    cv::Vec3d v = at_int(m, cv::Vec2f(1.0f, 1.0f));
    CHECK(v[0] == doctest::Approx(1.0));
    CHECK(v[1] == doctest::Approx(1.0));
}

TEST_CASE("grid_normal on a flat z=0 plane is +/- z")
{
    auto m = makePlanarGrid(10, 10);
    cv::Vec3f n = grid_normal(m, cv::Vec3f(5.0f, 5.0f, 0.0f));
    CHECK(std::abs(n[0]) < 1e-4f);
    CHECK(std::abs(n[1]) < 1e-4f);
    CHECK(std::abs(std::abs(n[2]) - 1.0f) < 1e-4f);
}

TEST_CASE("grid_normal clamps coords outside [1, dim-3] range")
{
    auto m = makePlanarGrid(10, 10);
    // far outside — should still yield a unit normal, not NaN
    cv::Vec3f n = grid_normal(m, cv::Vec3f(-100.0f, -100.0f, 0.0f));
    CHECK(std::isfinite(n[2]));
    CHECK(std::abs(std::abs(n[2]) - 1.0f) < 1e-4f);
}

TEST_CASE("grid_normal returns NaN when neighborhood has sentinel")
{
    auto m = makePlanarGrid(10, 10);
    m(5, 5) = cv::Vec3f(-1.f, -1.f, -1.f);
    cv::Vec3f n = grid_normal(m, cv::Vec3f(5.0f, 5.0f, 0.0f));
    CHECK(std::isnan(n[0]));
}

TEST_CASE("grid_normal_int on flat plane is unit-length")
{
    auto m = makePlanarGrid(8, 8);
    cv::Vec3f n = grid_normal_int(m, 4, 4);
    const float len = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    CHECK(len == doctest::Approx(1.0f));
}

TEST_CASE("grid_normal_int returns NaN with adjacent sentinel")
{
    auto m = makePlanarGrid(8, 8);
    m(4, 5) = cv::Vec3f(-1.f, -1.f, -1.f);
    cv::Vec3f n = grid_normal_int(m, 4, 4);
    CHECK(std::isnan(n[0]));
}

TEST_CASE("grid_normal_int returns NaN when degenerate (zero cross)")
{
    cv::Mat_<cv::Vec3f> m(5, 5, cv::Vec3f(0.f, 0.f, 0.f));
    cv::Vec3f n = grid_normal_int(m, 2, 2);
    CHECK(std::isnan(n[0]));
}

TEST_CASE("loc_valid Vec3f basic")
{
    auto m = makePlanarGrid(5, 5);
    // l is [y, x]; bounds rect is (0,0,rows-2,cols-2)
    CHECK(loc_valid(m, cv::Vec2d(1.0, 1.0)));
    CHECK_FALSE(loc_valid(m, cv::Vec2d(-1.0, 0.0)));
    // out of bounds
    CHECK_FALSE(loc_valid(m, cv::Vec2d(100.0, 100.0)));
}

TEST_CASE("loc_valid Vec3f rejects sentinel in 2x2 window")
{
    auto m = makePlanarGrid(5, 5);
    m(2, 2) = cv::Vec3f(-1.f, -1.f, -1.f);
    CHECK_FALSE(loc_valid(m, cv::Vec2d(2.0, 2.0)));
}

TEST_CASE("loc_valid Vec3d / float overloads")
{
    cv::Mat_<cv::Vec3d> md(4, 4, cv::Vec3d(0, 0, 0));
    CHECK(loc_valid(md, cv::Vec2d(1.0, 1.0)));
    md(1, 1) = cv::Vec3d(-1, -1, -1);
    CHECK_FALSE(loc_valid(md, cv::Vec2d(1.0, 1.0)));

    cv::Mat_<float> mf(4, 4, 0.0f);
    CHECK(loc_valid(mf, cv::Vec2d(1.0, 1.0)));
    mf(1, 1) = -1.0f;
    CHECK_FALSE(loc_valid(mf, cv::Vec2d(1.0, 1.0)));
    CHECK_FALSE(loc_valid(mf, cv::Vec2d(-1.0, 0.0)));
}

TEST_CASE("loc_valid_xy swaps axis order vs loc_valid")
{
    auto m = makePlanarGrid(5, 6); // rows=5, cols=6
    // valid interior in xy form
    CHECK(loc_valid_xy(m, cv::Vec2d(1.0, 1.0)));
    // float overload
    cv::Mat_<float> mf(4, 4, 0.0f);
    CHECK(loc_valid_xy(mf, cv::Vec2d(1.0, 1.0)));
    // Vec3d overload
    cv::Mat_<cv::Vec3d> md(4, 4, cv::Vec3d(0, 0, 0));
    CHECK(loc_valid_xy(md, cv::Vec2d(1.0, 1.0)));
}

TEST_CASE("tdist returns |distance - target|")
{
    cv::Vec3f a(0, 0, 0), b(3, 4, 0);
    CHECK(tdist(a, b, 5.0f) == doctest::Approx(0.0f));
    CHECK(tdist(a, b, 4.0f) == doctest::Approx(1.0f));
    CHECK(tdist(a, b, 7.0f) == doctest::Approx(2.0f));
}

TEST_CASE("tdist_sum aggregates squared per-target residuals")
{
    cv::Vec3f v(0, 0, 0);
    std::vector<cv::Vec3f> tgts{ cv::Vec3f(3, 4, 0), cv::Vec3f(0, 0, 0) };
    std::vector<float> tds{ 4.0f, 1.0f };
    // residuals: |5-4|=1, |0-1|=1; sum of squares = 2
    CHECK(tdist_sum(v, tgts, tds) == doctest::Approx(2.0f));
}

TEST_CASE("tdist_sum empty inputs is zero")
{
    cv::Vec3f v(1, 2, 3);
    std::vector<cv::Vec3f> tgts;
    std::vector<float> tds;
    CHECK(tdist_sum(v, tgts, tds) == doctest::Approx(0.0f));
}

TEST_CASE("clean_surface_outliers preserves a uniform grid")
{
    auto m = makePlanarGrid(8, 8);
    cv::Mat_<cv::Vec3f> cleaned = clean_surface_outliers(m, 5.0f, false);
    // Same shape, no points should be invalidated.
    REQUIRE(cleaned.rows == m.rows);
    REQUIRE(cleaned.cols == m.cols);
    for (int r = 0; r < cleaned.rows; ++r)
        for (int c = 0; c < cleaned.cols; ++c)
            CHECK(cleaned(r, c)[0] != -1.f);
}

TEST_CASE("clean_surface_outliers removes far-away points")
{
    auto m = makePlanarGrid(8, 8);
    // Inject a far outlier surrounded by sentinels so it has no close neighbor
    // (its only "neighbors" will be invalid; the function invalidates points
    // with zero valid neighbors).
    m(3, 3) = cv::Vec3f(1000.f, 1000.f, 1000.f);
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            if (dx != 0 || dy != 0)
                m(3 + dy, 3 + dx) = cv::Vec3f(-1.f, -1.f, -1.f);

    cv::Mat_<cv::Vec3f> cleaned = clean_surface_outliers(m, 5.0f, false);
    CHECK(cleaned(3, 3)[0] == -1.f);
}

TEST_CASE("clean_surface_outliers handles all-invalid input")
{
    cv::Mat_<cv::Vec3f> m(5, 5, cv::Vec3f(-1.f, -1.f, -1.f));
    cv::Mat_<cv::Vec3f> cleaned = clean_surface_outliers(m, 5.0f, false);
    REQUIRE(cleaned.rows == 5);
    REQUIRE(cleaned.cols == 5);
}

TEST_CASE("clean_surface_outliers print_stats path executes")
{
    auto m = makePlanarGrid(4, 4);
    // Just check it doesn't crash with print_stats=true; stdout is fine.
    cv::Mat_<cv::Vec3f> cleaned = clean_surface_outliers(m, 5.0f, true);
    REQUIRE(cleaned.rows == 4);
}
