#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/PlaneSurface.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <vector>

namespace {

bool nearlyEq(const cv::Vec3f& a, const cv::Vec3f& b, float eps = 1e-4f)
{
    return std::abs(a[0] - b[0]) < eps
        && std::abs(a[1] - b[1]) < eps
        && std::abs(a[2] - b[2]) < eps;
}

} // namespace

TEST_CASE("default-constructed plane has z-normal at origin")
{
    PlaneSurface p;
    CHECK(p.origin() == cv::Vec3f(0, 0, 0));
    cv::Vec3f n = p.normal(cv::Vec3f(0, 0, 0));
    CHECK(nearlyEq(n, cv::Vec3f(0, 0, 1)));
}

TEST_CASE("constructor sets origin & normalized normal")
{
    PlaneSurface p(cv::Vec3f(1, 2, 3), cv::Vec3f(0, 0, 5));
    CHECK(p.origin() == cv::Vec3f(1, 2, 3));
    CHECK(nearlyEq(p.normal(cv::Vec3f(0, 0, 0)), cv::Vec3f(0, 0, 1)));
}

TEST_CASE("setNormal normalizes input")
{
    PlaneSurface p;
    p.setNormal(cv::Vec3f(0, 0, 7));
    CHECK(nearlyEq(p.normal(cv::Vec3f(0, 0, 0)), cv::Vec3f(0, 0, 1)));
}

TEST_CASE("setOrigin / origin roundtrip")
{
    PlaneSurface p;
    p.setOrigin(cv::Vec3f(10, 20, 30));
    CHECK(p.origin() == cv::Vec3f(10, 20, 30));
}

TEST_CASE("pointDist of point on the plane is zero")
{
    PlaneSurface p(cv::Vec3f(0, 0, 5), cv::Vec3f(0, 0, 1));
    CHECK(p.pointDist(cv::Vec3f(10, 20, 5)) == doctest::Approx(0.0f));
}

TEST_CASE("pointDist returns absolute distance to plane")
{
    PlaneSurface p(cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 1));
    CHECK(p.pointDist(cv::Vec3f(0, 0, 7)) == doctest::Approx(7.0f));
    CHECK(p.pointDist(cv::Vec3f(0, 0, -3)) == doctest::Approx(3.0f));
}

TEST_CASE("scalarp is signed plane-offset")
{
    PlaneSurface p(cv::Vec3f(0, 0, 5), cv::Vec3f(0, 0, 1));
    CHECK(p.scalarp(cv::Vec3f(0, 0, 5)) == doctest::Approx(0.0f));
    CHECK(p.scalarp(cv::Vec3f(0, 0, 10)) == doctest::Approx(5.0f));
    CHECK(p.scalarp(cv::Vec3f(0, 0, 0)) == doctest::Approx(-5.0f));
}

TEST_CASE("project xy plane: identity on (x,y); z = signed distance")
{
    PlaneSurface p(cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 1));
    auto r = p.project(cv::Vec3f(3, 4, 0));
    CHECK(r[0] == doctest::Approx(3.0f).epsilon(1e-3));
    CHECK(r[1] == doctest::Approx(4.0f).epsilon(1e-3));
    CHECK(std::abs(r[2]) < 1e-3f);

    // Non-zero z should appear on the plane-normal axis after projection
    auto r2 = p.project(cv::Vec3f(0, 0, 5));
    CHECK(std::abs(r2[2] - 5.0f) < 1e-3f);
}

TEST_CASE("project applies render_scale * coord_scale")
{
    PlaneSurface p(cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 1));
    auto base = p.project(cv::Vec3f(1, 1, 0));
    auto scaled = p.project(cv::Vec3f(1, 1, 0), 2.0f, 3.0f);
    CHECK(scaled[0] == doctest::Approx(base[0] * 6.0f).epsilon(1e-3));
    CHECK(scaled[1] == doctest::Approx(base[1] * 6.0f).epsilon(1e-3));
}

TEST_CASE("move and loc are simple offsets")
{
    PlaneSurface p;
    cv::Vec3f ptr(1, 2, 3);
    p.move(ptr, cv::Vec3f(10, 20, 30));
    CHECK(ptr == cv::Vec3f(11, 22, 33));
    auto l = p.loc(cv::Vec3f(1, 2, 3), cv::Vec3f(4, 5, 6));
    CHECK(l == cv::Vec3f(5, 7, 9));
}

TEST_CASE("coord on xy plane equals (x,y,origin.z+z)")
{
    PlaneSurface p(cv::Vec3f(0, 0, 5), cv::Vec3f(0, 0, 1));
    auto c = p.coord(cv::Vec3f(3, 4, 0), cv::Vec3f(0, 0, 0));
    CHECK(c[0] == doctest::Approx(3.0f));
    CHECK(c[1] == doctest::Approx(4.0f));
    CHECK(c[2] == doctest::Approx(5.0f));

    // Offset along z should move along the plane normal
    auto c2 = p.coord(cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 2));
    CHECK(c2[2] == doctest::Approx(7.0f));
}

TEST_CASE("valid always returns true")
{
    PlaneSurface p;
    CHECK(p.valid(cv::Vec3f(0, 0, 0)));
    CHECK(p.valid(cv::Vec3f(1e6f, 1e6f, 1e6f)));
}

TEST_CASE("gen with z-normal yields planar (x,y,z) grid")
{
    PlaneSurface p(cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 1));
    cv::Mat_<cv::Vec3f> coords;
    p.gen(&coords, nullptr, cv::Size(8, 4), cv::Vec3f(0, 0, 0), 1.0f, cv::Vec3f(0, 0, 0));
    REQUIRE(coords.rows == 4);
    REQUIRE(coords.cols == 8);
    // All z should be 0; xy should span a regular grid.
    for (int r = 0; r < coords.rows; ++r) {
        for (int c = 0; c < coords.cols; ++c) {
            CHECK(std::abs(coords(r, c)[2]) < 1e-4f);
        }
    }
}

TEST_CASE("gen scale parameter changes spacing")
{
    PlaneSurface p(cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 1));
    cv::Mat_<cv::Vec3f> c1, c2;
    p.gen(&c1, nullptr, cv::Size(4, 1), cv::Vec3f(0, 0, 0), 1.0f, cv::Vec3f(0, 0, 0));
    p.gen(&c2, nullptr, cv::Size(4, 1), cv::Vec3f(0, 0, 0), 0.5f, cv::Vec3f(0, 0, 0));
    // doubled inverse scale (1/0.5 = 2) → step is 2x larger
    float step1 = cv::norm(c1(0, 1) - c1(0, 0));
    float step2 = cv::norm(c2(0, 1) - c2(0, 0));
    CHECK(step2 == doctest::Approx(step1 * 2.0f).epsilon(1e-3));
}

TEST_CASE("gen with null coords and non-null normals does not crash")
{
    PlaneSurface p;
    cv::Mat_<cv::Vec3f> normals;
    p.gen(nullptr, &normals, cv::Size(4, 4), cv::Vec3f(0, 0, 0), 1.0f, cv::Vec3f(0, 0, 1));
    // Behavior intentionally not asserted strongly — we just exercise the path.
    CHECK(true);
}

TEST_CASE("setFromNormalAndUp builds orthonormal basis")
{
    PlaneSurface p;
    p.setFromNormalAndUp(cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 1), cv::Vec3f(0, 1, 0));
    auto vx = p.basisX();
    auto vy = p.basisY();
    CHECK(std::abs(cv::norm(vx) - 1.0) < 1e-4);
    CHECK(std::abs(cv::norm(vy) - 1.0) < 1e-4);
    CHECK(std::abs(vx.dot(vy)) < 1e-4f);
    CHECK(std::abs(vx.dot(cv::Vec3f(0, 0, 1))) < 1e-4f);
    CHECK(std::abs(vy.dot(cv::Vec3f(0, 0, 1))) < 1e-4f);
}

TEST_CASE("setFromNormalAndUp handles upHint parallel to normal (fallback)")
{
    PlaneSurface p;
    // upHint exactly along normal — must fall back without producing NaN basis
    p.setFromNormalAndUp(cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 1), cv::Vec3f(0, 0, 1));
    auto vx = p.basisX();
    auto vy = p.basisY();
    CHECK(std::isfinite(vx[0]));
    CHECK(std::abs(cv::norm(vx) - 1.0) < 1e-4);
    CHECK(std::abs(cv::norm(vy) - 1.0) < 1e-4);
}

TEST_CASE("setInPlaneRotation stores and rotates basis")
{
    PlaneSurface p(cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 1));
    auto vx0 = p.basisX();
    p.setInPlaneRotation(static_cast<float>(M_PI) / 2.0f);
    CHECK(p.inPlaneRotation() == doctest::Approx(M_PI / 2.0));
    auto vx1 = p.basisX();
    // After 90° rotation around z, vx should now point along ±y (perpendicular to original vx)
    CHECK(std::abs(vx0.dot(vx1)) < 1e-3f);
}
