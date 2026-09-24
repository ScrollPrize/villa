#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"

#include "../../apps/src/vc_tifxyz_topology_impl.hpp"

#include <opencv2/core/mat.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

using vc_topology::Census;
using vc_topology::Params;
using vc_topology::census;
using vc_topology::valid_point;

namespace {

const cv::Vec3f INVALID(-1.f, -1.f, -1.f);

cv::Mat_<cv::Vec3f> blank(int rows, int cols)
{
    cv::Mat_<cv::Vec3f> P(rows, cols);
    P.setTo(INVALID);
    return P;
}

void put_plane(cv::Mat_<cv::Vec3f>& P, int row0, int col0, int rows, int cols,
               double z0, double xshift = 0.0)
{
    for (int v = 0; v < rows; ++v)
        for (int u = 0; u < cols; ++u)
            P(row0 + v, col0 + u) = cv::Vec3f(
                (float)(4.0 * (col0 + u) + xshift),
                (float)(4.0 * (row0 + v)), (float)z0);
}

Census run(const cv::Mat_<cv::Vec3f>& P, double tear_factor = 4.0,
           int max_sites = 200)
{
    Params p;
    p.tear_factor = tear_factor;
    p.max_sites = max_sites;
    return census(P, p);
}

}  // namespace

TEST_CASE("a plain sheet is one component with nothing else to report")
{
    cv::Mat_<cv::Vec3f> P = blank(14, 18);
    put_plane(P, 2, 3, 10, 12, 5.0);
    const Census c = run(P);
    CHECK(c.valid_vertices == 10 * 12);
    CHECK(c.valid_quads == 9 * 11);
    CHECK(c.component_count == 1);
    CHECK(c.island_quads == 0);
    CHECK(c.hole_count == 0);
    CHECK(c.tear_edges == 0);
    CHECK(c.fold_quads == 0);
    CHECK(c.degenerate_quads == 0);
    CHECK(c.isolated_vertices == 0);
    CHECK(c.median_step_u == doctest::Approx(4.0));
    CHECK(c.median_step_v == doctest::Approx(4.0));
    CHECK(c.area_vx2 == doctest::Approx(9.0 * 11.0 * 16.0));
}

TEST_CASE("two patches separated by invalid cells are two components")
{
    cv::Mat_<cv::Vec3f> P = blank(24, 12);
    put_plane(P, 0, 0, 8, 12, 5.0);
    put_plane(P, 14, 0, 6, 12, 5.0);
    const Census c = run(P);
    CHECK(c.component_count == 2);
    REQUIRE(c.components.size() == 2u);
    CHECK(c.components[0].quads == 7 * 11);
    CHECK(c.components[1].quads == 5 * 11);
    CHECK(c.island_quads == 5 * 11);
    CHECK(c.island_area_vx2 == doctest::Approx(5.0 * 11.0 * 16.0));
    CHECK(c.hole_count == 0);
}

TEST_CASE("a single missing cell is a hole, and the padding around the sheet "
          "is not")
{
    cv::Mat_<cv::Vec3f> P = blank(12, 14);
    put_plane(P, 1, 1, 10, 12, 5.0);
    P(5, 6) = INVALID;
    const Census c = run(P);
    CHECK(c.hole_count == 1);
    CHECK(c.hole_cells == 1);
    REQUIRE(c.holes.size() == 1u);
    CHECK(c.holes[0].cells == 1);
    CHECK(c.holes[0].v0 == 5);
    CHECK(c.holes[0].u0 == 6);
    CHECK(c.holes[0].v1 == 5);
    CHECK(c.holes[0].u1 == 6);
    CHECK(valid_point(cv::Vec3f((float)c.holes[0].site.x,
                                (float)c.holes[0].site.y,
                                (float)c.holes[0].site.z)));
    CHECK(c.valid_quads == 9 * 11 - 4);
    CHECK(c.component_count == 1);
}

TEST_CASE("a bay open to the sheet's edge is not a hole")
{
    cv::Mat_<cv::Vec3f> P = blank(12, 14);
    put_plane(P, 1, 1, 10, 12, 5.0);
    for (int v = 1; v <= 4; ++v)
        P(v, 6) = INVALID;
    const Census c = run(P);
    CHECK(c.hole_count == 0);
    CHECK(c.hole_cells == 0);
}

TEST_CASE("a hole big enough to cut the sheet gives both a hole and an "
          "island")
{
    cv::Mat_<cv::Vec3f> P = blank(12, 14);
    put_plane(P, 1, 1, 10, 12, 5.0);
    for (int v = 1; v <= 10; ++v)
        P(v, 6) = INVALID;
    const Census c = run(P);
    CHECK(c.hole_count == 0);
    CHECK(c.component_count == 2);
    CHECK(c.island_quads > 0);
}

TEST_CASE("a valid cell with no complete quad around it is isolated, not a "
          "component")
{
    cv::Mat_<cv::Vec3f> P = blank(12, 14);
    put_plane(P, 1, 1, 8, 8, 5.0);
    P(11, 13) = cv::Vec3f(9000.f, 9000.f, 9000.f);
    const Census c = run(P);
    CHECK(c.isolated_vertices == 1);
    CHECK(c.component_count == 1);
    CHECK(c.valid_vertices == 8 * 8 + 1);
}

TEST_CASE("a grid edge much longer than the surface's own step is a tear, "
          "and neighbouring torn edges are one site")
{
    cv::Mat_<cv::Vec3f> P = blank(12, 14);
    put_plane(P, 1, 1, 10, 12, 5.0);
    for (int v = 1; v <= 10; ++v)
        P(v, 6)[2] = 205.f;
    const Census c = run(P);
    CHECK(c.tear_edges > 0);
    CHECK(c.tear_site_count == 1);
    REQUIRE(c.tears.size() == 1u);
    CHECK(c.tears[0].max_len > 4.0 * 4.0);
    CHECK(c.max_edge_ratio > 4.0);
}

TEST_CASE("the tear threshold is relative, so a coarser surface is not all "
          "tears")
{
    cv::Mat_<cv::Vec3f> P = blank(12, 14);
    for (int v = 0; v < 10; ++v)
        for (int u = 0; u < 12; ++u)
            P(1 + v, 1 + u) = cv::Vec3f(40.f * u, 40.f * v, 5.f);
    const Census c = run(P);
    CHECK(c.median_step_u == doctest::Approx(40.0));
    CHECK(c.tear_edges == 0);
}

TEST_CASE("a quad folded back on itself is a fold; a strongly warped quad "
          "is not")
{
    cv::Mat_<cv::Vec3f> P = blank(6, 6);
    put_plane(P, 1, 1, 4, 4, 5.0);
    SUBCASE("saddle") {
        P(2, 2)[2] = 7.f;
        P(3, 3)[2] = 7.f;
        const Census c = run(P);
        CHECK(c.fold_quads == 0);
    }
    SUBCASE("crossed") {
        const cv::Vec3f a = P(2, 2), b = P(2, 3);
        P(2, 2) = b;
        P(2, 3) = a;
        const Census c = run(P);
        CHECK(c.fold_quads > 0);
        CHECK(c.fold_site_count > 0);
    }
}

TEST_CASE("PHerc0172 20251106170358 cell (286,566), folded 148 degrees "
          "across p00-p11 and 80 across p01-p10: counted, not located")
{
    cv::Mat_<cv::Vec3f> P = blank(6, 6);
    put_plane(P, 1, 1, 4, 4, 5.0);
    P(2, 2) = cv::Vec3f(1458.807251f, 4195.333008f, 8384.976562f);
    P(2, 3) = cv::Vec3f(1467.635376f, 4195.754395f, 8375.624023f);
    P(3, 2) = cv::Vec3f(1471.452881f, 4198.676758f, 8368.525391f);
    P(3, 3) = cv::Vec3f(1472.222290f, 4196.083984f, 8356.984375f);
    const Census c = run(P);
    CHECK(c.fold_quads_other_diagonal >= 1);
    bool at_quad = false;
    for (const auto& f : c.folds)
        if (f.v == 2 && f.u == 2) at_quad = true;
    CHECK_FALSE(at_quad);
}

TEST_CASE("PHercParis4 20260603222816 cell (96,764), a 0.7 by 20 voxel "
          "sliver 90.0005 degrees apart: not a fold")
{
    cv::Mat_<cv::Vec3f> P = blank(6, 6);
    put_plane(P, 1, 1, 4, 4, 5.0);
    P(2, 2) = cv::Vec3f(3893.86376953f, 4396.81982422f, 11223.69042969f);
    P(2, 3) = cv::Vec3f(3893.86425781f, 4396.60644531f, 11223.01953125f);
    P(3, 2) = cv::Vec3f(3893.99194336f, 4390.69873047f, 11204.37304688f);
    P(3, 3) = cv::Vec3f(3893.99707031f, 4390.48779297f, 11203.70214844f);
    const Census c = run(P);
    bool at_sliver = false;
    for (const auto& f : c.folds)
        if (f.v == 2 && f.u == 2) at_sliver = true;
    CHECK_FALSE(at_sliver);
}

TEST_CASE("a collapsed quad is degenerate, not folded")
{
    cv::Mat_<cv::Vec3f> P = blank(6, 6);
    put_plane(P, 1, 1, 4, 4, 5.0);
    P(2, 2) = P(1, 1);
    P(2, 3) = P(1, 1);
    P(3, 2) = P(1, 1);
    P(3, 3) = P(1, 1);
    const Census c = run(P);
    CHECK(c.degenerate_quads > 0);
}

TEST_CASE("an empty grid and an all-invalid grid census clean rather than "
          "failing")
{
    const Census e = run(cv::Mat_<cv::Vec3f>());
    CHECK(e.valid_vertices == 0);
    CHECK(e.valid_quads == 0);
    CHECK(e.component_count == 0);

    const Census i = run(blank(8, 8));
    CHECK(i.valid_vertices == 0);
    CHECK(i.valid_quads == 0);
    CHECK(i.component_count == 0);
    CHECK(i.hole_count == 0);
    CHECK(i.median_step_u == 0.0);
}

TEST_CASE("a non-finite coordinate is invalid, whatever its sentinel says")
{
    cv::Mat_<cv::Vec3f> P = blank(12, 14);
    put_plane(P, 1, 1, 10, 12, 5.0);
    P(5, 6) = cv::Vec3f(std::numeric_limits<float>::quiet_NaN(), 1.f, 5.f);
    const Census c = run(P);
    CHECK(c.valid_vertices == 10 * 12 - 1);
    CHECK(c.hole_count == 1);
    CHECK(std::isfinite(c.median_step_u));
    CHECK(c.median_step_u == doctest::Approx(4.0));
}

TEST_CASE("max_sites bounds the located sites and never the counts")
{
    cv::Mat_<cv::Vec3f> P = blank(16, 16);
    put_plane(P, 1, 1, 14, 14, 5.0);
    for (int v = 2; v <= 12; v += 2)
        for (int u = 2; u <= 12; u += 2)
            P(v, u) = INVALID;
    const Census full = run(P);
    const Census capped = run(P, 4.0, 3);
    CHECK(full.hole_count > 3);
    CHECK(capped.hole_count == full.hole_count);
    CHECK(capped.holes.size() == 3u);
}

TEST_CASE("a tear factor at or below one, or a negative site cap, is "
          "refused")
{
    cv::Mat_<cv::Vec3f> P = blank(6, 6);
    put_plane(P, 1, 1, 4, 4, 5.0);
    CHECK_THROWS_AS(run(P, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(run(P, 0.5), std::invalid_argument);
    CHECK_THROWS_AS(
        run(P, std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument);
    CHECK_THROWS_AS(run(P, 4.0, -1), std::invalid_argument);
}
