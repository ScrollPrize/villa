// Coverage for the grid-isometry block in GridIsometry.hpp (ScrollPrize/villa
// issue #2279). The point of the block is that a tifxyz grid's stretch against
// the step its meta.json promises is a standard, comparable number, so these
// tests pin the definitions on grids whose isometry is known by construction:
// an exact grid, a grid stretched along one axis, and a grid with an invalid
// cell that must be skipped exactly as the area metric skips it.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/GridIsometry.hpp"

#include <cmath>

using vc::surface::computeGridIsometry;
using vc::surface::computeGridIsometryStats;
using vc::surface::isometryStepsFromMeta;
using vc::surface::storeIsometryMeta;

namespace
{
// A flat grid in the xy-plane: x = col * du, y = row * dv, z = 0.
cv::Mat_<cv::Vec3f> flatGrid(int rows, int cols, float du, float dv)
{
    cv::Mat_<cv::Vec3f> points(rows, cols);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            points(y, x) = cv::Vec3f(static_cast<float>(x) * du, static_cast<float>(y) * dv, 0.0f);
        }
    }
    return points;
}
} // namespace

TEST_CASE("an exact grid is perfectly isometric")
{
    // 20 x 24 nodes spaced 5 voxels; scale 0.2 promises exactly that step.
    const auto points = flatGrid(20, 24, 5.0f, 5.0f);
    const auto s = computeGridIsometryStats(points, 5.0, 5.0);

    CHECK(s.step_u == doctest::Approx(5.0));
    CHECK(s.step_v == doctest::Approx(5.0));
    CHECK(s.edges == static_cast<std::size_t>(20 * 23 + 19 * 24));
    CHECK(s.within_10pct == doctest::Approx(1.0));
    CHECK(s.within_25pct == doctest::Approx(1.0));
    CHECK(s.median_abs_log_ratio == doctest::Approx(0.0));
    CHECK(s.area_ratio == doctest::Approx(1.0));
    CHECK(s.angle_p5 == doctest::Approx(90.0));
    CHECK(s.angle_p50 == doctest::Approx(90.0));
    CHECK(s.angle_p95 == doctest::Approx(90.0));
    CHECK(s.stretched_nodes == 0);
    CHECK(s.stretched_area_vx2 == doctest::Approx(0.0));
    CHECK(s.l2_mean == doctest::Approx(1.0));
    CHECK(s.l2_median == doctest::Approx(1.0));
    CHECK(s.linf == doctest::Approx(1.0));
    CHECK(s.area_error == doctest::Approx(0.0).scale(1.0)); // ~1e-16 of cancellation
}

TEST_CASE("a grid stretched 1.3x along one axis is half outside the 10% band")
{
    // 1.3x is beyond the 1.1 factor but still under the 1.5 stretch factor, so
    // half the edges are outside 10% and no node is flagged as stretched.
    const auto points = flatGrid(20, 24, 6.5f, 5.0f);
    const auto s = computeGridIsometryStats(points, 5.0, 5.0);

    const double total = static_cast<double>(20 * 23 + 19 * 24);
    const double vEdges = static_cast<double>(19 * 24);
    CHECK(s.within_10pct == doctest::Approx(vEdges / total));
    CHECK(s.within_25pct == doctest::Approx(vEdges / total));
    CHECK(s.median_abs_log_ratio == doctest::Approx(std::log(1.3)).epsilon(1e-3));
    CHECK(s.stretched_nodes == 0);
    CHECK(s.stretched_area_vx2 == doctest::Approx(0.0));
    CHECK(s.area_ratio == doctest::Approx(1.3));
    // The two singular values are 1.3 and 1, so L2 is their root-mean-square and
    // Linf is the larger one.
    CHECK(s.l2_mean == doctest::Approx(std::sqrt((1.3 * 1.3 + 1.0) / 2.0)));
    CHECK(s.l2_median == doctest::Approx((1.3 * 1.3 + 1.0) / 2.0));
    CHECK(s.linf == doctest::Approx(1.3));
}

TEST_CASE("stretched_area_vx2 is one promised cell per flagged node")
{
    // One interior node pulled out of an exact grid: the flagged area is the node
    // count times ONE cell (5 * 5), not the summed 3D area of the cells it
    // touches (issue #2279: "their area at one grid cell per node").
    auto points = flatGrid(5, 5, 5.0f, 5.0f);
    points(2, 2)[0] += 10.0f;
    const auto s = computeGridIsometryStats(points, 5.0, 5.0);

    CHECK(s.stretched_nodes > 0);
    CHECK(s.stretched_area_vx2 == doctest::Approx(static_cast<double>(s.stretched_nodes) * 25.0));
}

TEST_CASE("an invalid sentinel cell is skipped like the area metric")
{
    auto points = flatGrid(5, 5, 5.0f, 5.0f);
    points(2, 2) = cv::Vec3f(-1.0f, -1.0f, -1.0f);
    const auto s = computeGridIsometryStats(points, 5.0, 5.0);

    // 40 edges less the 4 that touch (2, 2); the 4 cells it corners are dropped
    // from the area/angle/flatboi metrics, leaving 12 cells.
    CHECK(s.edges == 36u);
    CHECK(s.within_10pct == doctest::Approx(1.0));
    CHECK(s.stretched_nodes == 0);
    CHECK(s.area_ratio == doctest::Approx(1.0));
    CHECK(s.l2_mean == doctest::Approx(1.0));
}

TEST_CASE("computeGridIsometry returns the issue's field list")
{
    const auto points = flatGrid(8, 8, 5.0f, 5.0f);
    const auto block = computeGridIsometry(points, 5.0, 5.0);

    CHECK(block["version"].get_int() == 1);
    REQUIRE(block.contains("step_vx"));
    CHECK(block["step_vx"].size() == 2u);
    CHECK(block["step_vx"][0].get_double() == doctest::Approx(5.0));
    REQUIRE(block.contains("angle_deg"));
    CHECK(block["angle_deg"].size() == 3u); // [p5, p50, p95], a list as in the reference
    for (const char* key : {"edges", "within_10pct", "within_25pct", "median_abs_log_ratio",
                            "area_ratio", "stretched_nodes", "stretched_area_vx2",
                            "l2_mean", "l2_median", "linf", "area_error"}) {
        CHECK(block.contains(key));
    }
}

TEST_CASE("storeIsometryMeta measures against the metadata scale")
{
    // The promised step comes from meta.json's scale (1 / scale), not the
    // surface's float32 copy, so the block is measured against the number a
    // reader sees.
    auto meta = utils::Json::object();
    meta["scale"] = utils::Json::array();
    meta["scale"].push_back(0.25);
    meta["scale"].push_back(0.125);

    double step_u = 0.0;
    double step_v = 0.0;
    REQUIRE(isometryStepsFromMeta(meta, step_u, step_v));
    CHECK(step_u == doctest::Approx(4.0));
    CHECK(step_v == doctest::Approx(8.0));

    const auto points = flatGrid(6, 6, 4.0f, 8.0f);
    storeIsometryMeta(meta, points, cv::Vec2f(0.25f, 0.125f));
    REQUIRE(meta.contains("isometry"));
    CHECK(meta["isometry"]["step_vx"][0].get_double() == doctest::Approx(4.0));
    CHECK(meta["isometry"]["within_10pct"].get_double() == doctest::Approx(1.0));
}
