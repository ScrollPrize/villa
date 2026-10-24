// Coverage for the #1379 guards in abfFlattenToNewSurface: the output grid is
// sized from the measured grid density when meta.json "scale" disagrees with
// the points, and a grid above kAbfMaxOutputCells is refused instead of
// asking OpenCV for a terabyte-sized allocation.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/flattening/ABFFlattening.hpp"

#include <limits>
#include <memory>

namespace {

// Flat rows x cols grid in the z=5 plane, `step` voxels between neighbours.
cv::Mat_<cv::Vec3f> planeGrid(int rows, int cols, float step)
{
    cv::Mat_<cv::Vec3f> pts(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            pts(r, c) = cv::Vec3f(100.f + c * step, 200.f + r * step, 5.f);
        }
    }
    return pts;
}

} // namespace

TEST_CASE("measuredGridScale: step 20 voxels gives 0.05 cells per voxel")
{
    const cv::Vec2f s = vc::measuredGridScale(planeGrid(40, 30, 20.f));
    CHECK(s[0] == doctest::Approx(0.05f));
    CHECK(s[1] == doctest::Approx(0.05f));
}

TEST_CASE("measuredGridScale: anisotropic steps are measured per axis")
{
    cv::Mat_<cv::Vec3f> pts(20, 20);
    for (int r = 0; r < pts.rows; ++r) {
        for (int c = 0; c < pts.cols; ++c) {
            pts(r, c) = cv::Vec3f(c * 10.f, r * 40.f, 0.f);
        }
    }
    const cv::Vec2f s = vc::measuredGridScale(pts);
    CHECK(s[0] == doctest::Approx(0.1f));
    CHECK(s[1] == doctest::Approx(0.025f));
}

TEST_CASE("measuredGridScale: invalid (-1) points are skipped")
{
    cv::Mat_<cv::Vec3f> pts = planeGrid(30, 30, 20.f);
    // Knock out every other column block; pairs touching -1 must be ignored.
    for (int r = 0; r < pts.rows; ++r) {
        for (int c = 0; c < pts.cols; c += 3) {
            pts(r, c) = cv::Vec3f(-1.f, -1.f, -1.f);
        }
    }
    const cv::Vec2f s = vc::measuredGridScale(pts);
    CHECK(s[0] == doctest::Approx(0.05f));
    CHECK(s[1] == doctest::Approx(0.05f));

    const cv::Mat_<cv::Vec3f> empty(10, 10, cv::Vec3f(-1.f, -1.f, -1.f));
    const cv::Vec2f none = vc::measuredGridScale(empty);
    CHECK(none[0] == 0.f);
    CHECK(none[1] == 0.f);
}

TEST_CASE("abfOutputGridSize: refuses grids above 1e9 cells")
{
    int w = 0;
    int h = 0;
    // UV range of the #1379 outer_shell (~18088 x 15160 voxels) at scale 0.05.
    REQUIRE(vc::abfOutputGridSize(cv::Vec2f(18087.6f, 15159.8f), cv::Vec2f(0.05f, 0.05f), w, h));
    CHECK(w == 906);
    CHECK(h == 759);

    // Same range at the broken declared scale 19.997: 361.7k x 303.2k cells.
    w = h = -7;
    CHECK_FALSE(vc::abfOutputGridSize(cv::Vec2f(18087.6f, 15159.8f), cv::Vec2f(19.997f, 19.997f), w, h));
    CHECK(w == -7);
    CHECK(h == -7);

    CHECK_FALSE(vc::abfOutputGridSize(cv::Vec2f(1e30f, 1e30f), cv::Vec2f(1e10f, 1e10f), w, h));
    CHECK_FALSE(vc::abfOutputGridSize(cv::Vec2f(100.f, 100.f),
                                      cv::Vec2f(std::numeric_limits<float>::infinity(), 1.f), w, h));

    // Degenerate ranges still produce the 2 x 2 minimum.
    REQUIRE(vc::abfOutputGridSize(cv::Vec2f(0.f, 0.f), cv::Vec2f(0.05f, 0.05f), w, h));
    CHECK(w == 2);
    CHECK(h == 2);
}

TEST_CASE("abfFlattenToNewSurface: wrong declared scale uses the measured density")
{
    // 40 x 30 points, 20 voxels apart: correct scale is 0.05.
    const cv::Mat_<cv::Vec3f> pts = planeGrid(30, 40, 20.f);

    vc::ABFConfig cfg;
    cfg.useABF = false;  // a plane needs only LSCM; keeps the test fast

    QuadSurface good(pts, cv::Vec2f(0.05f, 0.05f));
    std::unique_ptr<QuadSurface> goodFlat(vc::abfFlattenToNewSurface(good, cfg));
    REQUIRE(goodFlat);

    // Same points, scale stored as the step (19.997) like the #1379 shell.
    QuadSurface bad(pts, cv::Vec2f(19.997f, 19.997f));
    std::unique_ptr<QuadSurface> badFlat(vc::abfFlattenToNewSurface(bad, cfg));
    REQUIRE(badFlat);

    const cv::Size g = goodFlat->gridSize();
    const cv::Size b = badFlat->gridSize();
    // Without the guard the bad grid is ~400x larger per axis.
    CHECK(b.width <= g.width + 2);
    CHECK(b.height <= g.height + 2);
    CHECK(b.width >= g.width - 2);
    CHECK(b.height >= g.height - 2);
    // The output records the scale the grid was sized with, not the bad one.
    CHECK(badFlat->scale()[0] == doctest::Approx(0.05f).epsilon(0.01));
    CHECK(badFlat->scale()[1] == doctest::Approx(0.05f).epsilon(0.01));
    CHECK(goodFlat->scale()[0] == doctest::Approx(0.05f));
}
