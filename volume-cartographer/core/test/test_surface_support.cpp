// Coverage for vc::surface::onPredictionSupport (see #1675): the fraction of
// valid surface vertices landing on nonzero prediction voxels. A grown surface
// that follows its prediction scores ~1.0; one that cut across windings scores
// near the volume background rate (~0.06-0.10 measured on PHerc1203).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/SurfaceSupport.hpp"

#include <vector>

using vc::surface::onPredictionSupport;

// 8x8x8 toy prediction: a single lit "sheet" at z == 3, everything else 0.
struct ToyPrediction {
    std::vector<uint8_t> v = std::vector<uint8_t>(8 * 8 * 8, 0);
    ToyPrediction()
    {
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 8; ++x) v[(3 * 8 + y) * 8 + x] = 255;
    }
    uint8_t operator()(int z, int y, int x) const
    {
        if (z < 0 || z >= 8 || y < 0 || y >= 8 || x < 0 || x >= 8)
            return 0;
        return v[(z * 8 + y) * 8 + x];
    }
};

static cv::Mat_<cv::Vec3f> make_points(int rows, int cols, cv::Vec3f fill)
{
    cv::Mat_<cv::Vec3f> pts(rows, cols);
    pts.setTo(fill);
    return pts;
}

TEST_CASE("onPredictionSupport: vertices on the sheet score 1.0")
{
    ToyPrediction pred;
    auto pts = make_points(4, 5, cv::Vec3f(2.f, 2.f, 3.f));  // x, y, z
    CHECK(onPredictionSupport(pts, pred).fraction == doctest::Approx(1.0));
}

TEST_CASE("onPredictionSupport: vertices off the sheet score 0.0")
{
    ToyPrediction pred;
    auto pts = make_points(4, 5, cv::Vec3f(2.f, 2.f, 6.f));
    CHECK(onPredictionSupport(pts, pred).fraction == doctest::Approx(0.0));
}

TEST_CASE("onPredictionSupport: mixed vertices give the exact fraction")
{
    ToyPrediction pred;
    auto pts = make_points(2, 4, cv::Vec3f(1.f, 1.f, 3.f));
    pts(0, 0) = cv::Vec3f(1.f, 1.f, 0.f);
    pts(1, 3) = cv::Vec3f(1.f, 1.f, 7.f);
    CHECK(onPredictionSupport(pts, pred).fraction == doctest::Approx(0.75));
}

TEST_CASE("onPredictionSupport: invalid vertices are skipped")
{
    ToyPrediction pred;
    auto pts = make_points(2, 2, cv::Vec3f(2.f, 2.f, 3.f));
    pts(0, 0) = cv::Vec3f(-1.f, -1.f, -1.f);  // invalid: not counted
    pts(0, 1) = cv::Vec3f(2.f, 2.f, 0.f);     // valid, off sheet
    // 2 of 3 valid vertices on the sheet
    const auto res = onPredictionSupport(pts, pred);
    CHECK(res.fraction == doctest::Approx(2.0 / 3.0));
    CHECK(res.total == 3);
    CHECK(res.on == 2);
}

TEST_CASE("onPredictionSupport: no valid vertices returns 1.0 (nothing to judge)")
{
    ToyPrediction pred;
    auto pts = make_points(3, 3, cv::Vec3f(-1.f, -1.f, -1.f));
    CHECK(onPredictionSupport(pts, pred).fraction == doctest::Approx(1.0));
}

TEST_CASE("onPredictionSupport: samples the nearest voxel")
{
    ToyPrediction pred;
    auto pts = make_points(1, 2, cv::Vec3f(2.f, 2.f, 3.4f));  // rounds to z=3
    pts(0, 1) = cv::Vec3f(2.f, 2.f, 3.6f);                   // rounds to z=4
    CHECK(onPredictionSupport(pts, pred).fraction == doctest::Approx(0.5));
}
