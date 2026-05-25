#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"

#include <cmath>

namespace {

vc::lasagna::NormalSample normal(cv::Vec3d value, bool valid = true)
{
    return {value, valid, valid ? std::string{} : std::string{"missing"}};
}

vc::lasagna::LineModel simpleLine(cv::Vec3d n = {0.0, 0.0, 1.0})
{
    vc::lasagna::LineModel line;
    line.points = {
        {{0.0, 0.0, 0.0}, normal(n), true},
        {{10.0, 0.0, 0.0}, normal(n), true},
        {{20.0, 0.0, 0.0}, normal(n), true},
    };
    line.segmentSamples = {
        {{{0.0, {0.0, 0.0, 0.0}, normal(n)},
          {0.5, {5.0, 0.0, 0.0}, normal(n)},
          {1.0, {10.0, 0.0, 0.0}, normal(n)}}},
        {{{0.0, {10.0, 0.0, 0.0}, normal(n)},
          {0.5, {15.0, 0.0, 0.0}, normal(n)},
          {1.0, {20.0, 0.0, 0.0}, normal(n)}}},
    };
    return line;
}

void checkVec(const cv::Vec3f& actual, const cv::Vec3d& expected)
{
    CHECK(actual[0] == doctest::Approx(expected[0]));
    CHECK(actual[1] == doctest::Approx(expected[1]));
    CHECK(actual[2] == doctest::Approx(expected[2]));
}

bool finitePoint(const cv::Vec3f& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

} // namespace

TEST_CASE("LineViewBuilder creates ribbons from dense deduplicated samples")
{
    const auto views = vc::lasagna::buildLineViewSurfaces(simpleLine());

    REQUIRE(views.lineSurface);
    REQUIRE(views.lineSideSlice);
    const auto surfacePoints = views.lineSurface->rawPoints();
    const auto sideSlicePoints = views.lineSideSlice->rawPoints();

    REQUIRE(surfacePoints.rows == 5);
    REQUIRE(surfacePoints.cols == 3);
    REQUIRE(sideSlicePoints.rows == 5);
    REQUIRE(sideSlicePoints.cols == 3);

    checkVec(surfacePoints(0, 1), {0.0, 0.0, 0.0});
    checkVec(surfacePoints(1, 1), {5.0, 0.0, 0.0});
    checkVec(surfacePoints(2, 1), {10.0, 0.0, 0.0});
    checkVec(surfacePoints(3, 1), {15.0, 0.0, 0.0});
    checkVec(surfacePoints(4, 1), {20.0, 0.0, 0.0});
}

TEST_CASE("LineViewBuilder offsets line-surface along side and side-slice along normal")
{
    vc::lasagna::LineViewConfig config;
    config.surfaceHalfWidth = 2.0;
    config.sideSliceHalfDepth = 3.0;

    const auto views = vc::lasagna::buildLineViewSurfaces(simpleLine(), config);
    const auto surfacePoints = views.lineSurface->rawPoints();
    const auto sideSlicePoints = views.lineSideSlice->rawPoints();

    checkVec(surfacePoints(2, 0), {10.0, -2.0, 0.0});
    checkVec(surfacePoints(2, 1), {10.0, 0.0, 0.0});
    checkVec(surfacePoints(2, 2), {10.0, 2.0, 0.0});

    checkVec(sideSlicePoints(2, 0), {10.0, 0.0, -3.0});
    checkVec(sideSlicePoints(2, 1), {10.0, 0.0, 0.0});
    checkVec(sideSlicePoints(2, 2), {10.0, 0.0, 3.0});
}

TEST_CASE("LineViewBuilder creates one z slice per optimized control point")
{
    const auto views = vc::lasagna::buildLineViewSurfaces(simpleLine());

    REQUIRE(views.lineZSlices.size() == 3);
    for (const auto& slice : views.lineZSlices) {
        REQUIRE(slice);
    }
}

TEST_CASE("LineViewBuilder uses finite deterministic fallback frames")
{
    auto line = simpleLine({1.0, 0.0, 0.0});
    line.segmentSamples[0].samples[0].sampledNormal = normal({0.0, 0.0, 0.0}, false);
    line.segmentSamples[1].samples[2].sampledNormal = normal({0.0, 0.0, 0.0}, false);

    vc::lasagna::LineViewConfig config;
    config.surfaceHalfWidth = 5.0;
    const auto views = vc::lasagna::buildLineViewSurfaces(line, config);
    const auto points = views.lineSurface->rawPoints();

    REQUIRE(points.rows == 5);
    REQUIRE(points.cols == 3);
    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col < points.cols; ++col) {
            CHECK(finitePoint(points(row, col)));
        }
    }
    checkVec(points(2, 1), {10.0, 0.0, 0.0});
}
