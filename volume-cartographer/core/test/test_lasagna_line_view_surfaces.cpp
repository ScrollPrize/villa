#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"

#include <cmath>
#include <stdexcept>

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

TEST_CASE("LineViewBuilder creates ribbons from optimized control points")
{
    const auto views = vc::lasagna::buildLineViewSurfaces(simpleLine());

    REQUIRE(views.lineSurface);
    REQUIRE(views.lineSideSlice);
    const auto surfacePoints = views.lineSurface->rawPoints();
    const auto sideSlicePoints = views.lineSideSlice->rawPoints();

    REQUIRE(surfacePoints.rows == 21);
    REQUIRE(surfacePoints.cols == 3);
    REQUIRE(sideSlicePoints.rows == 21);
    REQUIRE(sideSlicePoints.cols == 3);

    checkVec(surfacePoints(10, 0), {0.0, 0.0, 0.0});
    checkVec(surfacePoints(10, 1), {10.0, 0.0, 0.0});
    checkVec(surfacePoints(10, 2), {20.0, 0.0, 0.0});
}

TEST_CASE("LineViewBuilder default strip spacing matches optimized step")
{
    const auto views = vc::lasagna::buildLineViewSurfaces(simpleLine());
    const auto surfacePoints = views.lineSurface->rawPoints();

    REQUIRE(surfacePoints.rows == 21);
    REQUIRE(surfacePoints.cols == 3);

    checkVec(surfacePoints(9, 1), {10.0, -10.0, 0.0});
    checkVec(surfacePoints(10, 1), {10.0, 0.0, 0.0});
    checkVec(surfacePoints(11, 1), {10.0, 10.0, 0.0});
}

TEST_CASE("LineViewBuilder offsets line-surface along side and side-slice along normal")
{
    vc::lasagna::LineViewConfig config;
    config.surfaceHalfWidth = 2.0;
    config.sideSliceHalfDepth = 3.0;
    config.crossSamples = 3;

    const auto views = vc::lasagna::buildLineViewSurfaces(simpleLine(), config);
    const auto surfacePoints = views.lineSurface->rawPoints();
    const auto sideSlicePoints = views.lineSideSlice->rawPoints();

    checkVec(surfacePoints(0, 1), {10.0, -2.0, 0.0});
    checkVec(surfacePoints(1, 1), {10.0, 0.0, 0.0});
    checkVec(surfacePoints(2, 1), {10.0, 2.0, 0.0});

    checkVec(sideSlicePoints(0, 1), {10.0, 0.0, -3.0});
    checkVec(sideSlicePoints(1, 1), {10.0, 0.0, 0.0});
    checkVec(sideSlicePoints(2, 1), {10.0, 0.0, 3.0});
}

TEST_CASE("LineViewBuilder uses fitted mesh normal for side-slice")
{
    vc::lasagna::LineViewConfig config;
    config.surfaceHalfWidth = 2.0;
    config.sideSliceHalfDepth = 3.0;
    config.crossSamples = 3;

    const auto views = vc::lasagna::buildLineViewSurfaces(simpleLine({1.0, 0.0, 1.0}),
                                                          config);
    const auto sideSlicePoints = views.lineSideSlice->rawPoints();

    checkVec(sideSlicePoints(0, 1), {10.0, 0.0, -3.0});
    checkVec(sideSlicePoints(1, 1), {10.0, 0.0, 0.0});
    checkVec(sideSlicePoints(2, 1), {10.0, 0.0, 3.0});

    const cv::Vec3f offset = sideSlicePoints(2, 1) - sideSlicePoints(1, 1);
    CHECK(offset[0] == doctest::Approx(0.0));
}

TEST_CASE("LineViewBuilder creates one z slice per optimized control point")
{
    const auto views = vc::lasagna::buildLineViewSurfaces(simpleLine());

    REQUIRE(views.lineZSlices.size() == 3);
    REQUIRE(views.lineUpVectors.size() == 3);
    for (const auto& slice : views.lineZSlices) {
        REQUIRE(slice);
        checkVec(slice->basisY(), {0.0, 0.0, 1.0});
    }
    for (const auto& up : views.lineUpVectors) {
        checkVec(up, {0.0, 0.0, 1.0});
    }
}

TEST_CASE("LineViewBuilder diagnostics flag sampled normal axis jumps")
{
    auto line = simpleLine();
    line.points[0].sampledNormal = normal({0.0, 0.0, 1.0});
    line.points[1].sampledNormal = normal({0.0, 1.0, 0.0});
    line.points[2].sampledNormal = normal({0.0, 0.0, 1.0});

    const auto diagnostics = vc::lasagna::diagnoseLineViewFrames(line);

    CHECK(diagnostics.minSampledAxisContinuityDot < 0.5);
    REQUIRE(!diagnostics.issues.empty());
    CHECK(diagnostics.issues.front().reason == "sampled_normal_axis_jump");
}

TEST_CASE("LineViewBuilder uses finite deterministic fallback frames")
{
    auto line = simpleLine({1.0, 0.0, 0.0});
    line.points[0].sampledNormal = normal({0.0, 0.0, 0.0}, false);
    line.points[2].sampledNormal = normal({0.0, 0.0, 0.0}, false);

    vc::lasagna::LineViewConfig config;
    config.surfaceHalfWidth = 5.0;
    const auto views = vc::lasagna::buildLineViewSurfaces(line, config);
    const auto points = views.lineSurface->rawPoints();

    REQUIRE(points.rows == 21);
    REQUIRE(points.cols == 3);
    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col < points.cols; ++col) {
            CHECK(finitePoint(points(row, col)));
        }
    }
    checkVec(points(10, 1), {10.0, 0.0, 0.0});
}

TEST_CASE("LineViewBuilder rejects empty models and invalid cross-sample counts")
{
    vc::lasagna::LineModel empty;
    CHECK_THROWS_AS(vc::lasagna::buildLineViewSurfaces(empty), std::invalid_argument);

    vc::lasagna::LineViewConfig config;
    config.crossSamples = 1;
    CHECK_THROWS_AS(vc::lasagna::buildLineViewSurfaces(simpleLine(), config), std::invalid_argument);
}

TEST_CASE("LineViewBuilder ignores dense segment samples for generated mesh rows")
{
    auto line = simpleLine();
    line.segmentSamples[0].samples[1].position = {500.0, 0.0, 0.0};

    const auto views = vc::lasagna::buildLineViewSurfaces(line);
    REQUIRE(views.lineSurface);
    const auto points = views.lineSurface->rawPoints();

    REQUIRE(points.rows == 21);
    REQUIRE(points.cols == 3);
    checkVec(points(10, 0), {0.0, 0.0, 0.0});
    checkVec(points(10, 1), {10.0, 0.0, 0.0});
    checkVec(points(10, 2), {20.0, 0.0, 0.0});
}

TEST_CASE("LineViewBuilder uses control point count even with duplicated segment-boundary samples")
{
    auto line = simpleLine();
    line.segmentSamples[0].samples.push_back({1.0, {10.0, 0.0, 0.0}, normal({0.0, 0.0, 1.0})});
    line.segmentSamples[1].samples.insert(
        line.segmentSamples[1].samples.begin(),
        {0.0, {10.0, 0.0, 0.0}, normal({0.0, 0.0, 1.0})});

    const auto views = vc::lasagna::buildLineViewSurfaces(line);
    const auto points = views.lineSurface->rawPoints();

    REQUIRE(points.cols == 3);
    checkVec(points(10, 1), {10.0, 0.0, 0.0});
}

TEST_CASE("LineViewBuilder falls back when all normals or tangents are degenerate")
{
    auto invalidNormalLine = simpleLine({0.0, 0.0, 0.0});
    for (auto& point : invalidNormalLine.points) {
        point.sampledNormal = normal({0.0, 0.0, 0.0}, false);
    }
    for (auto& segment : invalidNormalLine.segmentSamples) {
        for (auto& sample : segment.samples) {
            sample.sampledNormal = normal({0.0, 0.0, 0.0}, false);
        }
    }

    vc::lasagna::LineViewConfig config;
    config.surfaceHalfWidth = 4.0;
    config.sideSliceHalfDepth = 6.0;
    config.crossSamples = 3;
    auto views = vc::lasagna::buildLineViewSurfaces(invalidNormalLine, config);
    auto surfacePoints = views.lineSurface->rawPoints();
    auto sideSlicePoints = views.lineSideSlice->rawPoints();

    for (int row = 0; row < surfacePoints.rows; ++row) {
        for (int col = 0; col < surfacePoints.cols; ++col) {
            CHECK(finitePoint(surfacePoints(row, col)));
            CHECK(finitePoint(sideSlicePoints(row, col)));
        }
    }

    auto degenerateTangentLine = simpleLine();
    for (auto& point : degenerateTangentLine.points) {
        point.position = {5.0, 5.0, 5.0};
    }
    views = vc::lasagna::buildLineViewSurfaces(degenerateTangentLine, config);
    surfacePoints = views.lineSurface->rawPoints();
    REQUIRE(surfacePoints.rows == 3);
    REQUIRE(surfacePoints.cols == 3);
    CHECK(finitePoint(surfacePoints(0, 0)));
    CHECK(finitePoint(surfacePoints(0, 1)));
    CHECK(finitePoint(surfacePoints(0, 2)));
}
