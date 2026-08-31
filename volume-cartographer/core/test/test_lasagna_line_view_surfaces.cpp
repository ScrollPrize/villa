#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"

#include <cmath>
#include <limits>
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

// Five straight control points whose sampled normals alternate sign, so the
// display up sign depends entirely on which point is the display anchor.
vc::lasagna::LineModel alternatingNormalLine()
{
    vc::lasagna::LineModel line;
    for (int i = 0; i < 5; ++i) {
        const cv::Vec3d n{0.0, 0.0, (i % 2 == 0) ? 1.0 : -1.0};
        line.points.push_back({{10.0 * static_cast<double>(i), 0.0, 0.0}, normal(n), true});
    }
    return line;
}

cv::Vec3d toVec3d(const cv::Vec3f& v)
{
    return {static_cast<double>(v[0]),
            static_cast<double>(v[1]),
            static_cast<double>(v[2])};
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

vc::lasagna::LineViewConfig tenVoxelRibbonConfig()
{
    vc::lasagna::LineViewConfig config;
    config.targetSpacingBaseVoxels = 10.0;
    config.controlPointLinePositions = {0.0, 1.0, 2.0};
    return config;
}

} // namespace

TEST_CASE("LineViewBuilder creates ribbons from annotation control-point spans")
{
    const auto views = vc::lasagna::buildLineViewSurfaces(simpleLine(), tenVoxelRibbonConfig());

    REQUIRE(views.lineSurface);
    REQUIRE(views.lineSideSlice);
    const auto surfacePoints = views.lineSurface->rawPoints();
    const auto sideSlicePoints = views.lineSideSlice->rawPoints();

    REQUIRE(surfacePoints.rows == 7);
    REQUIRE(surfacePoints.cols == 3);
    REQUIRE(sideSlicePoints.rows == 7);
    REQUIRE(sideSlicePoints.cols == 3);

    checkVec(surfacePoints(3, 0), {0.0, 0.0, 0.0});
    checkVec(surfacePoints(3, 1), {10.0, 0.0, 0.0});
    checkVec(surfacePoints(3, 2), {20.0, 0.0, 0.0});
}

TEST_CASE("LineViewBuilder uses a fixed seven-row cross strip")
{
    vc::lasagna::LineViewConfig config;
    config.controlPointLinePositions = {0.0, 1.0, 2.0};
    const auto views = vc::lasagna::buildLineViewSurfaces(simpleLine(), config);
    const auto surfacePoints = views.lineSurface->rawPoints();
    const auto sideSlicePoints = views.lineSideSlice->rawPoints();

    REQUIRE(surfacePoints.rows == 7);
    REQUIRE(surfacePoints.cols == 3);
    REQUIRE(sideSlicePoints.rows == 7);
    REQUIRE(sideSlicePoints.cols == 3);

    checkVec(surfacePoints(0, 0), {0.0, -96.0, 0.0});
    checkVec(surfacePoints(2, 0), {0.0, -32.0, 0.0});
    checkVec(surfacePoints(3, 0), {0.0, 0.0, 0.0});
    checkVec(surfacePoints(4, 0), {0.0, 32.0, 0.0});
    checkVec(surfacePoints(6, 0), {0.0, 96.0, 0.0});
    checkVec(sideSlicePoints(0, 0), {0.0, 0.0, -96.0});
    checkVec(sideSlicePoints(2, 0), {0.0, 0.0, -32.0});
    checkVec(sideSlicePoints(3, 0), {0.0, 0.0, 0.0});
    checkVec(sideSlicePoints(4, 0), {0.0, 0.0, 32.0});
    checkVec(sideSlicePoints(6, 0), {0.0, 0.0, 96.0});
    CHECK(views.lineSurface->scale()[0] == doctest::Approx(1.0 / 32.0));
    CHECK(views.lineSurface->scale()[1] == doctest::Approx(1.0 / 32.0));
    CHECK(views.lineSideSlice->scale()[0] == doctest::Approx(1.0 / 32.0));
    CHECK(views.lineSideSlice->scale()[1] == doctest::Approx(1.0 / 32.0));
    CHECK(views.lineSurface->strictQuadRenderValidity());
    CHECK(views.lineSideSlice->strictQuadRenderValidity());
    const cv::Vec2d surfaceFirst = views.lineSurface->gridToSurface({0.0, 0.0});
    const cv::Vec2d surfaceLast = views.lineSurface->gridToSurface({0.0, 6.0});
    const cv::Vec2d sideFirst = views.lineSideSlice->gridToSurface({0.0, 0.0});
    const cv::Vec2d sideLast = views.lineSideSlice->gridToSurface({0.0, 6.0});
    CHECK(surfaceLast[1] - surfaceFirst[1] == doctest::Approx(192.0));
    CHECK(sideLast[1] - sideFirst[1] == doctest::Approx(192.0));
}

TEST_CASE("LineViewBuilder offsets line-surface along side and side-slice along normal")
{
    vc::lasagna::LineViewConfig config;
    config.targetSpacingBaseVoxels = 10.0;

    const auto views = vc::lasagna::buildLineViewSurfaces(simpleLine(), config);
    const auto surfacePoints = views.lineSurface->rawPoints();
    const auto sideSlicePoints = views.lineSideSlice->rawPoints();

    checkVec(surfacePoints(2, 1), {10.0, -32.0, 0.0});
    checkVec(surfacePoints(3, 1), {10.0, 0.0, 0.0});
    checkVec(surfacePoints(4, 1), {10.0, 32.0, 0.0});

    checkVec(sideSlicePoints(2, 1), {10.0, 0.0, -32.0});
    checkVec(sideSlicePoints(3, 1), {10.0, 0.0, 0.0});
    checkVec(sideSlicePoints(4, 1), {10.0, 0.0, 32.0});
}

TEST_CASE("LineViewBuilder uses fitted mesh normal for side-slice")
{
    vc::lasagna::LineViewConfig config;
    config.targetSpacingBaseVoxels = 10.0;

    const auto views = vc::lasagna::buildLineViewSurfaces(simpleLine({1.0, 0.0, 1.0}),
                                                          config);
    const auto sideSlicePoints = views.lineSideSlice->rawPoints();

    checkVec(sideSlicePoints(2, 1), {10.0, 0.0, -32.0});
    checkVec(sideSlicePoints(3, 1), {10.0, 0.0, 0.0});
    checkVec(sideSlicePoints(4, 1), {10.0, 0.0, 32.0});

    const cv::Vec3f offset = sideSlicePoints(4, 1) - sideSlicePoints(3, 1);
    CHECK(offset[0] == doctest::Approx(0.0));
}

TEST_CASE("LineViewBuilder creates one z slice per optimized line point")
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

TEST_CASE("LineViewBuilder flips frames and ups to agree with oriented point normals")
{
    vc::lasagna::LineViewConfig config;
    config.targetSpacingBaseVoxels = 10.0;

    const auto legacy = vc::lasagna::buildLineViewSurfaces(simpleLine(), config);

    SUBCASE("hints opposite to the legacy sign flip both strips and the ups")
    {
        config.orientedPointNormals = {{0.0f, 0.0f, -1.0f},
                                       {0.0f, 0.0f, -1.0f},
                                       {0.0f, 0.0f, -1.0f}};
        const auto views = vc::lasagna::buildLineViewSurfaces(simpleLine(), config);
        const auto surfacePoints = views.lineSurface->rawPoints();
        const auto sideSlicePoints = views.lineSideSlice->rawPoints();

        checkVec(sideSlicePoints(2, 1), {10.0, 0.0, 32.0});
        checkVec(sideSlicePoints(4, 1), {10.0, 0.0, -32.0});
        checkVec(surfacePoints(2, 1), {10.0, 32.0, 0.0});
        checkVec(surfacePoints(4, 1), {10.0, -32.0, 0.0});

        REQUIRE(views.lineUpVectors.size() == 3);
        for (const auto& up : views.lineUpVectors) {
            checkVec(up, {0.0, 0.0, -1.0});
        }
        REQUIRE(views.lineZSlices.size() == 3);
        for (const auto& slice : views.lineZSlices) {
            REQUIRE(slice);
            checkVec(slice->basisY(), {0.0, 0.0, -1.0});
        }
    }

    SUBCASE("hints agreeing with the legacy sign change nothing")
    {
        config.orientedPointNormals = {{0.0f, 0.0f, 1.0f},
                                       {0.0f, 0.0f, 1.0f},
                                       {0.0f, 0.0f, 1.0f}};
        const auto views = vc::lasagna::buildLineViewSurfaces(simpleLine(), config);
        checkVec(views.lineSideSlice->rawPoints()(0, 1), legacy.lineSideSlice->rawPoints()(0, 1));
        checkVec(views.lineSurface->rawPoints()(0, 1), legacy.lineSurface->rawPoints()(0, 1));
        for (size_t i = 0; i < views.lineUpVectors.size(); ++i) {
            checkVec(views.lineUpVectors[i], toVec3d(legacy.lineUpVectors[i]));
        }
    }

    SUBCASE("sparse invalid hints still decide the vote")
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        config.orientedPointNormals = {{nan, nan, nan},
                                       {0.0f, 0.0f, -1.0f},
                                       {nan, nan, nan}};
        const auto views = vc::lasagna::buildLineViewSurfaces(simpleLine(), config);
        checkVec(views.lineSideSlice->rawPoints()(2, 1), {10.0, 0.0, 32.0});
        for (const auto& up : views.lineUpVectors) {
            checkVec(up, {0.0, 0.0, -1.0});
        }
    }

    SUBCASE("all-invalid and wrong-length hints fall back to the legacy signs")
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        config.orientedPointNormals = {{nan, nan, nan}, {0.0f, 0.0f, 0.0f}, {nan, nan, nan}};
        auto views = vc::lasagna::buildLineViewSurfaces(simpleLine(), config);
        checkVec(views.lineSideSlice->rawPoints()(0, 1), legacy.lineSideSlice->rawPoints()(0, 1));
        checkVec(views.lineUpVectors[0], toVec3d(legacy.lineUpVectors[0]));

        config.orientedPointNormals = {{0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, -1.0f}};
        views = vc::lasagna::buildLineViewSurfaces(simpleLine(), config);
        checkVec(views.lineSideSlice->rawPoints()(0, 1), legacy.lineSideSlice->rawPoints()(0, 1));
        checkVec(views.lineUpVectors[0], toVec3d(legacy.lineUpVectors[0]));
    }
}

TEST_CASE("LineViewBuilder oriented point normals pin ups against a moving display anchor")
{
    // Alternating sampled normals: without hints the display anchor picks the
    // global up sign, so anchors 1 and 2 produce opposite ups.
    auto line = alternatingNormalLine();

    line.displayFrameAnchorIndex = 1;
    const auto anchorOne = vc::lasagna::buildLineViewSurfaces(line);
    line.displayFrameAnchorIndex = -1;
    const auto anchorFallback = vc::lasagna::buildLineViewSurfaces(line);

    REQUIRE(anchorOne.lineUpVectors.size() == 5);
    REQUIRE(anchorFallback.lineUpVectors.size() == 5);
    checkVec(anchorOne.lineUpVectors[0], {0.0, 0.0, -1.0});
    checkVec(anchorFallback.lineUpVectors[0], {0.0, 0.0, 1.0});

    vc::lasagna::LineViewConfig config;
    config.orientedPointNormals.assign(5, {0.0f, 0.0f, 1.0f});
    line.displayFrameAnchorIndex = 1;
    const auto pinnedOne = vc::lasagna::buildLineViewSurfaces(line, config);
    line.displayFrameAnchorIndex = -1;
    const auto pinnedFallback = vc::lasagna::buildLineViewSurfaces(line, config);

    REQUIRE(pinnedOne.lineUpVectors.size() == pinnedFallback.lineUpVectors.size());
    for (size_t i = 0; i < pinnedOne.lineUpVectors.size(); ++i) {
        checkVec(pinnedOne.lineUpVectors[i], toVec3d(pinnedFallback.lineUpVectors[i]));
        checkVec(pinnedOne.lineUpVectors[i], {0.0, 0.0, 1.0});
    }
}

TEST_CASE("LineViewBuilder diagnostics flag sampled normal axis jumps")
{
    auto line = simpleLine();
    line.points[0].sampledNormal = normal({0.0, 1.0, 0.0});
    line.points[1].sampledNormal = normal({0.0, 0.0, 1.0});
    line.points[2].sampledNormal = normal({0.0, 1.0, 0.0});

    const auto diagnostics = vc::lasagna::diagnoseLineViewFrames(line);

    CHECK(diagnostics.minSampledAxisContinuityDot < 0.5);
    CHECK(diagnostics.minDisplayUpContinuityDot > 0.99);
    CHECK(diagnostics.maxAbsDisplayUpRollDeltaRadians < 1.0e-9);
    REQUIRE(!diagnostics.issues.empty());
    CHECK(diagnostics.issues.front().reason == "sampled_normal_axis_jump");
}

TEST_CASE("LineViewBuilder uses transported up vectors for cross-slice orientation")
{
    auto line = simpleLine();
    line.points[0].sampledNormal = normal({0.0, 1.0, 0.0});
    line.points[1].sampledNormal = normal({0.0, 0.0, 1.0});
    line.points[2].sampledNormal = normal({0.0, 1.0, 0.0});

    const auto views = vc::lasagna::buildLineViewSurfaces(line);

    REQUIRE(views.lineUpVectors.size() == 3);
    for (const auto& up : views.lineUpVectors) {
        checkVec(up, {0.0, 0.0, 1.0});
    }
    for (const auto& slice : views.lineZSlices) {
        REQUIRE(slice);
        checkVec(slice->basisY(), {0.0, 0.0, 1.0});
    }
}

TEST_CASE("LineViewBuilder rejects invalid display-frame anchor normals")
{
    auto invalidAnchorLine = simpleLine();
    invalidAnchorLine.points[1].sampledNormal = normal({0.0, 0.0, 0.0}, false);
    CHECK_THROWS_AS(vc::lasagna::buildLineViewSurfaces(invalidAnchorLine), std::runtime_error);

    auto parallelAnchorLine = simpleLine();
    parallelAnchorLine.points[1].sampledNormal = normal({1.0, 0.0, 0.0});
    CHECK_THROWS_AS(vc::lasagna::buildLineViewSurfaces(parallelAnchorLine), std::runtime_error);
}

TEST_CASE("LineViewBuilder uses finite deterministic fallback frames")
{
    auto line = simpleLine();
    line.points[0].sampledNormal = normal({0.0, 0.0, 0.0}, false);
    line.points[2].sampledNormal = normal({0.0, 0.0, 0.0}, false);

    vc::lasagna::LineViewConfig config;
    config.targetSpacingBaseVoxels = 10.0;
    const auto views = vc::lasagna::buildLineViewSurfaces(line, config);
    const auto points = views.lineSurface->rawPoints();

    REQUIRE(points.rows == 7);
    REQUIRE(points.cols == 3);
    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col < points.cols; ++col) {
            CHECK(finitePoint(points(row, col)));
        }
    }
    checkVec(points(3, 1), {10.0, 0.0, 0.0});
}

TEST_CASE("LineViewBuilder rejects empty models")
{
    vc::lasagna::LineModel empty;
    CHECK_THROWS_AS(vc::lasagna::buildLineViewSurfaces(empty), std::invalid_argument);
}

TEST_CASE("LineViewBuilder ignores dense segment samples for generated mesh rows")
{
    auto line = simpleLine();
    line.segmentSamples[0].samples[1].position = {500.0, 0.0, 0.0};

    const auto views = vc::lasagna::buildLineViewSurfaces(line, tenVoxelRibbonConfig());
    REQUIRE(views.lineSurface);
    const auto points = views.lineSurface->rawPoints();

    REQUIRE(points.rows == 7);
    REQUIRE(points.cols == 3);
    checkVec(points(3, 0), {0.0, 0.0, 0.0});
    checkVec(points(3, 1), {10.0, 0.0, 0.0});
    checkVec(points(3, 2), {20.0, 0.0, 0.0});
}

TEST_CASE("LineViewBuilder uses control point count even with duplicated segment-boundary samples")
{
    auto line = simpleLine();
    line.segmentSamples[0].samples.push_back({1.0, {10.0, 0.0, 0.0}, normal({0.0, 0.0, 1.0})});
    line.segmentSamples[1].samples.insert(
        line.segmentSamples[1].samples.begin(),
        {0.0, {10.0, 0.0, 0.0}, normal({0.0, 0.0, 1.0})});

    const auto views = vc::lasagna::buildLineViewSurfaces(line, tenVoxelRibbonConfig());
    const auto points = views.lineSurface->rawPoints();

    REQUIRE(points.cols == 3);
    checkVec(points(3, 1), {10.0, 0.0, 0.0});
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

    CHECK_THROWS_AS(vc::lasagna::buildLineViewSurfaces(invalidNormalLine), std::runtime_error);

    auto degenerateTangentLine = simpleLine();
    for (auto& point : degenerateTangentLine.points) {
        point.position = {5.0, 5.0, 5.0};
    }
    CHECK_THROWS_AS(vc::lasagna::buildLineViewSurfaces(degenerateTangentLine),
                    std::invalid_argument);
}

TEST_CASE("LineViewBuilder resamples uneven control-point segments and maps positions")
{
    auto line = simpleLine();
    line.points[1].position = {15.0, 0.0, 0.0};
    line.points[2].position = {120.0, 0.0, 0.0};
    vc::lasagna::LineViewConfig config;
    config.targetSpacingBaseVoxels = 50.0;
    config.controlPointLinePositions = {0.0, 1.0, 2.0};

    const auto views = vc::lasagna::buildLineViewSurfaces(line, config);
    const auto points = views.lineSurface->rawPoints();
    REQUIRE(points.rows == 7);
    REQUIRE(points.cols == 4);
    CHECK(views.stripPositionMap.stripGridSpacingBaseVoxels == doctest::Approx(50.0));
    CHECK(views.lineSurface->scale()[0] == doctest::Approx(1.0 / 50.0));
    CHECK(views.lineSurface->scale()[1] == doctest::Approx(1.0 / 32.0));
    checkVec(points(3, 0), {0.0, 0.0, 0.0});
    checkVec(points(3, 1), {15.0, 0.0, 0.0});
    checkVec(points(3, 2), {67.5, 0.0, 0.0});
    checkVec(points(3, 3), {120.0, 0.0, 0.0});

    CHECK(views.stripPositionMap.originalPositionToStripGridColumn(1.0) ==
          doctest::Approx(1.0));
    CHECK(views.stripPositionMap.stripGridColumnToOriginalPosition(1.0) ==
          doctest::Approx(1.0));
    const double fractional = 1.5;
    CHECK(views.stripPositionMap.stripGridColumnToOriginalPosition(
              views.stripPositionMap.originalPositionToStripGridColumn(fractional)) ==
          doctest::Approx(fractional));
}

TEST_CASE("LineViewBuilder resamples by arc length only between annotation controls")
{
    vc::lasagna::LineModel line;
    for (const double x : {0.0, 4.0, 36.0, 116.0}) {
        line.points.push_back({{x, 0.0, 0.0}, normal({0.0, 0.0, 1.0}), true});
    }

    vc::lasagna::LineViewConfig config;
    config.controlPointLinePositions = {0.0, 1.0, 3.0};
    const auto views = vc::lasagna::buildLineViewSurfaces(line, config);
    const auto points = views.lineSurface->rawPoints();
    REQUIRE(points.rows == 7);
    REQUIRE(points.cols == 6);
    CHECK(views.stripPositionMap.stripGridSpacingBaseVoxels == doctest::Approx(32.0));
    CHECK(views.lineSurface->scale()[0] == doctest::Approx(1.0 / 32.0));
    checkVec(points(points.rows / 2, 0), {0.0, 0.0, 0.0});
    checkVec(points(points.rows / 2, 1), {4.0, 0.0, 0.0});
    checkVec(points(points.rows / 2, 2), {32.0, 0.0, 0.0});
    checkVec(points(points.rows / 2, 3), {60.0, 0.0, 0.0});
    checkVec(points(points.rows / 2, 4), {88.0, 0.0, 0.0});
    checkVec(points(points.rows / 2, 5), {116.0, 0.0, 0.0});

    CHECK(views.stripPositionMap.originalPositionToStripGridColumn(1.0) ==
          doctest::Approx(1.0));
    CHECK(views.stripPositionMap.originalPositionToStripGridColumn(2.0) ==
          doctest::Approx(2.0 + 4.0 / 28.0));
    for (const double position : {0.5, 1.5, 2.5}) {
        CHECK(views.stripPositionMap.stripGridColumnToOriginalPosition(
                  views.stripPositionMap.originalPositionToStripGridColumn(position)) ==
              doctest::Approx(position));
    }
}

TEST_CASE("LineViewBuilder follows optimized bends between control supports")
{
    vc::lasagna::LineModel line;
    for (const cv::Vec3d& point : std::vector<cv::Vec3d>{
             {0.0, 0.0, 0.0},
             {30.0, 0.0, 0.0},
             {30.0, 40.0, 0.0},
             {90.0, 40.0, 0.0}}) {
        line.points.push_back({point, normal({0.0, 0.0, 1.0}), true});
    }

    vc::lasagna::LineViewConfig config;
    config.controlPointLinePositions = {0.0, 2.0, 3.0};
    const auto views = vc::lasagna::buildLineViewSurfaces(line, config);
    const auto points = views.lineSurface->rawPoints();

    REQUIRE(points.cols == 5);
    const int center = points.rows / 2;
    checkVec(points(center, 0), {0.0, 0.0, 0.0});
    checkVec(points(center, 1), {30.0, 5.0, 0.0});
    checkVec(points(center, 2), {30.0, 40.0, 0.0});
    checkVec(points(center, 3), {60.0, 40.0, 0.0});
    checkVec(points(center, 4), {90.0, 40.0, 0.0});
    CHECK(views.stripPositionMap.originalPositionToStripGridColumn(2.0) ==
          doctest::Approx(2.0));
}

TEST_CASE("LineViewBuilder keeps fractional control positions as exact supports")
{
    vc::lasagna::LineModel line;
    for (const double x : {0.0, 20.0, 80.0}) {
        line.points.push_back({{x, 0.0, 0.0}, normal({0.0, 0.0, 1.0}), true});
    }

    vc::lasagna::LineViewConfig config;
    config.controlPointLinePositions = {0.5, 2.0};
    const auto views = vc::lasagna::buildLineViewSurfaces(line, config);
    const auto points = views.lineSurface->rawPoints();

    REQUIRE(points.cols == 4);
    const int center = points.rows / 2;
    checkVec(points(center, 0), {0.0, 0.0, 0.0});
    checkVec(points(center, 1), {10.0, 0.0, 0.0});
    checkVec(points(center, 2), {45.0, 0.0, 0.0});
    checkVec(points(center, 3), {80.0, 0.0, 0.0});
    CHECK(views.stripPositionMap.originalPositionToStripGridColumn(0.5) ==
          doctest::Approx(1.0));
}

TEST_CASE("LineViewBuilder preserves reversed uneven arclength orientation")
{
    auto line = simpleLine();
    line.points[0].position = {120.0, 0.0, 0.0};
    line.points[1].position = {15.0, 0.0, 0.0};
    line.points[2].position = {0.0, 0.0, 0.0};
    vc::lasagna::LineViewConfig config;
    config.targetSpacingBaseVoxels = 50.0;
    config.controlPointLinePositions = {0.0, 1.0, 2.0};

    const auto views = vc::lasagna::buildLineViewSurfaces(line, config);
    const auto points = views.lineSurface->rawPoints();
    REQUIRE(points.cols == 4);
    checkVec(points(points.rows / 2, 0), {120.0, 0.0, 0.0});
    checkVec(points(points.rows / 2, 1), {67.5, 0.0, 0.0});
    checkVec(points(points.rows / 2, 2), {15.0, 0.0, 0.0});
    checkVec(points(points.rows / 2, 3), {0.0, 0.0, 0.0});
    CHECK(views.stripPositionMap.originalPositionToStripGridColumn(1.0) ==
          doctest::Approx(2.0));
    CHECK(views.stripPositionMap.stripGridColumnToOriginalPosition(2.0) ==
          doctest::Approx(1.0));
}

TEST_CASE("LineViewBuilder preserves a control-point bend below target spacing")
{
    auto line = simpleLine();
    line.points[0].position = {0.0, 0.0, 0.0};
    line.points[1].position = {20.0, 0.0, 0.0};
    line.points[2].position = {20.0, 20.0, 0.0};
    vc::lasagna::LineViewConfig config;
    config.targetSpacingBaseVoxels = 50.0;
    config.controlPointLinePositions = {0.0, 1.0, 2.0};

    const auto views = vc::lasagna::buildLineViewSurfaces(line, config);
    const auto points = views.lineSurface->rawPoints();
    REQUIRE(points.cols == 3);
    checkVec(points(points.rows / 2, 0), {0.0, 0.0, 0.0});
    checkVec(points(points.rows / 2, 1), {20.0, 0.0, 0.0});
    checkVec(points(points.rows / 2, 2), {20.0, 20.0, 0.0});
    CHECK(views.stripPositionMap.originalPositionToStripGridColumn(1.0) ==
          doctest::Approx(1.0));
}

TEST_CASE("LineViewBuilder chooses closest spacing independently per control span")
{
    auto line = simpleLine();
    line.points[0].position = {0.0, 0.0, 0.0};
    line.points[1].position = {74.0, 0.0, 0.0};
    line.points[2].position = {174.0, 0.0, 0.0};
    vc::lasagna::LineViewConfig config;
    config.targetSpacingBaseVoxels = 50.0;
    config.controlPointLinePositions = {0.0, 1.0, 2.0};

    const auto views = vc::lasagna::buildLineViewSurfaces(line, config);
    const auto points = views.lineSurface->rawPoints();
    REQUIRE(points.cols == 5);
    checkVec(points(points.rows / 2, 0), {0.0, 0.0, 0.0});
    checkVec(points(points.rows / 2, 1), {37.0, 0.0, 0.0});
    checkVec(points(points.rows / 2, 2), {74.0, 0.0, 0.0});
    checkVec(points(points.rows / 2, 3), {124.0, 0.0, 0.0});
    checkVec(points(points.rows / 2, 4), {174.0, 0.0, 0.0});
    CHECK(views.stripPositionMap.originalPositionToStripGridColumn(1.0) ==
          doctest::Approx(2.0));
    CHECK(views.stripPositionMap.stripGridColumnToOriginalPosition(2.0) ==
          doctest::Approx(1.0));
}

TEST_CASE("LineViewBuilder mapping canonicalizes duplicate control points")
{
    auto line = simpleLine();
    line.points.insert(line.points.begin() + 1, line.points.front());
    const auto views = vc::lasagna::buildLineViewSurfaces(line, tenVoxelRibbonConfig());
    REQUIRE(views.stripPositionMap.valid());
    CHECK(views.stripPositionMap.originalPositionToStripGridColumn(0.0) == doctest::Approx(0.0));
    CHECK(views.stripPositionMap.originalPositionToStripGridColumn(1.0) == doctest::Approx(0.0));
    CHECK(views.stripPositionMap.stripGridColumnToOriginalPosition(0.0) == doctest::Approx(0.0));
    CHECK(views.stripPositionMap.stripGridColumnToOriginalPosition(1.0) == doctest::Approx(2.0));
}

TEST_CASE("LineViewBuilder validates target spacing")
{
    vc::lasagna::LineViewConfig config;
    config.targetSpacingBaseVoxels = 0.0;
    CHECK_THROWS_AS(vc::lasagna::buildLineViewSurfaces(simpleLine(), config),
                    std::invalid_argument);
}
