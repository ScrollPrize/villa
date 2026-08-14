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
    return config;
}

} // namespace

TEST_CASE("LineViewBuilder creates ribbons from optimized control points")
{
    const auto views = vc::lasagna::buildLineViewSurfaces(simpleLine(), tenVoxelRibbonConfig());

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

TEST_CASE("LineViewBuilder default strip spacing is a 50 base voxel target")
{
    const auto views = vc::lasagna::buildLineViewSurfaces(simpleLine());
    const auto surfacePoints = views.lineSurface->rawPoints();

    REQUIRE(surfacePoints.rows == 21);
    REQUIRE(surfacePoints.cols == 2);

    checkVec(surfacePoints(9, 0), {0.0, -50.0, 0.0});
    checkVec(surfacePoints(10, 0), {0.0, 0.0, 0.0});
    checkVec(surfacePoints(11, 0), {0.0, 50.0, 0.0});
    CHECK(views.lineSurface->scale()[0] == doctest::Approx(1.0 / 20.0));
    CHECK(views.lineSurface->scale()[1] == doctest::Approx(1.0 / 50.0));
}

TEST_CASE("LineViewBuilder offsets line-surface along side and side-slice along normal")
{
    vc::lasagna::LineViewConfig config;
    config.targetSpacingBaseVoxels = 10.0;
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
    config.targetSpacingBaseVoxels = 10.0;
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

TEST_CASE("LineViewBuilder flips frames and ups to agree with oriented point normals")
{
    vc::lasagna::LineViewConfig config;
    config.targetSpacingBaseVoxels = 10.0;
    config.surfaceHalfWidth = 2.0;
    config.sideSliceHalfDepth = 3.0;
    config.crossSamples = 3;

    const auto legacy = vc::lasagna::buildLineViewSurfaces(simpleLine(), config);

    SUBCASE("hints opposite to the legacy sign flip both strips and the ups")
    {
        config.orientedPointNormals = {{0.0f, 0.0f, -1.0f},
                                       {0.0f, 0.0f, -1.0f},
                                       {0.0f, 0.0f, -1.0f}};
        const auto views = vc::lasagna::buildLineViewSurfaces(simpleLine(), config);
        const auto surfacePoints = views.lineSurface->rawPoints();
        const auto sideSlicePoints = views.lineSideSlice->rawPoints();

        checkVec(sideSlicePoints(0, 1), {10.0, 0.0, 3.0});
        checkVec(sideSlicePoints(2, 1), {10.0, 0.0, -3.0});
        checkVec(surfacePoints(0, 1), {10.0, 2.0, 0.0});
        checkVec(surfacePoints(2, 1), {10.0, -2.0, 0.0});

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
        checkVec(views.lineSideSlice->rawPoints()(0, 1), {10.0, 0.0, 3.0});
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

    const auto views = vc::lasagna::buildLineViewSurfaces(line, tenVoxelRibbonConfig());
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

    const auto views = vc::lasagna::buildLineViewSurfaces(line, tenVoxelRibbonConfig());
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
    CHECK_THROWS_AS(vc::lasagna::buildLineViewSurfaces(invalidNormalLine, config), std::runtime_error);

    auto degenerateTangentLine = simpleLine();
    for (auto& point : degenerateTangentLine.points) {
        point.position = {5.0, 5.0, 5.0};
    }
    CHECK_THROWS_AS(vc::lasagna::buildLineViewSurfaces(degenerateTangentLine, config),
                    std::invalid_argument);
}

TEST_CASE("LineViewBuilder uniformly resamples uneven control points and maps positions")
{
    auto line = simpleLine();
    line.points[1].position = {15.0, 0.0, 0.0};
    line.points[2].position = {120.0, 0.0, 0.0};
    vc::lasagna::LineViewConfig config;
    config.targetSpacingBaseVoxels = 50.0;
    config.surfaceHalfWidth = 20.0;
    config.sideSliceHalfDepth = 10.0;
    config.crossSamples = 5;

    const auto views = vc::lasagna::buildLineViewSurfaces(line, config);
    const auto points = views.lineSurface->rawPoints();
    REQUIRE(points.cols == 4);
    CHECK(views.stripPositionMap.stripGridSpacingBaseVoxels == doctest::Approx(40.0));
    CHECK(views.lineSurface->scale()[0] == doctest::Approx(1.0 / 40.0));
    CHECK(views.lineSurface->scale()[1] == doctest::Approx(1.0 / 10.0));
    checkVec(points(2, 0), {0.0, 0.0, 0.0});
    checkVec(points(2, 1), {40.0, 0.0, 0.0});
    checkVec(points(2, 2), {80.0, 0.0, 0.0});
    checkVec(points(2, 3), {120.0, 0.0, 0.0});

    CHECK(views.stripPositionMap.originalPositionToStripGridColumn(1.0) ==
          doctest::Approx(15.0 / 40.0));
    CHECK(views.stripPositionMap.stripGridColumnToOriginalPosition(15.0 / 40.0) ==
          doctest::Approx(1.0));
    const double fractional = 1.5;
    CHECK(views.stripPositionMap.stripGridColumnToOriginalPosition(
              views.stripPositionMap.originalPositionToStripGridColumn(fractional)) ==
          doctest::Approx(fractional));
}

TEST_CASE("LineViewBuilder preserves reversed uneven arclength orientation")
{
    auto line = simpleLine();
    line.points[0].position = {120.0, 0.0, 0.0};
    line.points[1].position = {15.0, 0.0, 0.0};
    line.points[2].position = {0.0, 0.0, 0.0};
    vc::lasagna::LineViewConfig config;
    config.targetSpacingBaseVoxels = 50.0;

    const auto views = vc::lasagna::buildLineViewSurfaces(line, config);
    const auto points = views.lineSurface->rawPoints();
    REQUIRE(points.cols == 4);
    checkVec(points(points.rows / 2, 0), {120.0, 0.0, 0.0});
    checkVec(points(points.rows / 2, 1), {80.0, 0.0, 0.0});
    checkVec(points(points.rows / 2, 2), {40.0, 0.0, 0.0});
    checkVec(points(points.rows / 2, 3), {0.0, 0.0, 0.0});
    CHECK(views.stripPositionMap.originalPositionToStripGridColumn(1.0) ==
          doctest::Approx(105.0 / 40.0));
    CHECK(views.stripPositionMap.stripGridColumnToOriginalPosition(105.0 / 40.0) ==
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

TEST_CASE("LineViewBuilder validates target spacing and odd cross sample count")
{
    vc::lasagna::LineViewConfig config;
    config.targetSpacingBaseVoxels = 0.0;
    CHECK_THROWS_AS(vc::lasagna::buildLineViewSurfaces(simpleLine(), config),
                    std::invalid_argument);
    config.targetSpacingBaseVoxels = 50.0;
    config.crossSamples = 4;
    CHECK_THROWS_AS(vc::lasagna::buildLineViewSurfaces(simpleLine(), config),
                    std::invalid_argument);
}
