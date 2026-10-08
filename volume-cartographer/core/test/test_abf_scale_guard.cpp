#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/flattening/ABFFlattening.hpp"

#include <opencv2/core/mat.hpp>

namespace {

cv::Mat_<cv::Vec3f> makeGrid(int rows, int cols, float columnStep, float rowStep)
{
    cv::Mat_<cv::Vec3f> points(rows, cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            points(row, col) = cv::Vec3f(
                static_cast<float>(col) * columnStep,
                static_cast<float>(row) * rowStep,
                100.f);
        }
    }
    return points;
}

} // namespace

TEST_CASE("ABF scale guard accepts the documented cells-per-voxel convention")
{
    const auto points = makeGrid(12, 12, 20.f, 20.f);
    const auto check = vc::checkAbfInputScale(points, cv::Vec2f(0.05f, 0.05f));

    CHECK(check.checked);
    CHECK(check.consistent);
    CHECK(check.checkedColumns);
    CHECK(check.checkedRows);
    CHECK(check.columnMedianSpacingVoxels == doctest::Approx(20.f));
    CHECK(check.rowMedianSpacingVoxels == doctest::Approx(20.f));
    CHECK(check.columnScaleRatio == doctest::Approx(1.f));
    CHECK(check.rowScaleRatio == doctest::Approx(1.f));
}

TEST_CASE("ABF scale guard accepts anisotropic but self-consistent sampling")
{
    const auto points = makeGrid(12, 12, 16.6f, 8.0f);
    const auto check = vc::checkAbfInputScale(
        points, cv::Vec2f(1.f / 16.6f, 1.f / 8.0f));

    CHECK(check.checked);
    CHECK(check.consistent);
    CHECK(check.columnScaleRatio == doctest::Approx(1.f).epsilon(1e-5));
    CHECK(check.rowScaleRatio == doctest::Approx(1.f).epsilon(1e-5));
}

TEST_CASE("ABF scale guard rejects PHercParis4 outer-shell style reciprocal error")
{
    // villa#1379 measured about 21.26 voxels/cell horizontally and 20.49
    // vertically while meta.json stored ~20 cells/voxel instead of ~0.05.
    const auto points = makeGrid(12, 12, 21.26f, 20.49f);
    const auto check = vc::checkAbfInputScale(
        points, cv::Vec2f(19.997318f, 19.996687f));

    CHECK(check.checked);
    CHECK_FALSE(check.consistent);
    CHECK(check.columnScaleRatio > 400.f);
    CHECK(check.rowScaleRatio > 400.f);
    CHECK(check.failureReason.find("Refusing flattened output allocation") != std::string::npos);
    CHECK(check.failureReason.find("cells per voxel") != std::string::npos);
}

TEST_CASE("ABF scale guard also rejects a scale that is far too small")
{
    const auto points = makeGrid(12, 12, 16.6f, 8.0f);
    const auto check = vc::checkAbfInputScale(points, cv::Vec2f(0.0005f, 0.0005f));

    CHECK(check.checked);
    CHECK_FALSE(check.consistent);
    CHECK(check.columnScaleRatio < 0.25f);
    CHECK(check.rowScaleRatio < 0.25f);
}

TEST_CASE("ABF scale guard skips tiny grids without enough evidence")
{
    const auto points = makeGrid(2, 2, 20.f, 20.f);
    const auto check = vc::checkAbfInputScale(points, cv::Vec2f(20.f, 20.f));

    CHECK_FALSE(check.checked);
    CHECK(check.consistent);
    CHECK_FALSE(check.checkedColumns);
    CHECK_FALSE(check.checkedRows);
}

TEST_CASE("ABF scale guard rejects non-positive metadata")
{
    const auto points = makeGrid(12, 12, 20.f, 20.f);
    const auto check = vc::checkAbfInputScale(points, cv::Vec2f(0.f, 0.05f));

    CHECK(check.checked);
    CHECK_FALSE(check.consistent);
    CHECK(check.failureReason.find("finite positive") != std::string::npos);
}
