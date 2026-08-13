// Coverage for core/src/Render.cpp.
// render_binary_mask is Volume-free and easy to exercise on an in-memory
// QuadSurface. render_image_from_coords requires a real Volume, so we only
// hit the null-volume and empty-coords guards.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <stdexcept>

// Forward declare to avoid pulling Volume.hpp's heavy graph.
void render_binary_mask(QuadSurface* surf,
                        cv::Mat_<uint8_t>& mask,
                        cv::Mat_<cv::Vec3f>& coords_out,
                        float scale);
void render_image_from_coords(const cv::Mat_<cv::Vec3f>& coords,
                              cv::Mat_<uint8_t>& img,
                              class Volume* volume,
                              int level);

namespace {

cv::Mat_<cv::Vec3f> makePlanarGrid(int rows, int cols, float z = 0.f)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = cv::Vec3f(static_cast<float>(c), static_cast<float>(r), z);
    return m;
}

cv::Mat_<cv::Vec3f> makeSparseGrid(int rows, int cols, int patchH, int patchW)
{
    cv::Mat_<cv::Vec3f> m(rows, cols, cv::Vec3f(-1.f, -1.f, -1.f));
    const int r0 = (rows - patchH) / 2;
    const int c0 = (cols - patchW) / 2;
    for (int r = r0; r < r0 + patchH; ++r)
        for (int c = c0; c < c0 + patchW; ++c)
            m(r, c) = cv::Vec3f(c, r, 0);
    return m;
}

} // namespace

TEST_CASE("render_binary_mask at scale=1 matches raw grid dimensions")
{
    auto pts = makePlanarGrid(16, 24);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    cv::Mat_<uint8_t> mask;
    cv::Mat_<cv::Vec3f> coords;
    render_binary_mask(&qs, mask, coords, 1.0f);
    CHECK(mask.rows == 16);
    CHECK(mask.cols == 24);
    CHECK(coords.rows == 16);
    CHECK(coords.cols == 24);
    // Dense grid: every pixel valid.
    CHECK(cv::countNonZero(mask) == 16 * 24);
}

TEST_CASE("render_binary_mask upscale doubles target size")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    cv::Mat_<uint8_t> mask;
    cv::Mat_<cv::Vec3f> coords;
    render_binary_mask(&qs, mask, coords, 2.0f);
    CHECK(mask.rows == 16);
    CHECK(mask.cols == 16);
    CHECK(coords.rows == 16);
    CHECK(coords.cols == 16);
}

TEST_CASE("render_binary_mask preserves invalid regions on sparse grid")
{
    auto pts = makeSparseGrid(20, 20, 4, 4);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    cv::Mat_<uint8_t> mask;
    cv::Mat_<cv::Vec3f> coords;
    render_binary_mask(&qs, mask, coords, 1.0f);
    int valid = cv::countNonZero(mask);
    CHECK(valid == 16); // 4x4 patch
}

TEST_CASE("render_binary_mask downscale halves target size")
{
    auto pts = makePlanarGrid(16, 16);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    cv::Mat_<uint8_t> mask;
    cv::Mat_<cv::Vec3f> coords;
    render_binary_mask(&qs, mask, coords, 0.5f);
    CHECK(mask.rows == 8);
    CHECK(mask.cols == 8);
}

TEST_CASE("render_image_from_coords: null volume throws")
{
    cv::Mat_<cv::Vec3f> coords(4, 4, cv::Vec3f(0, 0, 0));
    cv::Mat_<uint8_t> img;
    CHECK_THROWS_AS(render_image_from_coords(coords, img, nullptr, 0),
                    std::runtime_error);
}

TEST_CASE("render_image_from_coords: empty coords releases img and returns")
{
    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<uint8_t> img(4, 4, uint8_t(99));
    // Pass a non-null volume pointer would be needed to test happy path,
    // but we can still exercise the empty-coords early return.
    // Use a non-null but garbage pointer — won't be dereferenced because
    // coords.empty() short-circuits.
    render_image_from_coords(coords, img, reinterpret_cast<Volume*>(0x1), 0);
    CHECK(img.empty());
}
