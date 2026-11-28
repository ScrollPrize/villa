#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/RenderCanvas.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <cstring>

// vc::RenderCanvas: the pixel grid vc_render_tifxyz writes. VC3D draws ink
// labels on this canvas, so its mapping must agree with the renderer exactly.

namespace {

// x = column, y = row.
cv::Mat_<cv::Vec3f> makeIdentityGrid(int rows, int cols)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = cv::Vec3f(float(c), float(r), 0.f);
    return m;
}

}  // namespace

TEST_CASE("renderCanvasSize matches vc_render_tifxyz's rounding")
{
    CHECK(vc::renderCanvasSize({152, 76}, {0.05f, 0.05f}, 1.0) == cv::Size(3040, 1520));
    CHECK(vc::renderCanvasSize({7, 5}, {2.0f, 2.0f}, 1.0) == cv::Size(4, 3));
    CHECK(vc::renderCanvasSize({1, 1}, {1000.f, 1000.f}, 1.0) == cv::Size(1, 1));
    CHECK(vc::renderCanvasSize({152, 76}, {0.05f, 0.05f}, 0.5) == cv::Size(1520, 760));
    CHECK(vc::renderCanvasSize({97, 97}, {0.3f, 0.3f}, 1.0) == cv::Size(323, 323));
    // Independent axes.
    CHECK(vc::renderCanvasSize({10, 20}, {0.5f, 0.25f}, 1.0) == cv::Size(20, 80));
}

TEST_CASE("renderCanvasOrigin is the renderer's centered origin, bit for bit")
{
    for (const cv::Size size : {cv::Size(1, 1), cv::Size(2, 3), cv::Size(3040, 1521), cv::Size(4097, 2048)}) {
        const cv::Vec2f o = vc::renderCanvasOrigin(size);
        const float u0 = -0.5f * (size.width - 1.0f);
        const float v0 = -0.5f * (size.height - 1.0f);
        CHECK(std::memcmp(&o[0], &u0, sizeof(float)) == 0);
        CHECK(std::memcmp(&o[1], &v0, sizeof(float)) == 0);
    }
}

TEST_CASE("RenderCanvas::gridToCanvas inverts canvasToGrid")
{
    const auto canvas = vc::RenderCanvas::make({152, 76}, {0.05f, 0.3f}, 1.0);
    for (const cv::Vec2d p : {cv::Vec2d(0, 0), cv::Vec2d(0.5, 0.5), cv::Vec2d(1234.25, 17.75), cv::Vec2d(3040, 253)}) {
        const cv::Vec2d back = canvas.gridToCanvas(canvas.canvasToGrid(p));
        CHECK(std::abs(back[0] - p[0]) <= 1e-9);
        CHECK(std::abs(back[1] - p[1]) <= 1e-9);
    }
    CHECK(canvas == vc::RenderCanvas::make({152, 76}, {0.05f, 0.3f}, 1.0));
    CHECK_FALSE(canvas == vc::RenderCanvas::make({152, 76}, {0.05f, 0.3f}, 0.5));
}

TEST_CASE("pixel centers map to the surface coordinates the renderer samples")
{
    // vc_render_tifxyz rasterizes output pixel (c, r) at surface coordinate
    // ((origin + (c, r)) / renderScale), with origin = renderCanvasOrigin(canvas).
    for (const float s : {0.05f, 0.25f, 2.0f}) {
        for (const double rs : {1.0, 0.5}) {
            QuadSurface surf(makeIdentityGrid(20, 30), cv::Vec2f(s, s));
            const auto canvas = vc::RenderCanvas::make({30, 20}, {s, s}, rs);
            const int W = canvas.size.width, H = canvas.size.height;
            const cv::Vec2f origin = vc::renderCanvasOrigin(canvas.size);
            for (const cv::Point px : {cv::Point(0, 0), cv::Point(W / 3, H / 2), cv::Point(W - 1, H - 1)}) {
                const cv::Vec2d grid = canvas.canvasToGrid({px.x + 0.5, px.y + 0.5});
                const cv::Vec2d surface = surf.gridToSurface(grid);
                INFO("scale " << s << " rs " << rs << " pixel " << px.x << "," << px.y);
                CHECK(std::abs(surface[0] - (double(origin[0]) + px.x) / rs) <= 1e-4);
                CHECK(std::abs(surface[1] - (double(origin[1]) + px.y) / rs) <= 1e-4);
            }
        }
    }
}

TEST_CASE("labels stay attached to grid positions when the surface origin shifts")
{
    const float s = 0.25f;
    QuadSurface surf(makeIdentityGrid(20, 30), cv::Vec2f(s, s));
    const auto canvas = vc::RenderCanvas::make({30, 20}, {s, s}, 1.0);
    const cv::Vec2d grid = canvas.canvasToGrid({11.5, 4.5});
    const cv::Vec2d before = surf.gridToSurface(grid);
    surf.shiftSurfaceOrigin({3.0, -2.0});
    const cv::Vec2d after = surf.gridToSurface(grid);
    CHECK(std::abs(after[0] - (before[0] - 3.0)) <= 1e-5);
    CHECK(std::abs(after[1] - (before[1] + 2.0)) <= 1e-5);
    const cv::Vec2d roundTrip = surf.surfaceToGrid(after);
    CHECK(std::abs(roundTrip[0] - grid[0]) <= 1e-5);
    CHECK(std::abs(roundTrip[1] - grid[1]) <= 1e-5);
}
