// vcr::grid::Surface interpolates a QuadSurface's vertex grid as QuadSurface::gen
// does: the same positions, normals and missing pixels, in both interpolation
// modes, with holes, strict quad validity and separate pieces. One deliberate
// difference: gen() blended a missing vertex's (-1, -1, -1) into the positions of
// its cells; vcr::grid blends only present vertices, so those cells are checked
// for that instead.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include <vc_reader/grid.hpp>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <random>

namespace {

cv::Mat_<cv::Vec3f> wavyGrid(int rows, int cols)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = cv::Vec3f(20.0f * c, 20.0f * r,
                                100.0f + 30.0f * std::sin(0.35f * c) * std::cos(0.27f * r));
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> row(0, rows - 1), col(0, cols - 1);
    for (int i = 0; i < 12; ++i)
        m(row(rng), col(rng)) = cv::Vec3f(-1, -1, -1);
    return m;
}

// gen()'s view at framebuffer offset `offset` and scale, as a raster over the grid.
vcr::grid::Raster rasterFor(const QuadSurface& s, cv::Size size, float scale, cv::Vec3f offset)
{
    const cv::Vec3f center = s.center();
    const cv::Vec2f gs = s.scale();
    vcr::grid::Raster r;
    r.x0 = (offset[0] / scale + center[0]) * gs[0];
    r.y0 = (offset[1] / scale + center[1]) * gs[1];
    r.sx = double(gs[0]) / double(scale);
    r.sy = double(gs[1]) / double(scale);
    r.width = size.width;
    r.height = size.height;
    return r;
}

void compare(QuadSurface& quad, const vcr::grid::SurfaceOptions& options)
{
    const auto& points = *quad.rawPointsPtr();
    vcr::grid::Surface surface({reinterpret_cast<const float*>(points.ptr()), points.rows, points.cols,
                                points.step1()},
                               options);
    for (const float scale : {0.5f, 1.3f, 4.0f}) {
        const cv::Size size(97, 61);
        const cv::Vec3f offset(-40.0f * scale, -25.0f * scale, 0.0f);
        cv::Mat_<cv::Vec3f> coords, normals;
        quad.gen(&coords, &normals, size, {0, 0, 0}, scale, offset);
        const auto raster = rasterFor(quad, size, scale, offset);
        std::vector<float> xyz(std::size_t(size.area()) * 3), nrm(xyz.size());
        surface.resample(raster, xyz.data(), nrm.data());
        int compared = 0, mismatched = 0, besideHoles = 0;
        // Whether the cell under a pixel (clamped like gen's bilinear lookup) has a missing vertex.
        const auto missing = [&](const cv::Vec3f& p) { return p[0] == -1.f && p[1] == -1.f && p[2] == -1.f; };
        const auto besideHole = [&](int x, int y) {
            const double gx = std::clamp(raster.x0 + x * raster.sx, 0.0, double(points.cols - 1));
            const double gy = std::clamp(raster.y0 + y * raster.sy, 0.0, double(points.rows - 1));
            const int x0 = int(gx), y0 = int(gy);
            const int x1 = std::min(x0 + 1, points.cols - 1), y1 = std::min(y0 + 1, points.rows - 1);
            return missing(points(y0, x0)) || missing(points(y0, x1)) || missing(points(y1, x0)) ||
                   missing(points(y1, x1));
        };
        for (int y = 0; y < size.height; ++y)
            for (int x = 0; x < size.width; ++x) {
                const cv::Vec3f g = coords(y, x), gn = normals(y, x);
                const float* p = &xyz[(std::size_t(y) * size.width + x) * 3];
                const float* n = &nrm[(std::size_t(y) * size.width + x) * 3];
                const bool gValid = std::isfinite(g[0]);
                if (gValid != std::isfinite(p[0])) {
                    ++mismatched;
                    continue;
                }
                if (!gValid)
                    continue;
                if (besideHole(x, y)) {
                    // Only present vertices (x, y >= 0, z >= 70 on this sheet) enter the position.
                    ++besideHoles;
                    if (!(p[0] >= 0.0f && p[1] >= 0.0f && p[2] >= 70.0f))
                        ++mismatched;
                    continue;
                }
                ++compared;
                for (int a = 0; a < 3; ++a) {
                    if (std::abs(g[a] - p[a]) > 1e-2f)
                        ++mismatched;
                    if (std::isfinite(gn[a]) != std::isfinite(n[a]) ||
                        (std::isfinite(gn[a]) && std::abs(gn[a] - n[a]) > 1e-4f))
                        ++mismatched;
                }
            }
        CAPTURE(scale);
        CHECK(compared > 1000);
        if (scale < 4.0f && !options.strictQuads)  // strict quads drop every cell with a hole
            CHECK(besideHoles > 0);
        CHECK(mismatched == 0);
    }
}

}  // namespace

TEST_CASE("vcr::grid::Surface matches QuadSurface::gen")
{
    for (const auto mode : {GenInterpolation::Linear, GenInterpolation::Smooth})
        for (const bool strict : {false, true})
            for (const bool pieces : {false, true}) {
                CAPTURE(int(mode));
                CAPTURE(strict);
                CAPTURE(pieces);
                auto grid = wavyGrid(40, 30);
                {
                    // A hole inside the views at scales 0.5 and 1.3 (they share their
                    // top-left corner with the small 4.0 view, which it stays out of).
                    QuadSurface probe(grid, cv::Vec2f(0.05f, 0.05f));
                    const auto r = rasterFor(probe, {97, 61}, 4.0f, {-160.0f, -100.0f, 0.0f});
                    grid(int(r.y0 + 1.5), int(r.x0 + 2.5)) = cv::Vec3f(-1, -1, -1);
                }
                QuadSurface quad(grid, cv::Vec2f(0.05f, 0.05f));
                quad.setGenInterpolation(mode);
                quad.setStrictQuadRenderValidity(strict);
                vcr::grid::SurfaceOptions options;
                options.interpolation = mode == GenInterpolation::Smooth ? vcr::grid::Interpolation::Smooth
                                                                         : vcr::grid::Interpolation::Linear;
                options.strictQuads = strict;
                if (pieces) {
                    quad.setComponents({{0, 13}, {13, 30}});
                    options.components = {{0, 13}, {13, 30}};
                }
                compare(quad, options);
            }
}
