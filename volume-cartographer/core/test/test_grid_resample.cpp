// vcr::grid::Surface interpolates a QuadSurface's vertex grid as QuadSurface::gen
// does: the same positions, normals and missing pixels, in both interpolation
// modes, with holes, strict quad validity and separate pieces.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include <vc_reader/grid.hpp>

#include <opencv2/core.hpp>

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
        int compared = 0, mismatched = 0;
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
                QuadSurface quad(wavyGrid(40, 30), cv::Vec2f(0.05f, 0.05f));
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
