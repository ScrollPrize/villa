#pragma once

#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

#include <algorithm>
#include <cmath>

namespace vc {

// Output size of vc_render_tifxyz for `grid` vertices stored at `gridScale`,
// rendered at `renderScale` (1 = full resolution). Exactly the renderer's
// arithmetic, in double with lround and a minimum of 1. No guard: callers
// check that gridScale is finite and > 0.
[[nodiscard]] inline cv::Size renderCanvasSize(cv::Size grid, cv::Vec2f gridScale, double renderScale)
{
    const double sx = renderScale / gridScale[0], sy = renderScale / gridScale[1];
    return {std::max(1, int(std::lround(grid.width * sx))), std::max(1, int(std::lround(grid.height * sy)))};
}

// The gen() offset vc_render_tifxyz passes for canvas pixel (0,0): it centers
// the output on the surface.
[[nodiscard]] inline cv::Vec2f renderCanvasOrigin(cv::Size canvas)
{
    return {-0.5f * (canvas.width - 1.0f), -0.5f * (canvas.height - 1.0f)};
}

// The pixel grid vc_render_tifxyz writes for one surface, as a continuous
// mapping. Canvas pixel (c, r) covers [c, c+1) x [r, r+1); its center samples
// the same grid position as the renderer's output pixel (c, r).
struct RenderCanvas {
    cv::Size grid;
    cv::Vec2f gridScale{1.f, 1.f};
    double renderScale = 1.0;
    cv::Size size;

    [[nodiscard]] static RenderCanvas make(cv::Size grid, cv::Vec2f gridScale, double renderScale = 1.0)
    {
        return {grid, gridScale, renderScale, renderCanvasSize(grid, gridScale, renderScale)};
    }

    // Grid units per canvas pixel: gridScale / renderScale.
    [[nodiscard]] cv::Vec2d pitch() const
    {
        return {double(gridScale[0]) / renderScale, double(gridScale[1]) / renderScale};
    }

    // Continuous canvas position (pixel corner convention) to grid position.
    [[nodiscard]] cv::Vec2d canvasToGrid(cv::Vec2d p) const
    {
        const cv::Vec2d s = pitch();
        return {0.5 * grid.width + s[0] * (p[0] - 0.5 * size.width), 0.5 * grid.height + s[1] * (p[1] - 0.5 * size.height)};
    }

    // Grid position to continuous canvas position; the inverse of canvasToGrid.
    [[nodiscard]] cv::Vec2d gridToCanvas(cv::Vec2d g) const
    {
        const cv::Vec2d s = pitch();
        return {0.5 * size.width + (g[0] - 0.5 * grid.width) / s[0], 0.5 * size.height + (g[1] - 0.5 * grid.height) / s[1]};
    }

    bool operator==(const RenderCanvas&) const = default;
};

}  // namespace vc
