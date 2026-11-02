#include "ViewerSampling.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Compositing.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <exception>

namespace vc3d {

bool ViewerLevels::has(int level) const
{
    return level >= 0 && level < count() && levels[std::size_t(level)].array;
}

std::array<float, 3> ViewerLevels::vertexSpacing() const
{
    for (const auto& l : levels) {
        if (!l.array)
            continue;
        const auto& scale = l.working.scale;  // ZYX
        const double finest = std::min({scale[0], scale[1], scale[2]});
        return {float(scale[2] / finest), float(scale[1] / finest), float(scale[0] / finest)};
    }
    return {1, 1, 1};
}

vcr::grid::Level ViewerLevels::gridLevel(int level) const
{
    const auto& l = levels[std::size_t(level)];
    return vcr::zarr::gridLevel(*l.array, l.working, gpuFailed->load() ? nullptr : l.cuda.get());
}

std::vector<vcr::grid::Level> ViewerLevels::gridLevels() const
{
    std::vector<vcr::grid::Level> out(levels.size());
    for (int level = 0; level < count(); ++level)
        if (has(level))
            out[std::size_t(level)] = gridLevel(level);
    return out;
}

vcr::grid::Request ViewerLevels::request(int startLevel, int fallbackLevels) const
{
    vcr::grid::Request request;
    const int start = std::clamp(startLevel, 0, std::max(0, count() - 1));
    const int last = std::min(count() - 1, start + std::max(0, fallbackLevels));
    for (int level = start; level <= last; ++level)
        if (has(level))
            request.levels.push_back(gridLevel(level));
    return request;
}

ViewerLevels openViewerLevels(OpenedVolumes& opened, const Volume& volume)
{
    ViewerLevels result;
    const auto& pyramid = opened.pyramid(volume);
    result.levels.resize(std::size_t(pyramid.count()));
    for (const int level : pyramid.levels) {
        try {
            result.levels[std::size_t(level)] = opened.level(volume, level);
        } catch (const std::exception& error) {
            Logger()->warn("viewer: level {} of {} unavailable: {}", level, volume.id(), error.what());
        }
    }
    return result;
}

float surfacePixelsPerVoxel(const QuadSurface& surface, const vcr::grid::Raster& viewport,
                            float nominalPixelsPerVoxel)
{
    // QuadSurface::pointAt reads a window of the grid: no whole-grid snapshot is built.
    const auto xy = vcr::grid::magnificationPoints(viewport);
    float du[27], dv[27];
    for (std::size_t i = 0; i < 9; ++i)
        surface.pointAt(xy[i * 2], xy[i * 2 + 1], nullptr, nullptr, du + i * 3, dv + i * 3);
    const float magnification = vcr::grid::pixelsPerVoxel(du, dv, 9, viewport.sx, viewport.sy);
    return magnification > 0 ? magnification : nominalPixelsPerVoxel;
}

std::shared_ptr<const vcr::grid::Surface> gridSurface(const QuadSurface& surface, const std::array<float, 3>& spacing)
{
    const auto* points = surface.rawPointsPtr();
    if (!points || points->empty())
        return nullptr;
    if (spacing == std::array<float, 3>{1, 1, 1})
        return surface.pointSurface();
    auto options = surface.gridOptions();
    options.spacing = spacing;
    std::vector<std::uint8_t> valid;
    return vcr::grid::Surface::snapshot(surface.gridVertices(valid, points), std::move(options));
}

cv::Rect gridWindowFor(const QuadSurface& surface, const vcr::grid::Raster& raster, int left, int top, int right, int bottom)
{
    const auto* points = surface.rawPointsPtr();
    const cv::Rect grid(0, 0, points->cols, points->rows);
    const auto span = [](double origin, double step, int before, int size, int after, int limit) {
        const double a = origin - before * step, b = origin + (size + after) * step;
        const int first = int(std::floor(std::clamp(std::min(a, b), 0.0, double(limit))));
        const int last = int(std::ceil(std::clamp(std::max(a, b), 0.0, double(limit))));
        return std::pair{first - QuadSurface::kGridWindowPad - 1, last + QuadSurface::kGridWindowPad + 2};
    };
    const auto [x0, x1] = span(raster.x0, raster.sx, left, raster.width, right, points->cols);
    const auto [y0, y1] = span(raster.y0, raster.sy, top, raster.height, bottom, points->rows);
    return cv::Rect(x0, y0, x1 - x0, y1 - y0) & grid;
}

namespace {

void renderOnCpu(const ViewerLevels& levels, const std::string& failure)
{
    if (!failure.empty() && !levels.gpuFailed->exchange(true))
        Logger()->warn("GPU frame failed, these levels render on the CPU from now on: {}", failure);
}

} // namespace

vcr::grid::Result sampleCached(const std::shared_ptr<const ViewerLevels>& levels, const vcr::grid::Frame& frame,
                               const vcr::grid::Request& request, ViewerLoad* load)
{
    const std::size_t reads = load ? load->pending.coarse.size() + load->pending.target.size() : 0;
    auto result = vcr::grid::sampleCached(frame, request, load ? &load->pending : nullptr);
    renderOnCpu(*levels, result.cudaFailure);
    if (load && load->pending.coarse.size() + load->pending.target.size() > reads &&
        std::find(load->levels.begin(), load->levels.end(), levels) == load->levels.end())
        load->levels.push_back(levels);
    return result;
}

void loadPending(const ViewerLoad& load, const std::function<bool()>& afterPart)
{
    const auto failure = vcr::grid::load(load.pending, afterPart);
    for (const auto& levels : load.levels)
        renderOnCpu(*levels, failure);
}

} // namespace vc3d
