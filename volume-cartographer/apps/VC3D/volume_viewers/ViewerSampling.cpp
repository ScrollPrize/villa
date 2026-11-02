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

namespace {
// Finer levels a frame may take resident data from when its own level lacks it.
constexpr int kFinerFallbackLevels = 2;
}

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

std::array<double, 3> ViewerLevels::voxelExtent(int level) const
{
    const auto& l = levels[std::size_t(level)];
    std::array<double, 3> extent{};
    for (std::size_t axis = 0; axis < 3; ++axis)
        extent[axis] = l.array->placement.scale[axis] / l.working.scale[axis];
    return extent;
}

vcr::grid::Level ViewerLevels::gridLevel(int level) const
{
    const auto& l = levels[std::size_t(level)];
    return vcr::zarr::gridLevel(*l.array, l.working, gpuFailed->load() ? nullptr : l.cuda.get());
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

double maximumBaseVoxelExtent(const ViewerLevels& levels, int level)
{
    if (!levels.has(level))
        return std::numeric_limits<double>::quiet_NaN();
    const auto extent = levels.voxelExtent(level);
    return *std::max_element(extent.begin(), extent.end());
}

int sourceLevelForView(const ViewerLevels& levels, float pixelsPerBaseVoxel, float qualityBias)
{
    if (!std::isfinite(pixelsPerBaseVoxel) || !std::isfinite(qualityBias) ||
        pixelsPerBaseVoxel <= 0.0f || qualityBias <= 0.0f)
        return 0;
    const double maximumExtent = 1.0 / (double(pixelsPerBaseVoxel) * double(qualityBias));
    int selected = -1;
    for (int level = 0; level < levels.count(); ++level) {
        if (!levels.has(level))
            continue;
        // At high zoom, keep the finest level even when none meets the threshold.
        if (selected < 0 || maximumBaseVoxelExtent(levels, level) <= maximumExtent)
            selected = level;
    }
    return std::max(selected, 0);
}

float surfacePixelsPerVoxel(const QuadSurface& surface, const vcr::grid::Raster& viewport,
                            float nominalPixelsPerVoxel)
{
    double magnification = 0;
    // Include off-center patches when the camera is over a hole or a boundary.
    for (const double y : {0.25, 0.5, 0.75}) {
        for (const double x : {0.25, 0.5, 0.75}) {
            cv::Vec3f du, dv;
            surface.pointAt(float(viewport.x0 + x * viewport.width * viewport.sx),
                            float(viewport.y0 + y * viewport.height * viewport.sy), nullptr, nullptr, du.val, dv.val);
            const cv::Vec3d dx = cv::Vec3d(du) * viewport.sx;
            const cv::Vec3d dy = cv::Vec3d(dv) * viewport.sy;
            const double a = dx.dot(dx), b = dx.dot(dy), c = dy.dot(dy);
            // Smaller singular value: resolve the most magnified direction,
            // including grids whose row and column tangents are not orthogonal.
            const double extent2 = 0.5 * (a + c - std::hypot(a - c, 2 * b));
            if (std::isfinite(extent2) && extent2 > 0)
                magnification = std::max(magnification, 1.0 / std::sqrt(extent2));
        }
    }
    return magnification > 0 ? float(magnification) : nominalPixelsPerVoxel;
}

int fallbackLevelCountForViewport(const ViewerLevels& levels, int startLevel, int viewportWidth,
                                  int viewportHeight, float pixelsPerBaseVoxel, int maximumFallbackLevels)
{
    if (startLevel < 0 || startLevel >= levels.count() || maximumFallbackLevels <= 0)
        return 0;
    const int lastLevel = std::min(levels.count() - 1, startLevel + maximumFallbackLevels);
    if (viewportWidth <= 0 || viewportHeight <= 0 || !std::isfinite(pixelsPerBaseVoxel) ||
        pixelsPerBaseVoxel <= 0.0f)
        return lastLevel - startLevel;
    const double viewportExtent = double(std::max(viewportWidth, viewportHeight)) / double(pixelsPerBaseVoxel);
    for (int level = startLevel; level <= lastLevel; ++level) {
        if (!levels.has(level))
            continue;
        // A representative chunk edge in working level-0 voxels: the mean over the axes.
        const auto extent = levels.voxelExtent(level);
        const auto& chunks = levels.levels[std::size_t(level)].array->meta.chunks;
        double chunkExtent = 0.0;
        for (std::size_t axis = 0; axis < 3; ++axis)
            chunkExtent += double(chunks[axis]) * extent[axis];
        if (chunkExtent / 3.0 >= viewportExtent)
            return level - startLevel;
    }
    return lastLevel - startLevel;
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

vcr::grid::Result sampleCached(const ViewerGeometry& geometry, const ViewerSample& what, ViewerLoad* load)
{
    const ViewerLevels& levels = *what.levels;
    vcr::grid::Request request;
    request.order = what.sampling;
    request.offsets = what.offsets;
    request.composite = what.composite;
    request.cachedOnly = true;
    // The view's level, then finer levels (resident after zooming in, and as good),
    // then coarser ones filling holes.
    const auto add = [&](int level) {
        if (levels.has(level))
            request.levels.push_back(levels.gridLevel(level));
    };
    const int start = std::clamp(what.startLevel, 0, std::max(0, levels.count() - 1));
    add(start);
    for (int level = start - 1; level >= std::max(0, start - kFinerFallbackLevels); --level)
        add(level);
    const std::size_t firstCoarser = request.levels.size();
    const int last = std::min(levels.count() - 1, start + std::max(0, what.fallbackLevels));
    for (int level = start + 1; level <= last; ++level)
        add(level);
    if (request.levels.empty())
        return {};
    auto result = geometry.plane ? vcr::grid::sample(*geometry.plane, request)
                                 : vcr::grid::sample(*geometry.surface, geometry.raster, request);
    if (!result.cudaFailure.empty() && !levels.gpuFailed->exchange(true))
        Logger()->warn("GPU frame failed, these levels render on the CPU from now on: {}", result.cudaFailure);
    if (!load)
        return result;

    // Pixels with a layer from a coarser level, or none, read the start level; pixels
    // with a layer no level held also read the coarsest level, for a quick preview.
    const std::size_t pixels = result.level.size();
    const bool coarser = request.levels.size() > firstCoarser;
    std::vector<std::uint8_t> target(pixels, 0), coarse(pixels, 0);
    bool anyTarget = false, anyCoarse = false;
    for (std::size_t q = 0; q < pixels; ++q) {
        const auto level = result.level[q];
        const bool missing = level == vcr::grid::Result::kMissing;
        if (!missing && (level == vcr::grid::Result::kNone || std::size_t(level) < firstCoarser))
            continue;
        target[q] = 1;
        anyTarget = true;
        if (missing && coarser) {
            coarse[q] = 1;
            anyCoarse = true;
        }
    }
    const auto read = [&](std::size_t index, std::vector<std::uint8_t> mask) {
        ViewerRead r{geometry, what.levels, {}, std::move(mask)};
        r.request.levels = {request.levels[index]};
        r.request.order = request.order;
        r.request.offsets = request.offsets;
        return r;
    };
    if (anyCoarse)
        load->coarse.push_back(read(request.levels.size() - 1, std::move(coarse)));
    if (anyTarget)
        load->target.push_back(read(0, std::move(target)));
    return result;
}

void loadPending(const ViewerLoad& load, const std::function<bool()>& afterPart)
{
    for (const auto* parts : {&load.coarse, &load.target}) {
        for (const auto& part : *parts) {
            auto request = part.request;
            request.pixels = part.pixels.data();
            const auto& geometry = part.geometry;
            const auto result = geometry.plane ? vcr::grid::sample(*geometry.plane, request)
                : vcr::grid::sample(*geometry.surface, geometry.raster, request);
            if (!result.cudaFailure.empty() && !part.levels->gpuFailed->exchange(true))
                Logger()->warn("GPU frame failed, these levels render on the CPU from now on: {}", result.cudaFailure);
        }
        if (!parts->empty() && !afterPart())
            return;
    }
}

} // namespace vc3d
