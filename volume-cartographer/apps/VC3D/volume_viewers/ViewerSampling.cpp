#include "ViewerSampling.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Compositing.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vc3d {

bool ViewerLevels::has(int level) const
{
    return level >= 0 && level < count() && levels[std::size_t(level)].array;
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
    vcr::grid::Level g{l.array->array.get(), l.array->meta.type, {}, {}};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        // index = (working.toLevel0(c) - offset) / scale, as one affine map
        g.scale[axis] = l.working.scale[axis] / l.array->placement.scale[axis];
        g.offset[axis] = (l.working.offset[axis] - l.array->placement.offset[axis]) / l.array->placement.scale[axis];
    }
    return g;
}

ViewerLevels openViewerLevels(OpenedVolumes& opened, const Volume& volume)
{
    ViewerLevels result;
    result.levels.resize(volume.numScales());
    for (int level = 0; level < result.count(); ++level) {
        if (!volume.hasScaleLevel(level))
            continue;
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

std::shared_ptr<const vcr::grid::Surface> gridSurface(const QuadSurface& surface)
{
    const auto* points = surface.rawPointsPtr();
    if (!points || points->empty())
        return nullptr;
    vcr::grid::SurfaceOptions options;
    options.interpolation = surface.genInterpolation() == GenInterpolation::Smooth
        ? vcr::grid::Interpolation::Smooth
        : vcr::grid::Interpolation::Linear;
    options.strictQuads = surface.strictQuadRenderValidity();
    options.components = surface.components();
    return std::make_shared<const vcr::grid::Surface>(
        vcr::grid::Vertices{reinterpret_cast<const float*>(points->ptr()), points->rows, points->cols,
                            points->step1()},
        std::move(options));
}

vcr::grid::Result ViewerGeometry::sample(const vcr::grid::Request& request) const
{
    return plane ? vcr::grid::sample(*plane, request) : vcr::grid::sample(*surface, raster, request);
}

vcr::grid::Result sampleCached(const ViewerGeometry& geometry, const ViewerSample& what, ViewerLoad* load)
{
    const ViewerLevels& levels = *what.levels;
    vcr::grid::Request request;
    request.order = what.sampling == vc::Sampling::Nearest ? 0 : 1;
    request.offsets = what.offsets;
    request.composite = what.composite;
    request.cachedOnly = true;
    std::size_t used = 0;  // levels in the request
    const int last = std::min(levels.count() - 1, what.startLevel + std::max(0, what.fallbackLevels));
    for (int level = std::max(0, what.startLevel); level <= last; ++level)
        if (levels.has(level)) {
            request.levels.push_back(levels.gridLevel(level));
            ++used;
        }
    if (request.levels.empty())
        return {};
    auto result = geometry.sample(request);
    if (!load)
        return result;

    // Pixels with a layer not from the start level read it; pixels with a layer no
    // level held also read the coarsest level, for a quick preview.
    const std::size_t pixels = std::size_t(result.width) * std::size_t(result.height);
    std::vector<std::uint8_t> target(pixels, 0), coarse(pixels, 0);
    bool anyTarget = false, anyCoarse = false;
    for (std::size_t k = 0; k < std::size_t(result.layers); ++k)
        for (std::size_t q = 0; q < pixels; ++q) {
            const auto level = result.level[k * pixels + q];
            if (level == vcr::grid::Result::kNone || level == 0)
                continue;
            target[q] = 1;
            anyTarget = true;
            if (level == vcr::grid::Result::kMissing && used > 1) {
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
        load->coarse.push_back(read(used - 1, std::move(coarse)));
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
            part.geometry.sample(request);
        }
        if (!parts->empty() && !afterPart())
            return;
    }
}

std::optional<vcr::composite::Options> compositeOptions(const CompositeParams& params)
{
    using Method = vcr::composite::Method;
    vcr::composite::Options options;
    if (params.method == "mean")
        options.method = Method::Mean;
    else if (params.method == "max")
        options.method = Method::Max;
    else if (params.method == "min")
        options.method = Method::Min;
    else if (params.method == "alpha")
        options.method = Method::Alpha;
    else if (params.method == "beerLambert")
        options.method = Method::BeerLambert;
    else
        return std::nullopt;
    options.isoCutoff = float(params.isoCutoff);
    options.alphaMin = params.alphaMin;
    options.alphaMax = params.alphaMax;
    options.alphaOpacity = params.alphaOpacity;
    options.alphaCutoff = params.alphaCutoff;
    options.extinction = params.blExtinction;
    options.emission = params.blEmission;
    options.ambient = params.blAmbient;
    return options;
}

void toDisplay(const vcr::grid::Result& result, int layer, uint8_t* values, uint8_t* coverage)
{
    const std::size_t pixels = std::size_t(result.width) * std::size_t(result.height);
    const float* v = result.values.data() + std::size_t(layer) * pixels;
    for (std::size_t q = 0; q < pixels; ++q) {
        if (!std::isfinite(v[q]))
            continue;
        values[q] = static_cast<uint8_t>(std::clamp(v[q], 0.0f, 255.0f));
        coverage[q] = 1;
    }
}

}  // namespace vc3d
