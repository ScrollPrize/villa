#include "vc/core/util/RenderLayers.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <vc_reader/affine.hpp>
#include <vc_reader/array.hpp>
#include <vc_reader/display.hpp>
#include <vc_reader/zarr.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace vc {

RenderLevelMap renderLevelMap(const vcr::zarr::LevelPlacement& working, const vcr::zarr::LevelPlacement& rendered)
{
    const auto m = rendered.indexMapFrom(working);
    RenderLevelMap map;
    for (int a = 0; a < 3; ++a) {
        map.scale[a] = float(m.scale[2 - a]);
        map.offset[a] = float(m.offset[2 - a]);
    }
    return map;
}

RenderSurface renderSurface(const QuadSurface& surf, vcr::grid::Interpolation interpolation, float scaleSeg,
                            bool hasAffine, const vcr::catalog::Affine& aff, const RenderLevelMap& level)
{
    RenderSurface r;
    std::vector<std::uint8_t> valid;
    (void)surf.gridVertices(valid);
    r.vertices = surf.rawPointsPtr()->clone();
    for (auto& v : r.vertices) {
        if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) {
            v = cv::Vec3f(NAN, NAN, NAN);
            continue;
        }
        v *= scaleSeg;
        if (hasAffine) {
            const auto q = aff.apply({v[0], v[1], v[2]});
            v = cv::Vec3f(float(q[0]), float(q[1]), float(q[2]));
        }
        for (int a = 0; a < 3; ++a)
            v[a] = v[a] * level.scale[a] + level.offset[a];
    }
    auto options = surf.gridOptions();
    options.interpolation = interpolation;
    r.grid = std::make_unique<vcr::grid::Surface>(
        vcr::grid::Vertices{reinterpret_cast<const float*>(r.vertices.ptr()), r.vertices.rows, r.vertices.cols,
                            r.vertices.step1(), nullptr, valid.data()},
        std::move(options));
    return r;
}

std::vector<float> renderLayerOffsets(int numSlices, double sliceStep, const std::vector<float>& accumOffsets)
{
    double center = 0.5 * (std::max(1, numSlices) - 1.0);
    size_t sps = std::max<size_t>(1, accumOffsets.size());
    std::vector<float> out;
    out.reserve(numSlices * sps);
    for (int zi = 0; zi < numSlices; zi++) {
        float base = float((zi - center) * sliceStep);
        if (accumOffsets.empty())
            out.push_back(base);
        else
            for (float ao : accumOffsets)
                out.push_back(base + ao);
    }
    return out;
}

vcr::grid::Raster renderRaster(const QuadSurface& surf, cv::Size canvas, float renderScale, const cv::Rect& pixels)
{
    // The tile's framebuffer pixels, the top-left one at canvas offset (u0, v0).
    const cv::Vec2f origin = renderCanvasOrigin(canvas);
    float u0 = origin[0], v0 = origin[1];
    u0 += float(pixels.x);
    v0 += float(pixels.y);
    return surf.gridRaster(double(u0 / renderScale), double(v0 / renderScale), renderScale, pixels.size());
}

vcr::grid::Result sampleRenderTile(const RenderSurface& surface, const vcr::grid::Raster& raster,
                                   const vcr::grid::Level& level, const std::vector<float>& offsets, bool flipNormals,
                                   int order, const std::optional<vcr::composite::Options>& composite,
                                   const std::atomic<bool>* cancel)
{
    vcr::grid::Request request;
    request.levels = {level};
    request.order = order;
    request.offsets = offsets;
    if (flipNormals)
        for (auto& o : request.offsets)
            o = -o;
    request.composite = composite;
    request.cancel = cancel;
    return vcr::grid::sample(*surface.grid, raster, request);
}

void renderLayerPixels(const vcr::grid::Result& result, int layer, std::uint8_t* out)
{
    const std::size_t n = std::size_t(result.width) * std::size_t(result.height);
    const float* v = result.values.data() + std::size_t(layer) * n;
    vcr::display::toUint8(v, n, out, nullptr, {1, vcr::display::Rounding::NearestEven, true});
}

void renderLayerPixels(const vcr::grid::Result& result, int layer, std::uint16_t* out)
{
    const std::size_t n = std::size_t(result.width) * std::size_t(result.height);
    const float* v = result.values.data() + std::size_t(layer) * n;
    vcr::display::toUint16(v, n, out);
}

cv::Mat sampleRenderLayers(const QuadSurface& surf, const RenderSurface& surface, const vcr::grid::Level& level,
                           const RenderCanvas& canvas, const cv::Rect& pixels, const std::vector<float>& offsets,
                           bool flipNormals, const std::atomic<bool>* cancel)
{
    const bool wide = level.type.itemsize == 2;
    if (!level.array || level.type.kind != 'u' || (level.type.itemsize != 1 && !wide))
        throw std::invalid_argument("sampleRenderLayers reads uint8 or uint16 volumes");
    if (level.linear || level.scale != std::array<double, 3>{1, 1, 1} || level.offset != std::array<double, 3>{0, 0, 0})
        throw std::invalid_argument("sampleRenderLayers: the level's placement belongs in the RenderSurface");
    // The canvas places its pixels on the grid: `surf` must have the canvas's grid and
    // scale, and its stored origin (shiftSurfaceOrigin moves the rasters off the canvas).
    const auto first = renderRaster(surf, canvas.size, float(canvas.renderScale), {0, 0, 1, 1});
    const cv::Vec2d center = canvas.canvasToGrid({0.5, 0.5}), pitch = canvas.pitch();
    if (canvas.grid != surface.vertices.size() || canvas.gridScale != surf.scale() ||
        !(std::abs(first.x0 - center[0]) < 0.5 * pitch[0] && std::abs(first.y0 - center[1]) < 0.5 * pitch[1]))
        throw std::invalid_argument("sampleRenderLayers: the canvas is not the surface's");
    if (pixels.empty() || (pixels & cv::Rect(cv::Point(), canvas.size)) != pixels)
        throw std::invalid_argument("sampleRenderLayers: the pixels are not inside the canvas");

    const int shape[3]{int(offsets.size()), pixels.height, pixels.width};
    cv::Mat out(3, shape, wide ? CV_16UC1 : CV_8UC1, cv::Scalar(0));
    // The renderer's tiles of the whole canvas that hold a pixel of the window.
    const int tx0 = pixels.x / kRenderTile, ty0 = pixels.y / kRenderTile;
    const int tilesX = (pixels.x + pixels.width - 1) / kRenderTile - tx0 + 1;
    const int tilesY = (pixels.y + pixels.height - 1) / kRenderTile - ty0 + 1;
    level.array->parallel_for(std::size_t(tilesX) * std::size_t(tilesY), [&](std::size_t t) {
        const int x0 = (tx0 + int(t % std::size_t(tilesX))) * kRenderTile;
        const int y0 = (ty0 + int(t / std::size_t(tilesX))) * kRenderTile;
        const cv::Rect tile(x0, y0, std::min(kRenderTile, canvas.size.width - x0),
                            std::min(kRenderTile, canvas.size.height - y0));
        const auto raster = renderRaster(surf, canvas.size, float(canvas.renderScale), tile);
        const auto layers = sampleRenderTile(surface, raster, level, offsets, flipNormals, 1, std::nullopt, cancel);
        const cv::Rect part = tile & pixels;
        cv::Mat image(tile.size(), out.type());
        for (int k = 0; k < shape[0]; ++k) {
            if (wide)
                renderLayerPixels(layers, k, image.ptr<std::uint16_t>());
            else
                renderLayerPixels(layers, k, image.ptr<std::uint8_t>());
            for (int y = part.y; y < part.y + part.height; ++y)
                std::memcpy(out.ptr(k, y - pixels.y, part.x - pixels.x), image.ptr(y - tile.y, part.x - tile.x),
                            std::size_t(part.width) * out.elemSize());
        }
    });
    return out;
}

}  // namespace vc
