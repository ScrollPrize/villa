#include <vc_reader/display.hpp>
#include "Fiber3DCT.hpp"
#include "Fiber3DGeometry.hpp"
#include "OpenedVolumes.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/CudaSampling.hpp"
#include <vc_reader/array.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc3d::fibers3d::ct
{
cv::Mat_<cv::Vec2b> sampleCoordinates(const cv::Mat_<cv::Vec3f>& points, OpenedVolumes& volumes, const Volume& volume, int level)
{
    const auto& pyramid = volumes.pyramid(volume);
    if (!pyramid.has(level))
        throw std::runtime_error("Requested CT level is unavailable");
    if (pyramid.dtype != "|u1" && pyramid.dtype != "<u2")
        throw std::runtime_error("CT display supports uint8 and uint16 volumes");
    const auto shape = volume.shapeXyz();
    const auto opened = volumes.level(volume, level);
    const auto& placement = opened.array->placement;
    cv::Mat_<cv::Vec2b> result(points.size(), cv::Vec2b{0,0});
    std::vector<double> coords;
    coords.reserve(points.total() * 3);
    std::vector<size_t> indices;
    indices.reserve(points.total());
    for (int r=0; r<points.rows; ++r)
        for (int c=0; c<points.cols; ++c) {
            const auto v = points(r,c);
            if (!(std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]) &&
                  v[0]>=0 && v[1]>=0 && v[2]>=0 && v[0]<=shape[0]-1 && v[1]<=shape[1]-1 && v[2]<=shape[2]-1))
                continue;
            result(r,c)[1] = 255;
            for (int axis=0; axis<3; ++axis)
                coords.push_back(placement.fromLevel0(axis, opened.working.toLevel0(axis, v[2-axis])));
            indices.push_back(size_t(r) * points.cols + c);
        }
    std::vector<float> values(indices.size());
    vc::cuda::sampleLinear(*opened.array->array, opened.array->meta.type, opened.cuda.get(), coords.data(), indices.size(), values.data());
    const double scale = pyramid.dtype == "<u2" ? 1./256. : 1.;
    std::vector<uint8_t> bytes(indices.size());
    vcr::display::toUint8(values.data(), values.size(), bytes.data(), nullptr,
        {scale, vcr::display::Rounding::NearestEven});
    for (size_t i=0; i<indices.size(); ++i)
        result(int(indices[i] / points.cols), int(indices[i] % points.cols))[0] = bytes[i];
    return result;
}

void sampleTile(Tile& tile, OpenedVolumes& volumes, const Volume& volume, int level, const std::atomic_bool* cancelled)
{
    if (tile.width < 2 || tile.height < 2 || tile.width > 1024 || tile.height > 1024 || tile.left.size() != size_t(tile.height) ||
        tile.right.size() != size_t(tile.height))
        throw std::invalid_argument("Invalid CT tile dimensions");
    tile.pixels.resize(size_t(tile.width) * tile.height * 2);
    for (int first = 0; first < tile.height; first += 32) {
        if (cancelled && cancelled->load(std::memory_order_relaxed))
            throw std::runtime_error("CT sampling cancelled");
        const int rows = std::min(32, tile.height - first);
        cv::Mat_<cv::Vec3f> coords(rows, tile.width);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < tile.width; ++c) {
                const auto p = interpolate(tile.left[first + r], tile.right[first + r], double(c) / (tile.width - 1));
                coords(r, c) = finite(p) ? cv::Vec3f(float(p[0]), float(p[1]), float(p[2])) : cv::Vec3f(-1, -1, -1);
            }
        const auto samples = sampleCoordinates(coords, volumes, volume, level);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < tile.width; ++c) {
                const size_t index = (size_t(first + r) * tile.width + c) * 2;
                tile.pixels[index] = samples(r, c)[0];
                tile.pixels[index + 1] = samples(r, c)[1];
            }
    }
}

int levelForSpacing(OpenedVolumes& volumes, const Volume& volume, double spacing)
{
    const auto& levels = volumes.pyramid(volume).levels;
    if (levels.empty())
        throw std::runtime_error("CT volume has no present levels");
    const int requested = int(std::clamp(std::floor(std::log2(std::max(1., spacing))), 0., 20.));
    int chosen = levels.front();
    for (int level : levels)
        if (level <= requested)
            chosen = level;
    return chosen;
}
}  // namespace vc3d::fibers3d::ct
