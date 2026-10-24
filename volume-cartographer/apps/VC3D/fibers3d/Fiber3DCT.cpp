#include "Fiber3DCT.hpp"
#include "Fiber3DGeometry.hpp"
#include "vc/core/types/Volume.hpp"
#include <stdexcept>

namespace vc3d::fibers3d::ct
{
void sampleTile(Tile& tile, Volume& volume, int level, const std::atomic_bool* cancelled)
{
    if (tile.width < 2 || tile.height < 2 || tile.width > 1024 || tile.height > 1024 ||
        tile.left.size() != size_t(tile.height) || tile.right.size() != size_t(tile.height))
        throw std::invalid_argument("Invalid CT tile dimensions");
    if (!volume.hasScaleLevel(level))
        throw std::runtime_error("Requested CT level is unavailable");
    const auto dtype = volume.dtype();
    if (dtype != vc::render::ChunkDtype::UInt8 && dtype != vc::render::ChunkDtype::UInt16)
        throw std::runtime_error("CT display supports uint8 and uint16 volumes");
    tile.pixels.resize(size_t(tile.width) * tile.height * 2);
    const auto shape = volume.shapeXyz();
    vc::SampleParams params;
    params.level = level;
    for (int first = 0; first < tile.height; first += 32) {
        if (cancelled && cancelled->load(std::memory_order_relaxed))
            throw std::runtime_error("CT sampling cancelled");
        const int rows = std::min(32, tile.height - first);
        cv::Mat_<cv::Vec3f> coords(rows, tile.width);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < tile.width; ++c) {
                const auto p = interpolate(tile.left[first + r], tile.right[first + r], double(c) / (tile.width - 1));
                const bool valid = finite(p) && p[0] >= 0 && p[1] >= 0 && p[2] >= 0 &&
                                   p[0] <= shape[0] - 1 && p[1] <= shape[1] - 1 && p[2] <= shape[2] - 1;
                coords(r, c) = valid ? cv::Vec3f(float(p[0]), float(p[1]), float(p[2])) : cv::Vec3f(-1, -1, -1);
                tile.pixels[(size_t(first + r) * tile.width + c) * 2 + 1] = valid ? 255 : 0;
            }
        cv::Mat_<uint8_t> bytes;
        if (dtype == vc::render::ChunkDtype::UInt16) {
            cv::Mat_<uint16_t> values;
            volume.sample(values, coords, params);
            values.convertTo(bytes, CV_8U, 1. / 256.);
        } else
            volume.sample(bytes, coords, params);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < tile.width; ++c)
                tile.pixels[(size_t(first + r) * tile.width + c) * 2] = bytes(r, c);
    }
}
}
