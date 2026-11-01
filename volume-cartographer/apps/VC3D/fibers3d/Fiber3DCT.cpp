#include "Fiber3DCT.hpp"
#include "Fiber3DGeometry.hpp"
#include "vc/core/types/Volume.hpp"
#include <stdexcept>
#include "vc/core/render/ChunkedPlaneSampler.hpp"

namespace vc3d::fibers3d::ct
{
cv::Mat_<uint8_t> displayBytes(const cv::Mat& values)
{
    if (values.type() != CV_8UC1 && values.type() != CV_16UC1)
        throw std::invalid_argument("CT display supports uint8 and uint16 values");
    cv::Mat_<uint8_t> bytes;
    values.convertTo(bytes, CV_8U, values.type() == CV_16UC1 ? 1./256. : 1.);
    return bytes;
}
cv::Mat_<cv::Vec2b> sampleCoordinates(const cv::Mat_<cv::Vec3f>& points, Volume& volume, int level)
{
    if (!volume.hasScaleLevel(level))
        throw std::runtime_error("Requested CT level is unavailable");
    const auto dtype = volume.dtype();
    if (dtype != vc::render::ChunkDtype::UInt8 && dtype != vc::render::ChunkDtype::UInt16)
        throw std::runtime_error("CT display supports uint8 and uint16 volumes");
    const auto shape = volume.shapeXyz();
    cv::Mat_<cv::Vec3f> coords = points.clone();
    cv::Mat_<cv::Vec2b> result(points.size(), cv::Vec2b{0,0});
    for (int r=0; r<coords.rows; ++r)
        for (int c=0; c<coords.cols; ++c) {
            const auto v = coords(r,c);
            const bool valid = std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]) &&
                v[0]>=0 && v[1]>=0 && v[2]>=0 && v[0]<=shape[0]-1 && v[1]<=shape[1]-1 && v[2]<=shape[2]-1;
            if (valid) result(r,c)[1] = 255;
            else coords(r,c) = {-1,-1,-1};
        }
    vc::SampleParams params;
    params.level = level;
    cv::Mat_<uint8_t> bytes;
    if (dtype == vc::render::ChunkDtype::UInt16) {
        cv::Mat_<uint16_t> values;
        volume.sample(values, coords, params);
        bytes = displayBytes(values);
    } else
        volume.sample(bytes, coords, params);
    for (int r=0; r<coords.rows; ++r)
        for (int c=0; c<coords.cols; ++c) result(r,c)[0] = bytes(r,c);
    return result;
}

void sampleTile(Tile& tile, Volume& volume, int level, const std::atomic_bool* cancelled)
{
    if (tile.width < 2 || tile.height < 2 || tile.width > 1024 || tile.height > 1024 || tile.left.size() != size_t(tile.height) ||
        tile.right.size() != size_t(tile.height))
        throw std::invalid_argument("Invalid CT tile dimensions");
    if (!volume.hasScaleLevel(level))
        throw std::runtime_error("Requested CT level is unavailable");
    const auto dtype = volume.dtype();
    if (dtype != vc::render::ChunkDtype::UInt8 && dtype != vc::render::ChunkDtype::UInt16)
        throw std::runtime_error("CT display supports uint8 and uint16 volumes");
    tile.pixels.resize(size_t(tile.width) * tile.height * 2);
    const auto shape = volume.shapeXyz();
    for (int first = 0; first < tile.height; first += 32) {
        if (cancelled && cancelled->load(std::memory_order_relaxed))
            throw std::runtime_error("CT sampling cancelled");
        const int rows = std::min(32, tile.height - first);
        cv::Mat_<cv::Vec3f> coords(rows, tile.width);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < tile.width; ++c) {
                const auto p = interpolate(tile.left[first + r], tile.right[first + r], double(c) / (tile.width - 1));
                const bool valid =
                    finite(p) && p[0] >= 0 && p[1] >= 0 && p[2] >= 0 && p[0] <= shape[0] - 1 && p[1] <= shape[1] - 1 && p[2] <= shape[2] - 1;
                coords(r, c) = valid ? cv::Vec3f(float(p[0]), float(p[1]), float(p[2])) : cv::Vec3f(-1, -1, -1);
                tile.pixels[(size_t(first + r) * tile.width + c) * 2 + 1] = valid ? 255 : 0;
            }
        const auto samples = sampleCoordinates(coords, volume, level);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < tile.width; ++c)
                tile.pixels[(size_t(first + r) * tile.width + c) * 2] = samples(r, c)[0];
    }
}
Progress sampleTileProgressive(Tile& tile, vc::render::IChunkedArray& cache, int level, const std::atomic_bool* cancelled)
{
    if (tile.width < 2 || tile.height < 2 || tile.width > 1024 || tile.height > 1024 || tile.left.size() != size_t(tile.height) ||
        tile.right.size() != size_t(tile.height))
        throw std::invalid_argument("Invalid CT tile dimensions");
    using Sampler = vc::render::ChunkedPlaneSampler;
    Progress progress;
    tile.pixels.assign(size_t(tile.width) * tile.height * 2, 0);
    const auto shape = cache.shape(0);  // ZYX
    Sampler::Options options(vc::Sampling::Trilinear, 32);
    options.parallelThreads = 2;
    options.request.foreground = true;
    options.queuedFallbackLevels = 2;
    for (int first = 0; first < tile.height; first += 128) {
        if (cancelled && cancelled->load(std::memory_order_relaxed))
            throw std::runtime_error("CT sampling cancelled");
        const int rows = std::min(128, tile.height - first);
        cv::Mat_<cv::Vec3f> coords(rows, tile.width);
        cv::Mat_<uint8_t> coverage(rows, tile.width, uint8_t(0)), values(rows, tile.width, uint8_t(0));
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < tile.width; ++c) {
                const auto p = interpolate(tile.left[first + r], tile.right[first + r], double(c) / (tile.width - 1));
                const bool valid =
                    finite(p) && p[0] >= 0 && p[1] >= 0 && p[2] >= 0 && p[0] <= shape[2] - 1 && p[1] <= shape[1] - 1 && p[2] <= shape[0] - 1;
                coords(r, c) = valid ? cv::Vec3f(float(p[0]), float(p[1]), float(p[2])) : cv::Vec3f(-1, -1, -1);
                if (valid)
                    ++progress.valid;
            }
        const auto fine = Sampler::sampleCoordsLevel(cache, level, coords, values, coverage, options);
        progress.requestedLevel += fine.coveredPixels;
        progress.errors += fine.errorChunks;
        // Same fine-to-coarse sampling as CT viewers: only missing pixels are
        // filled from coarser data, never MIP or a blend of physical layers.
        if (level + 1 < cache.numLevels())
            Sampler::sampleCoordsFineToCoarse(cache, level + 1, coords, values, coverage, options);
        // Parameterized surface samplers reserve (0,0,0) as a sentinel. For an
        // actual CT ribbon this is a valid point; the plane API supports it.
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < tile.width; ++c) {
                if (coords(r, c) == cv::Vec3f(0, 0, 0)) {
                    cv::Mat_<uint8_t> v(1, 1, uint8_t(0)), mask(1, 1, uint8_t(0));
                    const auto atOrigin = Sampler::samplePlaneLevel(cache, level, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, v, mask, options);
                    progress.requestedLevel += atOrigin.coveredPixels;
                    progress.errors += atOrigin.errorChunks;
                    if (level + 1 < cache.numLevels())
                        Sampler::samplePlaneFineToCoarse(cache, level + 1, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, v, mask, options);
                    values(r, c) = v(0, 0);
                    coverage(r, c) = mask(0, 0);
                }
                const size_t index = (size_t(first + r) * tile.width + c) * 2;
                if (coverage(r, c)) {
                    tile.pixels[index] = values(r, c);
                    tile.pixels[index + 1] = 255;
                    ++progress.covered;
                }
            }
    }
    return progress;
}

}  // namespace vc3d::fibers3d::ct
