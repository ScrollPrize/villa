#include "vc/core/util/SurfaceTexture.hpp"

#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/types/Volume.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace vc::core::util
{
namespace
{

constexpr int kTiffCompressionNone = 1;

}  // namespace

cv::Mat renderCoordsTextureFineToCoarse(
    const cv::Mat_<cv::Vec3f>& baseCoords,
    Volume& textureVolume,
    int textureLevel,
    int renderScale,
    const char* label)
{
    if (baseCoords.empty())
        throw std::runtime_error("surface has no points for texture rendering");
    renderScale = std::max(1, renderScale);
    cv::Mat_<cv::Vec3f> coords;
    if (renderScale == 1) {
        coords = baseCoords.clone();
    } else {
        cv::resize(baseCoords,
                   coords,
                   cv::Size(baseCoords.cols * renderScale,
                            baseCoords.rows * renderScale),
                   0.0,
                   0.0,
                   cv::INTER_LINEAR);
    }

    vc::render::IChunkedArray* cache = textureVolume.chunkedCache();
    if (!cache)
        throw std::runtime_error("texture volume has no chunk cache");
    if (cache->dtype() != vc::render::ChunkDtype::UInt8) {
        throw std::runtime_error(
            "line probe strip rendering uses the VC3D uint8 chunk sampler; "
            "choose a uint8 texture zarr");
    }

    cv::Mat_<uint8_t> sampled(coords.rows, coords.cols, uint8_t(0));
    cv::Mat_<uint8_t> coverage(coords.rows, coords.cols, uint8_t(0));
    const vc::render::ChunkedPlaneSampler::Options options(
        vc::Sampling::Trilinear, 32);
    const int startLevel = std::clamp(
        textureLevel, 0, cache->numLevels() - 1);

    for (int level = startLevel; level < cache->numLevels(); ++level) {
        std::vector<vc::render::ChunkKey> keys =
            vc::render::ChunkedPlaneSampler::collectCoordsDependencies(
                *cache, level, coords, coverage, options);
        if (!keys.empty())
            cache->prefetchChunks(keys, true);
    }

    const auto stats =
        vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
            *cache, startLevel, coords, sampled, coverage, options);
    const int covered = cv::countNonZero(coverage);
    const int total = coverage.rows * coverage.cols;
    std::cout << label << ": start_level=" << startLevel
              << " render_scale=" << renderScale
              << " size=" << sampled.cols << "x" << sampled.rows
              << " covered=" << covered << "/" << total
              << " requested_chunks=" << stats.requestedChunks
              << " error_chunks=" << stats.errorChunks << '\n';
    return sampled;
}


void writeUncompressedTextureTiff(
    const std::filesystem::path& path, const cv::Mat& image)
{
    if (image.empty())
        throw std::runtime_error("cannot write empty image: " + path.string());
    const std::vector<int> params{
        cv::IMWRITE_TIFF_COMPRESSION,
        kTiffCompressionNone,
    };
    if (!cv::imwrite(path.string(), image, params))
        throw std::runtime_error("failed to write image: " + path.string());
}

}  // namespace vc::core::util
