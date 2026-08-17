#include "vc/core/util/SurfaceTexture.hpp"

#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"

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

bool validSurfacePoint(const cv::Vec3f& point)
{
    return point[0] != -1.0f && std::isfinite(point[0]) &&
        std::isfinite(point[1]) && std::isfinite(point[2]);
}

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

cv::Mat renderSurfaceTextureFineToCoarse(
    const QuadSurface& surface,
    Volume& textureVolume,
    int textureLevel,
    int renderScale)
{
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->empty())
        throw std::runtime_error("surface has no points for texture rendering");
    return renderCoordsTextureFineToCoarse(
        *points,
        textureVolume,
        textureLevel,
        renderScale,
        "Strip texture sampling");
}

TexturedMesh texturedSurfaceMesh(const QuadSurface& surface)
{
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->empty())
        throw std::runtime_error("surface has no points for OBJ export");

    TexturedMesh mesh;
    constexpr size_t invalidIndex = std::numeric_limits<size_t>::max();
    std::vector<size_t> vertexIndex(
        static_cast<size_t>(points->rows * points->cols), invalidIndex);
    for (int row = 0; row < points->rows; ++row) {
        for (int col = 0; col < points->cols; ++col) {
            const cv::Vec3f& point = (*points)(row, col);
            if (!validSurfacePoint(point))
                continue;
            vertexIndex[static_cast<size_t>(row * points->cols + col)] =
                mesh.vertices.size();
            mesh.vertices.push_back({point[0], point[1], point[2]});
        }
    }

    std::vector<size_t> uvIndex(
        static_cast<size_t>(points->rows * points->cols), invalidIndex);
    const double colDenom = std::max(1, points->cols - 1);
    const double rowDenom = std::max(1, points->rows - 1);
    for (int row = 0; row < points->rows; ++row) {
        for (int col = 0; col < points->cols; ++col) {
            const cv::Vec3f& point = (*points)(row, col);
            if (!validSurfacePoint(point))
                continue;
            uvIndex[static_cast<size_t>(row * points->cols + col)] =
                mesh.textureCoordinates.size();
            mesh.textureCoordinates.push_back({
                static_cast<double>(col) / colDenom,
                1.0 - static_cast<double>(row) / rowDenom,
            });
        }
    }

    for (int row = 0; row + 1 < points->rows; ++row) {
        for (int col = 0; col + 1 < points->cols; ++col) {
            const auto index = [&](int r, int c) {
                return static_cast<size_t>(r * points->cols + c);
            };
            const size_t v00 = vertexIndex[index(row, col)];
            const size_t v01 = vertexIndex[index(row, col + 1)];
            const size_t v10 = vertexIndex[index(row + 1, col)];
            const size_t v11 = vertexIndex[index(row + 1, col + 1)];
            if (v00 == invalidIndex || v01 == invalidIndex ||
                v10 == invalidIndex || v11 == invalidIndex) {
                continue;
            }
            mesh.quads.push_back({
                {v00, v01, v11, v10},
                {uvIndex[index(row, col)], uvIndex[index(row, col + 1)],
                 uvIndex[index(row + 1, col + 1)],
                 uvIndex[index(row + 1, col)]},
            });
        }
    }
    return mesh;
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
