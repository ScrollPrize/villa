#include "vc/core/util/SurfaceTexture.hpp"

#include <vc_reader/grid.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace vc::core::util
{
namespace
{

constexpr int kTiffCompressionNone = 1;

}  // namespace

cv::Mat_<std::uint8_t> renderCoordsTexture(
    const cv::Mat_<cv::Vec3f>& coords,
    const vcr::grid::Level& level,
    int renderScale,
    const char* label)
{
    if (coords.empty())
        throw std::runtime_error("surface has no points for texture rendering");
    renderScale = std::max(1, renderScale);
    const vcr::grid::Surface surface({reinterpret_cast<const float*>(coords.data), coords.rows, coords.cols,
                                      coords.step1()});
    // Pixel centres of a renderScale-times upsampled image: pixel i at grid
    // column (i + 0.5) / renderScale - 0.5.
    const double step = 1.0 / renderScale;
    const vcr::grid::Raster raster{0.5 * step - 0.5, 0.5 * step - 0.5, step, step,
                                   coords.cols * renderScale, coords.rows * renderScale};
    vcr::grid::Request request;
    request.levels = {level};
    const auto result = vcr::grid::sample(surface, raster, request);

    cv::Mat_<std::uint8_t> sampled(raster.height, raster.width);
    int covered = 0;
    for (int i = 0; i < int(sampled.total()); ++i) {
        const float v = result.values[std::size_t(i)];
        covered += !std::isnan(v);
        sampled(i) = std::isnan(v) ? std::uint8_t(0) : std::uint8_t(std::clamp(v, 0.0f, 255.0f));
    }
    std::cout << label << ": render_scale=" << renderScale
              << " size=" << sampled.cols << "x" << sampled.rows
              << " covered=" << covered << "/" << sampled.total() << '\n';
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
