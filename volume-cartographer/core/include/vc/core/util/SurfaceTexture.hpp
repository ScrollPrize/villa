#pragma once

#include <filesystem>

#include <opencv2/core/mat.hpp>

class Volume;

namespace vc::core::util
{

[[nodiscard]] cv::Mat renderCoordsTextureFineToCoarse(
    const cv::Mat_<cv::Vec3f>& baseCoords,
    Volume& textureVolume,
    int textureLevel,
    int renderScale,
    const char* label);

void writeUncompressedTextureTiff(
    const std::filesystem::path& path, const cv::Mat& image);

}  // namespace vc::core::util
