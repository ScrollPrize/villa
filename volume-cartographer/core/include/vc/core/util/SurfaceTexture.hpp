#pragma once

#include "vc/core/util/TexturedMesh.hpp"

#include <filesystem>

#include <opencv2/core/mat.hpp>

class QuadSurface;
class Volume;

namespace vc::core::util
{

[[nodiscard]] cv::Mat renderCoordsTextureFineToCoarse(
    const cv::Mat_<cv::Vec3f>& baseCoords,
    Volume& textureVolume,
    int textureLevel,
    int renderScale,
    const char* label);

[[nodiscard]] cv::Mat renderSurfaceTextureFineToCoarse(
    const QuadSurface& surface,
    Volume& textureVolume,
    int textureLevel,
    int renderScale);

[[nodiscard]] TexturedMesh texturedSurfaceMesh(const QuadSurface& surface);

void writeUncompressedTextureTiff(
    const std::filesystem::path& path, const cv::Mat& image);

}  // namespace vc::core::util
