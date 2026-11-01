#pragma once

#include <cstdint>

#include <opencv2/core/mat.hpp>

namespace vcr::grid {
struct Level;
}

namespace vc::core::util
{

// A uint8 texture of a volume level along a grid of volume points (XYZ, in the
// coordinates `level` maps to its indices): the grid upsampled renderScale times
// with cv::resize's pixel centres, positions interpolated bilinearly between the
// vertices and sampled trilinearly with blocking reads, truncated to 8 bits; 0
// where the grid has no point.
[[nodiscard]] cv::Mat_<std::uint8_t> renderCoordsTexture(
    const cv::Mat_<cv::Vec3f>& coords,
    const vcr::grid::Level& level,
    int renderScale,
    const char* label);

}  // namespace vc::core::util
