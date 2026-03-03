#pragma once

#include <opencv2/core.hpp>

namespace vc {

// Build a 2D grid of 3D coordinates from an origin and two axis vectors.
// coords(r, c) = origin + axisU * c + axisV * r
cv::Mat_<cv::Vec3f> makeCoordGrid(
    const cv::Vec3f& origin, const cv::Vec3f& axisU, const cv::Vec3f& axisV, int w, int h);

}  // namespace vc
