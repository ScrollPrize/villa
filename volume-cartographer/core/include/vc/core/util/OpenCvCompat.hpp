#pragma once

#include <opencv2/core/version.hpp>

#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>
#endif

namespace vc::opencv {

// OpenCV 5 removed the public cv::DistanceTypes enum while retaining the
// integer distanceTransform API. These are the stable values used by OpenCV
// 4 and 5 for Manhattan and Euclidean distance, respectively.
inline constexpr int distanceL1 = 1;
inline constexpr int distanceL2 = 2;

}  // namespace vc::opencv
