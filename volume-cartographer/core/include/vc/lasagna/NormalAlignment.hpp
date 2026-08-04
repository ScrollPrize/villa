#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/core/types.hpp>

namespace vc::lasagna
{

inline double normalAlignmentMagnitudeErrorDegrees(double alignment)
{
    if (!std::isfinite(alignment))
        return std::numeric_limits<double>::quiet_NaN();
    return std::asin(std::clamp(std::abs(alignment), 0.0, 1.0)) * 180.0 / M_PI;
}

inline double normalAlignmentErrorDegrees(const cv::Vec3d& tangent, const cv::Vec3d& normal)
{
    const double tangentLength = std::sqrt(tangent.dot(tangent));
    const double normalLength = std::sqrt(normal.dot(normal));
    if (!std::isfinite(tangentLength) || !std::isfinite(normalLength) || tangentLength <= 1.0e-12 || normalLength <= 1.0e-12) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return normalAlignmentMagnitudeErrorDegrees(tangent.dot(normal) / (tangentLength * normalLength));
}

}  // namespace vc::lasagna
