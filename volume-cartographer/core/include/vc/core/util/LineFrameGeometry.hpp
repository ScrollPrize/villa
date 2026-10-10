#pragma once
#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>

namespace vc::geometry {
// Existing line-view convention: central chord, one-sided at endpoints.
template<class PointAt>
cv::Vec3d lineTangent(size_t count, size_t index, PointAt point)
{
    if (count < 2) return {1,0,0};
    auto v = cv::Vec3d(point(std::min(index+1,count-1))) -
             cv::Vec3d(point(index ? index-1 : 0));
    const double n = std::sqrt(v.dot(v));
    return std::isfinite(n) && n > 1e-12 ? v / n : cv::Vec3d(1,0,0);
}

// Minimal tangent rotation followed by projection. Zero means degenerate.
inline cv::Vec3d transportFrameNormal(cv::Vec3d normal, cv::Vec3d from, cv::Vec3d to)
{
    auto axis = from.cross(to);
    const double s = std::sqrt(axis.dot(axis));
    const double c = std::clamp(from.dot(to), -1.0, 1.0);
    if (s > 1e-12) {
        axis /= s;
        const double angle = std::atan2(s,c);
        normal = normal*std::cos(angle) + axis.cross(normal)*std::sin(angle) +
                 axis*(axis.dot(normal)*(1-std::cos(angle)));
    }
    normal -= to*normal.dot(to);
    const double n = std::sqrt(normal.dot(normal));
    return std::isfinite(n) && n > 1e-12 ? normal/n : cv::Vec3d(0,0,0);
}
}
