#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "vc_object_store/geometry/FiberAxialAlignment.hpp"

namespace vc::fiber_tracer {
// Arc-weighted midpoint quadrature; invariant to either tangent's sign.
template<class AxisAt>
double fiberAxialAlignment(const std::vector<cv::Vec3d>& points, AxisAt axisAt, double spacing=16.) {
    return vc::geometry::fiberAxialAlignment(points,axisAt,[](const auto& p){return cv::norm(p);},spacing);
}
}
