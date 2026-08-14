#pragma once

#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

struct FiberPrincipalAxis {
    cv::Vec3d axis{0.0, 0.0, 0.0};
    double largestEigenvalue = 0.0;
    double secondEigenvalue = 0.0;
    bool valid = false;
    bool unique = false;
};

[[nodiscard]] cv::Vec3d canonicalFiberAxis(cv::Vec3d axis);

[[nodiscard]] cv::Matx33d fiberAxisTensor(const cv::Vec3d& unitAxis, double weight = 1.0);

[[nodiscard]] FiberPrincipalAxis principalFiberAxis(const cv::Matx33d& tensor);

}  // namespace vc::fiber_tracer
