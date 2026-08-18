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

// Resolves a symmetric 3x3 tensor in closed form. Ambiguous top eigenvalues
// remain non-unique; the iterative solver is used only when a clear-gap
// eigenvector cannot be reconstructed to a bounded residual.
[[nodiscard]] FiberPrincipalAxis principalFiberAxisClosedForm(
    const cv::Matx33d& tensor,
    bool* usedIterativeFallback = nullptr);

}  // namespace vc::fiber_tracer
