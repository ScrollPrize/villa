#pragma once
#include "options.hpp"

namespace vcr::transforms {
// Separable Gaussian blur along an axis and its perpendicular direction(s).
// Each pass samples a clamped grid using bilinear or trilinear interpolation.
// The axis is fixed (yx/zyx), the first row of a Vectors target (so earlier
// geometry rotates it), or uniformly random when neither is given.
struct OrientedBlur {
    Scalar sigma_along = Uniform{0, 2}, sigma_across = Uniform{0, 1};
    std::vector<double> axis;
    std::string axis_from;
    ChannelOptions options;
};
} // namespace vcr::transforms
