#pragma once
#include "options.hpp"

namespace vcr::transforms {
// Specialized tracing operations share the same geometry/target machinery.
struct StripGeometry {
    Scalar shift_x = 0., shift_y = 0., degrees = 0., scale = 1., shear_x = 0., shear_y = 0.;
    Scalar smooth_offset = 0., stride = 16.; double p_flip_x = 0, p_flip_y = 0, p = 1; Shape shape;
};
struct SurfacePerturb {
    std::string on = "surface"; Scalar blobs = Uniform{1, 3}, sigma_fraction = Uniform{.04, .10}, amplitude = Uniform{.25, 1.25};
    std::size_t points_affected = 10; double radius_sigma = 2.5, max_displacement = 1.5, p = .35;
};
} // namespace vcr::transforms
