#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct Flip { std::string axes; double p_axis = .5, p = 1; };
struct Rotate90 { std::vector<std::string> planes; Choice turns{{1, 2, 3}}; bool resize = false; double p = 1; };
struct Transpose { std::string order; double p = 1; };
struct CropPad { Shape shape; bool random = false; double p = 1; };
struct Affine {
    Scalar degrees = 0., scale = 1., translate = 0., shear = 0.;
    std::vector<double> scale_factors;
    std::string rotation_axes = "zyx"; bool shared_scale = true, border = false; Shape shape; double p = 1;
};
struct Elastic { Scalar sigma = Uniform{5, 10}, magnitude = Uniform{1, 5}; bool border = false; double p = 1; };
struct SheetCompression { Scalar strength = Uniform{.1, .3}, intensity_sigma = 1., smoothing = Uniform{3, 8}; std::string axes, image = "image"; double threshold = -1, p = 1; };
// Resolution simulation deliberately has its own image/target routing, distinct
// from a geometric deformation (point coordinates do not move).
struct LowResolution { Scalar scale = Uniform{.5, 1}; std::string axes = "zyx"; bool shared_axes = true; ChannelOptions options; };
struct ThickSlice { Scalar scale = Uniform{.25, .6}; std::string axes; bool labels = true; double p = 1; };
} // namespace vcr::transforms
