#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct BrightnessMultiply { Scalar factor = Uniform{.75, 1.25}; ChannelOptions options; };
struct BrightnessAdd { Scalar shift = Normal{0, .1}; ChannelOptions options; };
struct Contrast { Scalar factor = Uniform{.75, 1.25}; bool preserve_range = true; ChannelOptions options; };
struct Gamma { Scalar gamma = Uniform{.7, 1.5}; double p_invert = 0, p_retain_stats = 0; ChannelOptions options; };
struct Invert { ChannelOptions options; };
struct GaussianNoise { Scalar std = Uniform{0, .1}; bool shared_noise = false; ChannelOptions options; };
struct RicianNoise { Scalar std = Uniform{0, .1}; bool shared_noise = false; ChannelOptions options; };
struct ClipOutliers { Scalar lower = .2, upper = 99.8; double p_retain_std = 1; ChannelOptions options; };
struct GaussianBlur { Scalar sigma = Uniform{.3, 1.5}; bool shared_axes = true; Boundary boundary = Boundary::Mirror; double truncate = 6; bool full_width = true; ChannelOptions options; };
struct MedianFilter { Scalar size = Uniform{3, 7}; bool quantize = false; ChannelOptions options; };
struct Sharpen { Scalar strength = Uniform{.1, .5}; double p_clamp = .5; ChannelOptions options; };
struct BrightnessGradient { Scalar strength = Uniform{-.5, .5}; bool mean_centered = true, clip = false; LocalOptions local; ChannelOptions options; };
struct LocalContrast { Scalar factor = Uniform{.5, 1.5}; LocalOptions local; ChannelOptions options; };
struct LocalGamma { Scalar gamma = Uniform{.5, 1.5}; LocalOptions local; ChannelOptions options; };
struct LocalSmoothing { Scalar strength = Uniform{.5, 1}, sigma = Uniform{1, 3}; LocalOptions local; ChannelOptions options; };
struct BlankRectangle {
    std::vector<Scalar> size{16.}; Scalar count = Uniform{1, 5}, value = 0.;
    std::string statistic = "mean"; bool square = false; ChannelOptions options;
};
struct SliceIllumination {
    char axis = 'z'; Scalar defects = Uniform{1, 5}, width = Uniform{1, 10}, reduction = Uniform{.1, .5};
    Scalar base_p = Uniform{.1, .3}, base_reduction = Uniform{.1, .5}; ChannelOptions options;
};
struct Smear { char axis = 'z'; std::vector<int> shift{10, 0}; double alpha = .5; int previous = 1; ChannelOptions options; };
} // namespace vcr::transforms
