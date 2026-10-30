#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct Vignetting { Scalar strength=Uniform{.1,.5}, radius=.8;std::vector<double> center; ChannelOptions options; };
} // namespace vcr::transforms
