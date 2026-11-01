#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct Illumination { Scalar strength=Uniform{-.3,.3}, radius=.4;int mode=0;std::vector<double> direction,center; ChannelOptions options; };
} // namespace vcr::transforms
