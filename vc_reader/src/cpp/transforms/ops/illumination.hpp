#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct Illumination { Scalar strength=Uniform{-.3,.3}, radius=.4;int mode=0; /* 0 linear, 1 corner, 2 gaussian */std::vector<double> direction,center; ChannelOptions options; };
} // namespace vcr::transforms
