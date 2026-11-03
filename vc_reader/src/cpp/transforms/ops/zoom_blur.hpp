#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct ZoomBlur { Scalar strength=Uniform{0,.2};std::vector<double> center;int samples=9; ChannelOptions options; };
} // namespace vcr::transforms
