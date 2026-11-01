#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct MotionBlur { Scalar length=Uniform{1,5};std::vector<double> direction;int samples=9; ChannelOptions options; };
} // namespace vcr::transforms
