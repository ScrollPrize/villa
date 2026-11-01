#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct RandomToneCurve { Scalar scale=.1; ChannelOptions options; };
} // namespace vcr::transforms
