#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct MultiplicativeNoise { Scalar low=.9,high=1.1;bool shared_noise=false; ChannelOptions options; };
} // namespace vcr::transforms
