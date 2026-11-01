#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct BiasField { Scalar std=Uniform{0,.3}, scale=.025; ChannelOptions options; };
} // namespace vcr::transforms
