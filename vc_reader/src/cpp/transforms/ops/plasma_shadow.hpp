#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct PlasmaShadow { Scalar strength=Uniform{.1,.5}, scale=.025, persistence=.5;int octaves=5; ChannelOptions options; };
} // namespace vcr::transforms
