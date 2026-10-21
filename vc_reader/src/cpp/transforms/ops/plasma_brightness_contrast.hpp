#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct PlasmaBrightnessContrast { Scalar brightness=Uniform{-.2,.2}, contrast=Uniform{0,.3}, scale=.025, persistence=.5;int octaves=5; ChannelOptions options; };
} // namespace vcr::transforms
