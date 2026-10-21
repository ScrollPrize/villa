#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct UnsharpMask {Scalar strength=Uniform{.1,.5},sigma=Uniform{.5,1.5};ChannelOptions options;};
} // namespace vcr::transforms
