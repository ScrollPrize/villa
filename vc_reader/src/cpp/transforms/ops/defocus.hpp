#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct Defocus { Scalar radius=Uniform{1,3}; ChannelOptions options; };
} // namespace vcr::transforms
