#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct RingingOvershoot {Scalar cutoff=Uniform{.5,.9},radius=4.;ChannelOptions options;};
} // namespace vcr::transforms
