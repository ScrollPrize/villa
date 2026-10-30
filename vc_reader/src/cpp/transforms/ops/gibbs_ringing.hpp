#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct GibbsRinging {Scalar cutoff=Uniform{.5,.9};ChannelOptions options;};
} // namespace vcr::transforms
