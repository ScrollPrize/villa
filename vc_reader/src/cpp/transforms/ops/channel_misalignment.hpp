#pragma once
#include "options.hpp"
#include "spatial.hpp"

namespace vcr::transforms {
struct ChannelMisalignment {Affine affine;ChannelOptions options;};
} // namespace vcr::transforms
