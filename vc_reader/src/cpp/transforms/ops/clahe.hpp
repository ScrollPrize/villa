#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct CLAHE {Scalar clip_limit=2.;Shape tile_shape;int bins=256;ChannelOptions options;};
} // namespace vcr::transforms
