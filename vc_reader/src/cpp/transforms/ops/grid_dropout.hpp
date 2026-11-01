#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct GridDropout {Shape unit_size;Scalar ratio=.25;std::string axes;volume_detail::EraseValues fill;ChannelOptions options;};
} // namespace vcr::transforms
