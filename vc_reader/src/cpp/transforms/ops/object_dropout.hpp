#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct ObjectDropout {std::string mask="mask";std::size_t mask_channel=0;double threshold=0;Scalar drop_probability=.2;volume_detail::EraseValues fill{0,0,true};ChannelOptions options;};
} // namespace vcr::transforms
