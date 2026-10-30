#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct ForegroundDropout {std::string mask="mask";std::size_t mask_channel=0;double threshold=0;Shape size;Scalar count=Uniform{1,3};volume_detail::EraseValues fill;ChannelOptions options;};
} // namespace vcr::transforms
