#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct FDA {volume_detail::ReferenceVolume reference;std::string reference_key;Scalar blend=Uniform{.5,1},fraction=.1;ChannelOptions options;};
} // namespace vcr::transforms
