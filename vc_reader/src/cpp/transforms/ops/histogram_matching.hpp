#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct HistogramMatching {volume_detail::ReferenceVolume reference;Scalar blend=Uniform{.5,1};int bins=256;ChannelOptions options;};
} // namespace vcr::transforms
