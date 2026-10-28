#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct PixelDistributionAdaptation {volume_detail::ReferenceVolume reference;Scalar blend=Uniform{.5,1};int mode=0;ChannelOptions options;};
} // namespace vcr::transforms
