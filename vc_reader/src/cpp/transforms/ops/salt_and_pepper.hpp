#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct SaltAndPepper { Scalar amount=.01,salt_fraction=.5;bool shared_noise=true; ChannelOptions options; };
} // namespace vcr::transforms
