#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct ShotNoise { Scalar photons=Uniform{50,200};bool shared_noise=false; ChannelOptions options; };
} // namespace vcr::transforms
