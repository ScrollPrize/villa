#pragma once
#include "options.hpp"
#include "spatial.hpp"

namespace vcr::transforms {
inline Affine default_channel_misalignment() { Affine a; a.degrees=Uniform{-5,5}; a.translate=Uniform{-2,2}; a.scale=Uniform{.95,1.05}; a.shared_scale=false; return a; }
struct ChannelMisalignment {Affine affine=default_channel_misalignment();ChannelOptions options;};
} // namespace vcr::transforms
