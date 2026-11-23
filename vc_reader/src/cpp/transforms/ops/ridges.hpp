#pragma once
#include "options.hpp"
namespace vcr::transforms {
// Hessian ridge response of a smoothed, normalized scalar image.
struct Ridges { double sigma=6, gamma=1.5, beta1=.5, beta2=.5; ChannelOptions options; };
}
