#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct AdditiveNoise { Scalar scale=.05,mean=0.,alpha=2.,beta=2.;int distribution=0; /* 0 uniform, 1 gaussian, 2 laplace, 3 beta */bool shared_noise=false; ChannelOptions options; };
} // namespace vcr::transforms
