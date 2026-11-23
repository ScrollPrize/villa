#pragma once
#include "options.hpp"

namespace vcr::transforms {
// CED on a single channel-first image or volume; each selected channel is independent.
// Scales and time steps use unit voxel spacing. lambda is in squared intensity units.
struct CoherenceEnhancingDiffusion {
    double sigma = 1, rho = 2, lambda = .01, exponent = 2, alpha = .01;
    double step_size = .1;
    int num_steps = 10;
    ChannelOptions options;
};
}
