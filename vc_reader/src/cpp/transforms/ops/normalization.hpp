#pragma once
#include "options.hpp"

namespace vcr::transforms {
enum class Normalization { MinMax, ZScore, Robust, Percentile, Fixed };
// Statistics use the selected voxels of each channel: those inside the image's
// validity mask when masked (all voxels if it has none, or no valid voxel), and
// for the default z-score only nonzero values. Every voxel is then transformed;
// padding is refilled from the validity mask afterwards.
struct Normalize {
    Normalization mode = Normalization::ZScore;
    bool nonzero = true;           // zscore: statistics exclude zeros, which stay zero
    bool masked = false;
    double eps = 0;                // zscore: lower bound of the standard deviation
    double lower = 1, upper = 99;  // robust/percentile: NumPy linear percentiles
    bool clip = true;              // robust: clip to the percentiles before median/MAD
    bool median = false;           // percentile: (x-median)/(span/2), else (x-lower)/span
    double center = 0, scale = 1;  // fixed: (clip(x, low, high)-center)/scale
    double low = -std::numeric_limits<double>::infinity(), high = std::numeric_limits<double>::infinity();
    ChannelOptions options;
};
} // namespace vcr::transforms
