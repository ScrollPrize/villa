#pragma once

#include <algorithm>
#include <cmath>

namespace vc::lasagna { struct LineOptimizationConfig; }

namespace vc3d::line_annotation {

inline constexpr int kDefaultExtrapolationDistanceBaseVoxels = 1200;

struct InitialLineDiscretization {
    int segmentsPerSide = 1;
    double segmentLength = 32.0;
};

inline InitialLineDiscretization initialLineDiscretization(int totalLengthVx)
{
    constexpr double segmentLength = 32.0;
    const double halfLength = std::max(1, totalLengthVx) * 0.5;
    const int segmentsPerSide = std::max(
        1, static_cast<int>(std::ceil(halfLength / segmentLength)));
    return {segmentsPerSide, halfLength / static_cast<double>(segmentsPerSide)};
}

// Shared by the GUI request builder and the real-data reoptimization CLI.
void configureFiberModeLasagnaDefaults(vc::lasagna::LineOptimizationConfig& config,
                                      int extrapolationDistanceBaseVoxels);

} // namespace vc3d::line_annotation
