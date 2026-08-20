#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace vc::fiber_tracer::detail {

inline constexpr size_t kFiberAnchorPeakGaussianTableIntervals = 2048;
inline constexpr float kFiberAnchorPeakGaussianTableMaximumExponent = 8.0F;
using FiberAnchorPeakGaussianTable =
    std::array<float, kFiberAnchorPeakGaussianTableIntervals + 1>;

[[nodiscard]] const FiberAnchorPeakGaussianTable&
fiberAnchorPeakGaussianTable();

[[nodiscard]] inline float fiberAnchorPeakGaussian(
    const FiberAnchorPeakGaussianTable& table,
    float exponent)
{
    if (!std::isfinite(exponent) || exponent < 0.0F ||
        exponent > kFiberAnchorPeakGaussianTableMaximumExponent) {
        return std::exp(-exponent);
    }
    constexpr float scale =
        static_cast<float>(kFiberAnchorPeakGaussianTableIntervals) /
        kFiberAnchorPeakGaussianTableMaximumExponent;
    const size_t index = std::min(
        kFiberAnchorPeakGaussianTableIntervals,
        static_cast<size_t>(exponent * scale + 0.5F));
    return table[index];
}

}  // namespace vc::fiber_tracer::detail
