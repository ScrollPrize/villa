#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer::detail
{

constexpr size_t kMaximumExactFloatGridExtent = size_t{1} << 24;

inline bool floatGridShapeExactlyRepresentable(
    const std::array<size_t, 3>& shapeZYX) noexcept
{
    for (const size_t extent : shapeZYX) {
        if (extent == 0 || extent > kMaximumExactFloatGridExtent)
            return false;
    }
    return true;
}

inline float checkedScaleFloatValue(
    float value, double scale, std::string_view name)
{
    const double scaled = static_cast<double>(value) * scale;
    if (!std::isfinite(scaled) ||
        std::abs(scaled) > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::overflow_error(std::string(name) + " is not finite float32");
    }
    return static_cast<float>(scaled);
}

inline float checkedScaleFloatIndex(
    size_t value, double scale, std::string_view name)
{
    if (value > kMaximumExactFloatGridExtent)
        throw std::overflow_error(std::string(name) + " is not exact float32");
    return checkedScaleFloatValue(static_cast<float>(value), scale, name);
}

inline cv::Vec3f checkedFloatPosition(
    const cv::Vec3f& value, std::string_view name)
{
    if (!std::isfinite(value[0]) || !std::isfinite(value[1]) ||
        !std::isfinite(value[2])) {
        throw std::overflow_error(std::string(name) + " is not finite float32");
    }
    return value;
}

inline cv::Vec3f checkedScaleFloatPosition(
    const cv::Vec3f& value, double scale, std::string_view name)
{
    return {
        checkedScaleFloatValue(value[0], scale, name),
        checkedScaleFloatValue(value[1], scale, name),
        checkedScaleFloatValue(value[2], scale, name),
    };
}

}  // namespace vc::fiber_tracer::detail
