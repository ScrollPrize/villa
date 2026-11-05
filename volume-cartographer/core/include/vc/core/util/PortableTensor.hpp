#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#ifndef VC_TRACE_HD
#ifdef __CUDACC__
#define VC_TRACE_HD __host__ __device__
#else
#define VC_TRACE_HD
#endif
#endif
namespace vc::core::portable {
inline constexpr float kFloatEpsilon=1.0e-6f;
template<class V>
VC_TRACE_HD static inline V normalizedFloatOrZero(const V& value)
{
    const float norm2 = value.dot(value);
    if (!(norm2 > kFloatEpsilon * kFloatEpsilon) || !std::isfinite(norm2))
        return {0.0f, 0.0f, 0.0f};
    return value * (1.0f / std::sqrt(norm2));
}
template<class V>
VC_TRACE_HD static inline V principalFloatTensorAxis(
    float a00,
    float a01,
    float a02,
    float a11,
    float a12,
    float a22,
    const V& hint)
{
    if (!std::isfinite(a00) || !std::isfinite(a01) || !std::isfinite(a02) ||
        !std::isfinite(a11) || !std::isfinite(a12) || !std::isfinite(a22)) {
        return {0.0f, 0.0f, 0.0f};
    }
    auto fallback = [&]() {
        V result{1.0f, 0.0f, 0.0f};
        if (a11 > a00 && a11 >= a22)
            result = {0.0f, 1.0f, 0.0f};
        else if (a22 > a00 && a22 > a11)
            result = {0.0f, 0.0f, 1.0f};
        return result;
    };

    V axis{};
    const float offDiagonal = a01 * a01 + a02 * a02 + a12 * a12;
    if (offDiagonal <= kFloatEpsilon * kFloatEpsilon) {
        axis = fallback();
    } else {
        const float q = (a00 + a11 + a22) / 3.0f;
        const float b00 = a00 - q;
        const float b11 = a11 - q;
        const float b22 = a22 - q;
        const float p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0f * offDiagonal;
        const float p = std::sqrt(std::max(0.0f, p2 / 6.0f));
        if (!(p > kFloatEpsilon)) {
            axis = fallback();
        } else {
            const float invP = 1.0f / p;
            const float c00 = b00 * invP;
            const float c01 = a01 * invP;
            const float c02 = a02 * invP;
            const float c11 = b11 * invP;
            const float c12 = a12 * invP;
            const float c22 = b22 * invP;
            const float detC =
                c00 * (c11 * c22 - c12 * c12) -
                c01 * (c01 * c22 - c12 * c02) +
                c02 * (c01 * c12 - c11 * c02);
            const float phi = std::acos(std::clamp(0.5f * detC, -1.0f, 1.0f)) / 3.0f;
            const float lambda = q + 2.0f * p * std::cos(phi);
            const V row0{a00 - lambda, a01, a02};
            const V row1{a01, a11 - lambda, a12};
            const V row2{a02, a12, a22 - lambda};
            const std::array<V, 3> candidates{
                row0.cross(row1), row0.cross(row2), row1.cross(row2)};
            float bestNorm2 = -1.0f;
            for (const auto& candidate : candidates) {
                const float norm2 = candidate.dot(candidate);
                if (norm2 > bestNorm2) {
                    bestNorm2 = norm2;
                    axis = candidate;
                }
            }
            if (!(bestNorm2 > kFloatEpsilon * kFloatEpsilon))
                axis = fallback();
        }
    }

    axis = normalizedFloatOrZero(axis);
    if (axis.dot(axis) <= kFloatEpsilon * kFloatEpsilon)
        return {0.0f, 0.0f, 0.0f};
    const V normalizedHint = normalizedFloatOrZero(hint);
    if (normalizedHint.dot(normalizedHint) > kFloatEpsilon * kFloatEpsilon) {
        if (axis.dot(normalizedHint) < 0.0f)
            axis *= -1.0f;
    } else if (axis[2] < 0.0f) {
        axis *= -1.0f;
    }
    return axis;
}
}
