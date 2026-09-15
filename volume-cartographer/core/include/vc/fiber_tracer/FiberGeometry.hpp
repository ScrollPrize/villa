#pragma once
#include <opencv2/core/types.hpp>
#include <optional>
#include <vector>
#include <cstddef>

namespace vc::fiber_tracer {
inline std::optional<std::vector<size_t>> orderedControlPointLineIndices(
    const std::vector<cv::Vec3d>& controlPoints,
    const std::vector<cv::Vec3d>& linePoints)
{
    constexpr double kMatchEpsilon = 1.0e-8;
    constexpr double kMaxDistanceSq = kMatchEpsilon * kMatchEpsilon;
    std::vector<size_t> lineIndices;
    lineIndices.reserve(controlPoints.size());
    size_t nextLineIndex = 0;
    for (const auto& control : controlPoints) {
        std::optional<size_t> matched;
        for (size_t j = nextLineIndex; j < linePoints.size(); ++j) {
            const cv::Vec3d delta = control - linePoints[j];
            if (delta.dot(delta) <= kMaxDistanceSq) {
                matched = j;
                break;
            }
        }
        if (!matched) {
            return std::nullopt;
        }
        lineIndices.push_back(*matched);
        nextLineIndex = *matched + 1;
    }
    return lineIndices;
}

inline std::vector<cv::Vec3d> linePointsBetweenOuterControlPoints(
    const std::vector<cv::Vec3d>& linePoints,
    const std::vector<cv::Vec3d>& controlPoints)
{
    if (controlPoints.empty()) {
        return {};
    }
    const auto indices = orderedControlPointLineIndices(controlPoints, linePoints);
    if (!indices || indices->empty()) {
        return {};
    }
    return std::vector<cv::Vec3d>(linePoints.begin() + static_cast<std::ptrdiff_t>(indices->front()),
                                  linePoints.begin() + static_cast<std::ptrdiff_t>(indices->back()) + 1);
}

}
