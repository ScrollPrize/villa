#pragma once

#include <cstddef>
#include <optional>
#include <vector>
#include <opencv2/core/types.hpp>

namespace vc3d::line_annotation
{

// The loader's exact ordered-membership scan. Shared by save/edit operations
// and read-only visualizations; do not replace it with nearest-point matching.
[[nodiscard]] inline std::optional<std::vector<size_t>> orderedControlPointLineIndices(
    const std::vector<cv::Vec3d>& controlPoints, const std::vector<cv::Vec3d>& linePoints)
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

}  // namespace vc3d::line_annotation
