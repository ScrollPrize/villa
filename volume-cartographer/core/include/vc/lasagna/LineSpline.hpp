#pragma once

#include <optional>
#include <vector>

#include <opencv2/core/types.hpp>

namespace vc::lasagna {

struct LineSplineRequest {
    std::vector<cv::Vec3d> controlPoints;
    std::optional<cv::Vec3d> leftDirection;
    std::optional<cv::Vec3d> rightDirection;
    double sampleSpacing = 1.0;
};

struct LineSplineResult {
    std::vector<cv::Vec3d> points;
    std::vector<int> controlPointIndices;
};

[[nodiscard]] LineSplineResult interpolateLineControlPoints(const LineSplineRequest& request);

} // namespace vc::lasagna
