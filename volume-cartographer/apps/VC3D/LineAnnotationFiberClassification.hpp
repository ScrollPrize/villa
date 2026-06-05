#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace vc3d::line_annotation {

enum class FiberHvTag {
    Unknown,
    H,
    V,
};

struct FiberHvClassification {
    double zDistance = 0.0;
    double fiberLength = 0.0;
    double horizontalScore = 0.0;
    double verticalScore = 0.0;
    double automaticCertainty = 0.0;
    FiberHvTag automaticTag = FiberHvTag::Unknown;
    bool valid = false;
};

inline std::string fiberHvTagToString(FiberHvTag tag)
{
    switch (tag) {
    case FiberHvTag::H:
        return "H";
    case FiberHvTag::V:
        return "V";
    case FiberHvTag::Unknown:
    default:
        return "unknown";
    }
}

inline FiberHvTag fiberHvTagFromString(const std::string& tag)
{
    if (tag == "H" || tag == "h" || tag == "horizontal") {
        return FiberHvTag::H;
    }
    if (tag == "V" || tag == "v" || tag == "vertical") {
        return FiberHvTag::V;
    }
    return FiberHvTag::Unknown;
}

inline double fiberLineLengthVx(const std::vector<cv::Vec3d>& points)
{
    double length = 0.0;
    for (size_t i = 1; i < points.size(); ++i) {
        const cv::Vec3d delta = points[i] - points[i - 1];
        const double step = std::sqrt(delta.dot(delta));
        if (std::isfinite(step)) {
            length += step;
        }
    }
    return length;
}

inline FiberHvClassification classifyFiberHv(const std::vector<cv::Vec3d>& points)
{
    FiberHvClassification classification;
    if (points.size() < 2) {
        return classification;
    }

    const double length = fiberLineLengthVx(points);
    classification.fiberLength = length;
    if (!std::isfinite(length) || length <= 0.0) {
        return classification;
    }

    const double zDistance = std::abs(points.back()[2] - points.front()[2]);
    classification.zDistance = zDistance;
    classification.verticalScore = std::clamp(zDistance / length, 0.0, 1.0);
    classification.horizontalScore = 1.0 - classification.verticalScore;
    classification.automaticTag = classification.verticalScore >= 0.5
        ? FiberHvTag::V
        : FiberHvTag::H;
    classification.automaticCertainty = std::clamp(std::abs(classification.verticalScore - 0.5) * 2.0,
                                                   0.0,
                                                   1.0);
    classification.valid = true;
    return classification;
}

} // namespace vc3d::line_annotation
