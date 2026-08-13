#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace vc::atlas {

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

inline std::string fiberHvTagToPointCollectionString(FiberHvTag tag)
{
    switch (tag) {
    case FiberHvTag::H:
        return "h";
    case FiberHvTag::V:
        return "v";
    case FiberHvTag::Unknown:
    default:
        return {};
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

inline FiberHvTag effectiveFiberHvTag(const FiberHvClassification& automatic,
                                      const std::string& manualTag)
{
    const FiberHvTag manual = fiberHvTagFromString(manualTag);
    if (manual != FiberHvTag::Unknown) {
        return manual;
    }
    return automatic.automaticTag;
}

inline bool firstFiberDisplaysAsH(const FiberHvClassification& first,
                                  const std::string& firstManualTag,
                                  const FiberHvClassification& second,
                                  const std::string& secondManualTag,
                                  bool firstTieBreak = true)
{
    const FiberHvTag firstManual = fiberHvTagFromString(firstManualTag);
    const FiberHvTag secondManual = fiberHvTagFromString(secondManualTag);
    if (firstManual != FiberHvTag::Unknown || secondManual != FiberHvTag::Unknown) {
        if (firstManual == FiberHvTag::H && secondManual != FiberHvTag::H) {
            return true;
        }
        if (secondManual == FiberHvTag::H && firstManual != FiberHvTag::H) {
            return false;
        }
        if (firstManual == FiberHvTag::V && secondManual != FiberHvTag::V) {
            return false;
        }
        if (secondManual == FiberHvTag::V && firstManual != FiberHvTag::V) {
            return true;
        }
    }

    if (std::isfinite(first.horizontalScore) &&
        std::isfinite(second.horizontalScore) &&
        first.horizontalScore != second.horizontalScore) {
        return first.horizontalScore > second.horizontalScore;
    }
    if (std::isfinite(first.verticalScore) &&
        std::isfinite(second.verticalScore) &&
        first.verticalScore != second.verticalScore) {
        return first.verticalScore < second.verticalScore;
    }
    return firstTieBreak;
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
    classification.automaticCertainty = std::clamp(
        std::abs(classification.verticalScore - 0.5) * 2.0,
        0.0,
        1.0);
    classification.valid = true;
    return classification;
}

} // namespace vc::atlas
