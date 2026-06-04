#pragma once

#include <algorithm>

namespace vc3d::line_annotation {

constexpr double kBottomCrossSliceLineStep = 10.0;

inline int shiftScrollLineStepSize(int viewerSliceStepSize)
{
    return std::max(1, viewerSliceStepSize);
}

inline double shiftedLinePosition(double currentPosition,
                                  int scrollSteps,
                                  int viewerSliceStepSize,
                                  int linePointCount)
{
    if (linePointCount <= 0) {
        return currentPosition;
    }
    const double maxLinePosition = static_cast<double>(linePointCount - 1);
    const double delta = static_cast<double>(scrollSteps) *
                         static_cast<double>(shiftScrollLineStepSize(viewerSliceStepSize));
    return std::clamp(currentPosition + delta, 0.0, maxLinePosition);
}

inline double bottomCrossSliceLinePosition(double centerPosition,
                                           int slot,
                                           int bottomCount,
                                           int linePointCount)
{
    if (linePointCount <= 0 || bottomCount <= 0) {
        return 0.0;
    }
    const double maxLinePosition = static_cast<double>(linePointCount - 1);
    const double centerOffset = static_cast<double>(slot - bottomCount / 2) *
                                kBottomCrossSliceLineStep;
    return std::clamp(centerPosition + centerOffset, 0.0, maxLinePosition);
}

} // namespace vc3d::line_annotation
