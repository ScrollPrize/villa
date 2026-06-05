#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace vc3d::line_annotation {

constexpr double kDefaultBottomCrossSliceLineStep = 10.0;
constexpr double kMinBottomCrossSliceLineStep = 0.25;
constexpr double kBottomCrossSliceLineStepFactor = 1.5;
constexpr double kControlPointSnapLinePositionThreshold = 0.25;

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
                                           int linePointCount,
                                           double lineStep = kDefaultBottomCrossSliceLineStep)
{
    if (linePointCount <= 0 || bottomCount <= 0) {
        return 0.0;
    }
    const double maxLinePosition = static_cast<double>(linePointCount - 1);
    lineStep = std::max(kMinBottomCrossSliceLineStep, lineStep);
    const double centerOffset = static_cast<double>(slot - bottomCount / 2) * lineStep;
    return std::clamp(centerPosition + centerOffset, 0.0, maxLinePosition);
}

inline double adjustedBottomCrossSliceLineStep(double currentLineStep,
                                               int scrollSteps,
                                               int linePointCount)
{
    currentLineStep = std::max(kMinBottomCrossSliceLineStep, currentLineStep);
    if (scrollSteps == 0) {
        return currentLineStep;
    }
    const double maxLineStep = std::max(kMinBottomCrossSliceLineStep,
                                        static_cast<double>(std::max(1, linePointCount - 1)));
    const double scale = std::pow(kBottomCrossSliceLineStepFactor, static_cast<double>(scrollSteps));
    return std::clamp(currentLineStep * scale, kMinBottomCrossSliceLineStep, maxLineStep);
}

template <typename LinePositionRange>
inline double snappedControlPointLinePosition(double position,
                                              const LinePositionRange& controlLinePositions,
                                              double threshold = kControlPointSnapLinePositionThreshold)
{
    double bestPosition = position;
    double bestDistance = std::numeric_limits<double>::infinity();
    for (const double controlLinePosition : controlLinePositions) {
        if (!std::isfinite(controlLinePosition)) {
            continue;
        }
        const double distance = std::abs(controlLinePosition - position);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestPosition = controlLinePosition;
        }
    }
    return bestDistance <= threshold ? bestPosition : position;
}

} // namespace vc3d::line_annotation
