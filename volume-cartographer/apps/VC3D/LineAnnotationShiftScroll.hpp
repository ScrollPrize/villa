#pragma once

#include "FiberSliceGeometry.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/core.hpp>

namespace vc3d::line_annotation {

constexpr double kDefaultBottomCrossSliceLineStep = 10.0;
constexpr double kMinBottomCrossSliceLineStep = 0.25;
constexpr double kBottomCrossSliceLineStepFactor = 1.5;
// Legacy snap threshold in line positions (vertex indices); the current cut
// uses the arclength threshold below, this one remains for callers without an
// arclength map (intersection inspection follow slices).
constexpr double kControlPointSnapLinePositionThreshold = 0.25;

// Along-line motion in the current cut is measured in base-voxel arclength,
// not vertex indices: the dense line is 4 vx per vertex in trace spans and
// ~32 vx in cspline/Lasagna spans, so index steps moved 8x faster in one than
// the other. One shift+wheel notch (at slice step size 1) moves one strip
// column; Space snaps onto a control within a quarter of a column.
constexpr double kShiftScrollLineStepBaseVoxels =
    vc::lasagna::kLineViewAlongSamplingDistanceBaseVoxels;
constexpr double kControlPointSnapArclengthBaseVoxels =
    vc::lasagna::kLineViewAlongSamplingDistanceBaseVoxels * 0.25;

// True when `cumulativeArclengths` can map positions of a line with
// `linePointCount` points (one entry per vertex, at least two).
inline bool lineArclengthsUsable(const std::vector<double>& cumulativeArclengths,
                                 size_t linePointCount)
{
    return cumulativeArclengths.size() == linePointCount && linePointCount >= 2 &&
           std::isfinite(cumulativeArclengths.back()) && cumulativeArclengths.back() > 0.0;
}

// The map on its own: at least two finite, increasing-to-positive entries.
inline bool lineArclengthsUsable(const std::vector<double>& cumulativeArclengths)
{
    return lineArclengthsUsable(cumulativeArclengths, cumulativeArclengths.size());
}

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

// Arclength-based sibling of shiftedLinePosition: each notch moves
// kShiftScrollLineStepBaseVoxels * step size along the optimized polyline.
// Without a usable arclength map it falls back to the index-based step.
inline double shiftedLinePositionByArclength(double currentPosition,
                                             int scrollSteps,
                                             int viewerSliceStepSize,
                                             const std::vector<double>& cumulativeArclengths)
{
    const int linePointCount = static_cast<int>(cumulativeArclengths.size());
    if (!lineArclengthsUsable(cumulativeArclengths)) {
        return shiftedLinePosition(currentPosition, scrollSteps, viewerSliceStepSize, linePointCount);
    }
    const double maxLinePosition = static_cast<double>(linePointCount - 1);
    const double deltaBaseVoxels = static_cast<double>(scrollSteps) *
                                   static_cast<double>(shiftScrollLineStepSize(viewerSliceStepSize)) *
                                   kShiftScrollLineStepBaseVoxels;
    const double currentArclength = vc3d::fiber_slice::arclengthAtLinePosition(
        cumulativeArclengths, std::clamp(currentPosition, 0.0, maxLinePosition));
    const double targetArclength = std::clamp(
        currentArclength + deltaBaseVoxels, 0.0, cumulativeArclengths.back());
    return std::clamp(vc3d::fiber_slice::linePositionAtArclength(cumulativeArclengths, targetArclength),
                      0.0,
                      maxLinePosition);
}

inline cv::Vec3f shiftedPlaneOriginAlongNormal(const cv::Vec3f& currentOrigin,
                                               const cv::Vec3f& planeNormal,
                                               int scrollSteps,
                                               int viewerSliceStepSize)
{
    const float n = cv::norm(planeNormal);
    if (!std::isfinite(currentOrigin[0]) ||
        !std::isfinite(currentOrigin[1]) ||
        !std::isfinite(currentOrigin[2]) ||
        !std::isfinite(planeNormal[0]) ||
        !std::isfinite(planeNormal[1]) ||
        !std::isfinite(planeNormal[2]) ||
        n <= 1.0e-6f) {
        return currentOrigin;
    }
    const float delta = static_cast<float>(scrollSteps * shiftScrollLineStepSize(viewerSliceStepSize));
    return currentOrigin + planeNormal * (delta / n);
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

// Arclength-based sibling of snappedControlPointLinePosition: snaps when the
// nearest control is within `thresholdBaseVoxels` along the optimized
// polyline. Without a usable arclength map it falls back to the index-based
// quarter-position threshold.
template <typename LinePositionRange>
inline double snappedControlPointLinePositionByArclength(
    double position,
    const LinePositionRange& controlLinePositions,
    const std::vector<double>& cumulativeArclengths,
    double thresholdBaseVoxels = kControlPointSnapArclengthBaseVoxels)
{
    if (!lineArclengthsUsable(cumulativeArclengths) || !std::isfinite(position)) {
        return snappedControlPointLinePosition(position, controlLinePositions);
    }
    const double targetArclength =
        vc3d::fiber_slice::arclengthAtLinePosition(cumulativeArclengths, position);
    double bestPosition = position;
    double bestDistance = std::numeric_limits<double>::infinity();
    for (const double controlLinePosition : controlLinePositions) {
        if (!std::isfinite(controlLinePosition)) {
            continue;
        }
        const double distance = std::abs(
            vc3d::fiber_slice::arclengthAtLinePosition(cumulativeArclengths, controlLinePosition) -
            targetArclength);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestPosition = controlLinePosition;
        }
    }
    return bestDistance <= thresholdBaseVoxels ? bestPosition : position;
}

} // namespace vc3d::line_annotation
