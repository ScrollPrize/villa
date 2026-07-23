#pragma once

#include <opencv2/core/types.hpp>

#include <QPoint>
#include <QPointF>
#include <QString>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

class CChunkedVolumeViewer;
class PlaneSurface;
class QuadSurface;
class QWidget;

namespace vc3d::line_annotation {

enum class GeneratedControlPointContextResult {
    None,
    Handled,
    NewLineAnnotationRequested,
};

enum class GeneratedCurrentLineMarkerState {
    Neutral,
    Allowed,
    Blocked,
};

struct GeneratedOverlay {
    struct ControlPointMarker {
        cv::Vec3f point{std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN()};
        double linePosition = std::numeric_limits<double>::quiet_NaN();
        struct BranchLink {
            uint64_t fiberId = 0;
            int controlPointIndex = -1;
            bool pending = false;
        };

        size_t controlIndex = std::numeric_limits<size_t>::max();
        bool isSeed = false;
        bool hasBranches = false;
        bool hasPendingLinks = false;
        bool isLinkCandidate = false;
        std::vector<uint64_t> branchIds;
        std::vector<BranchLink> branchLinks;
    };

    struct PredSnapMarker {
        cv::Vec3f controlPoint{std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::quiet_NaN()};
        cv::Vec3f snapPoint{std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::quiet_NaN()};
        double linePosition = std::numeric_limits<double>::quiet_NaN();
        size_t controlIndex = std::numeric_limits<size_t>::max();
        bool manual = false;
    };

    struct BranchLinkMarker {
        uint64_t linkedFiberId = 0;
        cv::Vec3f localControlPoint{std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::quiet_NaN()};
        cv::Vec3f linkedControlPoint{std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::quiet_NaN()};
        cv::Vec3f localDirection{std::numeric_limits<float>::quiet_NaN(),
                                 std::numeric_limits<float>::quiet_NaN(),
                                 std::numeric_limits<float>::quiet_NaN()};
        cv::Vec3f linkedDirection{std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::quiet_NaN()};
        cv::Vec3f planePoint{std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN()};
        bool estimated = false;
    };

    struct FiberIntersectionMarker {
        cv::Vec3f point{std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN()};
        uint64_t fiberId = 0;
        int segmentIndex = -1;
        double arclength = std::numeric_limits<double>::quiet_NaN();
        double distance = std::numeric_limits<double>::quiet_NaN();
        bool projectedBranchLink = false;
        bool pendingBranchLink = false;
        bool isLinkCandidateFiber = false;
        std::optional<cv::Vec3f> connectorStart;
    };

    std::vector<cv::Vec3f> linePoints;
    std::vector<std::vector<cv::Vec3f>> branchLinePoints;
    cv::Vec3f seedPoint{std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN()};
    cv::Vec3f pointMarker{std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::quiet_NaN()};
    int seedLineIndex = -1;
    std::vector<double> markerLinePositions;
    std::vector<ControlPointMarker> controlPoints;
    std::vector<PredSnapMarker> predSnapPoints;
    std::vector<BranchLinkMarker> branchLinks;
    std::vector<FiberIntersectionMarker> fiberIntersections;
    double currentLinePosition = std::numeric_limits<double>::quiet_NaN();
    GeneratedCurrentLineMarkerState currentLineMarkerState =
        GeneratedCurrentLineMarkerState::Neutral;
    bool emphasizedPointMarker = false;
    bool useSurfaceCenterLine = false;
    bool currentLineMarkerAsCross = false;
};

struct GeneratedSpanAlignmentMetric {
    int spanIndex = 0;
    int firstControlIndex = 0;
    int secondControlIndex = 0;
    double firstControlLinePosition = std::numeric_limits<double>::quiet_NaN();
    double secondControlLinePosition = std::numeric_limits<double>::quiet_NaN();
    double maxErrorDegrees = 0.0;
    bool available = false;
    bool pending = false;
    std::string error;
};

struct GeneratedViews {
    std::string lineSurfaceName;
    QString lineSurfaceTitle;
    std::string lineSideSliceName;
    QString lineSideSliceTitle;
    std::shared_ptr<QuadSurface> lineSideSlice;
    std::string currentCutName;
    std::shared_ptr<PlaneSurface> currentCutSurface;
    std::string sideCutName;
    std::shared_ptr<PlaneSurface> sideCutSurface;
    std::vector<cv::Vec3f> linePoints;
    std::vector<cv::Vec3f> lineUpVectors;
    std::vector<std::vector<cv::Vec3f>> branchLinePoints;
    cv::Vec3f seedPoint{std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN()};
    cv::Vec3f focusPoint{std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN()};
    int seedLineIndex = -1;
    int initialCenterIndex = 0;
    std::optional<std::pair<double, double>> initialStripLinePositionRange;
    bool initialCurrentCutFollowsStripMouse = true;
    std::vector<GeneratedOverlay::ControlPointMarker> controlPoints;
    std::vector<GeneratedOverlay::PredSnapMarker> predSnapPoints;
    std::vector<GeneratedOverlay::BranchLinkMarker> branchLinks;
    std::vector<GeneratedOverlay::FiberIntersectionMarker> fiberIntersections;
    std::vector<GeneratedSpanAlignmentMetric> spanAlignmentMetrics;
};

struct GeneratedControlPointLinePositionIndex {
    std::vector<size_t> sortedControlIndices;
};

enum class GeneratedCutRotationAxis {
    Horizontal,
    Vertical,
};

struct GeneratedCutFrame {
    cv::Vec3f horizontal{std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN()};
    cv::Vec3f vertical{std::numeric_limits<float>::quiet_NaN(),
                       std::numeric_limits<float>::quiet_NaN(),
                       std::numeric_limits<float>::quiet_NaN()};
    cv::Vec3f normal{std::numeric_limits<float>::quiet_NaN(),
                     std::numeric_limits<float>::quiet_NaN(),
                     std::numeric_limits<float>::quiet_NaN()};
};

struct GeneratedLineViewNavigationState {
    double currentLinePosition = 0.0;
    double bottomCenterPosition = 0.0;
    double bottomSliceLineStep = 10.0;
    cv::Matx33f currentCutManualRotation = cv::Matx33f::eye();
    bool currentCutManualRotationActive = false;
};

inline bool finiteGeneratedPoint(const cv::Vec3f& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

inline bool finiteStoredPoint(const cv::Vec3d& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

inline bool storedPointsApproximatelyEqual(const cv::Vec3d& a,
                                           const cv::Vec3d& b,
                                           double tolerance = 1.0e-6)
{
    if (!finiteStoredPoint(a) || !finiteStoredPoint(b)) {
        return false;
    }
    const cv::Vec3d delta = a - b;
    return delta.dot(delta) <= tolerance * tolerance;
}

inline std::optional<cv::Vec3d> storedSinglePointFiberSeed(
    const std::vector<cv::Vec3d>& controlPoints,
    const std::vector<cv::Vec3d>& linePoints)
{
    std::optional<cv::Vec3d> controlSeed;
    size_t finiteControlCount = 0;
    for (const cv::Vec3d& point : controlPoints) {
        if (!finiteStoredPoint(point)) {
            continue;
        }
        ++finiteControlCount;
        if (finiteControlCount == 1) {
            controlSeed = point;
        }
    }

    std::optional<cv::Vec3d> lineSeed;
    size_t finiteLineCount = 0;
    for (const cv::Vec3d& point : linePoints) {
        if (!finiteStoredPoint(point)) {
            continue;
        }
        ++finiteLineCount;
        if (finiteLineCount == 1) {
            lineSeed = point;
        }
    }

    if (finiteControlCount > 1 || finiteLineCount > 1) {
        return std::nullopt;
    }
    if (!controlSeed && !lineSeed) {
        return std::nullopt;
    }
    if (controlSeed && lineSeed &&
        !storedPointsApproximatelyEqual(*controlSeed, *lineSeed)) {
        return std::nullopt;
    }
    return controlSeed ? controlSeed : lineSeed;
}

inline cv::Vec3f normalizedGeneratedVectorOrNan(const cv::Vec3f& vector)
{
    const float n = cv::norm(vector);
    if (!finiteGeneratedPoint(vector) || n <= 1.0e-6f) {
        return {std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()};
    }
    return vector * (1.0f / n);
}

inline cv::Vec3f generatedMatrixColumn(const cv::Matx33f& matrix, int column)
{
    return {matrix(0, column), matrix(1, column), matrix(2, column)};
}

inline cv::Matx33f generatedCutAxisRotation(GeneratedCutRotationAxis axis, float radians)
{
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    if (axis == GeneratedCutRotationAxis::Horizontal) {
        return {1.0f, 0.0f, 0.0f,
                0.0f, c, -s,
                0.0f, s, c};
    }
    return {c, 0.0f, s,
            0.0f, 1.0f, 0.0f,
            -s, 0.0f, c};
}

inline cv::Matx33f accumulatedGeneratedCutRotation(const cv::Matx33f& current,
                                                   GeneratedCutRotationAxis axis,
                                                   float radians)
{
    return current * generatedCutAxisRotation(axis, radians);
}

inline GeneratedCutFrame generatedCutFrameWithManualRotation(const cv::Vec3f& tangent,
                                                             const cv::Vec3f& upHint,
                                                             const cv::Matx33f& manualRotation)
{
    const cv::Vec3f normal = normalizedGeneratedVectorOrNan(tangent);
    cv::Vec3f vertical = upHint - normal * upHint.dot(normal);
    vertical = normalizedGeneratedVectorOrNan(vertical);
    if (!finiteGeneratedPoint(normal) || !finiteGeneratedPoint(vertical)) {
        return {};
    }
    const cv::Vec3f horizontal = normalizedGeneratedVectorOrNan(vertical.cross(normal));
    if (!finiteGeneratedPoint(horizontal)) {
        return {};
    }

    const cv::Matx33f base(horizontal[0], vertical[0], normal[0],
                           horizontal[1], vertical[1], normal[1],
                           horizontal[2], vertical[2], normal[2]);
    const cv::Matx33f rotated = base * manualRotation;
    GeneratedCutFrame frame;
    frame.horizontal = normalizedGeneratedVectorOrNan(generatedMatrixColumn(rotated, 0));
    frame.vertical = normalizedGeneratedVectorOrNan(generatedMatrixColumn(rotated, 1));
    frame.normal = normalizedGeneratedVectorOrNan(generatedMatrixColumn(rotated, 2));
    return frame;
}

inline bool generatedCutFrameIsOrthonormal(const GeneratedCutFrame& frame,
                                           float tolerance = 1.0e-4f)
{
    if (!finiteGeneratedPoint(frame.horizontal) ||
        !finiteGeneratedPoint(frame.vertical) ||
        !finiteGeneratedPoint(frame.normal)) {
        return false;
    }
    return std::abs(cv::norm(frame.horizontal) - 1.0f) <= tolerance &&
           std::abs(cv::norm(frame.vertical) - 1.0f) <= tolerance &&
           std::abs(cv::norm(frame.normal) - 1.0f) <= tolerance &&
           std::abs(frame.horizontal.dot(frame.vertical)) <= tolerance &&
           std::abs(frame.horizontal.dot(frame.normal)) <= tolerance &&
           std::abs(frame.vertical.dot(frame.normal)) <= tolerance;
}

inline GeneratedLineViewNavigationState resetGeneratedLineViewNavigationState(
    double initialCurrentLinePosition,
    double initialBottomCenterPosition,
    double initialBottomSliceLineStep)
{
    GeneratedLineViewNavigationState state;
    state.currentLinePosition = initialCurrentLinePosition;
    state.bottomCenterPosition = initialBottomCenterPosition;
    state.bottomSliceLineStep = initialBottomSliceLineStep;
    state.currentCutManualRotation = cv::Matx33f::eye();
    state.currentCutManualRotationActive = false;
    return state;
}

inline bool validGeneratedLinePosition(double position, size_t pointCount)
{
    return std::isfinite(position) &&
           pointCount > 0 &&
           position >= 0.0 &&
           position <= static_cast<double>(pointCount - 1);
}

inline GeneratedSpanAlignmentMetric makeGeneratedSpanAlignmentMetric(
    int spanIndex,
    int firstControlIndex,
    int secondControlIndex,
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints)
{
    GeneratedSpanAlignmentMetric metric;
    metric.spanIndex = spanIndex;
    metric.firstControlIndex = firstControlIndex;
    metric.secondControlIndex = secondControlIndex;
    if (firstControlIndex >= 0 &&
        static_cast<size_t>(firstControlIndex) < controlPoints.size()) {
        metric.firstControlLinePosition =
            controlPoints[static_cast<size_t>(firstControlIndex)].linePosition;
    }
    if (secondControlIndex >= 0 &&
        static_cast<size_t>(secondControlIndex) < controlPoints.size()) {
        metric.secondControlLinePosition =
            controlPoints[static_cast<size_t>(secondControlIndex)].linePosition;
    }
    return metric;
}

inline std::optional<double> generatedSpanAlignmentMetricCenterLinePosition(
    const GeneratedSpanAlignmentMetric& metric)
{
    if (!std::isfinite(metric.firstControlLinePosition) ||
        !std::isfinite(metric.secondControlLinePosition)) {
        return std::nullopt;
    }
    return (metric.firstControlLinePosition + metric.secondControlLinePosition) * 0.5;
}

inline cv::Vec3f interpolatedGeneratedLinePoint(const std::vector<cv::Vec3f>& linePoints,
                                                double linePosition)
{
    if (linePoints.empty()) {
        return {std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()};
    }
    linePosition = std::clamp(linePosition, 0.0, static_cast<double>(linePoints.size() - 1));
    const int lower = static_cast<int>(std::floor(linePosition));
    const int upper = std::min<int>(lower + 1, static_cast<int>(linePoints.size()) - 1);
    const float t = static_cast<float>(linePosition - static_cast<double>(lower));
    return linePoints[static_cast<size_t>(lower)] * (1.0f - t) +
           linePoints[static_cast<size_t>(upper)] * t;
}

inline std::optional<std::pair<double, double>> generatedControlLinePositionRange(
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints)
{
    double first = std::numeric_limits<double>::infinity();
    double last = -std::numeric_limits<double>::infinity();
    int finiteCount = 0;
    for (const auto& control : controlPoints) {
        if (!std::isfinite(control.linePosition)) {
            continue;
        }
        ++finiteCount;
        first = std::min(first, control.linePosition);
        last = std::max(last, control.linePosition);
    }
    if (finiteCount < 2 || !std::isfinite(first) || !std::isfinite(last) || first >= last) {
        return std::nullopt;
    }
    return std::make_pair(first, last);
}

inline std::vector<double> finiteGeneratedControlPointLinePositions(
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints)
{
    std::vector<double> positions;
    positions.reserve(controlPoints.size());
    for (const auto& control : controlPoints) {
        if (std::isfinite(control.linePosition)) {
            positions.push_back(control.linePosition);
        }
    }
    std::sort(positions.begin(), positions.end());
    return positions;
}

inline GeneratedControlPointLinePositionIndex buildGeneratedControlPointLinePositionIndex(
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints)
{
    GeneratedControlPointLinePositionIndex index;
    index.sortedControlIndices.reserve(controlPoints.size());
    for (size_t i = 0; i < controlPoints.size(); ++i) {
        if (std::isfinite(controlPoints[i].linePosition)) {
            index.sortedControlIndices.push_back(i);
        }
    }
    std::sort(index.sortedControlIndices.begin(),
              index.sortedControlIndices.end(),
              [&controlPoints](size_t lhs, size_t rhs) {
                  const double lhsPosition = controlPoints[lhs].linePosition;
                  const double rhsPosition = controlPoints[rhs].linePosition;
                  if (lhsPosition == rhsPosition) {
                      return lhs < rhs;
                  }
                  return lhsPosition < rhsPosition;
              });
    return index;
}

inline std::vector<size_t> generatedControlPointCandidateIndicesInLinePositionWindow(
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints,
    const GeneratedControlPointLinePositionIndex& index,
    double linePosition,
    double radius)
{
    std::vector<size_t> candidates;
    if (!std::isfinite(linePosition) || !std::isfinite(radius) || radius < 0.0) {
        return candidates;
    }

    const double lower = linePosition - radius;
    const double upper = linePosition + radius;
    const auto positionForIndex = [&controlPoints](size_t controlIndex) {
        return controlPoints[controlIndex].linePosition;
    };
    const auto lowerIt = std::lower_bound(
        index.sortedControlIndices.begin(),
        index.sortedControlIndices.end(),
        lower,
        [&positionForIndex](size_t controlIndex, double value) {
            return positionForIndex(controlIndex) < value;
        });
    for (auto it = lowerIt; it != index.sortedControlIndices.end(); ++it) {
        const double position = positionForIndex(*it);
        if (!std::isfinite(position)) {
            continue;
        }
        if (position > upper) {
            break;
        }
        candidates.push_back(*it);
    }
    return candidates;
}

inline double medianGeneratedLinePointSpacing(const std::vector<cv::Vec3f>& linePoints)
{
    std::vector<double> spacings;
    if (linePoints.size() < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    spacings.reserve(linePoints.size() - 1);
    for (size_t i = 1; i < linePoints.size(); ++i) {
        if (!finiteGeneratedPoint(linePoints[i - 1]) || !finiteGeneratedPoint(linePoints[i])) {
            continue;
        }
        const double spacing = cv::norm(linePoints[i] - linePoints[i - 1]);
        if (std::isfinite(spacing) && spacing > 1.0e-6) {
            spacings.push_back(spacing);
        }
    }
    if (spacings.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const size_t middle = spacings.size() / 2;
    std::nth_element(spacings.begin(),
                     spacings.begin() + static_cast<std::ptrdiff_t>(middle),
                     spacings.end());
    double median = spacings[middle];
    if (spacings.size() % 2 == 0) {
        const auto lowerIt =
            std::max_element(spacings.begin(),
                             spacings.begin() + static_cast<std::ptrdiff_t>(middle));
        median = (*lowerIt + median) * 0.5;
    }
    return median;
}

inline double generatedLinePositionRadiusForVolumeThreshold(
    const std::vector<cv::Vec3f>& linePoints,
    double linePosition,
    float volumeThreshold)
{
    constexpr double kMinimumRadius = 0.5;
    if (!std::isfinite(linePosition) ||
        !std::isfinite(volumeThreshold) ||
        volumeThreshold <= 0.0f ||
        linePoints.size() < 2) {
        return kMinimumRadius;
    }

    const int lower = std::clamp(static_cast<int>(std::floor(linePosition)),
                                 0,
                                 static_cast<int>(linePoints.size()) - 1);
    double spacing = std::numeric_limits<double>::quiet_NaN();
    if (lower + 1 < static_cast<int>(linePoints.size()) &&
        finiteGeneratedPoint(linePoints[static_cast<size_t>(lower)]) &&
        finiteGeneratedPoint(linePoints[static_cast<size_t>(lower + 1)])) {
        spacing = cv::norm(linePoints[static_cast<size_t>(lower + 1)] -
                           linePoints[static_cast<size_t>(lower)]);
    }
    if (!std::isfinite(spacing) || spacing <= 1.0e-6) {
        spacing = medianGeneratedLinePointSpacing(linePoints);
    }
    if (!std::isfinite(spacing) || spacing <= 1.0e-6) {
        return kMinimumRadius;
    }
    return std::max(kMinimumRadius, static_cast<double>(volumeThreshold) / spacing);
}

inline std::optional<double> previousGeneratedControlPointLinePosition(
    double currentLinePosition,
    const std::vector<double>& controlLinePositions)
{
    if (!std::isfinite(currentLinePosition)) {
        return std::nullopt;
    }
    std::optional<double> previous;
    for (const double position : controlLinePositions) {
        if (!std::isfinite(position) || position >= currentLinePosition) {
            continue;
        }
        if (!previous || position > *previous) {
            previous = position;
        }
    }
    return previous;
}

inline std::optional<double> nextGeneratedControlPointLinePosition(
    double currentLinePosition,
    const std::vector<double>& controlLinePositions)
{
    if (!std::isfinite(currentLinePosition)) {
        return std::nullopt;
    }
    std::optional<double> next;
    for (const double position : controlLinePositions) {
        if (!std::isfinite(position) || position <= currentLinePosition) {
            continue;
        }
        if (!next || position < *next) {
            next = position;
        }
    }
    return next;
}

inline std::optional<double> closestGeneratedControlPointLinePosition(
    double currentLinePosition,
    const std::vector<double>& controlLinePositions)
{
    if (!std::isfinite(currentLinePosition)) {
        return std::nullopt;
    }
    std::optional<double> closest;
    double closestDistance = std::numeric_limits<double>::infinity();
    for (const double position : controlLinePositions) {
        if (!std::isfinite(position)) {
            continue;
        }
        const double distance = std::abs(position - currentLinePosition);
        if (distance < closestDistance) {
            closest = position;
            closestDistance = distance;
        }
    }
    return closest;
}

inline bool generatedControlPointPlacementWithinAnyDistance(
    double linePosition,
    const std::vector<double>& controlLinePositions,
    double maxDistance,
    double existingControlTolerance = 0.5)
{
    if (!std::isfinite(maxDistance) || maxDistance <= 0.0) {
        return true;
    }
    if (!std::isfinite(linePosition)) {
        return false;
    }

    bool hasFiniteControl = false;
    double nearestDistance = std::numeric_limits<double>::infinity();
    for (const double controlPosition : controlLinePositions) {
        if (!std::isfinite(controlPosition)) {
            continue;
        }
        hasFiniteControl = true;
        const double distance = std::abs(controlPosition - linePosition);
        if (distance <= existingControlTolerance) {
            return true;
        }
        nearestDistance = std::min(nearestDistance, distance);
    }
    if (!hasFiniteControl) {
        return controlLinePositions.empty();
    }
    return nearestDistance <= maxDistance + 1.0e-6;
}

inline bool generatedControlPointPlacementWithinPreviousDistance(
    double linePosition,
    const std::vector<double>& controlLinePositions,
    double maxDistance,
    double existingControlTolerance = 0.5)
{
    return generatedControlPointPlacementWithinAnyDistance(linePosition,
                                                          controlLinePositions,
                                                          maxDistance,
                                                          existingControlTolerance);
}

inline bool generatedControlPointPlacementWithinPreviousDistance(
    double linePosition,
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints,
    double maxDistance,
    double existingControlTolerance = 0.5)
{
    std::vector<double> positions;
    positions.reserve(controlPoints.size());
    for (const auto& control : controlPoints) {
        positions.push_back(control.linePosition);
    }
    return generatedControlPointPlacementWithinPreviousDistance(linePosition,
                                                               positions,
                                                               maxDistance,
                                                               existingControlTolerance);
}

inline bool generatedControlPointPlacementWithinAnyDistance(
    double linePosition,
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints,
    double maxDistance,
    double existingControlTolerance = 0.5)
{
    std::vector<double> positions;
    positions.reserve(controlPoints.size());
    for (const auto& control : controlPoints) {
        positions.push_back(control.linePosition);
    }
    return generatedControlPointPlacementWithinAnyDistance(linePosition,
                                                          positions,
                                                          maxDistance,
                                                          existingControlTolerance);
}

inline bool generatedLinePositionWithinAnyControlDistance(
    double linePosition,
    const std::vector<double>& controlLinePositions,
    double maxDistance)
{
    if (!std::isfinite(maxDistance) || maxDistance <= 0.0) {
        return true;
    }
    if (!std::isfinite(linePosition)) {
        return false;
    }

    constexpr double kExactControlTolerance = 1.0e-6;
    bool hasFiniteControl = false;
    double nearestDistance = std::numeric_limits<double>::infinity();
    for (const double controlPosition : controlLinePositions) {
        if (!std::isfinite(controlPosition)) {
            continue;
        }
        hasFiniteControl = true;
        const double distance = std::abs(controlPosition - linePosition);
        if (distance <= kExactControlTolerance) {
            return true;
        }
        nearestDistance = std::min(nearestDistance, distance);
    }
    if (!hasFiniteControl) {
        return controlLinePositions.empty();
    }
    return nearestDistance <= maxDistance + 1.0e-6;
}

inline bool generatedLinePositionWithinPreviousControlDistance(
    double linePosition,
    const std::vector<double>& controlLinePositions,
    double maxDistance)
{
    return generatedLinePositionWithinAnyControlDistance(linePosition,
                                                        controlLinePositions,
                                                        maxDistance);
}

inline bool generatedLinePositionWithinAnyControlDistance(
    double linePosition,
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints,
    double maxDistance)
{
    std::vector<double> positions;
    positions.reserve(controlPoints.size());
    for (const auto& control : controlPoints) {
        positions.push_back(control.linePosition);
    }
    return generatedLinePositionWithinAnyControlDistance(linePosition,
                                                        positions,
                                                        maxDistance);
}

inline bool generatedLinePositionWithinPreviousControlDistance(
    double linePosition,
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints,
    double maxDistance)
{
    return generatedLinePositionWithinAnyControlDistance(linePosition,
                                                        controlPoints,
                                                        maxDistance);
}

inline std::optional<size_t> nearestGeneratedControlPointIndex(
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints,
    const cv::Vec3f& point)
{
    if (!finiteGeneratedPoint(point)) {
        return std::nullopt;
    }
    std::optional<size_t> best;
    double bestDistanceSq = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < controlPoints.size(); ++i) {
        if (!finiteGeneratedPoint(controlPoints[i].point)) {
            continue;
        }
        const cv::Vec3f delta = controlPoints[i].point - point;
        const double distanceSq = static_cast<double>(delta.dot(delta));
        if (distanceSq < bestDistanceSq) {
            best = i;
            bestDistanceSq = distanceSq;
        }
    }
    return best;
}

inline bool generatedLineSegmentIsTail(
    double startPosition,
    double endPosition,
    const std::optional<std::pair<double, double>>& controlRange)
{
    if (!controlRange || !std::isfinite(startPosition) || !std::isfinite(endPosition)) {
        return false;
    }
    const double midpoint = (startPosition + endPosition) * 0.5;
    return midpoint < controlRange->first || midpoint > controlRange->second;
}

inline GeneratedOverlay makeGeneratedStripOverlay(
    const GeneratedViews& views,
    double currentLinePosition,
    const std::vector<double>& markerLinePositions)
{
    GeneratedOverlay overlay;
    overlay.linePoints = views.linePoints;
    overlay.branchLinePoints = views.branchLinePoints;
    overlay.seedPoint = views.seedPoint;
    overlay.seedLineIndex = views.controlPoints.empty() ? views.seedLineIndex : -1;
    overlay.useSurfaceCenterLine = true;
    overlay.currentLinePosition = currentLinePosition;
    overlay.controlPoints = views.controlPoints;
    overlay.predSnapPoints = views.predSnapPoints;
    overlay.markerLinePositions = markerLinePositions;
    return overlay;
}

inline GeneratedOverlay makeGeneratedStaticStripOverlay(const GeneratedViews& views)
{
    GeneratedOverlay overlay;
    overlay.linePoints = views.linePoints;
    overlay.branchLinePoints = views.branchLinePoints;
    overlay.seedPoint = views.seedPoint;
    overlay.seedLineIndex = views.controlPoints.empty() ? views.seedLineIndex : -1;
    overlay.useSurfaceCenterLine = true;
    overlay.controlPoints = views.controlPoints;
    overlay.predSnapPoints = views.predSnapPoints;
    return overlay;
}

inline GeneratedOverlay makeGeneratedDynamicStripOverlay(
    const GeneratedViews& views,
    double currentLinePosition,
    const std::vector<double>& markerLinePositions)
{
    GeneratedOverlay overlay;
    overlay.useSurfaceCenterLine = true;
    overlay.currentLinePosition = currentLinePosition;
    overlay.markerLinePositions = markerLinePositions;
    return overlay;
}

inline GeneratedOverlay makeGeneratedCrossSliceOverlay(
    const GeneratedViews& views,
    double linePosition,
    bool emphasized,
    std::optional<float> controlDistanceThreshold,
    const std::function<float(const cv::Vec3f&)>& pointDistance,
    const GeneratedControlPointLinePositionIndex* controlIndex = nullptr,
    std::optional<double> controlLinePositionRadius = std::nullopt)
{
    GeneratedOverlay overlay;
    overlay.branchLinePoints = views.branchLinePoints;
    overlay.pointMarker = emphasized && finiteGeneratedPoint(views.focusPoint)
        ? views.focusPoint
        : interpolatedGeneratedLinePoint(views.linePoints, linePosition);
    overlay.emphasizedPointMarker = emphasized;
    if (!controlDistanceThreshold || !pointDistance) {
        return overlay;
    }

    std::vector<size_t> candidateIndices;
    if (controlIndex && controlLinePositionRadius) {
        candidateIndices = generatedControlPointCandidateIndicesInLinePositionWindow(
            views.controlPoints,
            *controlIndex,
            linePosition,
            *controlLinePositionRadius);
    } else {
        candidateIndices.reserve(views.controlPoints.size());
        for (size_t i = 0; i < views.controlPoints.size(); ++i) {
            candidateIndices.push_back(i);
        }
    }

    for (const size_t controlIndexValue : candidateIndices) {
        if (controlIndexValue >= views.controlPoints.size()) {
            continue;
        }
        const auto& control = views.controlPoints[controlIndexValue];
        if (!finiteGeneratedPoint(control.point)) {
            continue;
        }
        const float distance = pointDistance(control.point);
        if (std::isfinite(distance) && std::abs(distance) <= *controlDistanceThreshold) {
            overlay.controlPoints.push_back(control);
            for (const auto& predSnap : views.predSnapPoints) {
                if (predSnap.controlIndex == controlIndexValue &&
                    finiteGeneratedPoint(predSnap.snapPoint)) {
                    overlay.predSnapPoints.push_back(predSnap);
                }
            }
        }
    }

    for (const auto& intersection : views.fiberIntersections) {
        if (!finiteGeneratedPoint(intersection.point)) {
            continue;
        }
        const float distance = pointDistance(intersection.point);
        if (std::isfinite(distance) && std::abs(distance) <= *controlDistanceThreshold) {
            overlay.fiberIntersections.push_back(intersection);
        }
    }
    return overlay;
}

struct GeneratedLinkCandidateMenuState {
    bool enabled = false;
    QString label;
};

struct GeneratedControlPointContextMenuOptions {
    QWidget* parent = nullptr;
    std::string surfaceName;
    CChunkedVolumeViewer* viewer = nullptr;
    QPointF scenePoint;
    QPoint globalPos;
    std::vector<GeneratedOverlay::ControlPointMarker> controlPoints;
    std::vector<GeneratedOverlay::FiberIntersectionMarker> fiberIntersections;
    size_t linePointCount = 0;
    double linePosition = std::numeric_limits<double>::quiet_NaN();
    bool stripViewer = false;
    bool linkWithCandidateEnabled = false;
    QString linkWithCandidateLabel;
    cv::Vec3f branchLinkDirection{std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::quiet_NaN()};
    std::function<void(double, cv::Vec3f)> deleteControlPoint;
    std::function<void(size_t, cv::Vec3f, bool, cv::Vec3f)> addBranch;
    std::function<void(uint64_t, int)> openBranch;
    std::function<void(size_t, uint64_t, int)> unlinkBranch;
    // (controlIndex, linkedFiberId, linkedControlPointIndex, newPendingState)
    std::function<void(size_t, uint64_t, int, bool)> setBranchLinkPending;
    std::function<void(size_t, cv::Vec3f)> designateLinkCandidate;
    std::function<void(size_t, cv::Vec3f)> linkWithCandidate;
    std::function<void(uint64_t, cv::Vec3f)> openNearbyAnnotation;
};

QPointF generatedStripLinePositionToScene(CChunkedVolumeViewer* viewer,
                                          QuadSurface* surface,
                                          double linePosition);
double generatedLinePositionFromStripScene(CChunkedVolumeViewer* viewer,
                                           const QPointF& scenePoint);
std::optional<float> generatedCrossSliceControlPointDistanceThreshold(CChunkedVolumeViewer* viewer);
GeneratedOverlay makeGeneratedCrossSliceOverlayForPlane(const GeneratedViews& views,
                                                        double linePosition,
                                                        bool emphasized,
                                                        CChunkedVolumeViewer* viewer,
                                                        PlaneSurface* plane,
                                                        const GeneratedControlPointLinePositionIndex* controlIndex = nullptr);
GeneratedOverlay makeGeneratedCrossSliceControlOverlayForPlane(const GeneratedViews& views,
                                                               double linePosition,
                                                               CChunkedVolumeViewer* viewer,
                                                               PlaneSurface* plane,
                                                               const GeneratedControlPointLinePositionIndex* controlIndex = nullptr);
void applyGeneratedOverlay(CChunkedVolumeViewer* viewer,
                           const std::string& surfaceName,
                           const GeneratedOverlay& overlay);
void clearGeneratedControlPointContextPreview(CChunkedVolumeViewer* viewer,
                                              const std::string& surfaceName);
GeneratedControlPointContextResult showGeneratedControlPointContextMenu(
    const GeneratedControlPointContextMenuOptions& options);

} // namespace vc3d::line_annotation
