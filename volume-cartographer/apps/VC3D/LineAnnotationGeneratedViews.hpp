#pragma once

#include "vc/lasagna/LineViewBuilder.hpp"

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
        // Same-orientation links (H-H / V-V) render in the orange warning
        // palette; H-V links keep the default blue/purple. Set by the
        // controller, which owns the fiber HV state.
        bool hasSameHvBranches = false;
        bool hasSameHvPendingLinks = false;
        bool isLinkCandidate = false;
        bool isSplitCandidate = false;
        bool hasTracedSegmentToNext = false;
        std::string interpolationGoal = "global";
        char interpolationModeMarker = 'L';
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
    // Present for strip overlays. Line positions above remain in original
    // LineModel point-index coordinates and are mapped only while projecting.
    vc::lasagna::LineStripPositionMap stripPositionMap;
};

struct GeneratedSpanAlignmentMetric {
    enum class Kind {
        LasagnaNormalAlignment,
        NativeMeetingError,
        NativeFailure,
        Cspline,
    };

    int spanIndex = 0;
    int firstControlIndex = 0;
    int secondControlIndex = 0;
    double firstControlLinePosition = std::numeric_limits<double>::quiet_NaN();
    double secondControlLinePosition = std::numeric_limits<double>::quiet_NaN();
    double maxErrorDegrees = 0.0;
    bool available = false;
    bool pending = false;
    std::string error;
    Kind kind = Kind::LasagnaNormalAlignment;
    double meetingErrorBaseVoxels =
        std::numeric_limits<double>::quiet_NaN();
    double meetingErrorRatio =
        std::numeric_limits<double>::quiet_NaN();
    std::string meetingSource;
    std::string failureCode;
    std::string failureDetail;
    char modeMarker = 'L';
    std::string message;
};

struct GeneratedViews {
    std::string lineSurfaceName;
    QString lineSurfaceTitle;
    std::shared_ptr<QuadSurface> lineSurface;
    std::string lineSideSliceName;
    QString lineSideSliceTitle;
    std::shared_ptr<QuadSurface> lineSideSlice;
    std::string currentCutName;
    std::shared_ptr<PlaneSurface> currentCutSurface;
    std::string sideCutName;
    std::shared_ptr<PlaneSurface> sideCutSurface;
    std::vector<cv::Vec3f> linePoints;
    std::vector<cv::Vec3f> lineUpVectors;
    vc::lasagna::LineStripPositionMap stripPositionMap;
    // Per-line-point sampled sheet normals, sign-oriented away from the
    // scroll center (NaN where the sample is invalid). Empty when
    // unavailable.
    std::vector<cv::Vec3f> lineNormals;
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

inline void replaceGeneratedBranchOverlayData(
    GeneratedViews& views,
    std::vector<GeneratedOverlay::ControlPointMarker> controlPoints,
    std::vector<std::vector<cv::Vec3f>> branchLinePoints,
    std::vector<GeneratedOverlay::BranchLinkMarker> branchLinks,
    std::vector<GeneratedSpanAlignmentMetric> spanAlignmentMetrics)
{
    views.controlPoints = std::move(controlPoints);
    views.branchLinePoints = std::move(branchLinePoints);
    views.branchLinks = std::move(branchLinks);
    views.fiberIntersections.clear();
    views.spanAlignmentMetrics = std::move(spanAlignmentMetrics);
}

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

// One sign (+1/-1) per fiber for the DISPLAYED tangent used to pose the
// current-cut and side-cut planes. Stored line-point order never changes.
// The current cut's screen x is (up x normal) with normal = sign * tangent, so
// pinning sign * mean((normal_i x tangent_i) . z) >= 0 puts increasing slice
// index on the same screen side for every circumferential fiber, whatever
// direction it was traced or merged in. For fibers running along the scroll
// axis the tangent's own z component decides instead, which pins the side
// cut's vertical (its up is the signed tangent). Per point the two votes
// measure the tangent's circumferential and axial magnitudes, so the larger
// mean identifies the fiber's dominant direction (switching conventions at
// ~45 degree pitch): a near-axial fiber's slight helical drift must not
// decide its sign.
inline float generatedDisplayTangentSign(const std::vector<cv::Vec3f>& linePoints,
                                         const std::vector<cv::Vec3f>& lineNormals)
{
    if (linePoints.size() < 2) {
        return 1.0f;
    }
    const bool haveNormals = lineNormals.size() == linePoints.size();
    double primary = 0.0;
    double fallback = 0.0;
    size_t tangentCount = 0;
    size_t normalPairCount = 0;
    for (size_t i = 0; i < linePoints.size(); ++i) {
        cv::Vec3f tangent;
        if (i == 0) {
            tangent = linePoints[1] - linePoints[0];
        } else if (i + 1 == linePoints.size()) {
            tangent = linePoints[i] - linePoints[i - 1];
        } else {
            tangent = linePoints[i + 1] - linePoints[i - 1];
        }
        tangent = normalizedGeneratedVectorOrNan(tangent);
        if (!finiteGeneratedPoint(tangent)) {
            continue;
        }
        ++tangentCount;
        fallback += static_cast<double>(tangent[2]);
        if (!haveNormals) {
            continue;
        }
        const cv::Vec3f normal = normalizedGeneratedVectorOrNan(lineNormals[i]);
        if (!finiteGeneratedPoint(normal)) {
            continue;
        }
        ++normalPairCount;
        primary += static_cast<double>(normal.cross(tangent)[2]);
    }
    // Compare per-vote means, not raw sums: primary only accumulates where a
    // sampled normal is valid, so on a sparse-normal fiber a raw fallback sum
    // over every tangent would drown out a decisive primary vote. The means
    // are per-point direction magnitudes in [-1, 1] and comparable directly;
    // the tie band keeps rounding noise from masquerading as a decision.
    constexpr double kTie = 1.0e-3;
    const double meanPrimary =
        normalPairCount > 0 ? primary / static_cast<double>(normalPairCount) : 0.0;
    const double meanFallback =
        tangentCount > 0 ? fallback / static_cast<double>(tangentCount) : 0.0;
    if (std::abs(meanPrimary) > std::max(kTie, std::abs(meanFallback))) {
        return meanPrimary > 0.0 ? 1.0f : -1.0f;
    }
    if (std::abs(meanFallback) > kTie) {
        return meanFallback > 0.0 ? 1.0f : -1.0f;
    }
    return 1.0f;
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

inline constexpr double kGeneratedParallaxGhostMinimumOpacity = 0.3;
inline constexpr double kGeneratedParallaxGhostMaximumOpacity = 0.85;
// Fraction of the visibility distance over which a ghost fades out at the far
// edge, so it eases in and out instead of popping at the cutoff.
inline constexpr double kGeneratedParallaxGhostEdgeFadeFraction = 0.25;

// Parallax slide-in cue for the nearest control point on one side of the current
// cut. All positions, deltas and the slide range are line-position units (one unit
// is one index step in GeneratedViews::linePoints, roughly 30 base voxels of arc
// length); nothing here is expressed in voxels or scene pixels. The viewer-side
// geometry (scene offset, viewport width) stays in the dialog.
struct GeneratedParallaxGhost {
    size_t controlIndex = 0;
    double linePosition = 0.0;
    // Signed, clamped to [-1, 1]; positive means the control point is ahead.
    double offsetFraction = 0.0;
    // Ramps from kGeneratedParallaxGhostMinimumOpacity at or beyond the slide
    // range up to kGeneratedParallaxGhostMaximumOpacity as the delta closes.
    double opacity = 0.0;
};

// direction is +1 for the nearest control point strictly ahead of
// currentLinePosition and -1 for the nearest one strictly behind it. A ghost
// only exists while the control point is within maxDistanceLinePositions of the
// current position; its opacity fades to zero over the outer
// kGeneratedParallaxGhostEdgeFadeFraction of that distance. Returns nullopt
// when no such control point exists or when any input is unusable.
inline std::optional<GeneratedParallaxGhost> generatedParallaxGhost(
    const std::vector<GeneratedOverlay::ControlPointMarker>& controls,
    const GeneratedControlPointLinePositionIndex& index,
    double currentLinePosition,
    int direction,
    double slideRangeLinePositions,
    double maxDistanceLinePositions)
{
    if (controls.empty() || index.sortedControlIndices.empty()) {
        return std::nullopt;
    }
    if (!std::isfinite(currentLinePosition)) {
        return std::nullopt;
    }
    if (!std::isfinite(slideRangeLinePositions) || slideRangeLinePositions <= 0.0) {
        return std::nullopt;
    }
    if (!std::isfinite(maxDistanceLinePositions) || maxDistanceLinePositions <= 0.0) {
        return std::nullopt;
    }
    if (direction != 1 && direction != -1) {
        return std::nullopt;
    }

    const auto& indices = index.sortedControlIndices;
    // Out-of-range entries sort last and are rejected by the scan below; the
    // sentinel keeps both binary-search comparators consistently ordered.
    const auto positionForIndex = [&controls](size_t controlIndex) {
        return controlIndex < controls.size()
                   ? controls[controlIndex].linePosition
                   : std::numeric_limits<double>::infinity();
    };
    const auto usable = [&controls, &positionForIndex](size_t controlIndex) {
        return controlIndex < controls.size() && std::isfinite(positionForIndex(controlIndex));
    };

    std::optional<size_t> selected;
    if (direction > 0) {
        auto it = std::upper_bound(
            indices.begin(),
            indices.end(),
            currentLinePosition,
            [&positionForIndex](double value, size_t controlIndex) {
                return value < positionForIndex(controlIndex);
            });
        for (; it != indices.end(); ++it) {
            if (usable(*it) && positionForIndex(*it) > currentLinePosition) {
                selected = *it;
                break;
            }
        }
    } else {
        auto it = std::lower_bound(
            indices.begin(),
            indices.end(),
            currentLinePosition,
            [&positionForIndex](size_t controlIndex, double value) {
                return positionForIndex(controlIndex) < value;
            });
        while (it != indices.begin()) {
            --it;
            if (usable(*it) && positionForIndex(*it) < currentLinePosition) {
                selected = *it;
                break;
            }
        }
    }
    if (!selected) {
        return std::nullopt;
    }

    GeneratedParallaxGhost ghost;
    ghost.controlIndex = *selected;
    ghost.linePosition = positionForIndex(*selected);
    const double delta = ghost.linePosition - currentLinePosition;
    if (std::abs(delta) > maxDistanceLinePositions) {
        return std::nullopt;
    }
    ghost.offsetFraction = std::clamp(delta / slideRangeLinePositions, -1.0, 1.0);
    const double proximity = 1.0 - std::abs(ghost.offsetFraction);
    ghost.opacity = kGeneratedParallaxGhostMinimumOpacity +
                    proximity * (kGeneratedParallaxGhostMaximumOpacity -
                                 kGeneratedParallaxGhostMinimumOpacity);
    const double edgeFadeSpan =
        maxDistanceLinePositions * kGeneratedParallaxGhostEdgeFadeFraction;
    ghost.opacity *= std::clamp(
        (maxDistanceLinePositions - std::abs(delta)) / edgeFadeSpan, 0.0, 1.0);
    return ghost;
}

// ---------------------------------------------------------------------------
// Arrow-key panning between control points.
//
// One signed-velocity integrator drives the whole gesture: a tap ramps up and
// brakes into the first control point ahead, a hold cruises straight through
// the intermediate ones, a live speed change simply moves the cruise target,
// and pressing the opposite arrow decelerates through zero into the reverse
// ramp. Everything below is pure arithmetic so it can be exercised without Qt.
// ---------------------------------------------------------------------------

// Seconds spent ramping from rest to the cruise speed (acceleration = cruise / this).
inline constexpr double kGeneratedArrowPanRampSeconds = 0.25;
// Cruise-speed bounds and default, in line positions per second (1 unit ~ 30 voxels).
inline constexpr double kGeneratedArrowPanMinimumSpeed = 1.0;
inline constexpr double kGeneratedArrowPanMaximumSpeed = 500.0;
inline constexpr double kGeneratedArrowPanDefaultSpeed = 12.0;
// Multiplicative step applied by the Up/Down arrows.
inline constexpr double kGeneratedArrowPanSpeedStep = 1.25;
// Distance below which a stop target counts as reached.
inline constexpr double kGeneratedArrowPanLandingEpsilon = 1.0e-9;

struct GeneratedArrowPanState {
    double position = 0.0;
    // Signed, in line positions per second.
    double velocity = 0.0;
    // True once the step consumed the stop target exactly.
    bool landed = false;
};

// One integrator step. `direction` is the travel direction (not the key state):
// it stays set while a released tap coasts into its target. `stopTarget`, when
// present, is braked into using the v^2 / (2a) trigger and landed on exactly.
inline GeneratedArrowPanState generatedArrowPanStep(double position,
                                                    double velocity,
                                                    int direction,
                                                    double cruiseSpeed,
                                                    double acceleration,
                                                    double dtSeconds,
                                                    const std::optional<double>& stopTarget)
{
    GeneratedArrowPanState next;
    next.position = position;
    next.velocity = std::isfinite(velocity) ? velocity : 0.0;
    if (!std::isfinite(position)) {
        next.velocity = 0.0;
        return next;
    }
    if (!std::isfinite(dtSeconds) || dtSeconds <= 0.0) {
        return next;
    }
    if (!std::isfinite(cruiseSpeed) || cruiseSpeed <= 0.0 ||
        !std::isfinite(acceleration) || acceleration <= 0.0) {
        next.velocity = 0.0;
        return next;
    }

    const int travel = (direction > 0) ? 1 : ((direction < 0) ? -1 : 0);
    const bool haveTarget = stopTarget.has_value() && std::isfinite(*stopTarget);
    if (haveTarget && travel != 0) {
        // Target already reached (or behind us): land instead of running off.
        const double signedRemaining = (*stopTarget - position) * static_cast<double>(travel);
        if (signedRemaining <= kGeneratedArrowPanLandingEpsilon) {
            next.position = *stopTarget;
            next.velocity = 0.0;
            next.landed = true;
            return next;
        }
    }

    double desiredVelocity = static_cast<double>(travel) * cruiseSpeed;
    double rate = acceleration;
    bool braking = false;
    if (haveTarget) {
        const double remaining = *stopTarget - position;
        // A reversal keeps the old velocity while the direction already points
        // the other way; brake only when the target is ahead of the motion.
        const double heading = (next.velocity != 0.0) ? next.velocity : desiredVelocity;
        if (heading != 0.0 && remaining != 0.0 && ((remaining > 0.0) == (heading > 0.0))) {
            const double brakingDistance =
                (next.velocity * next.velocity) / (2.0 * acceleration);
            if (std::abs(remaining) <= brakingDistance) {
                desiredVelocity = 0.0;
                braking = true;
                // Never undershoot: brake at least as hard as the exact profile.
                rate = std::max(acceleration,
                                (next.velocity * next.velocity) / (2.0 * std::abs(remaining)));
            }
        }
    }

    const double maxDelta = rate * dtSeconds;
    next.velocity += std::clamp(desiredVelocity - next.velocity, -maxDelta, maxDelta);
    next.position = position + next.velocity * dtSeconds;

    if (haveTarget) {
        const double moved = next.position - position;
        const double remaining = *stopTarget - position;
        if (moved != 0.0 && ((remaining > 0.0) == (moved > 0.0)) &&
            std::abs(moved) >= std::abs(remaining)) {
            next.position = *stopTarget;
            next.velocity = 0.0;
            next.landed = true;
        } else if (braking && next.velocity == 0.0) {
            // Braking decayed to a standstill less than half a tick short of the
            // target; snap so the gesture always terminates on the control point.
            next.position = *stopTarget;
            next.landed = true;
        }
    }
    return next;
}

// Next control point strictly in `direction` from `currentPosition`, but never
// short of `minimumTarget` (the first control point the gesture promised when
// the key went down, or the far end while the key is still held). Falls back to
// `minimumTarget` when nothing further exists; a non-finite `minimumTarget`
// means "no floor and no fallback", which yields nullopt with no candidate.
inline std::optional<double> generatedArrowPanStopTarget(
    const std::vector<double>& sortedControlLinePositions,
    double currentPosition,
    int direction,
    double minimumTarget)
{
    if (!std::isfinite(currentPosition) || direction == 0) {
        return std::nullopt;
    }
    const bool haveMinimum = std::isfinite(minimumTarget);
    std::optional<double> best;
    for (const double position : sortedControlLinePositions) {
        if (!std::isfinite(position)) {
            continue;
        }
        if (direction > 0) {
            if (position <= currentPosition || (haveMinimum && position < minimumTarget)) {
                continue;
            }
            if (!best || position < *best) {
                best = position;
            }
        } else {
            if (position >= currentPosition || (haveMinimum && position > minimumTarget)) {
                continue;
            }
            if (!best || position > *best) {
                best = position;
            }
        }
    }
    if (!best && haveMinimum) {
        best = minimumTarget;
    }
    return best;
}

// One extra pan target beyond the outermost control point in `direction`: the
// outer control point plus the max-control-point-distance allowance, clamped
// to `lineEndPosition` (the end of the extrapolated line) - whichever is
// shorter. `maxControlPointDistance` uses the same line-position
// interpretation as the current-line marker state; a non-finite or <= 0 value
// means unlimited (the line end alone bounds the hop). Returns nullopt when
// there are no finite control positions or no room beyond the outer one.
inline std::optional<double> generatedArrowPanBoundaryTarget(
    const std::vector<double>& sortedControlLinePositions,
    int direction,
    double lineEndPosition,
    double maxControlPointDistance)
{
    if (direction == 0 || !std::isfinite(lineEndPosition)) {
        return std::nullopt;
    }
    std::optional<double> outer;
    for (const double position : sortedControlLinePositions) {
        if (!std::isfinite(position)) {
            continue;
        }
        if (!outer || (direction > 0 ? position > *outer : position < *outer)) {
            outer = position;
        }
    }
    if (!outer) {
        return std::nullopt;
    }
    double limit = lineEndPosition;
    if (std::isfinite(maxControlPointDistance) && maxControlPointDistance > 0.0) {
        limit = (direction > 0) ? std::min(limit, *outer + maxControlPointDistance)
                                : std::max(limit, *outer - maxControlPointDistance);
    }
    if (direction > 0 ? limit <= *outer : limit >= *outer) {
        return std::nullopt;
    }
    return limit;
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
    overlay.stripPositionMap = views.stripPositionMap;
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
    overlay.stripPositionMap = views.stripPositionMap;
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
    overlay.stripPositionMap = views.stripPositionMap;
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
    vc::lasagna::LineStripPositionMap stripPositionMap;
    bool linkWithCandidateEnabled = false;
    QString linkWithCandidateLabel;
    bool mergeWithCandidateEnabled = false;
    QString mergeWithCandidateLabel;
    bool splitFromCandidateEnabled = false;
    QString splitFromCandidateLabel;
    QString splitFromCandidateAndLinkLabel;
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
    std::function<void(size_t, cv::Vec3f)> mergeWithCandidate;
    std::function<void(size_t, cv::Vec3f)> designateSplitCandidate;
    std::function<void(size_t, cv::Vec3f)> splitFromCandidate;
    std::function<void(size_t, cv::Vec3f)> splitFromCandidateAndLink;
    std::function<void(uint64_t, cv::Vec3f)> openNearbyAnnotation;
    std::function<void(size_t, size_t, std::string)> setSegmentInterpolationGoal;
};

QPointF generatedStripLinePositionToScene(CChunkedVolumeViewer* viewer,
                                          QuadSurface* surface,
                                          double linePosition,
                                          const vc::lasagna::LineStripPositionMap* positionMap = nullptr);
double generatedLinePositionFromStripScene(CChunkedVolumeViewer* viewer,
                                           const QPointF& scenePoint,
                                           const vc::lasagna::LineStripPositionMap* positionMap = nullptr);
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
