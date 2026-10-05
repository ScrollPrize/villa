#pragma once

#include "vc/lasagna/LineViewBuilder.hpp"
#include "vc/fiber_tracer/FiberDisplay.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core/types.hpp>

#include <QPoint>
#include <QPointF>
#include <QString>

class QPainterPath;

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
class QColor;
class QuadSurface;
class QWidget;

namespace vc3d::line_annotation {

struct GeneratedStripFrame {
    cv::Vec3d along, across, normal;
};
inline std::optional<GeneratedStripFrame> generatedStripFrame(QuadSurface* surface,
                                                            const cv::Vec2d& uv)
{
    if (!surface || !surface->rawPointsPtr() || surface->rawPointsPtr()->empty())
        return std::nullopt;
    const cv::Vec3f ptr{float(uv[0]*surface->scale()[0]),
                       float(uv[1]*surface->scale()[1]),0};
    const auto across=vc::fiber_tracer::displayUnit(cv::Vec3d(
        surface->coord(ptr,{0,1,0})-surface->coord(ptr,{0,-1,0})));
    const double col=surface->surfaceToGrid(uv)[0];
    const double row=surface->rawPointsPtr()->rows/2;
    const auto left=surface->gridToSurface({std::max(0.0,col-0.5),row});
    const auto right=surface->gridToSurface({
        std::min(double(surface->rawPointsPtr()->cols-1),col+0.5),row});
    const auto along=vc::fiber_tracer::displayUnit(cv::Vec3d(
        surface->sampleAtSurface(right).volume-surface->sampleAtSurface(left).volume));
    const auto normal=across && along ?
        vc::fiber_tracer::displayUnit(along->cross(*across)) : std::nullopt;
    if (!normal) return std::nullopt;
    return GeneratedStripFrame{across->cross(*normal),*across,*normal};
}

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
            // Adjacent-winding link (FiberBranchRef::adjacent).
            bool adjacent = false;
        };

        size_t controlIndex = std::numeric_limits<size_t>::max();
        bool isSeed = false;
        // Tagged kollesis_termination: hollow yellow ring. A linked tagged
        // point keeps the link-state fill inside the yellow ring.
        bool isKollesisTermination = false;
        // Tagged break: dotted amber ring.
        bool isBreak = false;
        // The span this point owns (to the next control in line order)
        // carries the gap span tag: drawn as the dotted amber line, closed
        // to placement. Read from the span descriptor, never inferred from
        // two break rings, so what is drawn is what the file says.
        bool hasGapToNext = false;
        // The span this point owns carries the damaged span tag: drawn as
        // alternating amber and red dashes, nothing else changes.
        bool hasDamagedToNext = false;
        bool hasBranches = false;
        bool hasPendingLinks = false;
        // Same-orientation links (H-H / V-V) render in the orange warning
        // palette; H-V links keep the default blue/purple. Set by the
        // controller, which owns the fiber HV state.
        bool hasSameHvBranches = false;
        bool hasSameHvPendingLinks = false;
        // An adjacent-winding link on this point: the marker is a triangle
        // in the link-state colour instead of a circle.
        bool hasAdjacentLinks = false;
        bool isLinkCandidate = false;
        // With isLinkCandidate: designated as an ADJACENT link candidate, a
        // green triangle rather than a green circle.
        bool isAdjacentLinkCandidate = false;
        bool hasTracedSegmentToNext = false;
        std::string interpolationGoal = "global";
        char interpolationModeMarker = 'L';
        std::vector<uint64_t> branchIds;
        std::vector<BranchLink> branchLinks;
        std::optional<cv::Vec3d> direction;
        // Arc length from the line start to this control, and the whole
        // line's arc length, both measured on the line the control's
        // linePosition indexes (the session's live line, which may be a
        // provisional splice the dialog does not hold yet). Non-finite when
        // the producer did not know the line.
        double arcLength = std::numeric_limits<double>::quiet_NaN();
        double lineArcLength = std::numeric_limits<double>::quiet_NaN();
        // The control's point lies on the line at its linePosition (every
        // control except one edited off the line without re-optimization).
        // On the strips such a control is drawn at its arc length along the
        // strip's centre line, which stays right while the strip still shows
        // the previous frame; an off-line control is drawn at its own point.
        bool onLine = false;
        // Revision of the line the marker's linePosition and arcLength index
        // (the session's lineRevision); 0 when unknown. A control set whose
        // revision differs from the displayed line's (GeneratedViews::
        // lineRevision) indexes a line that is not on screen yet.
        uint64_t lineRevision = 0;
        // Session-lifetime identity of the control (LineControlPoint::
        // identity); 0 when the producer has none. Views match controls
        // across publishes by this, never by point or index.
        uint64_t identity = 0;
        // Set only while the marker's linePosition indexes a line other than
        // the displayed one (a provisional publish, re-expressed on the
        // displayed line): the control's position ON the displayed line, for
        // everything that draws spans between controls on it.
        double displayedLinePosition = std::numeric_limits<double>::quiet_NaN();
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
        // Set with projectedBranchLink when the local and linked fibers share
        // an H/V classification (the orange warning palette).
        bool sameHvBranchLink = false;
        bool isLinkCandidateFiber = false;
        std::optional<cv::Vec3f> connectorStart;
    };

    std::vector<cv::Vec3f> linePoints;
    std::vector<std::vector<cv::Vec3f>> branchLinePoints;
    // Line-position range outside which linePoints segments are tails (not
    // drawn). Defaults to the span of controlPoints; a caller that shows only
    // a subset of the controls sets the full fiber's range here so interior
    // spans are not mistaken for tails.
    std::optional<std::pair<double, double>> lineTailControlRange;
    // Line-position ranges [first, second] of the fiber's gap spans: the
    // owner control's span descriptor carries the gap tag. Drawn as a dotted
    // amber line in place of the fiber's own line. Computed by the overlay
    // builders from the FULL control list (generatedGapLineRanges) before any
    // visibility filtering, so a hidden neighbour cannot move a range's end.
    std::vector<std::pair<double, double>> gapLineRanges;
    // Same, for spans carrying the damaged span tag: drawn as alternating
    // amber and red dashes in place of the fiber's own line.
    std::vector<std::pair<double, double>> damagedLineRanges;
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
    // Strip overlays: the surface to project through instead of the viewer's
    // current one. Set while a strip still shows the frame of its previous
    // surface (overlay swap pending): the viewer has already adopted the new
    // surface, whose grid origin and scale differ, while what is on screen is
    // the old one, so the held overlay must be placed through the held surface.
    std::shared_ptr<QuadSurface> projectionSurface;
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
    // The span carries the gap span tag (shown in the label so the metadata
    // can be read as text, not only as the dotted amber line).
    bool gap = false;
    // The span carries the damaged span tag.
    bool damaged = false;
};

// Positions cross from the stored fiber grid to the viewer grid together;
// directions and line indices are independent of that uniform scale.
inline void scaleGeneratedMarkerForVolume(GeneratedOverlay::BranchLinkMarker& marker,
                                          double scale)
{
    marker.localControlPoint *= static_cast<float>(scale);
    marker.linkedControlPoint *= static_cast<float>(scale);
    marker.planePoint *= static_cast<float>(scale);
}

inline void scaleGeneratedMarkerForVolume(GeneratedOverlay::PredSnapMarker& marker,
                                          double scale)
{
    marker.controlPoint *= static_cast<float>(scale);
    marker.snapPoint *= static_cast<float>(scale);
}

struct GeneratedViews {
    // Revision of the line these views were built from; 0 when unknown.
    uint64_t lineRevision = 0;
    double fiberWidth = 0.0; // Display-volume voxels, not persisted units.
    double fiberWidthGapFraction = vc::fiber_tracer::kDefaultFiberWidthGapFraction;
    double fiberBaseToVolumeScale = 1.0;
    bool hasManualDisplayNormals = false;
    std::vector<double> controlAngleOffsetsDegrees;
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
    // Optional directed display correction; lineNormals still owns the
    // fiber-wide viewer orientation convention.
    std::vector<cv::Vec3f> displayLineNormals;
    // Unwrapped winding angle (radians) of each line point about the scroll
    // center, or empty when no center reference exists. See
    // unwrappedGeneratedWindingAngles.
    std::vector<double> lineWindingAngles;
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
    // Fit the complete line in both strips once instead of restoring saved zooms.
    bool initialFitWholeLine = false;
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
    return cv::Vec3f(vc::fiber_tracer::displayVectorAt(linePoints, linePosition));
}

// The side cut shows the stretch of the fiber within this winding distance of
// the current position, on either side: half a wrap.
inline constexpr double kGeneratedSideCutHalfWrapAngle = 3.14159265358979323846;

// Unwrapped winding angle (radians) of each line point about the scroll
// center, accumulated along the line so adjacent wraps differ by ~2*pi instead
// of aliasing onto the same value. towardCenter(point) returns the vector from
// the point to the center at that point's z (non-finite when unknown). A point
// without a usable direction gets NaN and does not break the chain: the next
// finite angle continues from the last finite one. No towardCenter: all NaN.
// pointToCenterFrameScale maps all three query coordinates into the center
// provider's grid (including z for a center that varies along the scroll).
inline std::vector<double> unwrappedGeneratedWindingAngles(
    const std::vector<cv::Vec3f>& linePoints,
    const std::function<cv::Vec3f(const cv::Vec3f&)>& towardCenter,
    float pointToCenterFrameScale = 1.0f)
{
    constexpr double kTwoPi = 2.0 * kGeneratedSideCutHalfWrapAngle;
    std::vector<double> angles(linePoints.size(), std::numeric_limits<double>::quiet_NaN());
    if (!towardCenter) {
        return angles;
    }
    std::optional<double> previous;
    for (size_t i = 0; i < linePoints.size(); ++i) {
        const cv::Vec3f& point = linePoints[i];
        if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2])) {
            continue;
        }
        const cv::Vec3f toCenter = towardCenter(point * pointToCenterFrameScale);
        if (!std::isfinite(toCenter[0]) || !std::isfinite(toCenter[1])) {
            continue;
        }
        // Radial direction, center -> point, in the slice plane (z is the axis).
        const double dx = -static_cast<double>(toCenter[0]);
        const double dy = -static_cast<double>(toCenter[1]);
        if (dx * dx + dy * dy <= 1.0e-12) {
            continue;
        }
        double angle = std::atan2(dy, dx);
        if (previous) {
            // Nearest equivalent to the previous angle: remainder lands in [-pi, pi].
            angle = *previous + std::remainder(angle - *previous, kTwoPi);
        }
        angles[i] = angle;
        previous = angle;
    }
    return angles;
}

// Inclusive index range [first, last] of the contiguous stretch of the line
// around linePosition whose winding angle stays within maxAngleDelta of the
// angle at linePosition. Without usable angles (empty, size mismatch, or no
// finite angle at the position) the whole line qualifies. NaN angles inside
// the stretch are kept so isolated unknown points do not split the run.
inline std::pair<size_t, size_t> generatedLineIndexRangeWithinWinding(
    const std::vector<double>& angles,
    size_t pointCount,
    double linePosition,
    double maxAngleDelta)
{
    if (pointCount == 0) {
        return {0, 0};
    }
    const std::pair<size_t, size_t> full{0, pointCount - 1};
    if (angles.size() != pointCount || !std::isfinite(linePosition) ||
        !(maxAngleDelta >= 0.0)) {
        return full;
    }
    const double clamped = std::clamp(linePosition, 0.0, static_cast<double>(pointCount - 1));
    const size_t lower = static_cast<size_t>(std::floor(clamped));
    const size_t upper = std::min(lower + 1, pointCount - 1);
    double reference = std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(angles[lower]) && std::isfinite(angles[upper])) {
        const double t = clamped - static_cast<double>(lower);
        reference = angles[lower] * (1.0 - t) + angles[upper] * t;
    } else if (std::isfinite(angles[lower])) {
        reference = angles[lower];
    } else if (std::isfinite(angles[upper])) {
        reference = angles[upper];
    }
    if (!std::isfinite(reference)) {
        return full;
    }
    const auto within = [&](size_t index) {
        return !std::isfinite(angles[index]) ||
               std::abs(angles[index] - reference) <= maxAngleDelta;
    };
    size_t first = lower;
    while (first > 0 && within(first - 1)) {
        --first;
    }
    size_t last = upper;
    while (last + 1 < pointCount && within(last + 1)) {
        ++last;
    }
    return {first, last};
}

// Content-anchored remap of a fractional line position across a line-geometry
// change: re-optimization renumbers and moves the points, so the old numeric
// position is ambiguous on the new line. The position's 3D point on the old
// polyline is located on the new polyline instead (nearest vertex, refined by
// projecting onto that vertex's adjacent segments), so the returned position
// names the same fiber spot. Among near-ties in distance the vertex closest
// to the old position in INDEX wins: a spiral fiber's adjacent wraps pass
// within a few voxels of each other, and where the edit moved the local
// geometry by a comparable amount, plain nearest-point could jump the anchor
// onto the other wrap. Falls back to the clamped input position when either
// polyline is unusable.
//
// remappedGeneratedLinePositionFromAnchor is the anchor-based core, shared with
// the controller: a pane reports control-point edits at a line position it
// measured on the DISPLAYED line, and the controller resolves that position on
// the session line through the position's own 3D line point. Deliberately not
// through the clicked point: a click is meant to be off the line, and where
// another pass of the same fiber runs through the cut plane the clicked point
// can be nearer to that pass than to the local one.
template <typename Point>
inline double remappedGeneratedLinePositionFromAnchor(const std::vector<Point>& newLinePoints,
                                                      const Point& anchor,
                                                      double oldPosition)
{
    using Scalar = typename Point::value_type;
    const auto finite = [](const Point& p) {
        return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
    };
    if (newLinePoints.empty()) {
        return 0.0;
    }
    const double maxNewPosition = static_cast<double>(newLinePoints.size() - 1);
    if (!std::isfinite(oldPosition)) {
        return 0.0;
    }
    const double fallback = std::clamp(oldPosition, 0.0, maxNewPosition);
    if (!finite(anchor)) {
        return fallback;
    }
    std::optional<size_t> nearestIndex;
    double nearestDistanceSq = std::numeric_limits<double>::max();
    for (size_t i = 0; i < newLinePoints.size(); ++i) {
        const Point& point = newLinePoints[i];
        if (!finite(point)) {
            continue;
        }
        const Point delta = point - anchor;
        const double distanceSq = static_cast<double>(delta.dot(delta));
        if (distanceSq < nearestDistanceSq) {
            nearestDistanceSq = distanceSq;
            nearestIndex = i;
        }
    }
    if (!nearestIndex) {
        return fallback;
    }
    // Continuity tiebreak (see above): among vertices within twice the
    // nearest distance, prefer the one whose index is closest to the old
    // position. Outside edited regions the true match is at distance ~0, so
    // the band is empty of impostors and this is a no-op.
    {
        constexpr double kTieDistanceSqFactor = 4.0;  // (2x distance)^2
        const double tieThresholdSq =
            nearestDistanceSq * kTieDistanceSqFactor + 1.0e-12;
        double chosenIndexDelta = std::abs(
            static_cast<double>(*nearestIndex) - oldPosition);
        for (size_t i = 0; i < newLinePoints.size(); ++i) {
            const Point& point = newLinePoints[i];
            if (!finite(point)) {
                continue;
            }
            const Point delta = point - anchor;
            const double distanceSq = static_cast<double>(delta.dot(delta));
            if (distanceSq > tieThresholdSq) {
                continue;
            }
            const double indexDelta =
                std::abs(static_cast<double>(i) - oldPosition);
            if (indexDelta < chosenIndexDelta) {
                chosenIndexDelta = indexDelta;
                nearestIndex = i;
                nearestDistanceSq = distanceSq;
            }
        }
    }
    double bestPosition = static_cast<double>(*nearestIndex);
    double bestDistanceSq = nearestDistanceSq;
    // Fractional refinement: project the anchor onto the two segments adjacent
    // to the nearest vertex; each candidate segment must have both endpoints
    // finite (the nearest vertex already is).
    for (const size_t segmentStart :
         {*nearestIndex > 0 ? *nearestIndex - 1 : *nearestIndex, *nearestIndex}) {
        if (segmentStart + 1 >= newLinePoints.size()) {
            continue;
        }
        const Point& a = newLinePoints[segmentStart];
        const Point& b = newLinePoints[segmentStart + 1];
        if (!finite(a) || !finite(b)) {
            continue;
        }
        const Point segment = b - a;
        const double lengthSq = static_cast<double>(segment.dot(segment));
        if (!(lengthSq > 0.0)) {
            continue;
        }
        const double t = std::clamp(
            static_cast<double>((anchor - a).dot(segment)) / lengthSq, 0.0, 1.0);
        const Point projected = a + segment * static_cast<Scalar>(t);
        const Point delta = projected - anchor;
        const double distanceSq = static_cast<double>(delta.dot(delta));
        if (distanceSq < bestDistanceSq) {
            bestDistanceSq = distanceSq;
            bestPosition = static_cast<double>(segmentStart) + t;
        }
    }
    return std::clamp(bestPosition, 0.0, maxNewPosition);
}

inline double remappedGeneratedLinePosition(const std::vector<cv::Vec3f>& oldLinePoints,
                                            const std::vector<cv::Vec3f>& newLinePoints,
                                            double oldPosition)
{
    if (newLinePoints.empty() || !std::isfinite(oldPosition)) {
        return 0.0;
    }
    const cv::Vec3f anchor = interpolatedGeneratedLinePoint(oldLinePoints, oldPosition);
    return remappedGeneratedLinePositionFromAnchor(newLinePoints, anchor, oldPosition);
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

// The integrator is unit-agnostic; the dialog runs it in base-voxel arclength
// along the optimized polyline (LineStripPositionMap::originalArclengths) so
// the pan covers the same physical distance per second in 4 vx trace spans and
// ~32 vx cspline spans alike, and converts the result back to a line position.

// Seconds spent ramping from rest to the cruise speed (acceleration = cruise / this).
inline constexpr double kGeneratedArrowPanRampSeconds = 0.25;
// Cruise-speed bounds and default, in base voxels of arclength per second. The
// default matches the previous 12 positions/s at one strip column (8 vx) per
// position.
inline constexpr double kGeneratedArrowPanMinimumSpeed = 8.0;
inline constexpr double kGeneratedArrowPanMaximumSpeed = 4000.0;
inline constexpr double kGeneratedArrowPanDefaultSpeed = 96.0;
// Multiplicative step applied by the Up/Down arrows.
inline constexpr double kGeneratedArrowPanSpeedStep = 1.25;
// Distance below which a stop target counts as reached, in the integrator's
// unit (base voxels in the dialog): far below anything visible, well above
// double rounding on arclengths of ~1e4.
inline constexpr double kGeneratedArrowPanLandingEpsilon = 1.0e-3;

// Camera baseline an overlay group was built against: the scene position of
// a fixed reference surface point and the camera scale. A strip viewer maps
// surface to scene as (surface - cameraPointer) * scale + viewportCenter, so
// while the scale is unchanged every camera move (pan, linked-camera echo,
// viewport recenter) shifts all overlay items by one common scene delta.
struct GeneratedOverlayCameraBaseline {
    QPointF referenceScene{std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::quiet_NaN()};
    double scale = std::numeric_limits<double>::quiet_NaN();
};

// The scene delta that moves an overlay built at `baseline` to the camera
// whose reference point now sits at `referenceScene`, or nullopt when the
// items must be rebuilt instead: the scale changed (a zoom), or either state
// is unknown.
inline std::optional<QPointF> generatedOverlayPanTranslation(
    const GeneratedOverlayCameraBaseline& baseline,
    const QPointF& referenceScene,
    double scale)
{
    const auto finitePoint = [](const QPointF& point) {
        return std::isfinite(point.x()) && std::isfinite(point.y());
    };
    if (!finitePoint(baseline.referenceScene) || !std::isfinite(baseline.scale) ||
        !finitePoint(referenceScene) || !std::isfinite(scale)) {
        return std::nullopt;
    }
    if (scale != baseline.scale) {
        return std::nullopt;
    }
    return referenceScene - baseline.referenceScene;
}

struct GeneratedArrowPanState {
    double position = 0.0;
    // Signed, in the integrator's unit per second.
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

// Defined further down (with the other tagged-end helpers); declared here for
// the overlay builders' blocked-marker state.
inline bool generatedLinePositionBeyondKollesisTermination(
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints,
    double linePosition);

// The gap spans of a fiber as line-position ranges: a control whose span
// descriptor carries the gap tag (hasGapToNext) spans to the next control in
// line position order. Read from the span tag, not from the break rings, so
// the drawn line reports the file's span metadata. Must be given the complete
// control list. Inline: this header is compiled into QtCore-only tests
// without the .cpp.
template <typename SpanFlag>
inline std::vector<std::pair<double, double>> generatedSpanLineRanges(
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints,
    SpanFlag ownerHasFlag)
{
    std::vector<const GeneratedOverlay::ControlPointMarker*> sorted;
    sorted.reserve(controlPoints.size());
    for (const auto& control : controlPoints) {
        if (std::isfinite(control.linePosition)) {
            sorted.push_back(&control);
        }
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto* a, const auto* b) {
        return a->linePosition < b->linePosition;
    });
    // On the displayed line: a provisional publish's positions index the
    // controller's line, which is not what the spans are drawn on.
    const auto shownPosition = [](const GeneratedOverlay::ControlPointMarker& m) {
        return std::isfinite(m.displayedLinePosition) ? m.displayedLinePosition : m.linePosition;
    };
    std::vector<std::pair<double, double>> ranges;
    for (size_t i = 1; i < sorted.size(); ++i) {
        const double first = shownPosition(*sorted[i - 1]);
        const double second = shownPosition(*sorted[i]);
        if (ownerHasFlag(*sorted[i - 1]) && first < second) {
            ranges.emplace_back(first, second);
        }
    }
    return ranges;
}

inline std::vector<std::pair<double, double>> generatedGapLineRanges(
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints)
{
    return generatedSpanLineRanges(controlPoints, [](const GeneratedOverlay::ControlPointMarker& m) {
        return m.hasGapToNext;
    });
}

// The damaged spans, same rule (a span is never both: the gap wins).
inline std::vector<std::pair<double, double>> generatedDamagedLineRanges(
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints)
{
    return generatedSpanLineRanges(controlPoints, [](const GeneratedOverlay::ControlPointMarker& m) {
        return m.hasDamagedToNext && !m.hasGapToNext;
    });
}

// Strictly inside a gap span: nothing may be placed there until a break is
// removed. The endpoints themselves stay available (a click there replaces
// the break point).
inline bool generatedLinePositionInsideGap(
    const std::vector<std::pair<double, double>>& gapLineRanges,
    double linePosition)
{
    if (!std::isfinite(linePosition)) {
        return false;
    }
    for (const auto& [first, second] : gapLineRanges) {
        if (linePosition > first && linePosition < second) {
            return true;
        }
    }
    return false;
}

// Whether the dense line segment between two line positions belongs to a gap
// span: its midpoint lies within a gap range (the dense points at the
// endpoints are the controls themselves, so the midpoint test assigns every
// segment between them and nothing outside).
inline bool generatedLineSegmentInGap(
    double previousLinePosition,
    double currentLinePosition,
    const std::vector<std::pair<double, double>>& gapLineRanges)
{
    const double midpoint = 0.5 * (previousLinePosition + currentLinePosition);
    if (!std::isfinite(midpoint)) {
        return false;
    }
    for (const auto& [first, second] : gapLineRanges) {
        if (midpoint >= first && midpoint <= second) {
            return true;
        }
    }
    return false;
}

// The marker state the shared renderer draws when the overlay's builder has
// no dialog to ask: Blocked strictly inside a gap span or beyond a kollesis
// termination (the two placement rules that need no extrapolation-distance
// setting), Neutral otherwise. The dialog's own current-cut marker adds the
// Allowed state from its extrapolation limit; the intersection-inspection
// panes only ever see this.
inline GeneratedCurrentLineMarkerState generatedBlockedLineMarkerState(
    const std::vector<GeneratedOverlay::ControlPointMarker>& fullControlPoints,
    const std::vector<std::pair<double, double>>& gapLineRanges,
    double linePosition)
{
    if (generatedLinePositionInsideGap(gapLineRanges, linePosition) ||
        generatedLinePositionBeyondKollesisTermination(fullControlPoints, linePosition)) {
        return GeneratedCurrentLineMarkerState::Blocked;
    }
    return GeneratedCurrentLineMarkerState::Neutral;
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
    overlay.gapLineRanges = generatedGapLineRanges(views.controlPoints);
    overlay.damagedLineRanges = generatedDamagedLineRanges(views.controlPoints);
    overlay.currentLineMarkerState = generatedBlockedLineMarkerState(
        views.controlPoints, overlay.gapLineRanges, currentLinePosition);
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
    overlay.gapLineRanges = generatedGapLineRanges(views.controlPoints);
    overlay.damagedLineRanges = generatedDamagedLineRanges(views.controlPoints);
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
    overlay.currentLineMarkerState = generatedBlockedLineMarkerState(
        views.controlPoints, generatedGapLineRanges(views.controlPoints), currentLinePosition);
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
    // From the full list, before the plane-distance filter below keeps only
    // the nearby controls.
    overlay.gapLineRanges = generatedGapLineRanges(views.controlPoints);
    overlay.damagedLineRanges = generatedDamagedLineRanges(views.controlPoints);
    overlay.currentLineMarkerState = generatedBlockedLineMarkerState(
        views.controlPoints, overlay.gapLineRanges, linePosition);
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

// The link-state palette shared by linked control points, their connector
// lines and the linked fiber's projected X marker: purple approved, blue
// pending, orange same-H/V approved, light orange same-H/V pending.
// Defined in the .cpp: this header is also compiled into QtCore-only tests.
[[nodiscard]] QColor generatedLinkStateColor(bool pending, bool sameHv, int alpha);

// The kollesis-termination ring colour: the ordinary control-point yellow.
// The tag is told apart by form, not hue: a tagged point draws as a hollow
// yellow ring (a linked one keeps its link-state fill inside the ring).
// Shared with the overview bar and the Fiber Map so the tag looks the same
// everywhere.
[[nodiscard]] QColor generatedKollesisTerminationColor(int alpha);

// The current-position marker colour per placement state: green allowed,
// red blocked, cyan neutral. Shared by the renderer, the dialog's fast
// overlays and the overview bar. Defined in the .cpp (QtCore-only tests).
[[nodiscard]] QColor generatedCurrentLineMarkerColor(GeneratedCurrentLineMarkerState state,
                                                     int alpha);

// The break colour: amber, told apart from the cyan line and the yellow
// control points by hue and from every solid stroke by form (a break point is
// a dotted ring, a gap span a dotted line). Shared with the overview bar and
// the Fiber Map so the tag looks the same everywhere.
[[nodiscard]] QColor generatedBreakColor(int alpha);

// The dash pattern of the gap and damaged span lines, in pen widths: dashes
// three long, six apart. Shared with the overview bar so they look the same.
inline constexpr qreal kSpanDashOn = 3.0;
inline constexpr qreal kSpanDashOff = 6.0;

// The span line colours: a gap span's dashes in a pastel red, a damaged
// span's in a pastel pink (same dash pattern, told apart by hue). The break
// rings keep the amber of generatedBreakColor.
[[nodiscard]] QColor generatedGapLineColor(int alpha);
[[nodiscard]] QColor generatedDamagedColor(int alpha);


namespace detail
{

struct GeneratedControlPointExtent {
    const GeneratedOverlay::ControlPointMarker* first = nullptr;
    const GeneratedOverlay::ControlPointMarker* last = nullptr;
};

inline GeneratedControlPointExtent generatedControlPointExtent(
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints)
{
    GeneratedControlPointExtent extent;
    for (const auto& control : controlPoints) {
        if (!std::isfinite(control.linePosition)) {
            continue;
        }
        if (!extent.first || control.linePosition < extent.first->linePosition) {
            extent.first = &control;
        }
        if (!extent.last || control.linePosition > extent.last->linePosition) {
            extent.last = &control;
        }
    }
    return extent;
}

} // namespace detail

// A kollesis termination may only sit on a fiber end: the control point with
// the smallest or the largest line position (a single point is both). Markers
// without a finite line position do not take part. Defined inline: this
// header is compiled into QtCore-only tests without the .cpp.
inline bool generatedControlPointIsEndpoint(
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints,
    size_t controlIndex)
{
    const auto extent = detail::generatedControlPointExtent(controlPoints);
    if (!extent.first || !extent.last) {
        return false;
    }
    const auto atIndex = [controlIndex](const GeneratedOverlay::ControlPointMarker* marker) {
        return marker && marker->controlIndex == controlIndex;
    };
    if (atIndex(extent.first) || atIndex(extent.last)) {
        return true;
    }
    // Ties on the extreme line position (collapsed points not yet re-fit)
    // count too: any of them is the fiber's end.
    for (const auto& control : controlPoints) {
        if (control.controlIndex != controlIndex || !std::isfinite(control.linePosition)) {
            continue;
        }
        return control.linePosition == extent.first->linePosition ||
               control.linePosition == extent.last->linePosition;
    }
    return false;
}

// The placement rule itself, on parallel per-control-point vectors so the
// dialog (overlay markers) and the controller (session controls) enforce the
// same thing: true when linePosition lies strictly before a tagged first
// control point or strictly after a tagged last one. The fiber is declared to
// end there, so no control point may be placed beyond it; placing at the
// endpoint's own position (which replaces it) stays allowed. Entries without
// a finite position do not take part; a size mismatch means no tags.
inline bool generatedLinePositionBeyondTaggedEnd(const std::vector<double>& controlLinePositions,
                                                 const std::vector<bool>& tagged,
                                                 double linePosition)
{
    if (!std::isfinite(linePosition) || tagged.size() != controlLinePositions.size()) {
        return false;
    }
    bool haveExtent = false;
    double first = 0.0;
    double last = 0.0;
    for (const double position : controlLinePositions) {
        if (!std::isfinite(position)) {
            continue;
        }
        if (!haveExtent) {
            first = last = position;
            haveExtent = true;
            continue;
        }
        first = std::min(first, position);
        last = std::max(last, position);
    }
    if (!haveExtent) {
        return false;
    }
    const auto taggedAt = [&](double extremePosition) {
        for (size_t i = 0; i < tagged.size(); ++i) {
            if (tagged[i] && controlLinePositions[i] == extremePosition) {
                return true;
            }
        }
        return false;
    };
    if (linePosition < first && taggedAt(first)) {
        return true;
    }
    return linePosition > last && taggedAt(last);
}

// generatedLinePositionBeyondTaggedEnd over overlay markers.
inline bool generatedLinePositionBeyondKollesisTermination(
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints,
    double linePosition)
{
    std::vector<double> positions;
    std::vector<bool> tagged;
    positions.reserve(controlPoints.size());
    tagged.reserve(controlPoints.size());
    for (const auto& control : controlPoints) {
        positions.push_back(control.linePosition);
        tagged.push_back(control.isKollesisTermination);
    }
    return generatedLinePositionBeyondTaggedEnd(positions, tagged, linePosition);
}

// Whether a control's neighbour in line-position order (either side) is a
// kollesis termination. A break is refused at or immediately next to a
// termination, so the span between them can never become a gap at the
// sheet join. Inline: compiled into QtCore-only tests.
inline bool generatedLineOrderNeighbourIsKollesisTermination(
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints,
    size_t controlIndex)
{
    std::vector<const GeneratedOverlay::ControlPointMarker*> sorted;
    sorted.reserve(controlPoints.size());
    for (const auto& control : controlPoints) {
        if (std::isfinite(control.linePosition) &&
            control.controlIndex != std::numeric_limits<size_t>::max()) {
            sorted.push_back(&control);
        }
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto* a, const auto* b) {
        return a->linePosition < b->linePosition;
    });
    for (size_t rank = 0; rank < sorted.size(); ++rank) {
        if (sorted[rank]->controlIndex != controlIndex) {
            continue;
        }
        return (rank > 0 && sorted[rank - 1]->isKollesisTermination) ||
               (rank + 1 < sorted.size() && sorted[rank + 1]->isKollesisTermination);
    }
    return false;
}

// What a strip click (or hover) addresses, decided by its scene x alone so
// the height of the mouse over the strip never matters. `rank` indexes the
// line-ordered control points; a span runs from `rank` to `rank + 1`.
struct GeneratedStripContextTarget {
    enum class Kind { ControlPoint, Span };
    Kind kind = Kind::ControlPoint;
    size_t rank = 0;

    bool operator==(const GeneratedStripContextTarget& other) const
    {
        return kind == other.kind && rank == other.rank;
    }
};

// Each span is divided along its drawn length: the quarter next to either
// control point belongs to that point, the middle half is the span.
constexpr double kGeneratedStripContextControlFraction = 0.25;

// A control's position on the DISPLAYED line: its displayedLinePosition
// while it is a provisional control re-expressed on that line, else its own
// linePosition (which then indexes the displayed line).
// A table a binary search may run on: every entry finite and nondecreasing
// (std::is_sorted alone lets interior NaNs through).
inline bool generatedFiniteSortedTable(const std::vector<double>& table)
{
    for (size_t i = 0; i < table.size(); ++i) {
        if (!std::isfinite(table[i]) || (i > 0 && table[i] < table[i - 1])) {
            return false;
        }
    }
    return true;
}

inline double generatedShownLinePosition(const GeneratedOverlay::ControlPointMarker& control)
{
    return std::isfinite(control.displayedLinePosition) ? control.displayedLinePosition
                                                        : control.linePosition;
}

// The strip grid column at an arc length along the centre line, from the
// map's per-column arc lengths (nondecreasing); clamped to the strip.
inline double generatedStripGridColumnForArcLength(
    const vc::lasagna::LineStripPositionMap& positionMap,
    double arcLength)
{
    const auto& arcs = positionMap.stripGridArclengths;
    if (!positionMap.valid() || arcs.empty() || !std::isfinite(arcLength) ||
        !generatedFiniteSortedTable(arcs)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (arcLength <= arcs.front()) {
        return 0.0;
    }
    if (arcLength >= arcs.back() || !std::isfinite(arcs.back())) {
        return static_cast<double>(arcs.size() - 1);
    }
    const auto upper = std::upper_bound(arcs.begin(), arcs.end(), arcLength);
    const size_t b = static_cast<size_t>(upper - arcs.begin());
    if (b == 0 || b >= arcs.size()) {
        // A map with non-finite or unsorted entries: no interpolation.
        return b == 0 ? 0.0 : static_cast<double>(arcs.size() - 1);
    }
    const size_t a = b - 1;
    const double span = arcs[b] - arcs[a];
    const double t = span > 0.0 && std::isfinite(span) ? (arcLength - arcs[a]) / span : 0.0;
    return static_cast<double>(a) + std::clamp(t, 0.0, 1.0);
}

// The strip grid column a control is drawn at on this map's strip: by arc
// length when the control's arc length is measured on this map's line, else
// by its position on the displayed line. The one mapping every strip-side
// consumer (markers, hover zones, direction picking) must share.
inline double generatedStripControlGridColumn(const GeneratedOverlay::ControlPointMarker& control,
                                              const vc::lasagna::LineStripPositionMap& positionMap)
{
    if (std::isfinite(control.arcLength) && positionMap.valid() && control.lineRevision != 0 &&
        control.lineRevision == positionMap.lineRevision) {
        const double column = generatedStripGridColumnForArcLength(positionMap, control.arcLength);
        if (std::isfinite(column)) {
            return column;
        }
    }
    const double shown = generatedShownLinePosition(control);
    return positionMap.valid() ? positionMap.originalPositionToStripGridColumn(shown) : shown;
}

// The strip's controls in line order with the strip grid column of each
// control's line position on the centre line: the space the click zones are
// measured in. Grid columns are nondecreasing in line order by construction
// (the position map maps arc length monotonically; without a map the column
// is the line position itself) and scene x is affine in the grid column
// under the strip camera, so the quarter rule holds in either. The index
// needs no camera and no projection: it is rebuilt only when the controls or
// the map change, never per hover or pan tick.
struct GeneratedStripContextIndex {
    // Into the control vector the index was built from, line order.
    std::vector<size_t> controlIndices;
    // Nondecreasing, parallel to controlIndices.
    std::vector<double> gridColumns;

    bool empty() const { return gridColumns.empty(); }
};

inline GeneratedStripContextIndex buildGeneratedStripContextIndex(
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints,
    size_t linePointCount,
    const vc::lasagna::LineStripPositionMap& positionMap)
{
    std::vector<std::pair<double, size_t>> ordered;
    for (size_t i = 0; i < controlPoints.size(); ++i) {
        const auto& control = controlPoints[i];
        // Validity and the fallback column are judged on the displayed line
        // (a provisional control's own index may lie beyond it).
        const double shown = generatedShownLinePosition(control);
        if (control.controlIndex == std::numeric_limits<size_t>::max() ||
            !validGeneratedLinePosition(shown, linePointCount)) {
            continue;
        }
        const double column = positionMap.valid()
            ? positionMap.originalPositionToStripGridColumn(shown)
            : shown;
        if (!std::isfinite(column)) {
            continue;
        }
        ordered.push_back({control.linePosition, i});
    }
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    GeneratedStripContextIndex index;
    index.controlIndices.reserve(ordered.size());
    index.gridColumns.reserve(ordered.size());
    for (const auto& [linePosition, i] : ordered) {
        const auto& control = controlPoints[i];
        double column = generatedStripControlGridColumn(control, positionMap);
        // Monotonic by construction; this only absorbs rounding in the map.
        if (!index.gridColumns.empty()) {
            column = std::max(column, index.gridColumns.back());
        }
        index.controlIndices.push_back(i);
        index.gridColumns.push_back(column);
    }
    return index;
}

// `columns` is a GeneratedStripContextIndex's nondecreasing grid columns,
// `column` the pointer's. The span containing the column (found by binary
// search) decides: within a quarter of its length of either end it is that
// control, the middle half is the span. Equal columns (two controls on one
// strip column) make a zero-length span that claims nothing; on their shared
// column, and within the quarter zone next to it, the first of them wins.
// Beyond the ends the end control. Empty input or a non-finite column yields
// no target.
inline std::optional<GeneratedStripContextTarget> generatedStripContextTarget(
    const std::vector<double>& columns,
    double column)
{
    if (columns.empty() || !std::isfinite(column)) {
        return std::nullopt;
    }
    using Kind = GeneratedStripContextTarget::Kind;
    // The first rank sharing the column of `rank`.
    const auto firstOf = [&](size_t rank) {
        return static_cast<size_t>(
            std::lower_bound(columns.begin(), columns.end(), columns[rank]) - columns.begin());
    };
    const size_t last = columns.size() - 1;
    if (column <= columns.front()) {
        return GeneratedStripContextTarget{Kind::ControlPoint, 0};
    }
    if (column >= columns.back()) {
        return GeneratedStripContextTarget{Kind::ControlPoint, firstOf(last)};
    }
    // First column strictly greater than the pointer's: the span from the
    // previous rank to this one contains it, with positive length.
    const size_t rank = static_cast<size_t>(
        std::upper_bound(columns.begin(), columns.end(), column) - columns.begin());
    const double a = columns[rank - 1];
    const double b = columns[rank];
    const double t = (column - a) / (b - a);
    if (t < kGeneratedStripContextControlFraction) {
        return GeneratedStripContextTarget{Kind::ControlPoint, firstOf(rank - 1)};
    }
    if (t > 1.0 - kGeneratedStripContextControlFraction) {
        return GeneratedStripContextTarget{Kind::ControlPoint, rank};
    }
    return GeneratedStripContextTarget{Kind::Span, rank - 1};
}

// ---- Overview bar layout --------------------------------------------------
// The overview bar draws each control at a fraction of its width. While a
// solve is running or queued the line the controller publishes runs ahead of
// the one on screen (a placement publishes its spliced controls before any
// landing, landings resample and regrow tails), so drawn from live positions
// the dots wander until the end. The bar therefore keeps the FRACTIONS of
// the last settled geometry, keyed by each control's volume point, and
// everything is measured in ARC LENGTH on the displayed line: a control still
// present keeps its fraction; a newly placed one is placed by its arc length
// between its matched neighbours (which is where the current-position marker
// stood when it was placed, see generatedDisplaySpaceControlArcLengths) and
// keeps that fraction until the geometry settles; the marker, the gap and
// damaged pieces and the bar's clicks map through the same anchors, by arc
// length, so they agree with the dots whatever the line's sampling.
struct GeneratedOverviewAnchor {
    // The control's identity (ControlPointMarker::identity); 0 when the
    // producer has none, then the point stands in for it.
    uint64_t identity = 0;
    cv::Vec3f point{std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN()};
    // On the line the anchors were computed for.
    double linePosition = 0.0;
    // Arc length from that line's start to the control (display units).
    double arcLength = 0.0;
    // Across the bar, 0..1, nondecreasing in line order.
    double fraction = 0.0;
};

struct GeneratedOverviewLayout {
    std::vector<GeneratedOverviewAnchor> anchors;
    // Arc length of the whole line the anchors belong to.
    double totalArcLength = 0.0;

    bool empty() const { return anchors.empty(); }
};

// Cumulative arc length per line point (non-finite points add nothing).
inline std::vector<double> generatedCumulativeArcLength(const std::vector<cv::Vec3f>& linePoints)
{
    std::vector<double> cumulative(linePoints.size(), 0.0);
    for (size_t i = 1; i < linePoints.size(); ++i) {
        double step = 0.0;
        if (finiteGeneratedPoint(linePoints[i]) && finiteGeneratedPoint(linePoints[i - 1])) {
            // In double: float coordinates at the volume's scale overflow a
            // float difference long before they are implausible.
            const cv::Vec3d delta = cv::Vec3d(linePoints[i]) - cv::Vec3d(linePoints[i - 1]);
            step = cv::norm(delta);
            if (!std::isfinite(step)) {
                step = 0.0;
            }
        }
        cumulative[i] = cumulative[i - 1] + step;
    }
    return cumulative;
}

// Arc length at a (fractional) line position, clamped to the line.
inline double generatedArcLengthAt(const std::vector<double>& cumulative, double linePosition)
{
    if (cumulative.empty() || !std::isfinite(linePosition)) {
        return 0.0;
    }
    const double last = static_cast<double>(cumulative.size() - 1);
    const double p = std::clamp(linePosition, 0.0, last);
    const size_t a = static_cast<size_t>(std::floor(p));
    const size_t b = std::min(a + 1, cumulative.size() - 1);
    const double t = p - static_cast<double>(a);
    return cumulative[a] * (1.0 - t) + cumulative[b] * t;
}

// The (fractional) line position at an arc length; flat stretches (repeated
// points) resolve to their first position.
inline double generatedLinePositionAtArcLength(const std::vector<double>& cumulative, double arcLength)
{
    if (cumulative.empty() || !std::isfinite(arcLength) || !generatedFiniteSortedTable(cumulative)) {
        return 0.0;
    }
    if (arcLength <= cumulative.front()) {
        return 0.0;
    }
    // First point whose arc length reaches `arcLength`: exactly on a point
    // (or on a run of repeated points) that is the point itself, the first
    // of the run; otherwise interpolate from the previous one.
    const auto lower = std::lower_bound(cumulative.begin(), cumulative.end(), arcLength);
    const size_t b = static_cast<size_t>(lower - cumulative.begin());
    if (b >= cumulative.size()) {
        return static_cast<double>(cumulative.size() - 1);
    }
    if (cumulative[b] == arcLength || b == 0) {
        return static_cast<double>(b);
    }
    const size_t a = b - 1;
    const double span = cumulative[b] - cumulative[a];
    return static_cast<double>(a) + (span > 0.0 ? (arcLength - cumulative[a]) / span : 0.0);
}

namespace overview_detail {
inline std::vector<const GeneratedOverlay::ControlPointMarker*> lineOrderedControls(
    const std::vector<GeneratedOverlay::ControlPointMarker>& controls)
{
    std::vector<const GeneratedOverlay::ControlPointMarker*> ordered;
    for (const auto& control : controls) {
        if (std::isfinite(control.linePosition)) {
            ordered.push_back(&control);
        }
    }
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const auto* a, const auto* b) { return a->linePosition < b->linePosition; });
    return ordered;
}
inline bool samePoint(const cv::Vec3f& a, const cv::Vec3f& b)
{
    constexpr float kToleranceVx = 1.0e-3f;
    return finiteGeneratedPoint(a) && finiteGeneratedPoint(b) &&
           std::abs(a[0] - b[0]) <= kToleranceVx && std::abs(a[1] - b[1]) <= kToleranceVx &&
           std::abs(a[2] - b[2]) <= kToleranceVx;
}
inline double lerp(double a, double b, double t)
{
    return a + (b - a) * std::clamp(t, 0.0, 1.0);
}
// A control's arc length: the one its producer measured, else measured on
// `cumulative` (then assumed to be the control's line).
inline double controlArcLength(const GeneratedOverlay::ControlPointMarker& control,
                               const std::vector<double>& cumulative)
{
    return std::isfinite(control.arcLength) ? control.arcLength
                                            : generatedArcLengthAt(cumulative, control.linePosition);
}
inline double lineArcLength(const std::vector<const GeneratedOverlay::ControlPointMarker*>& ordered,
                            const std::vector<double>& cumulative)
{
    for (const auto* control : ordered) {
        if (std::isfinite(control->lineArcLength)) {
            return control->lineArcLength;
        }
    }
    return cumulative.empty() ? 0.0 : cumulative.back();
}
inline double arcFraction(double arcLength, double totalArcLength)
{
    return totalArcLength > 0.0 ? std::clamp(arcLength / totalArcLength, 0.0, 1.0) : 0.0;
}
// The anchor for a control: by identity when both sides have one (exact,
// one anchor per control); otherwise, for producers without identities, the
// first unused anchor within tolerance of the point. Anchors already `used`
// are skipped (one control per anchor).
inline const GeneratedOverviewAnchor* findAnchor(const std::vector<GeneratedOverviewAnchor>& anchors,
                                                 uint64_t identity,
                                                 const cv::Vec3f& point,
                                                 std::vector<bool>* used = nullptr)
{
    const GeneratedOverviewAnchor* best = nullptr;
    size_t bestIndex = 0;
    // Exact identity first, over all anchors; the point only stands in
    // where one side has no identity (an anonymous anchor must not shadow
    // an identified control's own anchor).
    if (identity != 0) {
        for (size_t i = 0; i < anchors.size(); ++i) {
            if (used && i < used->size() && (*used)[i]) {
                continue;
            }
            if (anchors[i].identity == identity) {
                best = &anchors[i];
                bestIndex = i;
                break;
            }
        }
    }
    if (!best) {
        for (size_t i = 0; i < anchors.size(); ++i) {
            if (used && i < used->size() && (*used)[i]) {
                continue;
            }
            if ((identity == 0 || anchors[i].identity == 0) && samePoint(anchors[i].point, point)) {
                best = &anchors[i];
                bestIndex = i;
                break;
            }
        }
    }
    if (best && used) {
        if (used->size() < anchors.size()) {
            used->resize(anchors.size(), false);
        }
        (*used)[bestIndex] = true;
    }
    return best;
}
// Anchors fit for the piecewise mappings: finite, nondecreasing arc length
// and fraction (an unfit anchor is dropped, order kept).
inline std::vector<GeneratedOverviewAnchor> mappingAnchors(const std::vector<GeneratedOverviewAnchor>& anchors)
{
    std::vector<GeneratedOverviewAnchor> fit;
    for (const auto& anchor : anchors) {
        if (!std::isfinite(anchor.arcLength) || !std::isfinite(anchor.fraction)) {
            continue;
        }
        GeneratedOverviewAnchor a = anchor;
        if (!fit.empty()) {
            a.arcLength = std::max(a.arcLength, fit.back().arcLength);
            a.fraction = std::max(a.fraction, fit.back().fraction);
        }
        fit.push_back(a);
    }
    return fit;
}
} // namespace overview_detail

// The layout of a settled geometry: every control at its arc-length fraction
// of the line it indexes (`linePoints`, consistent with the controls in a
// settled publish; the controls' own arc lengths take precedence).
inline GeneratedOverviewLayout generatedOverviewSettledLayout(
    const std::vector<GeneratedOverlay::ControlPointMarker>& controls,
    const std::vector<cv::Vec3f>& linePoints)
{
    using namespace overview_detail;
    GeneratedOverviewLayout layout;
    const auto ordered = lineOrderedControls(controls);
    const auto cumulative = generatedCumulativeArcLength(linePoints);
    layout.totalArcLength = lineArcLength(ordered, cumulative);
    if (!std::isfinite(layout.totalArcLength) || layout.totalArcLength < 0.0) {
        layout.totalArcLength = 0.0;
    }
    for (const auto* control : ordered) {
        double arc = controlArcLength(*control, cumulative);
        if (!std::isfinite(arc)) {
            continue;
        }
        // Invariants the mappings rely on: arcs within the line and
        // nondecreasing in line order (inconsistent metadata is clamped).
        arc = std::clamp(arc, 0.0, std::max(layout.totalArcLength, 0.0));
        if (!layout.anchors.empty()) {
            arc = std::max(arc, layout.anchors.back().arcLength);
        }
        layout.anchors.push_back({control->identity, control->point, control->linePosition, arc,
                                  arcFraction(arc, layout.totalArcLength)});
    }
    return layout;
}

// An arc length's fraction across the bar: piecewise linear in arc length
// through the anchors (sorted by arc length), the tails stretched to the
// bar's ends; without anchors the plain arc-length fraction.
inline double generatedOverviewFraction(const std::vector<GeneratedOverviewAnchor>& rawAnchors,
                                        double arcLength,
                                        double totalArcLength)
{
    using namespace overview_detail;
    if (!std::isfinite(arcLength)) {
        return 0.0;
    }
    if (!std::isfinite(totalArcLength)) {
        totalArcLength = 0.0;
    }
    const auto anchors = mappingAnchors(rawAnchors);
    if (anchors.empty()) {
        return arcFraction(arcLength, totalArcLength);
    }
    const auto& first = anchors.front();
    const auto& last = anchors.back();
    if (arcLength <= first.arcLength) {
        return first.arcLength > 0.0 ? lerp(0.0, first.fraction, arcLength / first.arcLength)
                                     : first.fraction;
    }
    if (arcLength >= last.arcLength) {
        const double span = totalArcLength - last.arcLength;
        return span > 0.0 ? lerp(last.fraction, 1.0, (arcLength - last.arcLength) / span)
                          : last.fraction;
    }
    const auto upper = std::upper_bound(
        anchors.begin(), anchors.end(), arcLength,
        [](double value, const GeneratedOverviewAnchor& anchor) { return value < anchor.arcLength; });
    const auto& b = *upper;
    const auto& a = *(upper - 1);
    const double span = b.arcLength - a.arcLength;
    return span > 0.0 ? lerp(a.fraction, b.fraction, (arcLength - a.arcLength) / span) : a.fraction;
}

// The inverse: the arc length drawn at `fraction` of the bar.
inline double generatedOverviewArcLength(const std::vector<GeneratedOverviewAnchor>& rawAnchors,
                                         double fraction,
                                         double totalArcLength)
{
    using namespace overview_detail;
    if (!std::isfinite(fraction)) {
        return 0.0;
    }
    if (!std::isfinite(totalArcLength)) {
        totalArcLength = 0.0;
    }
    fraction = std::clamp(fraction, 0.0, 1.0);
    const auto anchors = mappingAnchors(rawAnchors);
    if (anchors.empty()) {
        return fraction * std::max(totalArcLength, 0.0);
    }
    const auto& first = anchors.front();
    const auto& last = anchors.back();
    if (fraction <= first.fraction) {
        return first.fraction > 0.0 ? lerp(0.0, first.arcLength, fraction / first.fraction)
                                    : first.arcLength;
    }
    if (fraction >= last.fraction) {
        const double span = 1.0 - last.fraction;
        return span > 0.0 ? lerp(last.arcLength, totalArcLength, (fraction - last.fraction) / span)
                          : last.arcLength;
    }
    const auto upper = std::upper_bound(
        anchors.begin(), anchors.end(), fraction,
        [](double value, const GeneratedOverviewAnchor& anchor) { return value < anchor.fraction; });
    const auto& b = *upper;
    const auto& a = *(upper - 1);
    const double span = b.fraction - a.fraction;
    return span > 0.0 ? lerp(a.arcLength, b.arcLength, (fraction - a.fraction) / span) : a.arcLength;
}

// The anchors while the geometry is in flight. `known` holds fractions by
// control point (the settled layout's, plus the fractions already given to
// controls placed since). A control found there keeps its fraction. Any
// other gets its first fraction from `positionMapping` when given: the
// anchors the current-position marker maps through (the displayed line's
// controls with their known fractions), so a new control lands exactly where
// the marker stood at its arc length, even when it replaces a control whose
// anchor it does not inherit. Without a mapping it is placed between its
// nearest known line-order neighbours by the ratio of ARC LENGTHS (the
// controls' arc lengths must all be on one line: the displayed line, see
// generatedDisplaySpaceControlArcLengths), toward the line's start before
// the first known control and toward its end past the last; with nothing
// known at all, at its arc-length fraction. Fractions are made nondecreasing
// in line order.
inline std::vector<GeneratedOverviewAnchor> generatedOverviewFrozenAnchors(
    const std::vector<GeneratedOverviewAnchor>& known,
    const std::vector<GeneratedOverlay::ControlPointMarker>& controls,
    const std::vector<cv::Vec3f>& linePoints,
    const std::vector<GeneratedOverviewAnchor>* positionMapping = nullptr)
{
    using namespace overview_detail;
    const auto ordered = lineOrderedControls(controls);
    const auto cumulative = generatedCumulativeArcLength(linePoints);
    double total = lineArcLength(ordered, cumulative);
    if (!std::isfinite(total) || total < 0.0) {
        total = 0.0;
    }
    std::vector<GeneratedOverviewAnchor> anchors(ordered.size());
    std::vector<bool> matched(ordered.size(), false);
    std::vector<bool> usedKnown(known.size(), false);
    for (size_t i = 0; i < ordered.size(); ++i) {
        anchors[i].identity = ordered[i]->identity;
        anchors[i].point = ordered[i]->point;
        anchors[i].linePosition = ordered[i]->linePosition;
        double arc = controlArcLength(*ordered[i], cumulative);
        if (!std::isfinite(arc)) {
            arc = i > 0 ? anchors[i - 1].arcLength : 0.0;
        }
        arc = std::clamp(arc, 0.0, total);
        anchors[i].arcLength = i > 0 ? std::max(arc, anchors[i - 1].arcLength) : arc;
        if (const auto* anchor = findAnchor(known, ordered[i]->identity, ordered[i]->point, &usedKnown)) {
            anchors[i].fraction = anchor->fraction;
            matched[i] = true;
        }
    }
    for (size_t i = 0; i < anchors.size(); ++i) {
        if (matched[i]) {
            continue;
        }
        const double arc = anchors[i].arcLength;
        if (positionMapping && !positionMapping->empty()) {
            anchors[i].fraction = generatedOverviewFraction(*positionMapping, arc, total);
            continue;
        }
        std::optional<size_t> prev;
        std::optional<size_t> next;
        for (size_t j = i; j-- > 0;) {
            if (matched[j]) { prev = j; break; }
        }
        for (size_t j = i + 1; j < anchors.size(); ++j) {
            if (matched[j]) { next = j; break; }
        }
        if (prev && next) {
            const double span = anchors[*next].arcLength - anchors[*prev].arcLength;
            anchors[i].fraction = span > 0.0
                ? lerp(anchors[*prev].fraction, anchors[*next].fraction,
                       (arc - anchors[*prev].arcLength) / span)
                : anchors[*prev].fraction;
        } else if (prev) {
            const double span = total - anchors[*prev].arcLength;
            anchors[i].fraction = span > 0.0
                ? lerp(anchors[*prev].fraction, 1.0, (arc - anchors[*prev].arcLength) / span)
                : anchors[*prev].fraction;
        } else if (next) {
            const double span = anchors[*next].arcLength;
            anchors[i].fraction = span > 0.0
                ? lerp(0.0, anchors[*next].fraction, arc / span)
                : anchors[*next].fraction;
        } else {
            anchors[i].fraction = arcFraction(arc, total);
        }
    }
    for (size_t i = 0; i < anchors.size(); ++i) {
        anchors[i].fraction = std::clamp(anchors[i].fraction, 0.0, 1.0);
        if (i > 0) {
            anchors[i].fraction = std::max(anchors[i].fraction, anchors[i - 1].fraction);
        }
    }
    return anchors;
}

// When the overview bar may adopt the live geometry as its settled layout.
// Every publish precedes the controller's report on it, so a publish alone
// never counts as settled; a placement request of the dialog that is still
// out keeps the layout it was made against; controls re-expressed on the
// displayed line, or indexing another line, are never a layout.
struct GeneratedOverviewGateState {
    bool layoutEmpty = true;
    bool controlsRebased = false;
    bool controlsIndexDisplayedLine = true;
    bool solveRunning = false;
    bool solvePending = false;
    bool autoReoptimize = true;
    bool geometryUnconfirmed = false;
    bool placementOutstanding = false;
};

inline bool generatedOverviewGeometryInFlight(const GeneratedOverviewGateState& gate)
{
    // Queued edits only ever dispatch in auto mode; in manual mode the
    // spliced line is the geometry until the user asks for a solve.
    return gate.solveRunning || (gate.solvePending && gate.autoReoptimize);
}

inline bool generatedOverviewAdopts(const GeneratedOverviewGateState& gate)
{
    if (gate.controlsRebased || !gate.controlsIndexDisplayedLine) {
        return false;
    }
    if (gate.layoutEmpty) {
        return true;
    }
    return !generatedOverviewGeometryInFlight(gate) && !gate.geometryUnconfirmed &&
           !gate.placementOutstanding;
}

// A control point placement the dialog has requested but whose publish has
// not arrived: the point it asked for, the arc length (on the line the
// dialog showed, revision `lineRevision`) of the position it was placed at,
// the current-position marker's spot.
struct GeneratedPendingPlacement {
    cv::Vec3f point{std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN()};
    // The point ON the displayed line at the placed-at position: what the
    // request is re-placed through on a later displayed line (the clicked
    // point may sit across the strip, nearest to another pass of the line).
    cv::Vec3f anchor{std::numeric_limits<float>::quiet_NaN(),
                     std::numeric_limits<float>::quiet_NaN(),
                     std::numeric_limits<float>::quiet_NaN()};
    // The request this entry belongs to; retired when the request returns
    // without having placed a control (rejected, failed), so no later
    // control can take its arc length.
    uint64_t token = 0;
    double arcLength = std::numeric_limits<double>::quiet_NaN();
    // The placed-at position on that same displayed line (to re-place the
    // request on a later displayed line through its 3D point).
    double linePosition = std::numeric_limits<double>::quiet_NaN();
    // The overview bar fraction the marker stood at when the request was
    // made: the control's fraction until the geometry settles, whatever
    // landings do to the mapping meanwhile. NaN when unknown.
    double fraction = std::numeric_limits<double>::quiet_NaN();
    uint64_t lineRevision = 0;
    // Resolved entries only: the control that took this arc length, and
    // whether it came from a placement of this dialog (drawn on the centre
    // line) rather than from an estimate.
    uint64_t identity = 0;
    bool fromPlacement = false;
};

// Re-expresses controls published for a line that is NOT on screen (their
// lineRevision differs from `displayedRevision`) in the DISPLAYED line's
// arc-length space, `displayed` being that line's layout: a control present
// there takes its displayed arc length; a control resolved by an earlier
// publish (`resolved`, by its own point, for this displayed revision) takes
// the arc length it was given then; a control the dialog asked to place
// takes the recorded arc length of the spot it was placed at (the nearest
// pending placement within tolerance, recorded for this displayed revision;
// consumed into `resolved` under the control's point, so no other control
// can claim it and the control keeps it across later publishes); any other
// control is interpolated between its nearest resolved line-order neighbours
// by the ratio of its own live arc lengths (and recorded in `resolved` too).
// Arc lengths are made nondecreasing in line order, lineArcLength becomes
// the displayed total and lineRevision the displayed revision: the controls
// then read as the displayed line's.
inline std::vector<GeneratedOverlay::ControlPointMarker> generatedDisplaySpaceControlArcLengths(
    const GeneratedOverviewLayout& displayed,
    std::vector<GeneratedOverlay::ControlPointMarker> controls,
    std::vector<GeneratedPendingPlacement>& pending,
    std::vector<GeneratedPendingPlacement>& resolved,
    uint64_t displayedRevision,
    const std::vector<cv::Vec3f>& displayedLinePoints = {})
{
    using namespace overview_detail;
    if (displayed.empty()) {
        return controls;
    }
    const double displayedTotal =
        std::isfinite(displayed.totalArcLength) ? std::max(displayed.totalArcLength, 0.0) : 0.0;
    // `onLine` was judged on the controller's line. A control not already on
    // the displayed line (a new one) is on the displayed centre line only if
    // the displayed point at its displayed arc length is where it is; off it
    // (a click across the strip) its own point must be drawn, not the line.
    const auto displayedCumulative = generatedCumulativeArcLength(displayedLinePoints);
    std::vector<size_t> order;
    for (size_t i = 0; i < controls.size(); ++i) {
        if (std::isfinite(controls[i].linePosition)) {
            order.push_back(i);
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return controls[a].linePosition < controls[b].linePosition;
    });
    std::vector<double> liveArc(order.size()), displayArc(order.size(), 0.0);
    std::vector<bool> resolvedHere(order.size(), false);
    std::vector<bool> usedDisplayed(displayed.anchors.size(), false);
    std::vector<bool> usedResolved(resolved.size(), false);
    std::vector<bool> fromPlacement(order.size(), false);
    double liveTotal = std::numeric_limits<double>::quiet_NaN();
    for (size_t k = 0; k < order.size(); ++k) {
        const auto& control = controls[order[k]];
        liveArc[k] = control.arcLength;
        if (std::isfinite(control.lineArcLength) && !std::isfinite(liveTotal)) {
            liveTotal = control.lineArcLength;
        }
        if (const auto* anchor = findAnchor(displayed.anchors, control.identity, control.point, &usedDisplayed)) {
            displayArc[k] = anchor->arcLength;
            resolvedHere[k] = true;
            continue;
        }
        // Resolved earlier in this flight: exact identity first, over all
        // entries; the point stands in only where one side is anonymous (an
        // anonymous entry must not shadow an identified control's own).
        const auto takeResolved = [&](size_t r) {
            displayArc[k] = resolved[r].arcLength;
            resolvedHere[k] = true;
            fromPlacement[k] = resolved[r].fromPlacement;
            usedResolved[r] = true;
        };
        const auto eligible = [&](size_t r) {
            return !usedResolved[r] && resolved[r].lineRevision == displayedRevision &&
                   std::isfinite(resolved[r].arcLength);
        };
        if (control.identity != 0) {
            for (size_t r = 0; r < resolved.size() && !resolvedHere[k]; ++r) {
                if (eligible(r) && resolved[r].identity == control.identity) {
                    takeResolved(r);
                }
            }
        }
        for (size_t r = 0; r < resolved.size() && !resolvedHere[k]; ++r) {
            if (eligible(r) && (control.identity == 0 || resolved[r].identity == 0) &&
                samePoint(resolved[r].point, control.point)) {
                takeResolved(r);
            }
        }
    }
    // Pending placements: controls still unresolved, in line order, take the
    // nearest placement within tolerance of their point (ties: the one
    // recorded first along the line, so two controls at one point of a
    // returning line get their own), one placement each.
    {
        std::vector<size_t> candidates;
        for (size_t p = 0; p < pending.size(); ++p) {
            if (pending[p].lineRevision == displayedRevision && std::isfinite(pending[p].arcLength) &&
                finiteGeneratedPoint(pending[p].point)) {
                candidates.push_back(p);
            }
        }
        std::stable_sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) {
            return pending[a].arcLength < pending[b].arcLength;
        });
        std::vector<bool> taken(pending.size(), false);
        for (size_t k = 0; k < order.size(); ++k) {
            if (resolvedHere[k]) {
                continue;
            }
            const auto& control = controls[order[k]];
            if (!finiteGeneratedPoint(control.point)) {
                continue;
            }
            constexpr float kPlacementToleranceVx = 0.5f;
            std::optional<size_t> best;
            float bestDistanceSq = kPlacementToleranceVx * kPlacementToleranceVx;
            for (size_t p : candidates) {
                if (taken[p]) {
                    continue;
                }
                const cv::Vec3f delta = pending[p].point - control.point;
                const float distanceSq = delta.dot(delta);
                if (distanceSq < bestDistanceSq || (!best && distanceSq <= bestDistanceSq)) {
                    bestDistanceSq = distanceSq;
                    best = p;
                }
            }
            if (best) {
                const size_t p = *best;
                displayArc[k] = pending[p].arcLength;
                resolvedHere[k] = true;
                fromPlacement[k] = true;
                taken[p] = true;
                GeneratedPendingPlacement done = pending[p];
                done.point = control.point;
                done.lineRevision = displayedRevision;
                done.identity = control.identity;
                done.fromPlacement = true;
                resolved.push_back(done);
            }
        }
        for (size_t p = pending.size(); p-- > 0;) {
            if (taken[p]) {
                pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(p));
            }
        }
    }
    std::vector<bool> interpolated(order.size(), false);
    for (size_t k = 0; k < order.size(); ++k) {
        if (resolvedHere[k]) {
            continue;
        }
        interpolated[k] = true;
        std::optional<size_t> prev;
        std::optional<size_t> next;
        for (size_t j = k; j-- > 0;) {
            if (resolvedHere[j]) { prev = j; break; }
        }
        for (size_t j = k + 1; j < order.size(); ++j) {
            if (resolvedHere[j]) { next = j; break; }
        }
        const bool haveLive = std::isfinite(liveArc[k]);
        if (prev && haveLive && std::isfinite(liveArc[*prev])) {
            const double liveSpan = next && std::isfinite(liveArc[*next])
                ? liveArc[*next] - liveArc[*prev]
                : (std::isfinite(liveTotal) ? liveTotal - liveArc[*prev] : 0.0);
            const double displayTo = next ? displayArc[*next] : displayedTotal;
            displayArc[k] = liveSpan > 0.0
                ? lerp(displayArc[*prev], displayTo, (liveArc[k] - liveArc[*prev]) / liveSpan)
                : displayArc[*prev];
        } else if (next && haveLive && std::isfinite(liveArc[*next])) {
            const double liveSpan = liveArc[*next];
            displayArc[k] = liveSpan > 0.0
                ? lerp(displayArc[*next], 0.0, (liveArc[*next] - liveArc[k]) / liveSpan)
                : displayArc[*next];
        } else {
            displayArc[k] = std::isfinite(liveArc[k]) ? liveArc[k] : 0.0;
        }
    }
    for (size_t k = 0; k < order.size(); ++k) {
        displayArc[k] = std::isfinite(displayArc[k]) ? std::clamp(displayArc[k], 0.0, displayedTotal) : 0.0;
        if (k > 0) {
            displayArc[k] = std::max(displayArc[k], displayArc[k - 1]);
        }
        auto& control = controls[order[k]];
        control.arcLength = displayArc[k];
        control.lineArcLength = displayedTotal;
        control.lineRevision = displayedRevision;
        if (fromPlacement[k]) {
            // (see below: placed by this dialog, drawn on the centre line)
            control.onLine = true;
        }
        if (!displayedCumulative.empty()) {
            control.displayedLinePosition =
                generatedLinePositionAtArcLength(displayedCumulative, displayArc[k]);
            if (fromPlacement[k]) {
                // A control this dialog placed is drawn ON the displayed
                // centre line at the spot it was placed at: the solve now
                // running pulls the line through the point, so that is where
                // it ends up along the line; its across-strip offset is a
                // transient the stale strip cannot show faithfully anyway
                // (and the 3D projection onto that strip fails too often to
                // be relied on for it).
                control.onLine = true;
            } else if (!findAnchor(displayed.anchors, control.identity, control.point) &&
                       finiteGeneratedPoint(control.point)) {
                // A provisional control from elsewhere: on the displayed
                // centre line only if the displayed point at its displayed
                // arc is (about) where it is.
                const cv::Vec3f onDisplayed = interpolatedGeneratedLinePoint(
                    displayedLinePoints, control.displayedLinePosition);
                constexpr float kOnLineToleranceVx = 0.5f;
                control.onLine = finiteGeneratedPoint(onDisplayed) &&
                                 cv::norm(control.point - onDisplayed) <= kOnLineToleranceVx;
            }
        }
        if (interpolated[k] && finiteGeneratedPoint(control.point)) {
            // Keep the estimate: a later publish of the same control must
            // not re-estimate it from arc lengths that moved meanwhile.
            GeneratedPendingPlacement estimate;
            estimate.point = control.point;
            estimate.arcLength = displayArc[k];
            estimate.linePosition = control.displayedLinePosition;
            estimate.lineRevision = displayedRevision;
            estimate.identity = control.identity;
            resolved.push_back(estimate);
        }
    }
    return controls;
}

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
    // Set when the request names a control point explicitly (the overview
    // bar's dot, forwarded as a synthetic strip click): on a strip the target
    // is the control nearest this line position, never a span, whatever the
    // click's scene x resolves to against the markers on screen.
    std::optional<double> pinnedControlLinePosition;
    vc::lasagna::LineStripPositionMap stripPositionMap;
    bool linkWithCandidateEnabled = false;
    QString linkWithCandidateLabel;
    bool mergeWithCandidateEnabled = false;
    QString mergeWithCandidateLabel;
    cv::Vec3f branchLinkDirection{std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::quiet_NaN()};
    // Shown only while a link candidate is designated (empty label = hidden).
    QString newLinkedToCandidateLabel;
    // Fiber file stem for menu labels, resolved when the menu opens.
    std::function<QString(uint64_t)> fiberDisplayNameForId;
    std::function<void(double, cv::Vec3f)> deleteControlPoint;
    std::function<void(size_t)> clearControlCorrections;
    // (clicked volume point, link direction): start a new fiber seeded at the
    // click whose seed control point is linked to the designated candidate.
    std::function<void(cv::Vec3f, cv::Vec3f)> newLineAnnotationLinkedToCandidate;
    std::function<void(uint64_t, int)> openBranch;
    std::function<void(size_t, uint64_t, int)> unlinkBranch;
    // (controlIndex, linkedFiberId, linkedControlPointIndex, newPendingState)
    std::function<void(size_t, uint64_t, int, bool)> setBranchLinkPending;
    std::function<void(size_t, cv::Vec3f)> designateLinkCandidate;
    // Same as designateLinkCandidate, for a link across adjacent windings.
    std::function<void(size_t, cv::Vec3f)> designateAdjacentLinkCandidate;
    std::function<void(size_t, cv::Vec3f)> linkWithCandidate;
    std::function<void(size_t, cv::Vec3f)> mergeWithCandidate;
    std::function<void(uint64_t, cv::Vec3f)> openNearbyAnnotation;
    // --- Span menu (strip viewers only: a Ctrl+right-click on the centre
    // line away from every control point). Each takes the two controls of
    // the span, in line-position order. The interpolation goal lives here
    // and nowhere else.
    std::function<void(size_t, size_t, std::string)> setSegmentInterpolationGoal;
    // (first, second, linkHalves): remove the span, saving both halves as
    // new fibers and closing this one; linkHalves records a pending link
    // between the two new ends ("same winding").
    std::function<void(size_t, size_t, bool)> splitSpan;
    // (first, second, enabled): make the span a gap by tagging both ends as
    // breaks (or undo that where no other gap depends on an end).
    std::function<void(size_t, size_t, bool)> setSpanGap;
    // (first, second, enabled): toggle the damaged span tag.
    std::function<void(size_t, size_t, bool)> setSpanDamaged;
    // (controlIndex, enabled): toggle the kollesis_termination tag on the
    // point. The menu item is checkable and reflects the marker's state.
    std::function<void(size_t, bool)> setKollesisTermination;
    // (controlIndex, enabled): toggle the break tag on the point. Checkable;
    // adding it is disabled while the point is a kollesis termination.
    std::function<void(size_t, bool)> setBreak;
};

// The marker of an adjacent-winding link: an upright triangle whose
// circumradius is the circle radius the point would otherwise draw with, so
// it reads at the same size next to the circles. Shared by every view that
// draws control markers. Declared with QPainterPath incomplete: this header
// is also compiled into QtCore-only tests, so it must not pull in QtGui;
// callers include <QPainterPath> themselves (the forward declaration sits
// at global scope, above the namespace).
QPainterPath generatedTriangleMarkerPath(const QPointF& center, qreal radius);

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
// The viewer overlay-group key applyGeneratedOverlay registers an overlay
// under. Single source for registration and for anything that later addresses
// the group (translateOverlayGroup during a pan).
inline std::string generatedOverlayGroupKey(const std::string& surfaceName)
{
    return "line_annotation_overlay_" + surfaceName;
}
// Registers the overlay's items on the viewer and returns the group key they
// were registered under (empty when nothing was registered).
std::string applyGeneratedOverlay(CChunkedVolumeViewer* viewer,
                                  const std::string& surfaceName,
                                  const GeneratedOverlay& overlay);
void clearGeneratedControlPointContextPreview(CChunkedVolumeViewer* viewer,
                                              const std::string& surfaceName);
// The strip grid column under a scene point (clamped to the strip), the
// space the click zones are measured in. O(1): an affine camera transform.
double generatedStripGridColumnFromScene(CChunkedVolumeViewer* viewer, const QPointF& scenePoint);
// The target a Ctrl+right-click at `scenePoint` would open the menu for: a
// column lookup and a binary search, no projection of any control.
std::optional<GeneratedStripContextTarget> resolveGeneratedStripContextTarget(
    CChunkedVolumeViewer* viewer,
    const GeneratedStripContextIndex& index,
    const QPointF& scenePoint);
// The hover glow on a strip for `target` (none clears it), drawn under its
// own overlay key so the menu preview and the generated overlays are left
// alone. Projects only the one or two controls the target consists of.
std::string generatedStripContextHoverKey(const std::string& surfaceName);
void drawGeneratedStripContextHover(CChunkedVolumeViewer* viewer,
                                    const std::string& surfaceName,
                                    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints,
                                    const GeneratedStripContextIndex& index,
                                    const vc::lasagna::LineStripPositionMap& positionMap,
                                    const std::optional<GeneratedStripContextTarget>& target);
void clearGeneratedStripContextHover(CChunkedVolumeViewer* viewer,
                                     const std::string& surfaceName);
GeneratedControlPointContextResult showGeneratedControlPointContextMenu(
    const GeneratedControlPointContextMenuOptions& options);

} // namespace vc3d::line_annotation
