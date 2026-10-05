#include "LineAnnotationGeneratedViews.hpp"

#include <QPainterPath>

#include "overlays/ViewerOverlayControllerBase.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"

#include <QAction>
#include <QApplication>
#include <QMenu>
#include <QRect>
#include <QWidget>

#include <cmath>
#include <array>

namespace vc3d::line_annotation {

namespace {

bool finiteScenePoint(const QPointF& point)
{
    return std::isfinite(point.x()) && std::isfinite(point.y());
}

QPointF generatedStripControlPointToScene(
    CChunkedVolumeViewer* viewer,
    QuadSurface* surface,
    const GeneratedOverlay::ControlPointMarker& control,
    const vc::lasagna::LineStripPositionMap& controlPositionMap)
{
    // The strip is uniformly parameterized by centerline arclength. The
    // explicit map converts the model point-index coordinate before the O(1)
    // centerline lookup. volumeToScene runs QuadSurface::pointTo — an O(strip length)
    // gradient descent from the strip center — which made every overlay
    // rebuild cost O(controlPoints x lineLength) and dominated zoom/pan
    // lag on many-control-point fibers; keep it only for points that are
    // genuinely off the centerline: manual/no-reoptimization edits retain
    // the clicked off-line volume position until the next optimization,
    // and their markers (and hit tests) must show the true position, not
    // the centerline. Preconditions are re-checked here because the O(1)
    // helper reports failure as a default (finite) QPointF, which must
    // not shadow the fallback.
    if (viewer && surface && std::isfinite(control.linePosition)) {
        const auto* points = surface->rawPointsPtr();
        const cv::Vec2f scale = surface->scale();
        const double gridColumn = controlPositionMap.valid()
            ? controlPositionMap.originalPositionToStripGridColumn(control.linePosition)
            : control.linePosition;
        const int column = static_cast<int>(std::lround(gridColumn));
        if (points && !points->empty() && scale[0] != 0.0f && scale[1] != 0.0f &&
            column >= 0 && column < points->cols) {
            const cv::Vec3f centerlinePoint = (*points)(points->rows / 2, column);
            constexpr float kOnCenterlineToleranceVx = 1.0e-3f;
            const bool onCenterline = !finiteGeneratedPoint(control.point) ||
                (finiteGeneratedPoint(centerlinePoint) &&
                 cv::norm(control.point - centerlinePoint) <= kOnCenterlineToleranceVx);
            if (onCenterline) {
                const QPointF positionScene = generatedStripLinePositionToScene(
                    viewer, surface, control.linePosition, &controlPositionMap);
                if (finiteScenePoint(positionScene)) {
                    return positionScene;
                }
            }
        }
    }
    if (viewer && finiteGeneratedPoint(control.point)) {
        const QPointF pointScene = viewer->volumeToScene(control.point);
        if (finiteScenePoint(pointScene)) {
            return pointScene;
        }
    }
    return generatedStripLinePositionToScene(viewer, surface, control.linePosition,
                                             &controlPositionMap);
}


} // namespace

QColor generatedCurrentLineMarkerColor(GeneratedCurrentLineMarkerState state,
                                       int alpha)
{
    switch (state) {
    case GeneratedCurrentLineMarkerState::Allowed:
        return QColor(40, 220, 120, alpha);
    case GeneratedCurrentLineMarkerState::Blocked:
        return QColor(255, 70, 70, alpha);
    case GeneratedCurrentLineMarkerState::Neutral:
    default:
        return QColor(0, 245, 255, alpha);
    }
}

QColor generatedKollesisTerminationColor(int alpha)
{
    return QColor(255, 230, 0, alpha);
}

QColor generatedBreakColor(int alpha)
{
    return QColor(255, 196, 0, alpha);
}

QColor generatedGapLineColor(int alpha)
{
    return QColor(235, 120, 120, alpha);
}

QColor generatedDamagedColor(int alpha)
{
    return QColor(255, 170, 205, alpha);
}

QPainterPath generatedTriangleMarkerPath(const QPointF& center, qreal radius)
{
    // Vertices at -90, 30 and 150 degrees: apex up.
    constexpr qreal kCos30 = 0.86602540378;
    QPolygonF triangle;
    triangle << QPointF(center.x(), center.y() - radius)
             << QPointF(center.x() + kCos30 * radius, center.y() + 0.5 * radius)
             << QPointF(center.x() - kCos30 * radius, center.y() + 0.5 * radius);
    QPainterPath path;
    path.addPolygon(triangle);
    path.closeSubpath();
    return path;
}

QColor generatedLinkStateColor(bool pending, bool sameHv, int alpha)
{
    if (sameHv) {
        return pending ? QColor(255, 190, 120, alpha) : QColor(255, 140, 0, alpha);
    }
    return pending ? QColor(80, 150, 255, alpha) : QColor(210, 95, 255, alpha);
}

QPointF generatedStripLinePositionToScene(CChunkedVolumeViewer* viewer,
                                          QuadSurface* surface,
                                          double linePosition,
                                          const vc::lasagna::LineStripPositionMap* positionMap)
{
    if (!viewer || !surface) {
        return {};
    }
    const auto* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return {};
    }
    const double gridColumn = positionMap && positionMap->valid()
        ? positionMap->originalPositionToStripGridColumn(linePosition)
        : linePosition;
    if (!std::isfinite(gridColumn)) {
        return {};
    }
    const cv::Vec2d surfacePoint = surface->gridToSurface(
        {gridColumn, static_cast<double>(points->rows / 2)});
    return viewer->surfaceCoordsToScene(static_cast<float>(surfacePoint[0]),
                                        static_cast<float>(surfacePoint[1]));
}

double generatedStripGridColumnFromScene(CChunkedVolumeViewer* viewer, const QPointF& scenePoint)
{
    if (!viewer) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    auto* quad = dynamic_cast<QuadSurface*>(viewer->currentSurface());
    const auto* points = quad ? quad->rawPointsPtr() : nullptr;
    if (!points || points->cols <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const cv::Vec2f surfacePoint = viewer->sceneToSurfaceCoords(scenePoint);
    if (!std::isfinite(surfacePoint[0]) || !std::isfinite(surfacePoint[1])) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const cv::Vec2d gridPoint = quad->surfaceToGrid(
        {static_cast<double>(surfacePoint[0]), static_cast<double>(surfacePoint[1])});
    return std::clamp(gridPoint[0], 0.0, static_cast<double>(points->cols - 1));
}

double generatedLinePositionFromStripScene(CChunkedVolumeViewer* viewer,
                                           const QPointF& scenePoint,
                                           const vc::lasagna::LineStripPositionMap* positionMap)
{
    const double gridColumn = generatedStripGridColumnFromScene(viewer, scenePoint);
    if (!std::isfinite(gridColumn)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return positionMap && positionMap->valid()
        ? positionMap->stripGridColumnToOriginalPosition(gridColumn)
        : gridColumn;
}

std::optional<float> generatedCrossSliceControlPointDistanceThreshold(CChunkedVolumeViewer* viewer)
{
    if (!viewer || !viewer->graphicsView() || !viewer->graphicsView()->viewport()) {
        return std::nullopt;
    }

    auto* view = viewer->graphicsView();
    const QRect viewportRect = view->viewport()->rect();
    if (viewportRect.width() <= 0 || viewportRect.height() <= 0) {
        return std::nullopt;
    }

    const QPointF topLeftScene = view->mapToScene(viewportRect.topLeft());
    const QPointF topRightScene = view->mapToScene(viewportRect.topRight());
    const QPointF bottomLeftScene = view->mapToScene(viewportRect.bottomLeft());
    const QPointF bottomRightScene = view->mapToScene(viewportRect.bottomRight());
    if (!finiteScenePoint(topLeftScene) ||
        !finiteScenePoint(topRightScene) ||
        !finiteScenePoint(bottomLeftScene) ||
        !finiteScenePoint(bottomRightScene)) {
        return std::nullopt;
    }

    const cv::Vec3f topLeft = viewer->sceneToVolume(topLeftScene);
    const cv::Vec3f topRight = viewer->sceneToVolume(topRightScene);
    const cv::Vec3f bottomLeft = viewer->sceneToVolume(bottomLeftScene);
    const cv::Vec3f bottomRight = viewer->sceneToVolume(bottomRightScene);
    if (!finiteGeneratedPoint(topLeft) ||
        !finiteGeneratedPoint(topRight) ||
        !finiteGeneratedPoint(bottomLeft) ||
        !finiteGeneratedPoint(bottomRight)) {
        return std::nullopt;
    }

    const float visibleWidthVx = std::max(cv::norm(topRight - topLeft),
                                          cv::norm(bottomRight - bottomLeft));
    const float visibleHeightVx = std::max(cv::norm(bottomLeft - topLeft),
                                           cv::norm(bottomRight - topRight));
    if (!std::isfinite(visibleWidthVx) ||
        !std::isfinite(visibleHeightVx) ||
        visibleWidthVx <= 0.0f ||
        visibleHeightVx <= 0.0f) {
        return std::nullopt;
    }

    return std::min(visibleWidthVx, visibleHeightVx) * 0.05f;
}

GeneratedOverlay makeGeneratedCrossSliceOverlayForPlane(const GeneratedViews& views,
                                                        double linePosition,
                                                        bool emphasized,
                                                        CChunkedVolumeViewer* viewer,
                                                        PlaneSurface* plane,
                                                        const GeneratedControlPointLinePositionIndex* controlIndex)
{
    const std::optional<float> threshold =
        plane ? generatedCrossSliceControlPointDistanceThreshold(viewer) : std::nullopt;
    const std::optional<double> linePositionRadius =
        threshold ? std::optional<double>(generatedLinePositionRadiusForVolumeThreshold(
                        views.linePoints,
                        linePosition,
                        *threshold))
                  : std::nullopt;
    GeneratedOverlay overlay = makeGeneratedCrossSliceOverlay(
        views,
        linePosition,
        emphasized,
        threshold,
        [plane](const cv::Vec3f& point) {
            return plane ? plane->pointDist(point) : std::numeric_limits<float>::quiet_NaN();
        },
        controlIndex,
        linePositionRadius);
    return overlay;
}

GeneratedOverlay makeGeneratedCrossSliceControlOverlayForPlane(
    const GeneratedViews& views,
    double linePosition,
    CChunkedVolumeViewer* viewer,
    PlaneSurface* plane,
    const GeneratedControlPointLinePositionIndex* controlIndex)
{
    GeneratedOverlay overlay =
        makeGeneratedCrossSliceOverlayForPlane(views,
                                               linePosition,
                                               false,
                                               viewer,
                                               plane,
                                               controlIndex);
    overlay.pointMarker = {std::numeric_limits<float>::quiet_NaN(),
                           std::numeric_limits<float>::quiet_NaN(),
                           std::numeric_limits<float>::quiet_NaN()};
    overlay.emphasizedPointMarker = false;
    return overlay;
}

std::string applyGeneratedOverlay(CChunkedVolumeViewer* viewer,
                                  const std::string& surfaceName,
                                  const GeneratedOverlay& overlay)
{
    if (!viewer) {
        return {};
    }

    const std::string key = generatedOverlayGroupKey(surfaceName);
    std::vector<ViewerOverlayControllerBase::OverlayPrimitive> primitives;
    size_t branchPointCount = 0;
    for (const auto& branch : overlay.branchLinePoints) {
        branchPointCount += branch.size();
    }
    const bool drawDirectBranchLinks = !overlay.useSurfaceCenterLine;
    primitives.reserve(3 + branchPointCount + overlay.controlPoints.size() +
                       overlay.predSnapPoints.size() * 2 +
                       (drawDirectBranchLinks ? overlay.branchLinks.size() * 4 : 0) +
                       overlay.fiberIntersections.size() * 3);

    ViewerOverlayControllerBase::OverlayStyle lineStyle;
    lineStyle.penColor = QColor(0, 220, 255, 190);
    lineStyle.penWidth = 1.0;
    lineStyle.z = 150.0;

    ViewerOverlayControllerBase::OverlayStyle branchLineStyle;
    branchLineStyle.penColor = QColor(190, 90, 255, 210);
    branchLineStyle.penWidth = 1.25;
    branchLineStyle.z = 149.0;

    // A gap span (both endpoint controls tagged break) replaces the fiber's
    // own line with a dotted amber one. Same z as the line it stands in for.
    // Dashes three pen widths long, six apart (in pen widths): long enough
    // to read as a broken line at any zoom, unlike the fine dots of the rings.
    ViewerOverlayControllerBase::OverlayStyle gapLineStyle;
    gapLineStyle.penColor = generatedGapLineColor(230);
    gapLineStyle.penWidth = 1.5;
    gapLineStyle.penStyle = Qt::CustomDashLine;
    gapLineStyle.penCap = Qt::FlatCap;
    gapLineStyle.dashPattern = {kSpanDashOn, kSpanDashOff};
    gapLineStyle.z = 150.0;

    // A damaged span: the same dashes in the pastel pink.
    ViewerOverlayControllerBase::OverlayStyle damagedLineStyle = gapLineStyle;
    damagedLineStyle.penColor = generatedDamagedColor(230);
    // Which of the fiber's span styles a stretch of line takes.
    enum class SpanStyle { Line, Gap, Damaged };
    const auto spanStyleAt = [&overlay](double previousLinePosition, double currentLinePosition) {
        if (generatedLineSegmentInGap(previousLinePosition, currentLinePosition, overlay.gapLineRanges)) {
            return SpanStyle::Gap;
        }
        if (generatedLineSegmentInGap(previousLinePosition, currentLinePosition,
                                      overlay.damagedLineRanges)) {
            return SpanStyle::Damaged;
        }
        return SpanStyle::Line;
    };
    const auto pushStyledStrip = [&](std::vector<QPointF> points, SpanStyle spanStyle) {
        if (points.size() < 2) {
            return;
        }
        switch (spanStyle) {
        case SpanStyle::Gap:
            primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{
                std::move(points), false, gapLineStyle});
            break;
        case SpanStyle::Damaged:
            primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{
                std::move(points), false, damagedLineStyle});
            break;
        case SpanStyle::Line:
            primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{
                std::move(points), false, lineStyle});
            break;
        }
    };
    const auto pushStyledSurfaceStrip = [&](float x0, float x1, float y, SpanStyle spanStyle) {
        const std::vector<cv::Vec2f> points{cv::Vec2f(x0, y), cv::Vec2f(x1, y)};
        switch (spanStyle) {
        case SpanStyle::Gap:
            primitives.push_back(ViewerOverlayControllerBase::SurfaceLineStripPrimitive{
                points, false, gapLineStyle});
            break;
        case SpanStyle::Damaged:
            primitives.push_back(ViewerOverlayControllerBase::SurfaceLineStripPrimitive{
                points, false, damagedLineStyle});
            break;
        case SpanStyle::Line:
            primitives.push_back(ViewerOverlayControllerBase::SurfaceLineStripPrimitive{
                points, false, lineStyle});
            break;
        }
    };

    ViewerOverlayControllerBase::OverlayStyle seedStyle;
    seedStyle.penColor = QColor(255, 230, 0, 220);
    seedStyle.brushColor = QColor(255, 230, 0, 170);
    seedStyle.penWidth = 1.5;
    seedStyle.z = 161.0;

    ViewerOverlayControllerBase::OverlayStyle controlPointStyle = seedStyle;
    controlPointStyle.z = 160.0;

    ViewerOverlayControllerBase::OverlayStyle branchControlPointStyle = controlPointStyle;
    branchControlPointStyle.penColor = QColor(210, 95, 255, 245);
    branchControlPointStyle.brushColor = QColor(210, 95, 255, 175);
    branchControlPointStyle.penWidth = 2.0;
    branchControlPointStyle.z = 162.0;

    ViewerOverlayControllerBase::OverlayStyle pendingBranchControlPointStyle = branchControlPointStyle;
    pendingBranchControlPointStyle.penColor = QColor(80, 150, 255, 245);
    pendingBranchControlPointStyle.brushColor = QColor(80, 150, 255, 175);
    pendingBranchControlPointStyle.z = 162.5;

    ViewerOverlayControllerBase::OverlayStyle sameHvBranchControlPointStyle = branchControlPointStyle;
    sameHvBranchControlPointStyle.penColor = QColor(255, 140, 0, 245);
    sameHvBranchControlPointStyle.brushColor = QColor(255, 140, 0, 175);

    ViewerOverlayControllerBase::OverlayStyle sameHvPendingBranchControlPointStyle =
        pendingBranchControlPointStyle;
    sameHvPendingBranchControlPointStyle.penColor = QColor(255, 190, 120, 245);
    sameHvPendingBranchControlPointStyle.brushColor = QColor(255, 190, 120, 175);

    ViewerOverlayControllerBase::OverlayStyle linkCandidateControlPointStyle = branchControlPointStyle;
    linkCandidateControlPointStyle.penColor = QColor(60, 235, 120, 245);
    linkCandidateControlPointStyle.brushColor = QColor(60, 235, 120, 175);
    linkCandidateControlPointStyle.z = 163.0;


    // The transient candidate designations outrank the tag: a candidate keeps
    // its own colour and size (the fast current-cut overlay does the same).
    const auto drawsKollesisRing = [](const GeneratedOverlay::ControlPointMarker& control) {
        return control.isKollesisTermination && !control.isLinkCandidate;
    };
    // A break point: dotted amber ring, same size step as the kollesis ring.
    // A point somehow carrying both tags (an edited file) draws as the
    // kollesis termination.
    const auto drawsBreakRing = [&drawsKollesisRing](
                                    const GeneratedOverlay::ControlPointMarker& control) {
        return control.isBreak && !control.isLinkCandidate && !drawsKollesisRing(control);
    };
    const auto drawsTagRing = [&](const GeneratedOverlay::ControlPointMarker& control) {
        return drawsKollesisRing(control) || drawsBreakRing(control);
    };
    // Triangle instead of circle: an adjacent-winding link on the point, or
    // the point designated as an adjacent link candidate (a split candidate
    // keeps its own red circle).
    auto drawsTriangle = [](const GeneratedOverlay::ControlPointMarker& control) {
        return control.isAdjacentLinkCandidate ||
               (control.hasAdjacentLinks && !control.isLinkCandidate);
    };
    auto controlStyleForMarker = [&](const GeneratedOverlay::ControlPointMarker& control)
        -> ViewerOverlayControllerBase::OverlayStyle {
        if (control.isLinkCandidate) {
            return linkCandidateControlPointStyle;
        }
        const bool linked = control.hasPendingLinks || control.hasBranches;
        ViewerOverlayControllerBase::OverlayStyle style;
        if (control.hasPendingLinks) {
            style = control.hasSameHvPendingLinks ? sameHvPendingBranchControlPointStyle
                                                  : pendingBranchControlPointStyle;
        } else if (control.hasBranches) {
            style = control.hasSameHvBranches ? sameHvBranchControlPointStyle
                                              : branchControlPointStyle;
        } else {
            style = control.isSeed ? seedStyle : controlPointStyle;
        }
        if (drawsKollesisRing(control)) {
            // Hollow ring: "yellow means control point, hollow means the
            // fiber ends here". A linked termination keeps the link-state
            // fill inside the ring so the link still reads.
            style.penColor = generatedKollesisTerminationColor(245);
            style.penWidth = 2.5;
            if (!linked) {
                style.brushColor = Qt::transparent;
            }
        } else if (drawsBreakRing(control)) {
            // Dotted amber ring: "the papyrus breaks at this point". A linked
            // break keeps its link-state fill inside the ring as well.
            style.penColor = generatedBreakColor(245);
            style.penWidth = 2.5;
            style.penStyle = Qt::DotLine;
            if (!linked) {
                style.brushColor = Qt::transparent;
            }
        }
        return style;
    };

    ViewerOverlayControllerBase::OverlayStyle markerStyle;
    markerStyle.penColor = QColor(0, 220, 255, 210);
    markerStyle.brushColor = QColor(0, 220, 255, 150);
    markerStyle.penWidth = 1.0;
    markerStyle.z = 151.0;

    ViewerOverlayControllerBase::OverlayStyle currentMarkerStyle = markerStyle;
    currentMarkerStyle.penColor =
        generatedCurrentLineMarkerColor(overlay.currentLineMarkerState, 245);
    currentMarkerStyle.brushColor =
        generatedCurrentLineMarkerColor(overlay.currentLineMarkerState, 210);
    currentMarkerStyle.penWidth = 1.5;
    currentMarkerStyle.z = 153.0;

    ViewerOverlayControllerBase::OverlayStyle predSnapLineStyle;
    predSnapLineStyle.penColor = QColor(255, 120, 40, 185);
    predSnapLineStyle.penWidth = 1.0;
    predSnapLineStyle.z = 158.0;

    ViewerOverlayControllerBase::OverlayStyle predSnapPointStyle;
    predSnapPointStyle.penColor = QColor(255, 120, 40, 225);
    predSnapPointStyle.brushColor = QColor(255, 120, 40, 165);
    predSnapPointStyle.penWidth = 1.0;
    predSnapPointStyle.z = 159.0;

    ViewerOverlayControllerBase::OverlayStyle branchLinkStyle;
    branchLinkStyle.penColor = QColor(255, 60, 180, 225);
    branchLinkStyle.brushColor = QColor(255, 60, 180, 165);
    branchLinkStyle.penWidth = 1.4;
    branchLinkStyle.z = 164.0;

    ViewerOverlayControllerBase::OverlayStyle estimatedBranchLinkStyle = branchLinkStyle;
    estimatedBranchLinkStyle.penColor = QColor(255, 150, 210, 185);
    estimatedBranchLinkStyle.brushColor = QColor(255, 150, 210, 130);
    estimatedBranchLinkStyle.penStyle = Qt::DashLine;

    ViewerOverlayControllerBase::OverlayStyle fiberIntersectionStyle;
    fiberIntersectionStyle.penColor = QColor(255, 245, 75, 245);
    fiberIntersectionStyle.brushColor = Qt::transparent;
    fiberIntersectionStyle.penWidth = 1.25;
    fiberIntersectionStyle.penCap = Qt::FlatCap;
    fiberIntersectionStyle.z = 168.0;

    ViewerOverlayControllerBase::OverlayStyle linkCandidateFiberIntersectionStyle =
        fiberIntersectionStyle;
    linkCandidateFiberIntersectionStyle.penColor = QColor(60, 235, 120, 245);
    linkCandidateFiberIntersectionStyle.penWidth = 1.75;
    linkCandidateFiberIntersectionStyle.z = 168.5;

    ViewerOverlayControllerBase::OverlayStyle branchLinkFiberIntersectionStyle =
        fiberIntersectionStyle;
    branchLinkFiberIntersectionStyle.penColor = QColor(210, 95, 255, 245);
    branchLinkFiberIntersectionStyle.penWidth = 1.75;
    branchLinkFiberIntersectionStyle.z = 168.25;

    auto addVolumePointMarker = [&](const cv::Vec3f& point,
                                    qreal radius,
                                    const ViewerOverlayControllerBase::OverlayStyle& style) {
        if (!finiteGeneratedPoint(point)) {
            return;
        }
        primitives.push_back(ViewerOverlayControllerBase::VolumePointPrimitive{
            point,
            radius,
            style});
    };
    auto addFiberIntersectionMarker =
        [&](const QPointF& scenePoint,
            const ViewerOverlayControllerBase::OverlayStyle& style) {
        if (!finiteScenePoint(scenePoint)) {
            return;
        }
        constexpr qreal kIntersectionArm = 7.5;
        primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{
            {scenePoint + QPointF(-kIntersectionArm, -kIntersectionArm),
             scenePoint + QPointF(kIntersectionArm, kIntersectionArm)},
            false,
            style});
        primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{
            {scenePoint + QPointF(-kIntersectionArm, kIntersectionArm),
             scenePoint + QPointF(kIntersectionArm, -kIntersectionArm)},
            false,
            style});
    };

    std::vector<std::pair<QPointF, double>> sceneLine;
    QPointF seedScene;
    bool hasSeedScene = false;

    if (overlay.useSurfaceCenterLine) {
        auto* quad = dynamic_cast<QuadSurface*>(viewer->currentSurface());
        const auto* points = quad ? quad->rawPointsPtr() : nullptr;
        if (points && !points->empty()) {
            const double maximumLinePosition = overlay.stripPositionMap.valid()
                ? static_cast<double>(overlay.stripPositionMap.originalArclengths.size() - 1)
                : static_cast<double>(points->cols - 1);
            const cv::Vec2f scale = quad->scale();
            if (scale[0] != 0.0f && scale[1] != 0.0f && !overlay.linePoints.empty()) {
                const float centerRow = static_cast<float>(points->rows / 2);
                const float surfaceY = (centerRow - static_cast<float>(points->rows) / 2.0f) / scale[1];
                // The centre line, split at the gap spans: each gap piece is
                // the dotted amber line, everything between stays the fiber's
                // line. Positions map through the strip position map to grid
                // columns and then through the surface's own grid-to-surface
                // mapping, the same route the control point markers take
                // (generatedStripLinePositionToScene), so the pieces end on
                // the markers. The half-width formula of startX/endX is not
                // that mapping: the strip surface's centre need not sit at
                // half its width.
                const double centerGridRow = static_cast<double>(points->rows / 2);
                const auto surfaceXForLinePosition = [&](double linePosition) {
                    const double gridColumn = overlay.stripPositionMap.valid()
                        ? overlay.stripPositionMap.originalPositionToStripGridColumn(linePosition)
                        : linePosition;
                    if (!std::isfinite(gridColumn)) {
                        return std::numeric_limits<float>::quiet_NaN();
                    }
                    return static_cast<float>(quad->gridToSurface({gridColumn, centerGridRow})[0]);
                };
                // The line's own ends in that same frame (the whole strip
                // width), so a clamp cannot shift a piece off its markers.
                const float lineStartX =
                    static_cast<float>(quad->gridToSurface({0.0, centerGridRow})[0]);
                const float lineEndX = static_cast<float>(
                    quad->gridToSurface({static_cast<double>(points->cols - 1), centerGridRow})[0]);
                struct StyledPiece {
                    float x0;
                    float x1;
                    SpanStyle spanStyle;
                    bool operator<(const StyledPiece& other) const { return x0 < other.x0; }
                };
                std::vector<StyledPiece> pieces;
                const auto collectPieces = [&](const std::vector<std::pair<double, double>>& ranges,
                                               SpanStyle spanStyle) {
                    for (const auto& [first, second] : ranges) {
                        const float x0 = surfaceXForLinePosition(
                            std::clamp(first, 0.0, maximumLinePosition));
                        const float x1 = surfaceXForLinePosition(
                            std::clamp(second, 0.0, maximumLinePosition));
                        if (std::isfinite(x0) && std::isfinite(x1) && x1 > x0) {
                            pieces.push_back({std::clamp(x0, lineStartX, lineEndX),
                                              std::clamp(x1, lineStartX, lineEndX),
                                              spanStyle});
                        }
                    }
                };
                collectPieces(overlay.gapLineRanges, SpanStyle::Gap);
                collectPieces(overlay.damagedLineRanges, SpanStyle::Damaged);
                std::sort(pieces.begin(), pieces.end());
                float cursorX = lineStartX;
                for (const auto& piece : pieces) {
                    if (piece.x0 > cursorX) {
                        pushStyledSurfaceStrip(cursorX, piece.x0, surfaceY, SpanStyle::Line);
                    }
                    const float from = std::max(cursorX, piece.x0);
                    if (piece.x1 > from) {
                        pushStyledSurfaceStrip(from, piece.x1, surfaceY, piece.spanStyle);
                    }
                    cursorX = std::max(cursorX, piece.x1);
                }
                if (lineEndX > cursorX) {
                    pushStyledSurfaceStrip(cursorX, lineEndX, surfaceY, SpanStyle::Line);
                }
            }
            if (overlay.controlPoints.empty() &&
                overlay.seedLineIndex >= 0 &&
                static_cast<double>(overlay.seedLineIndex) <= maximumLinePosition) {
                seedScene = generatedStripLinePositionToScene(
                    viewer, quad, overlay.seedLineIndex, &overlay.stripPositionMap);
                hasSeedScene = finiteScenePoint(seedScene);
            }
            for (const double position : overlay.markerLinePositions) {
                if (!std::isfinite(position) ||
                    position < 0.0 ||
                    position > maximumLinePosition ||
                    std::abs(position - overlay.currentLinePosition) < 1.0e-6) {
                    continue;
                }
                const QPointF markerScene = generatedStripLinePositionToScene(
                    viewer, quad, position, &overlay.stripPositionMap);
                if (finiteScenePoint(markerScene)) {
                    primitives.push_back(ViewerOverlayControllerBase::CirclePrimitive{
                        markerScene,
                        2.5,
                        true,
                        markerStyle});
                }
            }
            for (const auto& control : overlay.controlPoints) {
                if (!std::isfinite(control.linePosition) ||
                    control.linePosition < 0.0 ||
                    control.linePosition > maximumLinePosition) {
                    continue;
                }
                const QPointF controlScene =
                    generatedStripControlPointToScene(viewer, quad, control,
                                                      overlay.stripPositionMap);
                if (finiteScenePoint(controlScene)) {
                    if (control.direction) {
                        const double col=overlay.stripPositionMap.valid()
                            ? overlay.stripPositionMap.originalPositionToStripGridColumn(control.linePosition)
                            : control.linePosition;
                        const auto uv=quad->gridToSurface({col,double(quad->rawPointsPtr()->rows/2)});
                        if (const auto frame=generatedStripFrame(quad,uv)) {
                            const QPointF origin=viewer->surfaceCoordsToScene(uv[0],uv[1]);
                            const QPointF projected=viewer->surfaceCoordsToScene(
                                uv[0]+control.direction->dot(frame->along),
                                uv[1]+control.direction->dot(frame->across))-origin;
                            const auto transform=viewer->graphicsView()->viewportTransform();
                            const QPointF pixels=transform.map(origin+projected)-transform.map(origin);
                            const double length=std::hypot(pixels.x(),pixels.y());
                            if (length>1e-6) {
                                const QPointF half=projected*(40.0/length);
                                auto style=lineStyle;
                                style.penColor=QColor(0,245,255);
                                style.penWidth=1.5;
                                style.z=170;
                                primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{
                                    {controlScene-half,controlScene+half},false,style});
                            }
                        }
                    }
                    const qreal radius =
                        (control.hasBranches ? 6.25 : (control.isSeed ? 5.5 : 5.0)) +
                        (drawsTagRing(control) ? 1.0 : 0.0);
                    if (drawsTriangle(control)) {
                        primitives.push_back(ViewerOverlayControllerBase::PainterPathPrimitive{
                            generatedTriangleMarkerPath(controlScene, radius),
                            controlStyleForMarker(control)});
                    } else {
                        primitives.push_back(ViewerOverlayControllerBase::CirclePrimitive{
                            controlScene, radius, true, controlStyleForMarker(control)});
                    }
                }
            }
            for (const auto& predSnap : overlay.predSnapPoints) {
                if (!finiteGeneratedPoint(predSnap.snapPoint)) {
                    continue;
                }
                const QPointF controlScene = finiteGeneratedPoint(predSnap.controlPoint)
                    ? viewer->volumeToScene(predSnap.controlPoint)
                    : generatedStripLinePositionToScene(
                          viewer, quad, predSnap.linePosition, &overlay.stripPositionMap);
                const QPointF snapScene = viewer->volumeToScene(predSnap.snapPoint);
                if (finiteScenePoint(controlScene) && finiteScenePoint(snapScene)) {
                    primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{
                        {controlScene, snapScene},
                        false,
                        predSnapLineStyle});
                    primitives.push_back(ViewerOverlayControllerBase::CirclePrimitive{
                        snapScene,
                        3.0,
                        true,
                        predSnapPointStyle});
                }
            }
            if (std::isfinite(overlay.currentLinePosition)) {
                const QPointF markerScene =
                    generatedStripLinePositionToScene(
                        viewer, quad, overlay.currentLinePosition, &overlay.stripPositionMap);
                if (finiteScenePoint(markerScene)) {
                    if (overlay.currentLineMarkerAsCross) {
                        constexpr qreal kCrossRadius = 5.5;
                        auto crossStyle = currentMarkerStyle;
                        crossStyle.brushColor = Qt::transparent;
                        crossStyle.penCap = Qt::RoundCap;
                        crossStyle.penJoin = Qt::RoundJoin;
                        crossStyle.penWidth = 2.0;
                        crossStyle.z = 170.0;
                        primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{
                            {markerScene + QPointF{-kCrossRadius, -kCrossRadius},
                             markerScene + QPointF{kCrossRadius, kCrossRadius}},
                            false,
                            crossStyle});
                        primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{
                            {markerScene + QPointF{-kCrossRadius, kCrossRadius},
                             markerScene + QPointF{kCrossRadius, -kCrossRadius}},
                            false,
                            crossStyle});
                    } else {
                        primitives.push_back(ViewerOverlayControllerBase::CirclePrimitive{
                            markerScene,
                            4.0,
                            true,
                            currentMarkerStyle});
                    }
                }
            }
        }
    } else if (!overlay.linePoints.empty()) {
        sceneLine.reserve(overlay.linePoints.size());
        for (size_t pointIndex = 0; pointIndex < overlay.linePoints.size(); ++pointIndex) {
            const auto& point = overlay.linePoints[pointIndex];
            if (!finiteGeneratedPoint(point)) {
                continue;
            }
            const QPointF scenePoint = viewer->volumeToScene(point);
            if (finiteScenePoint(scenePoint)) {
                sceneLine.push_back({scenePoint, static_cast<double>(pointIndex)});
            }
        }
    }

    if (!overlay.useSurfaceCenterLine) {
        for (const auto& predSnap : overlay.predSnapPoints) {
            if (!finiteGeneratedPoint(predSnap.controlPoint) ||
                !finiteGeneratedPoint(predSnap.snapPoint)) {
                continue;
            }
            const QPointF controlScene = viewer->volumeToScene(predSnap.controlPoint);
            const QPointF snapScene = viewer->volumeToScene(predSnap.snapPoint);
            if (finiteScenePoint(controlScene) && finiteScenePoint(snapScene)) {
                primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{
                    {controlScene, snapScene},
                    false,
                    predSnapLineStyle});
                primitives.push_back(ViewerOverlayControllerBase::CirclePrimitive{
                    snapScene,
                    4.0,
                    true,
                    predSnapPointStyle});
            }
        }
        for (const auto& control : overlay.controlPoints) {
            const qreal radius = (control.hasBranches ? 12.0 : (control.isSeed ? 11.0 : 10.0)) +
                                 (drawsTagRing(control) ? 2.0 : 0.0);
            if (drawsTriangle(control) && finiteGeneratedPoint(control.point)) {
                // Adjacent-winding links and the adjacent candidate are
                // triangles; the path is built in scene space because the
                // point primitives only know circles.
                const QPointF controlScene = viewer->volumeToScene(control.point);
                if (finiteScenePoint(controlScene)) {
                    primitives.push_back(ViewerOverlayControllerBase::PainterPathPrimitive{
                        generatedTriangleMarkerPath(controlScene, radius),
                        controlStyleForMarker(control)});
                }
                continue;
            }
            addVolumePointMarker(control.point, radius, controlStyleForMarker(control));
        }
    }

    if (drawDirectBranchLinks) {
        for (const auto& link : overlay.branchLinks) {
            const cv::Vec3f visiblePoint = finiteGeneratedPoint(link.planePoint)
                ? link.planePoint
                : link.linkedControlPoint;
            if (!finiteGeneratedPoint(link.localControlPoint) ||
                !finiteGeneratedPoint(visiblePoint)) {
                continue;
            }
            const QPointF localScene = viewer->volumeToScene(link.localControlPoint);
            const QPointF visibleScene = viewer->volumeToScene(visiblePoint);
            if (!finiteScenePoint(localScene) || !finiteScenePoint(visibleScene)) {
                continue;
            }
            const auto& style = link.estimated ? estimatedBranchLinkStyle : branchLinkStyle;
            primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{
                {localScene, visibleScene},
                false,
                style});
            addFiberIntersectionMarker(visibleScene, fiberIntersectionStyle);
        }
    }

    for (const auto& intersection : overlay.fiberIntersections) {
        if (!finiteGeneratedPoint(intersection.point)) {
            continue;
        }
        const QPointF scenePoint = viewer->volumeToScene(intersection.point);
        // Connector and projected X follow the linked control point's palette
        // (pending / same-H/V); the link candidate's green keeps precedence
        // on the X.
        auto linkXStyle = branchLinkFiberIntersectionStyle;
        linkXStyle.penColor = generatedLinkStateColor(intersection.pendingBranchLink,
                                                      intersection.sameHvBranchLink,
                                                      245);
        // Pending glyphs keep drawing over approved ones where they overlap.
        linkXStyle.z = intersection.pendingBranchLink ? 168.3 : 168.25;
        if (intersection.connectorStart &&
            finiteGeneratedPoint(*intersection.connectorStart)) {
            const QPointF connectorScene = viewer->volumeToScene(*intersection.connectorStart);
            if (finiteScenePoint(connectorScene) && finiteScenePoint(scenePoint)) {
                auto connectorStyle = branchLinkStyle;
                connectorStyle.penColor = generatedLinkStateColor(
                    intersection.pendingBranchLink, intersection.sameHvBranchLink, 225);
                connectorStyle.brushColor = generatedLinkStateColor(
                    intersection.pendingBranchLink, intersection.sameHvBranchLink, 165);
                primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{
                    {connectorScene, scenePoint},
                    false,
                    connectorStyle});
            }
        }
        addFiberIntersectionMarker(scenePoint,
                                   intersection.isLinkCandidateFiber
                                       ? linkCandidateFiberIntersectionStyle
                                       : (intersection.projectedBranchLink
                                              ? linkXStyle
                                              : fiberIntersectionStyle));
    }

    if (!overlay.useSurfaceCenterLine) {
        for (const auto& branch : overlay.branchLinePoints) {
            if (branch.size() < 2) {
                continue;
            }
            std::vector<QPointF> branchScene;
            branchScene.reserve(branch.size());
            for (const auto& point : branch) {
                if (!finiteGeneratedPoint(point)) {
                    continue;
                }
                const QPointF scenePoint = viewer->volumeToScene(point);
                if (finiteScenePoint(scenePoint)) {
                    branchScene.push_back(scenePoint);
                }
            }
            if (branchScene.size() >= 2) {
                primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{
                    std::move(branchScene),
                    false,
                    branchLineStyle});
            }
        }
    }

    if (!overlay.useSurfaceCenterLine && sceneLine.size() >= 2) {
        const auto controlRange = overlay.lineTailControlRange.has_value()
            ? overlay.lineTailControlRange
            : generatedControlLinePositionRange(overlay.controlPoints);
        // Consecutive non-tail segments accumulate into one polyline
        // primitive per run (same pattern as the branch lines above): a
        // primitive per segment meant one QGraphicsPathItem per segment,
        // and rebuilding thousands of scene items per overlay refresh
        // dominated zoom/pan lag on long fibers.
        // A run also ends where the line enters or leaves a gap or damaged
        // span, so each draws in its own style.
        std::vector<QPointF> run;
        SpanStyle runStyle = SpanStyle::Line;
        const auto flushRun = [&]() {
            pushStyledStrip(std::move(run), runStyle);
            run = {};
        };
        for (size_t i = 1; i < sceneLine.size(); ++i) {
            const auto& previous = sceneLine[i - 1];
            const auto& current = sceneLine[i];
            if (generatedLineSegmentIsTail(previous.second, current.second, controlRange)) {
                flushRun();
                continue;
            }
            const SpanStyle spanStyle = spanStyleAt(previous.second, current.second);
            if (!run.empty() && spanStyle != runStyle) {
                flushRun();
            }
            if (run.empty()) {
                run.push_back(previous.first);
                runStyle = spanStyle;
            }
            run.push_back(current.first);
        }
        flushRun();
    }

    if (finiteGeneratedPoint(overlay.pointMarker)) {
        addVolumePointMarker(overlay.pointMarker,
                             overlay.emphasizedPointMarker ? 2.5 : 2.0,
                             overlay.emphasizedPointMarker ? currentMarkerStyle : markerStyle);
    }

    if (!hasSeedScene && finiteGeneratedPoint(overlay.seedPoint)) {
        seedScene = viewer->volumeToScene(overlay.seedPoint);
        hasSeedScene = finiteScenePoint(seedScene);
    }

    if (hasSeedScene) {
        const bool emphasizedSeed = overlay.emphasizedPointMarker &&
                                    !finiteGeneratedPoint(overlay.pointMarker);
        const qreal radius = emphasizedSeed ? 6.0 : 4.0;
        if (emphasizedSeed) {
            seedStyle.penColor = QColor(255, 245, 0, 255);
            seedStyle.brushColor = QColor(255, 245, 0, 220);
            seedStyle.penWidth = 2.0;
        }
        if (!overlay.useSurfaceCenterLine && finiteGeneratedPoint(overlay.seedPoint)) {
            addVolumePointMarker(overlay.seedPoint, radius, seedStyle);
        } else {
            primitives.push_back(ViewerOverlayControllerBase::CirclePrimitive{
                seedScene,
                radius,
                true,
                seedStyle});
        }
    }

    ViewerOverlayControllerBase::applyPrimitives(viewer, key, std::move(primitives));
    return key;
}

void clearGeneratedControlPointContextPreview(CChunkedVolumeViewer* viewer,
                                              const std::string& surfaceName)
{
    if (!viewer) {
        return;
    }
    ViewerOverlayControllerBase::applyPrimitives(
        viewer,
        "line_annotation_control_context_" + surfaceName,
        {});
}

namespace {

// The span menu: `ownerRank` indexes sortedControls (by line position); the
// span runs from that control to the next.
std::function<bool(QAction*)> appendGeneratedSpanContextActions(
    QMenu& menu,
    const GeneratedControlPointContextMenuOptions& options,
    const std::vector<const GeneratedOverlay::ControlPointMarker*>& sortedControls,
    size_t ownerRank)
{
    const auto& owner = *sortedControls[ownerRank];
    const auto& next = *sortedControls[ownerRank + 1];

    QString state = QWidget::tr("%1, goal %2")
                        .arg(QChar(owner.interpolationModeMarker))
                        .arg(QString::fromStdString(owner.interpolationGoal));
    if (owner.hasGapToNext) {
        state += QWidget::tr(", gap");
    } else if (owner.hasDamagedToNext) {
        state += QWidget::tr(", damaged");
    }
    QAction* header = menu.addAction(
        QWidget::tr("Span CP %1 to CP %2 (%3)")
            .arg(QString::number(owner.controlIndex), QString::number(next.controlIndex), state));
    header->setEnabled(false);
    menu.addSeparator();

    std::vector<std::pair<QAction*, std::string>> goalActions;
    {
        QMenu* goalMenu = menu.addMenu(QWidget::tr("Interpolation goal"));
        goalMenu->setEnabled(static_cast<bool>(options.setSegmentInterpolationGoal));
        const std::array<std::pair<const char*, const char*>, 4> goals{{
            {"Global", "global"},
            {"Cubic spline", "cspline"},
            {"Lasagna", "lasagna"},
            {"Fiber trace", "trace"},
        }};
        for (const auto& [label, value] : goals) {
            QAction* action = goalMenu->addAction(QWidget::tr(label));
            action->setCheckable(true);
            action->setChecked(owner.interpolationGoal == value);
            goalActions.push_back({action, value});
        }
    }

    QAction* gapAction = nullptr;
    if (options.setSpanGap) {
        // Making the span a gap tags both ends as breaks. A break at or
        // immediately next to a kollesis termination is refused altogether.
        const bool nextToKollesis =
            owner.isKollesisTermination || next.isKollesisTermination ||
            (ownerRank > 0 && sortedControls[ownerRank - 1]->isKollesisTermination) ||
            (ownerRank + 2 < sortedControls.size() &&
             sortedControls[ownerRank + 2]->isKollesisTermination);
        const bool blocked = !owner.hasGapToNext && nextToKollesis;
        gapAction = menu.addAction(
            blocked ? QWidget::tr("Gap (at or next to a kollesis termination)")
                    : QWidget::tr("Gap"));
        gapAction->setCheckable(true);
        gapAction->setChecked(owner.hasGapToNext);
        gapAction->setEnabled(!blocked);
    }
    QAction* damagedAction = nullptr;
    if (options.setSpanDamaged) {
        // Never on a gap span: the papyrus there is missing, not hurt.
        const bool blocked = owner.hasGapToNext;
        damagedAction = menu.addAction(
            blocked ? QWidget::tr("Damaged (gap span)") : QWidget::tr("Damaged"));
        damagedAction->setCheckable(true);
        damagedAction->setChecked(owner.hasDamagedToNext);
        damagedAction->setEnabled(!blocked);
    }

    QAction* splitAction = nullptr;
    QAction* splitAndLinkAction = nullptr;
    if (options.splitSpan) {
        menu.addSeparator();
        // Each half keeps at least 2 control points: the prefix ends at the
        // owner, the suffix starts at the next control.
        const bool enabled = ownerRank + 1 >= 2 && sortedControls.size() - (ownerRank + 1) >= 2;
        splitAction = menu.addAction(
            enabled ? QWidget::tr("Split, different windings")
                    : QWidget::tr("Split (needs 2 control points per half)"));
        splitAction->setEnabled(enabled);
        splitAndLinkAction = menu.addAction(QWidget::tr("Split and link, same winding"));
        splitAndLinkAction->setEnabled(enabled);
    }

    return [=, &options](QAction* selected) {
        if (!selected) {
            return false;
        }
        for (const auto& [action, goal] : goalActions) {
            if (selected == action) {
                options.setSegmentInterpolationGoal(owner.controlIndex, next.controlIndex, goal);
                return true;
            }
        }
        if (gapAction && selected == gapAction && gapAction->isEnabled()) {
            options.setSpanGap(owner.controlIndex, next.controlIndex, !owner.hasGapToNext);
            return true;
        }
        if (damagedAction && selected == damagedAction && damagedAction->isEnabled()) {
            options.setSpanDamaged(owner.controlIndex, next.controlIndex, !owner.hasDamagedToNext);
            return true;
        }
        if (splitAction && selected == splitAction && splitAction->isEnabled()) {
            options.splitSpan(owner.controlIndex, next.controlIndex, false);
            return true;
        }
        if (splitAndLinkAction && selected == splitAndLinkAction && splitAndLinkAction->isEnabled()) {
            options.splitSpan(owner.controlIndex, next.controlIndex, true);
            return true;
        }
        return false;
    };
}

// The glow of a strip target: a soft translucent halo, two widening layers
// of the same warm colour, laid under the markers (z between the line at 150
// and the control points at 160) so the markers keep their own look. Shared
// by the hover and the preview shown while the menu is open. Only the one or
// two controls the target consists of are projected (the marker position,
// which follows a point edited off the centre line).
std::vector<ViewerOverlayControllerBase::OverlayPrimitive> stripContextTargetPrimitives(
    CChunkedVolumeViewer* viewer,
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints,
    const GeneratedStripContextIndex& index,
    const vc::lasagna::LineStripPositionMap& positionMap,
    const GeneratedStripContextTarget& target)
{
    std::vector<ViewerOverlayControllerBase::OverlayPrimitive> primitives;
    if (!viewer) {
        return primitives;
    }
    auto* quad = dynamic_cast<QuadSurface*>(viewer->currentSurface());
    const auto marker = [&](size_t rank) -> const GeneratedOverlay::ControlPointMarker* {
        if (rank >= index.controlIndices.size() ||
            index.controlIndices[rank] >= controlPoints.size()) {
            return nullptr;
        }
        return &controlPoints[index.controlIndices[rank]];
    };
    const auto markerScene = [&](const GeneratedOverlay::ControlPointMarker& control) {
        return generatedStripControlPointToScene(viewer, quad, control, positionMap);
    };

    // Outer halo and inner core: the halo wide and faint, the core narrower
    // and a little stronger, both well short of opaque.
    ViewerOverlayControllerBase::OverlayStyle halo;
    halo.penColor = QColor(255, 150, 60, 40);
    halo.z = 155.0;
    ViewerOverlayControllerBase::OverlayStyle core;
    core.penColor = QColor(255, 150, 60, 80);
    core.z = 155.5;
    if (target.kind == GeneratedStripContextTarget::Kind::Span) {
        const auto* first = marker(target.rank);
        const auto* second = marker(target.rank + 1);
        if (!first || !second) {
            return primitives;
        }
        const QPointF a = markerScene(*first);
        const QPointF b = markerScene(*second);
        if (!finiteScenePoint(a) || !finiteScenePoint(b)) {
            return primitives;
        }
        const std::vector<QPointF> points{a, b};
        halo.penWidth = 14.0;
        core.penWidth = 7.0;
        primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{points, false, halo});
        primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{points, false, core});
        return primitives;
    }
    const auto* control = marker(target.rank);
    if (!control) {
        return primitives;
    }
    const QPointF center = markerScene(*control);
    if (!finiteScenePoint(center)) {
        return primitives;
    }
    // The marker radius drawn by applyGeneratedOverlay; the glow is a filled
    // disc behind it, feathered by the second, larger and fainter disc.
    const qreal radius = control->hasBranches ? 6.25 : (control->isSeed ? 5.5 : 5.0);
    halo.brushColor = halo.penColor;
    halo.penStyle = Qt::NoPen;
    core.brushColor = core.penColor;
    core.penStyle = Qt::NoPen;
    primitives.push_back(ViewerOverlayControllerBase::CirclePrimitive{center, radius + 9.0, true, halo});
    primitives.push_back(ViewerOverlayControllerBase::CirclePrimitive{center, radius + 5.0, true, core});
    return primitives;
}

// Nearest fiber-intersection "X" marker to the click, within a scene-space
// threshold matched to the drawn glyph (arm length 7.5 scene units).
const GeneratedOverlay::FiberIntersectionMarker* nearbyFiberIntersection(
    const GeneratedControlPointContextMenuOptions& options)
{
    constexpr double kFiberIntersectionHitThreshold = 12.0;
    const GeneratedOverlay::FiberIntersectionMarker* nearby = nullptr;
    double bestDistanceSq = kFiberIntersectionHitThreshold * kFiberIntersectionHitThreshold;
    for (const auto& intersection : options.fiberIntersections) {
        if (intersection.fiberId == 0 ||
            intersection.projectedBranchLink ||
            !finiteGeneratedPoint(intersection.point)) {
            continue;
        }
        const QPointF intersectionScene = options.viewer->volumeToScene(intersection.point);
        if (!finiteScenePoint(intersectionScene)) {
            continue;
        }
        const QPointF delta = intersectionScene - options.scenePoint;
        const double distanceSq = delta.x() * delta.x() + delta.y() * delta.y();
        if (distanceSq < bestDistanceSq) {
            bestDistanceSq = distanceSq;
            nearby = &intersection;
        }
    }
    return nearby;
}

// The actions that act on the CLICKED LOCATION rather than on a control
// point or span: offered by the span menu and the control-point menu alike,
// so a click anywhere on a strip can start an annotation or jump to the
// fiber marker under it.
struct ClickLocationActions {
    QAction* openNearbyAnnotation = nullptr;
    QAction* newLineAnnotation = nullptr;
    QAction* newLinkedLineAnnotation = nullptr;
    const GeneratedOverlay::FiberIntersectionMarker* nearbyIntersection = nullptr;

    void addOpenNearby(QMenu& menu,
                       const GeneratedControlPointContextMenuOptions& options,
                       const std::function<QString(uint64_t)>& fiberName)
    {
        nearbyIntersection = nearbyFiberIntersection(options);
        if (options.openNearbyAnnotation && nearbyIntersection) {
            openNearbyAnnotation = menu.addAction(
                QWidget::tr("Go to nearby annotation (%1)")
                    .arg(fiberName(nearbyIntersection->fiberId)));
        }
    }

    void addNewAnnotations(QMenu& menu, const GeneratedControlPointContextMenuOptions& options)
    {
        const bool canSampleClickedVolume =
            options.viewer->sampleSceneVolume(options.scenePoint).has_value();
        newLineAnnotation = menu.addAction(QWidget::tr("New line annotation"));
        newLineAnnotation->setEnabled(canSampleClickedVolume);
        // Only while a link candidate is designated.
        if (options.newLineAnnotationLinkedToCandidate &&
            !options.newLinkedToCandidateLabel.isEmpty()) {
            newLinkedLineAnnotation = menu.addAction(options.newLinkedToCandidateLabel);
            newLinkedLineAnnotation->setEnabled(canSampleClickedVolume);
        }
    }

    // The result for a chosen action of this group; nothing when `selected`
    // is none of them.
    std::optional<GeneratedControlPointContextResult> handle(
        QAction* selected, const GeneratedControlPointContextMenuOptions& options) const
    {
        if (!selected) {
            return std::nullopt;
        }
        if (newLineAnnotation && selected == newLineAnnotation && newLineAnnotation->isEnabled()) {
            return GeneratedControlPointContextResult::NewLineAnnotationRequested;
        }
        if (newLinkedLineAnnotation && selected == newLinkedLineAnnotation &&
            newLinkedLineAnnotation->isEnabled()) {
            const auto clickedVolumePoint = options.viewer->sampleSceneVolume(options.scenePoint);
            if (clickedVolumePoint) {
                options.newLineAnnotationLinkedToCandidate(clickedVolumePoint->position,
                                                           options.branchLinkDirection);
            }
            return GeneratedControlPointContextResult::Handled;
        }
        if (openNearbyAnnotation && selected == openNearbyAnnotation && nearbyIntersection) {
            options.openNearbyAnnotation(nearbyIntersection->fiberId, nearbyIntersection->point);
            return GeneratedControlPointContextResult::Handled;
        }
        return std::nullopt;
    }
};

} // namespace

std::string generatedStripContextHoverKey(const std::string& surfaceName)
{
    return "line_annotation_context_hover_" + surfaceName;
}

std::optional<GeneratedStripContextTarget> resolveGeneratedStripContextTarget(
    CChunkedVolumeViewer* viewer,
    const GeneratedStripContextIndex& index,
    const QPointF& scenePoint)
{
    if (!viewer || index.empty() || !finiteScenePoint(scenePoint)) {
        return std::nullopt;
    }
    return generatedStripContextTarget(index.gridColumns,
                                       generatedStripGridColumnFromScene(viewer, scenePoint));
}

void drawGeneratedStripContextHover(CChunkedVolumeViewer* viewer,
                                    const std::string& surfaceName,
                                    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints,
                                    const GeneratedStripContextIndex& index,
                                    const vc::lasagna::LineStripPositionMap& positionMap,
                                    const std::optional<GeneratedStripContextTarget>& target)
{
    if (!viewer) {
        return;
    }
    std::vector<ViewerOverlayControllerBase::OverlayPrimitive> primitives;
    if (target) {
        primitives = stripContextTargetPrimitives(viewer, controlPoints, index, positionMap, *target);
    }
    ViewerOverlayControllerBase::applyPrimitives(
        viewer, generatedStripContextHoverKey(surfaceName), std::move(primitives));
}

void clearGeneratedStripContextHover(CChunkedVolumeViewer* viewer,
                                     const std::string& surfaceName)
{
    if (!viewer) {
        return;
    }
    ViewerOverlayControllerBase::applyPrimitives(viewer, generatedStripContextHoverKey(surfaceName), {});
}

GeneratedControlPointContextResult showGeneratedControlPointContextMenu(
    const GeneratedControlPointContextMenuOptions& options)
{
    if (!options.viewer ||
        options.controlPoints.empty() ||
        options.linePointCount == 0 ||
        !validGeneratedLinePosition(options.linePosition, options.linePointCount)) {
        return GeneratedControlPointContextResult::None;
    }

    // On a strip the click's scene x alone decides the target: the quarter of
    // a span next to a control point is that point, the middle half is the
    // span, off the ends the nearest end point. The cut views draw no spans
    // and keep the nearest marker in 2D.
    std::optional<GeneratedStripContextIndex> strip;
    std::vector<const GeneratedOverlay::ControlPointMarker*> sortedControls;
    std::optional<GeneratedStripContextTarget> stripTarget;
    if (options.stripViewer) {
        strip = buildGeneratedStripContextIndex(
            options.controlPoints, options.linePointCount, options.stripPositionMap);
        for (size_t i : strip->controlIndices) {
            sortedControls.push_back(&options.controlPoints[i]);
        }
        if (options.pinnedControlLinePosition && std::isfinite(*options.pinnedControlLinePosition) &&
            !sortedControls.empty()) {
            size_t bestRank = 0;
            double bestDistance = std::numeric_limits<double>::infinity();
            for (size_t rank = 0; rank < sortedControls.size(); ++rank) {
                const double distance =
                    std::abs(sortedControls[rank]->linePosition - *options.pinnedControlLinePosition);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestRank = rank;
                }
            }
            stripTarget = GeneratedStripContextTarget{
                GeneratedStripContextTarget::Kind::ControlPoint, bestRank};
        } else {
            stripTarget = resolveGeneratedStripContextTarget(options.viewer, *strip, options.scenePoint);
        }
    }

    clearGeneratedControlPointContextPreview(options.viewer, options.surfaceName);

    const auto fiberName = [&options](uint64_t fiberId) {
        return options.fiberDisplayNameForId
            ? options.fiberDisplayNameForId(fiberId)
            : QWidget::tr("Fiber %1").arg(static_cast<qulonglong>(fiberId));
    };

    if (stripTarget && stripTarget->kind == GeneratedStripContextTarget::Kind::Span) {
        ViewerOverlayControllerBase::applyPrimitives(
            options.viewer,
            "line_annotation_control_context_" + options.surfaceName,
            stripContextTargetPrimitives(options.viewer, options.controlPoints, *strip,
                                         options.stripPositionMap, *stripTarget));
        QMenu menu(options.parent);
        const auto handleSpanAction =
            appendGeneratedSpanContextActions(menu, options, sortedControls, stripTarget->rank);
        // The click's own actions, as in the control-point menu.
        menu.addSeparator();
        ClickLocationActions clickActions;
        clickActions.addOpenNearby(menu, options, fiberName);
        clickActions.addNewAnnotations(menu, options);
        QAction* selected = menu.exec(options.globalPos);
        clearGeneratedControlPointContextPreview(options.viewer, options.surfaceName);
        if (const auto result = clickActions.handle(selected, options)) {
            return *result;
        }
        handleSpanAction(selected);
        return GeneratedControlPointContextResult::Handled;
    }

    size_t selectedIndex = 0;
    QPointF targetScene;
    bool haveSelection = false;
    if (stripTarget) {
        selectedIndex = strip->controlIndices[stripTarget->rank];
        auto* quad = dynamic_cast<QuadSurface*>(options.viewer->currentSurface());
        targetScene = generatedStripControlPointToScene(
            options.viewer, quad, options.controlPoints[selectedIndex], options.stripPositionMap);
        haveSelection = true;
    } else {
        double bestDistanceSq = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < options.controlPoints.size(); ++i) {
            const auto& control = options.controlPoints[i];
            if (!validGeneratedLinePosition(control.linePosition, options.linePointCount)) {
                continue;
            }

            QPointF controlScene;
            if (options.stripViewer) {
                auto* quad = dynamic_cast<QuadSurface*>(options.viewer->currentSurface());
                controlScene = generatedStripControlPointToScene(
                    options.viewer, quad, control, options.stripPositionMap);
            } else {
                controlScene = options.viewer->volumeToScene(control.point);
            }
            if (!finiteScenePoint(controlScene)) {
                continue;
            }

            const QPointF delta = controlScene - options.scenePoint;
            const double distanceSq = delta.x() * delta.x() + delta.y() * delta.y();
            if (distanceSq < bestDistanceSq) {
                haveSelection = true;
                bestDistanceSq = distanceSq;
                selectedIndex = i;
                targetScene = controlScene;
            }
        }
    }
    if (!haveSelection) {
        return GeneratedControlPointContextResult::None;
    }
    const auto& selectedControl = options.controlPoints[selectedIndex];

    if (stripTarget) {
        // Same glow the hover showed, now held while the menu is open.
        ViewerOverlayControllerBase::applyPrimitives(
            options.viewer,
            "line_annotation_control_context_" + options.surfaceName,
            stripContextTargetPrimitives(options.viewer, options.controlPoints, *strip,
                                         options.stripPositionMap, *stripTarget));
    } else if (finiteScenePoint(options.scenePoint) && finiteScenePoint(targetScene)) {
        ViewerOverlayControllerBase::OverlayStyle previewStyle;
        previewStyle.penColor = QColor(255, 120, 40, 245);
        previewStyle.brushColor = QColor(255, 120, 40, 190);
        previewStyle.penWidth = 2.5;
        previewStyle.z = 180.0;

        std::vector<ViewerOverlayControllerBase::OverlayPrimitive> primitives;
        primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{
            {options.scenePoint, targetScene},
            false,
            previewStyle});
        primitives.push_back(ViewerOverlayControllerBase::CirclePrimitive{
            targetScene,
            selectedControl.hasBranches ? 7.0 : (selectedControl.isSeed ? 6.5 : 6.0),
            true,
            previewStyle});
        ViewerOverlayControllerBase::applyPrimitives(
            options.viewer,
            "line_annotation_control_context_" + options.surfaceName,
            std::move(primitives));
    }

    const size_t selectedControlIndex =
        selectedControl.controlIndex == std::numeric_limits<size_t>::max()
            ? selectedIndex
            : selectedControl.controlIndex;

    QMenu menu(options.parent);
    menu.addAction(QWidget::tr("CP %1").arg(QString::number(selectedControlIndex)))
        ->setEnabled(false);
    // Ordered by use, most frequent first, in four sections: navigation and
    // linking, the point tags, new annotations, then the corrections reset.
    // Actions tied to a state (links on the point, a designated candidate, a
    // nearby fiber marker) appear only in that state.
    const bool haveControlIndex = selectedControlIndex != std::numeric_limits<size_t>::max();
    menu.addSeparator();

    // Section 1: go to, delete, link.
    std::vector<std::pair<QAction*, GeneratedOverlay::ControlPointMarker::BranchLink>> openBranchActions;
    if (!selectedControl.branchLinks.empty()) {
        QMenu* branchMenu = menu.addMenu(QWidget::tr("Go to linked annotation"));
        for (const auto& branch : selectedControl.branchLinks) {
            QAction* action = branchMenu->addAction(
                QWidget::tr("%1 / CP %2")
                    .arg(fiberName(branch.fiberId), QString::number(branch.controlPointIndex)));
            action->setEnabled(static_cast<bool>(options.openBranch));
            openBranchActions.push_back({action, branch});
        }
    }
    ClickLocationActions clickActions;
    clickActions.addOpenNearby(menu, options, fiberName);
    QAction* deleteAction = menu.addAction(QWidget::tr("Delete control point"));
    deleteAction->setEnabled(options.controlPoints.size() > 1);
    QAction* linkWithCandidateAction = nullptr;
    if (options.linkWithCandidate && !options.linkWithCandidateLabel.isEmpty()) {
        linkWithCandidateAction = menu.addAction(options.linkWithCandidateLabel);
        linkWithCandidateAction->setEnabled(
            options.linkWithCandidateEnabled && haveControlIndex && !selectedControl.hasBranches);
    }
    // No hasBranches gate: the merge consumes the very pending link the two
    // endpoint CPs typically already carry.
    QAction* mergeWithCandidateAction = nullptr;
    if (options.mergeWithCandidate && !options.mergeWithCandidateLabel.isEmpty()) {
        mergeWithCandidateAction = menu.addAction(options.mergeWithCandidateLabel);
        mergeWithCandidateAction->setEnabled(options.mergeWithCandidateEnabled && haveControlIndex);
    }
    QAction* designateLinkCandidateAction = nullptr;
    if (options.designateLinkCandidate) {
        designateLinkCandidateAction =
            menu.addAction(QWidget::tr("Designate as link candidate"));
        designateLinkCandidateAction->setEnabled(haveControlIndex && !selectedControl.hasBranches);
    }
    QAction* designateAdjacentLinkCandidateAction = nullptr;
    if (options.designateAdjacentLinkCandidate) {
        designateAdjacentLinkCandidateAction =
            menu.addAction(QWidget::tr("Designate as adjacent link candidate"));
        designateAdjacentLinkCandidateAction->setEnabled(
            haveControlIndex && !selectedControl.hasBranches);
    }
    std::vector<std::pair<QAction*, GeneratedOverlay::ControlPointMarker::BranchLink>> unlinkActions;
    if (options.unlinkBranch && !selectedControl.branchLinks.empty()) {
        if (selectedControl.branchLinks.size() == 1) {
            const auto& branch = selectedControl.branchLinks.front();
            QAction* action = menu.addAction(
                QWidget::tr("Unlink from %1 / CP %2")
                    .arg(fiberName(branch.fiberId), QString::number(branch.controlPointIndex)));
            unlinkActions.push_back({action, branch});
        } else {
            QMenu* unlinkMenu = menu.addMenu(QWidget::tr("Unlink"));
            for (const auto& branch : selectedControl.branchLinks) {
                QAction* action = unlinkMenu->addAction(
                    QWidget::tr("%1 / CP %2")
                        .arg(fiberName(branch.fiberId), QString::number(branch.controlPointIndex)));
                unlinkActions.push_back({action, branch});
            }
        }
    }
    std::vector<std::pair<QAction*, GeneratedOverlay::ControlPointMarker::BranchLink>> approveActions;
    std::vector<std::pair<QAction*, GeneratedOverlay::ControlPointMarker::BranchLink>> markPendingActions;
    if (options.setBranchLinkPending && !selectedControl.branchLinks.empty()) {
        auto addPendingChangeActions =
            [&menu, &fiberName](std::vector<std::pair<QAction*, GeneratedOverlay::ControlPointMarker::BranchLink>>& actions,
                    const std::vector<GeneratedOverlay::ControlPointMarker::BranchLink>& links,
                    const QString& singleFormat,
                    const QString& submenuTitle) {
                if (links.empty()) {
                    return;
                }
                if (links.size() == 1) {
                    const auto& branch = links.front();
                    QAction* action = menu.addAction(
                        singleFormat.arg(fiberName(branch.fiberId),
                                         QString::number(branch.controlPointIndex)));
                    actions.push_back({action, branch});
                } else {
                    QMenu* submenu = menu.addMenu(submenuTitle);
                    for (const auto& branch : links) {
                        QAction* action = submenu->addAction(
                            QWidget::tr("%1 / CP %2")
                                .arg(fiberName(branch.fiberId),
                                     QString::number(branch.controlPointIndex)));
                        actions.push_back({action, branch});
                    }
                }
            };
        std::vector<GeneratedOverlay::ControlPointMarker::BranchLink> pendingLinks;
        std::vector<GeneratedOverlay::ControlPointMarker::BranchLink> approvedLinks;
        for (const auto& branch : selectedControl.branchLinks) {
            (branch.pending ? pendingLinks : approvedLinks).push_back(branch);
        }
        addPendingChangeActions(approveActions,
                                pendingLinks,
                                QWidget::tr("Approve link to %1 / CP %2"),
                                QWidget::tr("Approve link"));
        addPendingChangeActions(markPendingActions,
                                approvedLinks,
                                QWidget::tr("Mark link as pending (%1 / CP %2)"),
                                QWidget::tr("Mark link as pending"));
    }

    // Section 2: the point tags.
    QAction* kollesisTerminationAction = nullptr;
    QAction* breakAction = nullptr;
    if (options.setKollesisTermination || options.setBreak) {
        menu.addSeparator();
    }
    if (options.setKollesisTermination) {
        // Only a fiber end can be a termination. An interior point that
        // somehow carries the tag (an edited file) can still shed it.
        const bool endpoint = haveControlIndex &&
            generatedControlPointIsEndpoint(options.controlPoints, selectedControlIndex);
        // Adding the tag to a break point is refused (one or the other);
        // removing a tag is always possible.
        const bool blockedByBreak = selectedControl.isBreak && !selectedControl.isKollesisTermination;
        const bool enabled = haveControlIndex && !blockedByBreak &&
            (endpoint || selectedControl.isKollesisTermination);
        kollesisTerminationAction = menu.addAction(
            enabled          ? QWidget::tr("Kollesis termination")
            : blockedByBreak ? QWidget::tr("Kollesis termination (point is a break)")
                             : QWidget::tr("Kollesis termination (fiber ends only)"));
        kollesisTerminationAction->setCheckable(true);
        kollesisTerminationAction->setChecked(selectedControl.isKollesisTermination);
        kollesisTerminationAction->setEnabled(enabled);
    }
    if (options.setBreak) {
        // Any point can be a break; adding the tag to a kollesis termination
        // is refused (one or the other). Removing a tag is always possible.
        // ... and never immediately next to one: the span between a break
        // and a termination could otherwise become a gap at the sheet join.
        const bool blockedByKollesis =
            !selectedControl.isBreak &&
            (selectedControl.isKollesisTermination ||
             generatedLineOrderNeighbourIsKollesisTermination(options.controlPoints,
                                                             selectedControlIndex));
        const bool enabled = haveControlIndex && !blockedByKollesis;
        breakAction = menu.addAction(
            blockedByKollesis ? QWidget::tr("Break (at or next to a kollesis termination)")
                              : QWidget::tr("Break"));
        breakAction->setCheckable(true);
        breakAction->setChecked(selectedControl.isBreak);
        breakAction->setEnabled(enabled);
    }

    // Section 3: new annotations at the clicked location (not the point).
    menu.addSeparator();
    clickActions.addNewAnnotations(menu, options);

    // Section 4: the corrections reset.
    QAction* clearCorrectionsAction = nullptr;
    if (options.clearControlCorrections) {
        menu.addSeparator();
        clearCorrectionsAction = menu.addAction(QWidget::tr("Clear CP normals and dirs"));
        clearCorrectionsAction->setEnabled(haveControlIndex);
    }
    QAction* selected = menu.exec(options.globalPos);
    clearGeneratedControlPointContextPreview(options.viewer, options.surfaceName);

    if (const auto result = clickActions.handle(selected, options)) {
        return *result;
    }
    if (clearCorrectionsAction && selected == clearCorrectionsAction) {
        options.clearControlCorrections(selectedControlIndex);
        return GeneratedControlPointContextResult::Handled;
    }
    if (selected == deleteAction && deleteAction->isEnabled()) {
        if (options.deleteControlPoint) {
            options.deleteControlPoint(selectedControl.linePosition, selectedControl.point);
        }
        return GeneratedControlPointContextResult::Handled;
    }
    if (kollesisTerminationAction && selected == kollesisTerminationAction &&
        kollesisTerminationAction->isEnabled()) {
        options.setKollesisTermination(selectedControlIndex,
                                       !selectedControl.isKollesisTermination);
        return GeneratedControlPointContextResult::Handled;
    }
    if (breakAction && selected == breakAction && breakAction->isEnabled()) {
        options.setBreak(selectedControlIndex, !selectedControl.isBreak);
        return GeneratedControlPointContextResult::Handled;
    }
    for (const auto& [action, branch] : openBranchActions) {
        if (selected == action && action->isEnabled()) {
            options.openBranch(branch.fiberId, branch.controlPointIndex);
            return GeneratedControlPointContextResult::Handled;
        }
    }
    for (const auto& [action, branch] : unlinkActions) {
        if (selected == action) {
            options.unlinkBranch(selectedControlIndex, branch.fiberId, branch.controlPointIndex);
            return GeneratedControlPointContextResult::Handled;
        }
    }
    for (const auto& [action, branch] : approveActions) {
        if (selected == action) {
            options.setBranchLinkPending(
                selectedControlIndex, branch.fiberId, branch.controlPointIndex, false);
            return GeneratedControlPointContextResult::Handled;
        }
    }
    for (const auto& [action, branch] : markPendingActions) {
        if (selected == action) {
            options.setBranchLinkPending(
                selectedControlIndex, branch.fiberId, branch.controlPointIndex, true);
            return GeneratedControlPointContextResult::Handled;
        }
    }
    if (designateLinkCandidateAction &&
        selected == designateLinkCandidateAction &&
        designateLinkCandidateAction->isEnabled()) {
        options.designateLinkCandidate(selectedControlIndex, selectedControl.point);
        return GeneratedControlPointContextResult::Handled;
    }
    if (designateAdjacentLinkCandidateAction &&
        selected == designateAdjacentLinkCandidateAction &&
        designateAdjacentLinkCandidateAction->isEnabled()) {
        options.designateAdjacentLinkCandidate(selectedControlIndex, selectedControl.point);
        return GeneratedControlPointContextResult::Handled;
    }
    if (linkWithCandidateAction &&
        selected == linkWithCandidateAction &&
        linkWithCandidateAction->isEnabled()) {
        options.linkWithCandidate(selectedControlIndex, selectedControl.point);
        return GeneratedControlPointContextResult::Handled;
    }
    if (mergeWithCandidateAction &&
        selected == mergeWithCandidateAction &&
        mergeWithCandidateAction->isEnabled()) {
        options.mergeWithCandidate(selectedControlIndex, selectedControl.point);
        return GeneratedControlPointContextResult::Handled;
    }
    return GeneratedControlPointContextResult::Handled;
}

} // namespace vc3d::line_annotation
