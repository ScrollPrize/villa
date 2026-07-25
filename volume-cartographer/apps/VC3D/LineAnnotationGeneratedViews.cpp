#include "LineAnnotationGeneratedViews.hpp"

#include "overlays/ViewerOverlayControllerBase.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"

#include <QAction>
#include <QMenu>
#include <QRect>
#include <QWidget>

#include <cmath>

namespace vc3d::line_annotation {
namespace {

bool finiteScenePoint(const QPointF& point)
{
    return std::isfinite(point.x()) && std::isfinite(point.y());
}

QPointF generatedStripControlPointToScene(
    CChunkedVolumeViewer* viewer,
    QuadSurface* surface,
    const GeneratedOverlay::ControlPointMarker& control)
{
    if (viewer && finiteGeneratedPoint(control.point)) {
        const QPointF pointScene = viewer->volumeToScene(control.point);
        if (finiteScenePoint(pointScene)) {
            return pointScene;
        }
    }
    return generatedStripLinePositionToScene(viewer, surface, control.linePosition);
}

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

} // namespace

QPointF generatedStripLinePositionToScene(CChunkedVolumeViewer* viewer,
                                          QuadSurface* surface,
                                          double linePosition)
{
    if (!viewer || !surface) {
        return {};
    }
    const auto* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return {};
    }
    const cv::Vec2f scale = surface->scale();
    if (scale[0] == 0.0f || scale[1] == 0.0f) {
        return {};
    }
    const float surfaceX = (static_cast<float>(linePosition) -
                            static_cast<float>(points->cols) / 2.0f) / scale[0];
    const float centerRow = static_cast<float>(points->rows / 2);
    const float surfaceY = (centerRow - static_cast<float>(points->rows) / 2.0f) / scale[1];
    return viewer->surfaceCoordsToScene(surfaceX, surfaceY);
}

double generatedLinePositionFromStripScene(CChunkedVolumeViewer* viewer,
                                           const QPointF& scenePoint)
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
    const cv::Vec2f scale = quad->scale();
    if (scale[0] == 0.0f || !std::isfinite(surfacePoint[0])) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double position = static_cast<double>(surfacePoint[0] * scale[0]) +
                            static_cast<double>(points->cols) / 2.0;
    return std::clamp(position, 0.0, static_cast<double>(points->cols - 1));
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

void applyGeneratedOverlay(CChunkedVolumeViewer* viewer,
                           const std::string& surfaceName,
                           const GeneratedOverlay& overlay)
{
    if (!viewer) {
        return;
    }

    const auto key = "line_annotation_overlay_" + surfaceName;
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

    ViewerOverlayControllerBase::OverlayStyle linkCandidateControlPointStyle = branchControlPointStyle;
    linkCandidateControlPointStyle.penColor = QColor(60, 235, 120, 245);
    linkCandidateControlPointStyle.brushColor = QColor(60, 235, 120, 175);
    linkCandidateControlPointStyle.z = 163.0;

    auto controlStyleForMarker = [&](const GeneratedOverlay::ControlPointMarker& control)
        -> const ViewerOverlayControllerBase::OverlayStyle& {
        if (control.isLinkCandidate) {
            return linkCandidateControlPointStyle;
        }
        if (control.hasPendingLinks) {
            return pendingBranchControlPointStyle;
        }
        if (control.hasBranches) {
            return branchControlPointStyle;
        }
        return control.isSeed ? seedStyle : controlPointStyle;
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

    ViewerOverlayControllerBase::OverlayStyle pendingBranchLinkFiberIntersectionStyle =
        branchLinkFiberIntersectionStyle;
    pendingBranchLinkFiberIntersectionStyle.penColor = QColor(80, 150, 255, 245);
    pendingBranchLinkFiberIntersectionStyle.z = 168.3;

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
            const cv::Vec2f scale = quad->scale();
            if (scale[0] != 0.0f && scale[1] != 0.0f && !overlay.linePoints.empty()) {
                const float centerRow = static_cast<float>(points->rows / 2);
                const float surfaceY = (centerRow - static_cast<float>(points->rows) / 2.0f) / scale[1];
                const float startX = -static_cast<float>(points->cols) / 2.0f / scale[0];
                const float endX = (static_cast<float>(points->cols - 1) -
                                    static_cast<float>(points->cols) / 2.0f) / scale[0];
                primitives.push_back(ViewerOverlayControllerBase::SurfaceLineStripPrimitive{
                    {cv::Vec2f(startX, surfaceY), cv::Vec2f(endX, surfaceY)},
                    false,
                    lineStyle});
            }
            if (overlay.controlPoints.empty() &&
                overlay.seedLineIndex >= 0 &&
                overlay.seedLineIndex < points->cols) {
                seedScene = generatedStripLinePositionToScene(viewer, quad, overlay.seedLineIndex);
                hasSeedScene = finiteScenePoint(seedScene);
            }
            for (const double position : overlay.markerLinePositions) {
                if (!std::isfinite(position) ||
                    position < 0.0 ||
                    position > static_cast<double>(points->cols - 1) ||
                    std::abs(position - overlay.currentLinePosition) < 1.0e-6) {
                    continue;
                }
                const QPointF markerScene = generatedStripLinePositionToScene(viewer, quad, position);
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
                    control.linePosition > static_cast<double>(points->cols - 1)) {
                    continue;
                }
                const QPointF controlScene =
                    generatedStripControlPointToScene(viewer, quad, control);
                if (finiteScenePoint(controlScene)) {
                    primitives.push_back(ViewerOverlayControllerBase::CirclePrimitive{
                        controlScene,
                        control.hasBranches ? 6.25 : (control.isSeed ? 5.5 : 5.0),
                        true,
                        controlStyleForMarker(control)});
                }
            }
            for (const auto& predSnap : overlay.predSnapPoints) {
                if (!finiteGeneratedPoint(predSnap.snapPoint)) {
                    continue;
                }
                const QPointF controlScene = finiteGeneratedPoint(predSnap.controlPoint)
                    ? viewer->volumeToScene(predSnap.controlPoint)
                    : generatedStripLinePositionToScene(viewer, quad, predSnap.linePosition);
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
                    generatedStripLinePositionToScene(viewer, quad, overlay.currentLinePosition);
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
            addVolumePointMarker(control.point,
                                 control.hasBranches ? 12.0 : (control.isSeed ? 11.0 : 10.0),
                                 controlStyleForMarker(control));
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
        if (intersection.connectorStart &&
            finiteGeneratedPoint(*intersection.connectorStart)) {
            const QPointF connectorScene = viewer->volumeToScene(*intersection.connectorStart);
            if (finiteScenePoint(connectorScene) && finiteScenePoint(scenePoint)) {
                primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{
                    {connectorScene, scenePoint},
                    false,
                    branchLinkStyle});
            }
        }
        addFiberIntersectionMarker(scenePoint,
                                   intersection.isLinkCandidateFiber
                                       ? linkCandidateFiberIntersectionStyle
                                       : (intersection.projectedBranchLink
                                              ? (intersection.pendingBranchLink
                                                     ? pendingBranchLinkFiberIntersectionStyle
                                                     : branchLinkFiberIntersectionStyle)
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
        const auto controlRange = generatedControlLinePositionRange(overlay.controlPoints);
        for (size_t i = 1; i < sceneLine.size(); ++i) {
            const auto& previous = sceneLine[i - 1];
            const auto& current = sceneLine[i];
            if (generatedLineSegmentIsTail(previous.second, current.second, controlRange)) {
                continue;
            }
            primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{
                {previous.first, current.first},
                false,
                lineStyle});
        }
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

GeneratedControlPointContextResult showGeneratedControlPointContextMenu(
    const GeneratedControlPointContextMenuOptions& options)
{
    if (!options.viewer ||
        options.controlPoints.empty() ||
        options.linePointCount == 0 ||
        !validGeneratedLinePosition(options.linePosition, options.linePointCount)) {
        return GeneratedControlPointContextResult::None;
    }

    size_t selectedIndex = 0;
    QPointF targetScene;
    bool haveSelection = false;
    double bestDistanceSq = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < options.controlPoints.size(); ++i) {
        const auto& control = options.controlPoints[i];
        if (!validGeneratedLinePosition(control.linePosition, options.linePointCount)) {
            continue;
        }

        QPointF controlScene;
        if (options.stripViewer) {
            auto* quad = dynamic_cast<QuadSurface*>(options.viewer->currentSurface());
            controlScene = generatedStripControlPointToScene(options.viewer, quad, control);
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
    if (!haveSelection) {
        return GeneratedControlPointContextResult::None;
    }
    const auto& selectedControl = options.controlPoints[selectedIndex];

    // Nearest fiber-intersection "X" marker to the click, within a scene-space
    // threshold matched to the drawn glyph (arm length 7.5 scene units).
    constexpr double kFiberIntersectionHitThreshold = 12.0;
    const GeneratedOverlay::FiberIntersectionMarker* nearbyIntersection = nullptr;
    double bestIntersectionDistanceSq =
        kFiberIntersectionHitThreshold * kFiberIntersectionHitThreshold;
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
        if (distanceSq < bestIntersectionDistanceSq) {
            bestIntersectionDistanceSq = distanceSq;
            nearbyIntersection = &intersection;
        }
    }

    clearGeneratedControlPointContextPreview(options.viewer, options.surfaceName);
    if (finiteScenePoint(options.scenePoint) && finiteScenePoint(targetScene)) {
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
    QAction* deleteAction = menu.addAction(QWidget::tr("Delete control point"));
    deleteAction->setEnabled(options.controlPoints.size() > 1);
    QAction* designateLinkCandidateAction = nullptr;
    if (options.designateLinkCandidate) {
        designateLinkCandidateAction =
            menu.addAction(QWidget::tr("Designate as link candidate"));
        designateLinkCandidateAction->setEnabled(
            selectedControlIndex != std::numeric_limits<size_t>::max() &&
            !selectedControl.hasBranches);
    }
    std::vector<std::pair<QAction*, GeneratedOverlay::ControlPointMarker::BranchLink>> openBranchActions;
    if (!selectedControl.branchLinks.empty()) {
        QMenu* branchMenu = menu.addMenu(QWidget::tr("Go to linked annotation"));
        for (const auto& branch : selectedControl.branchLinks) {
            QAction* action = branchMenu->addAction(
                QWidget::tr("Fiber %1 / CP %2")
                    .arg(static_cast<qulonglong>(branch.fiberId))
                    .arg(branch.controlPointIndex));
            action->setEnabled(static_cast<bool>(options.openBranch));
            openBranchActions.push_back({action, branch});
        }
    }
    std::vector<std::pair<QAction*, GeneratedOverlay::ControlPointMarker::BranchLink>> unlinkActions;
    if (options.unlinkBranch && !selectedControl.branchLinks.empty()) {
        if (selectedControl.branchLinks.size() == 1) {
            const auto& branch = selectedControl.branchLinks.front();
            QAction* action = menu.addAction(
                QWidget::tr("Unlink from Fiber %1 / CP %2")
                    .arg(static_cast<qulonglong>(branch.fiberId))
                    .arg(branch.controlPointIndex));
            unlinkActions.push_back({action, branch});
        } else {
            QMenu* unlinkMenu = menu.addMenu(QWidget::tr("Unlink"));
            for (const auto& branch : selectedControl.branchLinks) {
                QAction* action = unlinkMenu->addAction(
                    QWidget::tr("Fiber %1 / CP %2")
                        .arg(static_cast<qulonglong>(branch.fiberId))
                        .arg(branch.controlPointIndex));
                unlinkActions.push_back({action, branch});
            }
        }
    }
    std::vector<std::pair<QAction*, GeneratedOverlay::ControlPointMarker::BranchLink>> approveActions;
    std::vector<std::pair<QAction*, GeneratedOverlay::ControlPointMarker::BranchLink>> markPendingActions;
    if (options.setBranchLinkPending && !selectedControl.branchLinks.empty()) {
        auto addPendingChangeActions =
            [&menu](std::vector<std::pair<QAction*, GeneratedOverlay::ControlPointMarker::BranchLink>>& actions,
                    const std::vector<GeneratedOverlay::ControlPointMarker::BranchLink>& links,
                    const QString& singleFormat,
                    const QString& submenuTitle) {
                if (links.empty()) {
                    return;
                }
                if (links.size() == 1) {
                    const auto& branch = links.front();
                    QAction* action = menu.addAction(
                        singleFormat
                            .arg(static_cast<qulonglong>(branch.fiberId))
                            .arg(branch.controlPointIndex));
                    actions.push_back({action, branch});
                } else {
                    QMenu* submenu = menu.addMenu(submenuTitle);
                    for (const auto& branch : links) {
                        QAction* action = submenu->addAction(
                            QWidget::tr("Fiber %1 / CP %2")
                                .arg(static_cast<qulonglong>(branch.fiberId))
                                .arg(branch.controlPointIndex));
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
                                QWidget::tr("Approve link to Fiber %1 / CP %2"),
                                QWidget::tr("Approve link"));
        addPendingChangeActions(markPendingActions,
                                approvedLinks,
                                QWidget::tr("Mark link as pending (Fiber %1 / CP %2)"),
                                QWidget::tr("Mark link as pending"));
    }
    const bool canSampleClickedVolume =
        options.viewer->sampleSceneVolume(options.scenePoint).has_value();
    QAction* newLineAnnotationAction =
        menu.addAction(QWidget::tr("New line annotation"));
    newLineAnnotationAction->setEnabled(canSampleClickedVolume);
    QAction* newLinkedLineAnnotationAction = nullptr;
    if (options.addBranch) {
        newLinkedLineAnnotationAction = menu.addAction(
            QWidget::tr("New linked line annotation from control point"));
        newLinkedLineAnnotationAction->setEnabled(
            selectedControlIndex != std::numeric_limits<size_t>::max() &&
            canSampleClickedVolume &&
            !selectedControl.hasBranches);
    }
    QAction* linkWithCandidateAction = nullptr;
    if (options.linkWithCandidate && !options.linkWithCandidateLabel.isEmpty()) {
        linkWithCandidateAction = menu.addAction(options.linkWithCandidateLabel);
        linkWithCandidateAction->setEnabled(
            options.linkWithCandidateEnabled &&
            selectedControlIndex != std::numeric_limits<size_t>::max() &&
            !selectedControl.hasBranches);
    }
    QAction* openNearbyAnnotationAction = nullptr;
    if (options.openNearbyAnnotation && nearbyIntersection) {
        openNearbyAnnotationAction = menu.addAction(
            QWidget::tr("Go to nearby annotation (Fiber %1)")
                .arg(static_cast<qulonglong>(nearbyIntersection->fiberId)));
    }
    QAction* selected = menu.exec(options.globalPos);
    clearGeneratedControlPointContextPreview(options.viewer, options.surfaceName);

    if (selected == deleteAction && deleteAction->isEnabled()) {
        if (options.deleteControlPoint) {
            options.deleteControlPoint(selectedControl.linePosition, selectedControl.point);
        }
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
    if (selected == newLineAnnotationAction && newLineAnnotationAction->isEnabled()) {
        return GeneratedControlPointContextResult::NewLineAnnotationRequested;
    }
    if (newLinkedLineAnnotationAction &&
        selected == newLinkedLineAnnotationAction &&
        newLinkedLineAnnotationAction->isEnabled()) {
        const auto clickedVolumePoint = options.viewer->sampleSceneVolume(options.scenePoint);
        if (!clickedVolumePoint) {
            return GeneratedControlPointContextResult::Handled;
        }
        options.addBranch(selectedControlIndex,
                          clickedVolumePoint->position,
                          false,
                          options.branchLinkDirection);
        return GeneratedControlPointContextResult::Handled;
    }
    if (designateLinkCandidateAction &&
        selected == designateLinkCandidateAction &&
        designateLinkCandidateAction->isEnabled()) {
        options.designateLinkCandidate(selectedControlIndex, selectedControl.point);
        return GeneratedControlPointContextResult::Handled;
    }
    if (linkWithCandidateAction &&
        selected == linkWithCandidateAction &&
        linkWithCandidateAction->isEnabled()) {
        options.linkWithCandidate(selectedControlIndex, selectedControl.point);
        return GeneratedControlPointContextResult::Handled;
    }
    if (openNearbyAnnotationAction && selected == openNearbyAnnotationAction) {
        options.openNearbyAnnotation(nearbyIntersection->fiberId, nearbyIntersection->point);
        return GeneratedControlPointContextResult::Handled;
    }
    return GeneratedControlPointContextResult::Handled;
}

} // namespace vc3d::line_annotation
