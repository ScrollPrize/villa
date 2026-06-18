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

} // namespace

QColor generatedLineTailColor(const QColor& baseColor)
{
    QColor color;
    color.setRed(std::clamp(static_cast<int>(std::round(baseColor.red() * 0.75)), 0, 255));
    color.setGreen(std::clamp(static_cast<int>(std::round(baseColor.green() * 0.75)), 0, 255));
    color.setBlue(std::clamp(static_cast<int>(std::round(baseColor.blue() * 0.75)), 0, 255));
    color.setAlpha(std::clamp(static_cast<int>(std::round(baseColor.alpha() * 0.75)), 0, 255));
    return color;
}

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
    return makeGeneratedCrossSliceOverlay(
        views,
        linePosition,
        emphasized,
        threshold,
        [plane](const cv::Vec3f& point) {
            return plane ? plane->pointDist(point) : std::numeric_limits<float>::quiet_NaN();
        },
        controlIndex,
        linePositionRadius);
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

GeneratedOverlay makeGeneratedSurfaceCenterPointOverlay(bool emphasized)
{
    GeneratedOverlay overlay;
    overlay.pointMarker = {0.0f, 0.0f, 0.0f};
    overlay.pointMarkerInSurfaceCoords = true;
    overlay.emphasizedPointMarker = emphasized;
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
    primitives.reserve(3 + overlay.controlPoints.size() + overlay.predSnapPoints.size() * 2);

    ViewerOverlayControllerBase::OverlayStyle lineStyle;
    lineStyle.penColor = QColor(0, 220, 255, 190);
    lineStyle.penWidth = 1.0;
    lineStyle.z = 150.0;

    ViewerOverlayControllerBase::OverlayStyle tailLineStyle = lineStyle;
    tailLineStyle.penColor = generatedLineTailColor(lineStyle.penColor);
    tailLineStyle.z = lineStyle.z - 1.0;

    ViewerOverlayControllerBase::OverlayStyle seedStyle;
    seedStyle.penColor = QColor(255, 230, 0, 220);
    seedStyle.brushColor = QColor(255, 230, 0, 170);
    seedStyle.penWidth = 1.5;
    seedStyle.z = 161.0;

    ViewerOverlayControllerBase::OverlayStyle controlPointStyle = seedStyle;
    controlPointStyle.z = 160.0;

    ViewerOverlayControllerBase::OverlayStyle markerStyle;
    markerStyle.penColor = QColor(0, 220, 255, 210);
    markerStyle.brushColor = QColor(0, 220, 255, 150);
    markerStyle.penWidth = 1.0;
    markerStyle.z = 151.0;

    ViewerOverlayControllerBase::OverlayStyle currentMarkerStyle = markerStyle;
    currentMarkerStyle.penColor = QColor(0, 245, 255, 245);
    currentMarkerStyle.brushColor = QColor(0, 245, 255, 210);
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
                        control.isSeed ? 5.5 : 5.0,
                        true,
                        control.isSeed ? seedStyle : controlPointStyle});
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
                                 control.isSeed ? 11.0 : 10.0,
                                 control.isSeed ? seedStyle : controlPointStyle);
        }
    }

    if (!overlay.useSurfaceCenterLine && sceneLine.size() >= 2) {
        const auto controlRange = generatedControlLinePositionRange(overlay.controlPoints);
        for (size_t i = 1; i < sceneLine.size(); ++i) {
            const auto& previous = sceneLine[i - 1];
            const auto& current = sceneLine[i];
            const auto& style = generatedLineSegmentIsTail(previous.second,
                                                           current.second,
                                                           controlRange)
                ? tailLineStyle
                : lineStyle;
            primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{
                {previous.first, current.first},
                false,
                style});
        }
    }

    if (finiteGeneratedPoint(overlay.pointMarker)) {
        const qreal radius = overlay.emphasizedPointMarker ? 2.5 : 2.0;
        const auto& style = overlay.emphasizedPointMarker ? currentMarkerStyle : markerStyle;
        if (overlay.pointMarkerInSurfaceCoords) {
            primitives.push_back(ViewerOverlayControllerBase::SurfacePointPrimitive{
                {overlay.pointMarker[0], overlay.pointMarker[1]},
                radius,
                style});
        } else {
            addVolumePointMarker(overlay.pointMarker, radius, style);
        }
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
            selectedControl.isSeed ? 6.5 : 6.0,
            true,
            previewStyle});
        ViewerOverlayControllerBase::applyPrimitives(
            options.viewer,
            "line_annotation_control_context_" + options.surfaceName,
            std::move(primitives));
    }

    QMenu menu(options.parent);
    QAction* deleteAction = menu.addAction(QWidget::tr("Delete control point"));
    deleteAction->setEnabled(options.controlPoints.size() > 1);
    QAction* newLineAnnotationAction = menu.addAction(QWidget::tr("New line annotation"));
    newLineAnnotationAction->setEnabled(options.viewer->sampleSceneVolume(options.scenePoint).has_value());
    QAction* selected = menu.exec(options.globalPos);
    clearGeneratedControlPointContextPreview(options.viewer, options.surfaceName);

    if (selected == deleteAction && deleteAction->isEnabled()) {
        if (options.deleteControlPoint) {
            options.deleteControlPoint(selectedControl.linePosition, selectedControl.point);
        }
        return GeneratedControlPointContextResult::Handled;
    }
    if (selected == newLineAnnotationAction && newLineAnnotationAction->isEnabled()) {
        return GeneratedControlPointContextResult::NewLineAnnotationRequested;
    }
    return GeneratedControlPointContextResult::Handled;
}

} // namespace vc3d::line_annotation
