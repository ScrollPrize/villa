#include "SpiralOverlayController.hpp"

#include "../volume_viewers/CVolumeViewerView.hpp"
#include "../volume_viewers/VolumeViewerBase.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QFontMetricsF>

#include <algorithm>
#include <limits>

SpiralOverlayController::SpiralOverlayController(QObject* parent)
    : ViewerOverlayControllerBase("spiral_geometry", parent)
{
}

void SpiralOverlayController::publishRunDiff(std::shared_ptr<QuadSurface> surface, QImage image)
{
    _runDiffSurface = std::move(surface);
    _runDiffImage = std::move(image);
    refreshAll();
}

void SpiralOverlayController::publishLossMap(std::shared_ptr<QuadSurface> surface,
                                             QImage image, qreal opacity)
{
    _lossMapSurface = std::move(surface);
    _lossMapImage = std::move(image);
    _lossMapOpacity = std::clamp(opacity, 0.0, 1.0);
    refreshAll();
}

void SpiralOverlayController::publishWindingTransitions(
    std::shared_ptr<QuadSurface> surface,
    std::vector<WindingTransitionCurve> curves)
{
    _transitionSurface = std::move(surface);
    _transitionCurves = std::move(curves);
    refreshAll();
}

void SpiralOverlayController::setRunDiffVisible(bool visible)
{
    if (_runDiffVisible == visible) return;
    _runDiffVisible = visible;
    refreshAll();
}

void SpiralOverlayController::reset()
{
    _runDiffSurface.reset();
    _runDiffImage = {};
    _lossMapSurface.reset();
    _lossMapImage = {};
    _transitionSurface.reset();
    _transitionCurves.clear();
    refreshAll();
}

bool SpiralOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    const bool lossMapVisible = viewer && _lossMapSurface && !_lossMapImage.isNull()
        && viewer->currentSurface() == _lossMapSurface.get();
    return hasRunDiffFor(viewer) || lossMapVisible || hasTransitionsFor(viewer);
}

bool SpiralOverlayController::hasTransitionsFor(VolumeViewerBase* viewer) const
{
    return viewer && _transitionSurface && !_transitionCurves.empty()
        && viewer->currentSurface() == _transitionSurface.get();
}

bool SpiralOverlayController::hasRunDiffFor(VolumeViewerBase* viewer) const
{
    return _runDiffVisible && viewer && _runDiffSurface && !_runDiffImage.isNull()
        && viewer->currentSurface() == _runDiffSurface.get();
}

void SpiralOverlayController::collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder)
{
    if (!isOverlayEnabledFor(viewer)) return;

    if (hasRunDiffFor(viewer)) {
        const cv::Vec2f scale = _runDiffSurface->scale();
        const cv::Vec3f center = _runDiffSurface->center();
        if (std::abs(scale[0]) > 1e-6f && std::abs(scale[1]) > 1e-6f) {
            auto gridToScene = [viewer, scale, center](int row, int col) {
                const float surfaceX = static_cast<float>(col) / scale[0] - center[0];
                const float surfaceY = static_cast<float>(row) / scale[1] - center[1];
                return viewer->surfaceCoordsToScene(surfaceX, surfaceY);
            };
            const QPointF origin = gridToScene(0, 0);
            const QPointF columnStep = gridToScene(0, 1) - origin;
            const QPointF rowStep = gridToScene(1, 0) - origin;
            const qreal scaleX = std::hypot(columnStep.x(), columnStep.y());
            const qreal scaleY = std::hypot(rowStep.x(), rowStep.y());
            if (scaleX > 1e-6 && scaleY > 1e-6)
                builder.addImage(_runDiffImage, origin, scaleX, scaleY, 1.0, 65.0);
        }
    }

    if (viewer && _lossMapSurface && !_lossMapImage.isNull()
        && viewer->currentSurface() == _lossMapSurface.get()) {
        const cv::Vec2f scale = _lossMapSurface->scale();
        const cv::Vec3f center = _lossMapSurface->center();
        if (std::abs(scale[0]) > 1e-6f && std::abs(scale[1]) > 1e-6f) {
            auto gridToScene = [viewer, scale, center](int row, int col) {
                const float surfaceX = static_cast<float>(col) / scale[0] - center[0];
                const float surfaceY = static_cast<float>(row) / scale[1] - center[1];
                return viewer->surfaceCoordsToScene(surfaceX, surfaceY);
            };
            const QPointF origin = gridToScene(0, 0);
            const QPointF columnStep = gridToScene(0, 1) - origin;
            const QPointF rowStep = gridToScene(1, 0) - origin;
            const qreal scaleX = std::hypot(columnStep.x(), columnStep.y());
            const qreal scaleY = std::hypot(rowStep.x(), rowStep.y());
            if (scaleX > 1e-6 && scaleY > 1e-6)
                builder.addImage(_lossMapImage, origin, scaleX, scaleY,
                                 _lossMapOpacity, 66.0);
        }
    }

    if (hasTransitionsFor(viewer)) {
        const cv::Vec2f scale = _transitionSurface->scale();
        const cv::Vec3f center = _transitionSurface->center();
        if (std::abs(scale[0]) > 1e-6f && std::abs(scale[1]) > 1e-6f) {
            auto gridToSurface = [scale, center](const cv::Vec2f& gridColumnRow) {
                return cv::Vec2f(gridColumnRow[0] / scale[0] - center[0],
                                 gridColumnRow[1] / scale[1] - center[1]);
            };
            // The overlay is re-collected on every pan/zoom, so each label
            // can chase the viewport: it sits on the boundary point closest
            // to the viewport's vertical center and is clamped into view
            // while any part of its curve is on screen.
            QRectF visibleScene;
            if (auto* view = viewer->graphicsView(); view && view->viewport())
                visibleScene =
                    view->mapToScene(view->viewport()->rect()).boundingRect();
            const QFont labelFont;
            const QFontMetricsF labelMetrics(labelFont);
            for (const WindingTransitionCurve& curve : _transitionCurves) {
                // A jump straight to a non-adjacent winding is a mapping
                // anomaly worth spotting, so it gets the warning color.
                const bool adjacent =
                    curve.toWinding == curve.fromWinding + 1
                    || curve.fromWinding == curve.toWinding + 1;
                OverlayStyle style;
                style.penColor = adjacent ? QColor(255, 200, 40, 220)
                                          : QColor(255, 60, 60, 230);
                style.penWidth = 2.0;
                style.z = 67.0;
                QPointF labelAnchor;
                qreal labelCost = std::numeric_limits<qreal>::max();
                for (const std::vector<cv::Vec2f>& segment : curve.segments) {
                    if (segment.size() < 2) continue;
                    std::vector<cv::Vec2f> surfacePoints;
                    surfacePoints.reserve(segment.size());
                    for (const cv::Vec2f& point : segment)
                        surfacePoints.push_back(gridToSurface(point));
                    builder.addSurfaceLineStrip(surfacePoints, false, style);
                    for (const cv::Vec2f& surfacePoint : surfacePoints) {
                        const QPointF scene = viewer->surfaceCoordsToScene(
                            surfacePoint[0], surfacePoint[1]);
                        qreal cost = visibleScene.isValid()
                            ? std::abs(scene.y() - visibleScene.center().y())
                            : 0.0;
                        if (visibleScene.isValid()
                            && !visibleScene.contains(scene))
                            cost += 1.0e6;
                        if (cost < labelCost) {
                            labelCost = cost;
                            labelAnchor = scene;
                        }
                    }
                }
                if (labelCost == std::numeric_limits<qreal>::max()) continue;
                const QString label = QStringLiteral("%1 | %2")
                                          .arg(curve.fromWinding)
                                          .arg(curve.toWinding);
                // COutlinedTextItem is a QGraphicsTextItem, whose document
                // pads roughly 4 px per side; fold that into the centering.
                const qreal labelWidth =
                    labelMetrics.horizontalAdvance(label) + 8.0;
                const qreal labelHeight = labelMetrics.height() + 8.0;
                QPointF position(labelAnchor.x() - labelWidth / 2.0,
                                 labelAnchor.y() - labelHeight / 2.0);
                if (visibleScene.isValid()) {
                    position.setY(std::clamp(
                        position.y(), visibleScene.top() + 2.0,
                        visibleScene.bottom() - labelHeight - 2.0));
                }
                OverlayStyle textStyle;
                textStyle.penColor = style.penColor;
                textStyle.z = 68.0;
                builder.addText(position, label, labelFont, textStyle, true);
            }
        }
    }

}
