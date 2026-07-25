#include "SpiralOverlayController.hpp"

#include "../volume_viewers/VolumeViewerBase.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <algorithm>

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
    refreshAll();
}

bool SpiralOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    const bool lossMapVisible = viewer && _lossMapSurface && !_lossMapImage.isNull()
        && viewer->currentSurface() == _lossMapSurface.get();
    return hasRunDiffFor(viewer) || lossMapVisible;
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

}
