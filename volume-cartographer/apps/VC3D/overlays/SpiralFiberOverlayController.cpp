#include "SpiralFiberOverlayController.hpp"

#include "../volume_viewers/VolumeViewerBase.hpp"
#include "vc/core/types/FiberView.hpp"

#include <QPainterPath>

#include <algorithm>

SpiralFiberOverlayController::SpiralFiberOverlayController(QObject* parent)
    : ViewerOverlayControllerBase("spiral_fibers", parent)
{
}

void SpiralFiberOverlayController::setFibers(std::vector<Fiber> fibers)
{
    _fibers = std::move(fibers);
    refreshAll();
}

void SpiralFiberOverlayController::setVisible(bool visible)
{
    if (_visible == visible) return;
    _visible = visible;
    refreshAll();
}

void SpiralFiberOverlayController::setSlabHalfDepth(double voxels)
{
    // A zero-depth slab would only keep segments exactly crossing the plane.
    _halfDepth = std::max(voxels, 0.5);
    refreshAll();
}

bool SpiralFiberOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    return viewer && _visible && !_fibers.empty();
}

void SpiralFiberOverlayController::collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder)
{
    const auto region = planeViewRegion(viewer, _halfDepth);
    if (!region) return;
    QPainterPath horizontal, vertical, other;
    for (const auto& fiber : _fibers) {
        auto& path = fiber.family == QStringLiteral("H")   ? horizontal
                     : fiber.family == QStringLiteral("V") ? vertical
                                                           : other;
        // Contiguous clipped segments form one subpath; slab exits break it.
        bool open = false;
        QPointF last;
        for (std::size_t i = 1; i < fiber.line.size(); ++i) {
            const auto& from = fiber.line[i - 1];
            const auto& to = fiber.line[i];
            double lo = 0, hi = 0;
            if (!vc::fibers::clipSegment(region->project({from[0], from[1], from[2]}),
                                         region->project({to[0], to[1], to[2]}), region->local, lo, hi)) {
                open = false;
                continue;
            }
            const cv::Vec3d p = from + lo * (to - from), q = from + hi * (to - from);
            const QPointF a = viewer->volumeToScene(cv::Vec3f(p)), b = viewer->volumeToScene(cv::Vec3f(q));
            if (!open || last != a) path.moveTo(a);
            path.lineTo(b);
            last = b;
            open = true;
        }
    }
    const auto draw = [&](const QPainterPath& path, const QColor& color) {
        if (path.isEmpty()) return;
        OverlayStyle style;
        style.penColor = color;
        style.penWidth = 2.0;
        style.z = 45;
        builder.addPainterPath(path, style);
    };
    draw(horizontal, QColor("#40a5ff"));
    draw(vertical, QColor("#ffad43"));
    draw(other, QColor("#bbbbbb"));
}
