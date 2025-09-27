#include "BBoxOverlayController.hpp"

#include "../CVolumeViewer.hpp"

namespace
{
constexpr const char* kOverlayGroup = "bbox_overlays";
const QColor kActiveColor(255, 220, 0);
constexpr qreal kPersistentZ = 110.0;
constexpr qreal kActiveZ = 111.0;
}

BBoxOverlayController::BBoxOverlayController(QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroup, parent)
{
}

bool BBoxOverlayController::isOverlayEnabledFor(CVolumeViewer* viewer) const
{
    if (!viewer) {
        return false;
    }
    if (!viewer->selections().empty()) {
        return true;
    }
    return viewer->activeBBoxSceneRect().has_value();
}

void BBoxOverlayController::collectPrimitives(CVolumeViewer* viewer, OverlayBuilder& builder)
{
    if (!viewer) {
        return;
    }

    const auto selections = viewer->selections();
    for (const auto& [sceneRect, color] : selections) {
        OverlayStyle style;
        style.penColor = color;
        style.penWidth = 2.0;
        style.penStyle = Qt::DashLine;
        style.brushColor = Qt::transparent;
        style.z = kPersistentZ;
        builder.addRect(sceneRect.normalized(), false, style);
    }

    if (auto active = viewer->activeBBoxSceneRect()) {
        OverlayStyle style;
        style.penColor = kActiveColor;
        style.penWidth = 2.0;
        style.penStyle = Qt::DashLine;
        style.brushColor = Qt::transparent;
        style.z = kActiveZ;
        builder.addRect(active->normalized(), false, style);
    }
}

