#pragma once
#include "VolumeViewerBase.hpp"
#include <QTransform>

// Affine map of the surface's zero-offset plane into the current camera.
// Shared by retained surface overlays and the spiral brush projection.
inline QTransform surfaceToSceneTransform(const VolumeViewerBase* viewer)
{
    if (!viewer) return {};
    const QPointF origin = viewer->surfaceCoordsToScene(0.0f, 0.0f);
    const QPointF xStep = viewer->surfaceCoordsToScene(1.0f, 0.0f) - origin;
    const QPointF yStep = viewer->surfaceCoordsToScene(0.0f, 1.0f) - origin;
    return {xStep.x(), xStep.y(), yStep.x(), yStep.y(),
            origin.x(), origin.y()};
}
