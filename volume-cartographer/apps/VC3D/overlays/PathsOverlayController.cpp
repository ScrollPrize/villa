#include "PathsOverlayController.hpp"

#include "../CVolumeViewer.hpp"

#include "vc/core/util/Surface.hpp"

#include <cmath>

namespace
{
constexpr const char* kOverlayGroup = "drawing_paths";
}

PathsOverlayController::PathsOverlayController(QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroup, parent)
{
}

bool PathsOverlayController::isOverlayEnabledFor(CVolumeViewer* viewer) const
{
    if (!viewer) {
        return false;
    }
    const auto& paths = viewer->drawingPaths();
    return !paths.empty();
}

void PathsOverlayController::collectPrimitives(CVolumeViewer* viewer,
                                               OverlayBuilder& builder)
{
    if (!viewer) {
        return;
    }

    const auto& paths = viewer->drawingPaths();
    for (const auto& path : paths) {
        ViewerOverlayControllerBase::PointFilterOptions filter;
        filter.clipToSurface = true;
        filter.planeDistanceTolerance = 4.0f;
        filter.quadDistanceTolerance = 4.0f;
        filter.computeScenePoints = false;

        auto filtered = filterPoints(viewer, path.points, filter);
        if (filtered.volumePoints.empty()) {
            continue;
        }

        if (path.renderMode == PathRenderMode::LineStrip && filtered.volumePoints.size() < 2) {
            continue;
        }

        PathPrimitive adjusted = path;
        adjusted.points = std::move(filtered.volumePoints);
        builder.addPath(adjusted);
    }
}
