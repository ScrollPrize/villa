#include "PatchGraphOverlayController.hpp"

#include "../VolumeViewerBase.hpp"

#include <cstddef>
#include <utility>

namespace
{
constexpr const char* kOverlayGroup = "patch_graph_path";
}

PatchGraphOverlayController::PatchGraphOverlayController(QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroup, parent)
{
}

void PatchGraphOverlayController::setPath(std::vector<cv::Vec3f> path)
{
    _path = std::move(path);
    refreshAll();
}

void PatchGraphOverlayController::setHoverPoint(std::optional<cv::Vec3f> point)
{
    _hoverPoint = point;
    refreshAll();
}

void PatchGraphOverlayController::clearPath()
{
    _path.clear();
    _hoverPoint.reset();
    refreshAll();
}

bool PatchGraphOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    return viewer && (!_path.empty() || _hoverPoint.has_value());
}

void PatchGraphOverlayController::collectPrimitives(VolumeViewerBase* viewer,
                                                    OverlayBuilder& builder)
{
    if (!viewer) {
        return;
    }

    if (_hoverPoint) {
        PointFilterOptions hoverFilter;
        hoverFilter.clipToSurface = true;
        hoverFilter.planeDistanceTolerance = 6.0f;
        hoverFilter.quadDistanceTolerance = 6.0f;
        hoverFilter.computeScenePoints = false;

        auto hover = filterPoints(viewer, std::vector<cv::Vec3f>{*_hoverPoint}, hoverFilter);
        if (!hover.volumePoints.empty()) {
            PathPrimitive marker;
            marker.points = std::move(hover.volumePoints);
            marker.color = QColor(80, 220, 255);
            marker.opacity = 1.0;
            marker.renderMode = PathRenderMode::Points;
            marker.pointRadius = 5.0;
            marker.z = 40.0;
            builder.addPath(marker);
        }
    }

    if (_path.empty()) {
        return;
    }

    PointFilterOptions filter;
    filter.clipToSurface = true;
    filter.planeDistanceTolerance = 4.0f;
    filter.quadDistanceTolerance = 4.0f;
    filter.computeScenePoints = false;

    auto filtered = filterPoints(viewer, _path, filter);
    if (filtered.volumePoints.empty()) {
        return;
    }

    PathPrimitive path;
    path.color = QColor(255, 210, 40);
    path.lineWidth = 4.0;
    path.opacity = 0.95;
    path.renderMode = PathRenderMode::LineStrip;
    path.brushShape = PathBrushShape::Circle;
    path.z = 35.0;

    std::size_t runStart = 0;
    while (runStart < filtered.volumePoints.size()) {
        std::size_t runEnd = runStart + 1;
        while (runEnd < filtered.volumePoints.size() &&
               filtered.sourceIndices[runEnd] == filtered.sourceIndices[runEnd - 1] + 1) {
            ++runEnd;
        }

        if (runEnd - runStart >= 2) {
            path.points.assign(filtered.volumePoints.begin() + static_cast<std::ptrdiff_t>(runStart),
                               filtered.volumePoints.begin() + static_cast<std::ptrdiff_t>(runEnd));
            builder.addPath(path);
        }
        runStart = runEnd;
    }

    PathPrimitive points = path;
    points.points = std::move(filtered.volumePoints);
    points.renderMode = PathRenderMode::Points;
    points.pointRadius = 2.4;
    points.lineWidth = 1.0;
    points.z = 36.0;
    builder.addPath(points);
}
