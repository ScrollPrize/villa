#include "AtlasControlPointsOverlayController.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include "../volume_viewers/VolumeViewerBase.hpp"

#include <QColor>

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <utility>

namespace
{
constexpr const char* kOverlayGroup = "lasagna_atlas_control_points";

bool finiteSurfacePoint(const AtlasControlPointResult& point)
{
    return point.valid &&
           std::isfinite(point.modelH) &&
           std::isfinite(point.modelW);
}

std::optional<cv::Vec2f> gridPointToSurface(VolumeViewerBase* viewer,
                                            const AtlasControlPointResult& point)
{
    auto* quad = viewer ? dynamic_cast<QuadSurface*>(viewer->currentSurface()) : nullptr;
    if (!quad) {
        return std::nullopt;
    }
    const cv::Vec2f scale = quad->scale();
    if (std::abs(scale[0]) < 1e-6f || std::abs(scale[1]) < 1e-6f) {
        return std::nullopt;
    }
    const cv::Vec3f center = quad->center();
    const cv::Vec2f surface{
        point.modelW / scale[0] - center[0],
        point.modelH / scale[1] - center[1],
    };
    if (!std::isfinite(surface[0]) || !std::isfinite(surface[1])) {
        return std::nullopt;
    }
    return surface;
}

ViewerOverlayControllerBase::OverlayStyle lineStyle()
{
    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = QColor(70, 180, 255, 190);
    style.brushColor = Qt::transparent;
    style.penWidth = 1.6;
    style.z = 74.0;
    return style;
}

ViewerOverlayControllerBase::OverlayStyle pointStyle()
{
    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = QColor(255, 245, 120);
    style.brushColor = QColor(255, 245, 120, 230);
    style.penWidth = 0.0;
    style.z = 78.0;
    return style;
}

ViewerOverlayControllerBase::OverlayStyle selectedStyle()
{
    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = QColor(255, 70, 80);
    style.brushColor = QColor(255, 70, 80, 235);
    style.penWidth = 1.5;
    style.z = 88.0;
    return style;
}
}

AtlasControlPointsOverlayController::AtlasControlPointsOverlayController(QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroup, parent)
{
}

void AtlasControlPointsOverlayController::setResults(AtlasControlPointResults results)
{
    _results = std::move(results);
    _selected.reset();
    refreshAll();
}

void AtlasControlPointsOverlayController::clearResults()
{
    _results.clear();
    _selected.reset();
    refreshAll();
}

void AtlasControlPointsOverlayController::setOverlayEnabled(bool enabled)
{
    if (_enabled == enabled) {
        return;
    }
    _enabled = enabled;
    refreshAll();
}

void AtlasControlPointsOverlayController::setSelectedPoint(const QString& fiberId, int controlIndex)
{
    _selected = std::make_pair(fiberId, controlIndex);
    refreshAll();
}

void AtlasControlPointsOverlayController::clearSelectedPoint()
{
    if (!_selected) {
        return;
    }
    _selected.reset();
    refreshAll();
}

bool AtlasControlPointsOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    return viewer && _enabled && !_results.empty() && viewer->surfName() == "segmentation";
}

void AtlasControlPointsOverlayController::collectPrimitives(VolumeViewerBase* viewer,
                                                            OverlayBuilder& builder)
{
    if (!isOverlayEnabledFor(viewer)) {
        return;
    }

    std::map<QString, std::vector<const AtlasControlPointResult*>> byFiber;
    for (const auto& point : _results) {
        if (!finiteSurfacePoint(point)) {
            continue;
        }
        byFiber[point.fiberId].push_back(&point);
    }

    const auto fiberLineStyle = lineStyle();
    const auto controlPointStyle = pointStyle();
    const auto selectedPointStyle = selectedStyle();
    for (auto& [fiberId, points] : byFiber) {
        std::sort(points.begin(), points.end(), [](const auto* a, const auto* b) {
            if (a->sourceIndex != b->sourceIndex) {
                return a->sourceIndex < b->sourceIndex;
            }
            return a->controlIndex < b->controlIndex;
        });
        std::vector<cv::Vec2f> linePoints;
        linePoints.reserve(points.size());
        for (const auto* point : points) {
            if (const auto surface = gridPointToSurface(viewer, *point)) {
                linePoints.push_back(*surface);
            }
        }
        if (linePoints.size() >= 2) {
            builder.addSurfaceLineStrip(linePoints, false, fiberLineStyle);
        }
        for (const auto* point : points) {
            const auto surface = gridPointToSurface(viewer, *point);
            if (!surface) {
                continue;
            }
            const bool selected = _selected &&
                                  _selected->first == point->fiberId &&
                                  _selected->second == point->controlIndex;
            builder.addSurfacePoint(*surface,
                                    selected ? 5.5 : 3.5,
                                    selected ? selectedPointStyle : controlPointStyle);
        }
    }
}
