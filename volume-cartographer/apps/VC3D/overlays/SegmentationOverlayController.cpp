#include "SegmentationOverlayController.hpp"

#include "../CSurfaceCollection.hpp"
#include "../CVolumeViewer.hpp"

#include <QColor>

#include <algorithm>
#include <cmath>

#include "vc/core/util/Surface.hpp"

namespace
{
constexpr const char* kOverlayGroupKey = "segmentation_vertex_edit";
const QColor kActiveFill = QColor(255, 215, 0, 220);
const QColor kActiveBorder = QColor(255, 180, 0, 255);
const QColor kNeighborFill = QColor(0, 220, 255, 160);
const QColor kNeighborBorder = QColor(255, 255, 255, 210);
const QColor kGrowthFill = QColor(140, 255, 160, 210);
const QColor kGrowthBorder = QColor(80, 190, 90, 230);
const QColor kHoverFill = QColor(255, 255, 255, 200);
const QColor kHoverBorder = QColor(255, 255, 255, 240);
constexpr qreal kMarkerRadius = 5.0;
constexpr qreal kMarkerPenWidth = 2.0;
constexpr qreal kActivePenWidth = 2.5;
constexpr qreal kMaskZ = 60.0;
constexpr qreal kMarkerZ = 95.0;
constexpr qreal kRadiusCircleZ = 80.0;
}

SegmentationOverlayController::SegmentationOverlayController(CSurfaceCollection* surfaces, QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroupKey, parent)
    , _surfaces(surfaces)
{
    if (_surfaces) {
        connect(_surfaces, &CSurfaceCollection::sendSurfaceChanged,
                this, &SegmentationOverlayController::onSurfaceChanged);
    }
}

void SegmentationOverlayController::setEditingEnabled(bool enabled)
{
    if (_editingEnabled == enabled) {
        return;
    }
    _editingEnabled = enabled;
    refreshAll();
}

void SegmentationOverlayController::setEditManager(SegmentationEditManager* manager)
{
    _editManager = manager;
}

void SegmentationOverlayController::setGaussianParameters(float radiusSteps,
                                                          float sigmaSteps,
                                                          float gridStepWorld)
{
    _radiusSteps = std::max(radiusSteps, 0.0f);
    _sigmaSteps = std::max(sigmaSteps, 0.0f);
    _gridStepWorld = std::max(gridStepWorld, 1e-4f);
}

void SegmentationOverlayController::setActiveVertex(std::optional<VertexMarker> marker)
{
    _activeVertex = std::move(marker);
}

void SegmentationOverlayController::setTouchedVertices(const std::vector<VertexMarker>& markers)
{
    _touchedVertices = markers;
}

void SegmentationOverlayController::setMaskOverlay(const std::vector<cv::Vec3f>& points,
                                                   bool visible,
                                                   float pointRadius,
                                                   float opacity)
{
    _maskPoints = points;
    _maskVisible = visible;
    _maskPointRadius = pointRadius;
    _maskOpacity = opacity;
}

bool SegmentationOverlayController::isOverlayEnabledFor(CVolumeViewer* viewer) const
{
    Q_UNUSED(viewer);
    return _editingEnabled;
}

void SegmentationOverlayController::collectPrimitives(CVolumeViewer* viewer,
                                                      ViewerOverlayControllerBase::OverlayBuilder& builder)
{
    if (!viewer || !_editingEnabled) {
        return;
    }

    if (_maskVisible && !_maskPoints.empty()) {
        ViewerOverlayControllerBase::PathPrimitive maskPath;
        maskPath.points = _maskPoints;
        maskPath.renderMode = ViewerOverlayControllerBase::PathRenderMode::Points;
        maskPath.brushShape = ViewerOverlayControllerBase::PathBrushShape::Circle;
        maskPath.pointRadius = _maskPointRadius;
        maskPath.color = QColor(255, 140, 0);
        maskPath.opacity = _maskOpacity;
        maskPath.z = kMaskZ;
        builder.addPath(maskPath);
    }

    buildVertexMarkers(viewer, builder);
    buildRadiusOverlay(viewer, builder);
}

void SegmentationOverlayController::onSurfaceChanged(std::string name, Surface* surface)
{
    Q_UNUSED(surface);
    if (name == "segmentation") {
        refreshAll();
    }
}

void SegmentationOverlayController::buildRadiusOverlay(CVolumeViewer* viewer,
                                                       ViewerOverlayControllerBase::OverlayBuilder& builder) const
{
    if (!_activeVertex || !_activeVertex->isActive) {
        return;
    }

    const float radiusWorld = _radiusSteps * _gridStepWorld;
    if (radiusWorld <= 0.0f) {
        return;
    }

    const cv::Vec3f world = _activeVertex->world;
    const QPointF sceneCenter = viewer->volumePointToScene(world);

    cv::Vec3f offsetWorld = world;
    offsetWorld[0] += radiusWorld;
    const QPointF sceneEdge = viewer->volumePointToScene(offsetWorld);
    const qreal radiusPixels = std::hypot(sceneEdge.x() - sceneCenter.x(),
                                          sceneEdge.y() - sceneCenter.y());

    if (radiusPixels <= 1.0e-3) {
        return;
    }

    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = QColor(255, 255, 255, 120);
    style.penWidth = 1.5f;
    style.brushColor = QColor(Qt::transparent);
    style.z = kRadiusCircleZ;

    builder.addCircle(sceneCenter, radiusPixels, false, style);
}

void SegmentationOverlayController::buildVertexMarkers(CVolumeViewer* viewer,
                                                       ViewerOverlayControllerBase::OverlayBuilder& builder) const
{
    const auto buildStyle = [](const VertexMarker& marker) {
        ViewerOverlayControllerBase::OverlayStyle style;
        style.penWidth = marker.isActive ? kActivePenWidth : kMarkerPenWidth;
        style.z = kMarkerZ;

        if (marker.isActive) {
            style.penColor = kActiveBorder;
            style.brushColor = kActiveFill;
        } else if (marker.isGrowth) {
            style.penColor = kGrowthBorder;
            style.brushColor = kGrowthFill;
        } else {
            style.penColor = kNeighborBorder;
            style.brushColor = kNeighborFill;
        }

        return style;
    };

    auto appendMarker = [&](const VertexMarker& marker) {
        const QPointF scene = viewer->volumePointToScene(marker.world);
        const auto style = buildStyle(marker);
        builder.addCircle(scene, kMarkerRadius, true, style);
    };

    for (const auto& marker : _touchedVertices) {
        appendMarker(marker);
    }

    if (_activeVertex) {
        auto active = *_activeVertex;
        active.isActive = true;
        appendMarker(active);
    }
}
