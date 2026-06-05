#include "FiberSliceOverlayController.hpp"

#include "../volume_viewers/VolumeViewerBase.hpp"

#include <QColor>
#include <QPointF>
#include <QRectF>

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
constexpr const char* kOverlayGroup = "fiber_slice_overlay";
constexpr qreal kIntersectionMarkerBaseRadius = 3.0;

bool finiteScenePoint(const QPointF& point)
{
    return std::isfinite(point.x()) && std::isfinite(point.y());
}

cv::Vec3f toVec3f(const cv::Vec3d& point)
{
    return cv::Vec3f{
        static_cast<float>(point[0]),
        static_cast<float>(point[1]),
        static_cast<float>(point[2]),
    };
}
} // namespace

FiberSliceOverlayController::FiberSliceOverlayController(QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroup, parent)
{
}

void FiberSliceOverlayController::setSlice(VolumeViewerBase* viewer, SliceData data)
{
    if (_activeViewer && _activeViewer != viewer) {
        ViewerOverlayControllerBase::detachViewer(_activeViewer);
    }

    _activeViewer = viewer;
    _slice = std::move(data);

    if (_activeViewer) {
        attachViewer(_activeViewer);
        refreshViewer(_activeViewer);
    }
}

void FiberSliceOverlayController::clearSlice()
{
    VolumeViewerBase* oldViewer = _activeViewer;
    _activeViewer = nullptr;
    _slice.reset();
    if (oldViewer) {
        ViewerOverlayControllerBase::detachViewer(oldViewer);
    }
}

void FiberSliceOverlayController::detachViewer(VolumeViewerBase* viewer)
{
    if (_activeViewer == viewer) {
        _activeViewer = nullptr;
        _slice.reset();
    }
    ViewerOverlayControllerBase::detachViewer(viewer);
}

bool FiberSliceOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    return viewer && viewer == _activeViewer && _slice &&
           viewer->surfName() == _slice->surfaceName;
}

QPointF FiberSliceOverlayController::projectedVolumeToScene(VolumeViewerBase* viewer,
                                                            const cv::Vec3d& point) const
{
    if (!viewer || !_slice || !vc3d::fiber_slice::isFinitePoint(point)) {
        return {};
    }

    const cv::Vec3d projected = vc3d::fiber_slice::projectPointToPlane(point, _slice->plane);
    return volumeToScene(viewer, toVec3f(projected));
}

double FiberSliceOverlayController::currentViewportMinSpan(VolumeViewerBase* viewer) const
{
    const QRectF visible = visibleSceneRect(viewer);
    double minSpan = vc3d::fiber_slice::viewportMinVoxelSpan(visible.width(), visible.height());
    if (minSpan > 1.0) {
        return minSpan;
    }

    if (!_slice || _slice->fitSamples.empty()) {
        return 1.0;
    }

    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    for (const cv::Vec3d& sample : _slice->fitSamples) {
        const QPointF scenePoint = projectedVolumeToScene(viewer, sample);
        if (!finiteScenePoint(scenePoint)) {
            continue;
        }
        minX = std::min(minX, scenePoint.x());
        minY = std::min(minY, scenePoint.y());
        maxX = std::max(maxX, scenePoint.x());
        maxY = std::max(maxY, scenePoint.y());
    }

    if (!std::isfinite(minX) || !std::isfinite(minY) ||
        !std::isfinite(maxX) || !std::isfinite(maxY)) {
        return 1.0;
    }
    return std::max(1.0, vc3d::fiber_slice::viewportMinVoxelSpan(maxX - minX, maxY - minY));
}

void FiberSliceOverlayController::collectPrimitives(VolumeViewerBase* viewer,
                                                    OverlayBuilder& builder)
{
    namespace fslice = vc3d::fiber_slice;

    if (!isOverlayEnabledFor(viewer)) {
        return;
    }

    const auto selectedIt = std::find_if(_slice->fibers.begin(), _slice->fibers.end(),
                                         [this](const FiberData& fiber) {
                                             return fiber.id == _slice->selectedFiberId;
                                         });
    if (selectedIt == _slice->fibers.end()) {
        return;
    }

    const double minViewportSpan = currentViewportMinSpan(viewer);

    OverlayStyle lineStyle;
    lineStyle.penColor = QColor(250, 220, 70, 210);
    lineStyle.brushColor = Qt::transparent;
    lineStyle.penCap = Qt::RoundCap;
    lineStyle.penJoin = Qt::RoundJoin;
    lineStyle.z = 40.0;

    QPointF previousScene;
    double previousSize = 0.0;
    bool hasPrevious = false;
    for (const cv::Vec3d& point : selectedIt->linePoints) {
        if (!fslice::isFinitePoint(point)) {
            hasPrevious = false;
            previousSize = 0.0;
            continue;
        }

        const double distance = std::abs(fslice::signedDistanceToPlane(point, _slice->plane));
        const double size = fslice::distanceScaledSize(distance, minViewportSpan, 3.0, 0.75);
        const QPointF scenePoint = projectedVolumeToScene(viewer, point);
        if (!finiteScenePoint(scenePoint)) {
            hasPrevious = false;
            previousSize = 0.0;
            continue;
        }

        if (hasPrevious) {
            auto segmentStyle = lineStyle;
            segmentStyle.penWidth = (previousSize + size) * 0.5;
            builder.addLineStrip({previousScene, scenePoint}, false, segmentStyle);
        }

        previousScene = scenePoint;
        previousSize = size;
        hasPrevious = true;
    }

    OverlayStyle controlStyle;
    controlStyle.penColor = QColor(255, 220, 40, 255);
    controlStyle.brushColor = QColor(255, 220, 40, 220);
    controlStyle.penWidth = 1.0;
    controlStyle.z = 45.0;
    for (const cv::Vec3d& control : selectedIt->controlPoints) {
        if (!fslice::isFinitePoint(control)) {
            continue;
        }
        const QPointF scenePoint = projectedVolumeToScene(viewer, control);
        if (!finiteScenePoint(scenePoint)) {
            continue;
        }
        const double distance = std::abs(fslice::signedDistanceToPlane(control, _slice->plane));
        const double radius = fslice::distanceScaledSize(distance, minViewportSpan, 7.0, 4.0);
        builder.addPoint(scenePoint, radius, controlStyle);
    }

    OverlayStyle ellipseStyle;
    ellipseStyle.penColor = QColor(80, 210, 255, 130);
    ellipseStyle.brushColor = QColor(80, 210, 255, 90);
    ellipseStyle.penWidth = 0.75;
    ellipseStyle.z = 38.0;

    for (const FiberData& other : _slice->fibers) {
        if (other.id == _slice->selectedFiberId || other.linePoints.size() < 2) {
            continue;
        }
        for (size_t i = 1; i < other.linePoints.size(); ++i) {
            const auto crossing =
                fslice::segmentPlaneIntersection(other.linePoints[i - 1], other.linePoints[i], _slice->plane);
            if (!crossing) {
                continue;
            }
            const fslice::EllipseStyle ellipse =
                fslice::ellipseStyleForAngle(crossing->angleDegrees, kIntersectionMarkerBaseRadius);
            if (ellipse.opacity <= 0.01) {
                continue;
            }

            cv::Vec3d projectedTangent =
                crossing->tangent - _slice->plane.normal * crossing->tangent.dot(_slice->plane.normal);
            projectedTangent = fslice::normalizedOrZero(projectedTangent);
            const QPointF centerScene = projectedVolumeToScene(viewer, crossing->point);
            if (!finiteScenePoint(centerScene)) {
                continue;
            }

            double rotation = 0.0;
            if (cv::norm(projectedTangent) > 0.0) {
                const QPointF tangentScene =
                    projectedVolumeToScene(viewer, crossing->point + projectedTangent);
                const QPointF delta = tangentScene - centerScene;
                if (finiteScenePoint(tangentScene) && std::hypot(delta.x(), delta.y()) > 1.0e-6) {
                    rotation = std::atan2(delta.y(), delta.x());
                }
            }

            auto style = ellipseStyle;
            style.penColor.setAlphaF(std::clamp(ellipse.opacity * 0.55, 0.0, 1.0));
            style.brushColor.setAlphaF(std::clamp(ellipse.opacity * 0.38, 0.0, 1.0));
            builder.addRotatedEllipse(centerScene,
                                      ellipse.majorRadius,
                                      ellipse.minorRadius,
                                      rotation,
                                      true,
                                      style);
        }
    }
}
