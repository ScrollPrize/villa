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
    if (!viewer) {
        return;
    }

    _slices[viewer] = std::move(data);
    attachViewer(viewer);
    refreshViewer(viewer);
}

void FiberSliceOverlayController::clearSlice()
{
    std::vector<VolumeViewerBase*> oldViewers;
    oldViewers.reserve(_slices.size());
    for (const auto& entry : _slices) {
        oldViewers.push_back(entry.first);
    }
    _slices.clear();
    for (VolumeViewerBase* viewer : oldViewers) {
        ViewerOverlayControllerBase::detachViewer(viewer);
    }
}

void FiberSliceOverlayController::detachViewer(VolumeViewerBase* viewer)
{
    _slices.erase(viewer);
    ViewerOverlayControllerBase::detachViewer(viewer);
}

bool FiberSliceOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    if (!viewer) {
        return false;
    }
    const auto it = _slices.find(viewer);
    return it != _slices.end() && viewer->surfName() == it->second.surfaceName;
}

QPointF FiberSliceOverlayController::projectedVolumeToScene(VolumeViewerBase* viewer,
                                                            const SliceData& slice,
                                                            const cv::Vec3d& point) const
{
    if (!viewer || !vc3d::fiber_slice::isFinitePoint(point)) {
        return {};
    }

    const cv::Vec3d projected = vc3d::fiber_slice::projectPointToPlane(point, slice.plane);
    return volumeToScene(viewer, toVec3f(projected));
}

double FiberSliceOverlayController::currentViewportMinSpan(VolumeViewerBase* viewer,
                                                           const SliceData& slice) const
{
    const QRectF visible = visibleSceneRect(viewer);
    double minSpan = vc3d::fiber_slice::viewportMinVoxelSpan(visible.width(), visible.height());
    if (minSpan > 1.0) {
        return minSpan;
    }

    if (slice.fitSamples.empty()) {
        return 1.0;
    }

    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    for (const cv::Vec3d& sample : slice.fitSamples) {
        const QPointF scenePoint = projectedVolumeToScene(viewer, slice, sample);
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

    const auto sliceIt = _slices.find(viewer);
    if (sliceIt == _slices.end()) {
        return;
    }
    const SliceData& slice = sliceIt->second;

    const auto selectedIt = std::find_if(slice.fibers.begin(), slice.fibers.end(),
                                         [&slice](const FiberData& fiber) {
                                             return fiber.id == slice.selectedFiberId;
                                         });
    if (selectedIt == slice.fibers.end()) {
        return;
    }

    const double minViewportSpan = currentViewportMinSpan(viewer, slice);

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

        const double distance = std::abs(fslice::signedDistanceToPlane(point, slice.plane));
        const double size = fslice::distanceScaledSize(distance, minViewportSpan, 3.0, 0.75);
        const QPointF scenePoint = projectedVolumeToScene(viewer, slice, point);
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
        const QPointF scenePoint = projectedVolumeToScene(viewer, slice, control);
        if (!finiteScenePoint(scenePoint)) {
            continue;
        }
        const double distance = std::abs(fslice::signedDistanceToPlane(control, slice.plane));
        const double radius = fslice::distanceScaledSize(distance, minViewportSpan, 7.0, 4.0);
        builder.addPoint(scenePoint, radius, controlStyle);
    }

    OverlayStyle ellipseStyle;
    ellipseStyle.penColor = QColor(80, 210, 255, 130);
    ellipseStyle.brushColor = QColor(80, 210, 255, 90);
    ellipseStyle.penWidth = 0.75;
    ellipseStyle.z = 38.0;

    for (const FiberData& other : slice.fibers) {
        if (other.id == slice.selectedFiberId || other.linePoints.size() < 2) {
            continue;
        }
        for (size_t i = 1; i < other.linePoints.size(); ++i) {
            const auto crossing =
                fslice::segmentPlaneIntersection(other.linePoints[i - 1], other.linePoints[i], slice.plane);
            if (!crossing) {
                continue;
            }
            const fslice::EllipseStyle ellipse =
                fslice::ellipseStyleForAngle(crossing->angleDegrees, kIntersectionMarkerBaseRadius);
            if (ellipse.opacity <= 0.01) {
                continue;
            }

            cv::Vec3d projectedTangent =
                crossing->tangent - slice.plane.normal * crossing->tangent.dot(slice.plane.normal);
            projectedTangent = fslice::normalizedOrZero(projectedTangent);
            const QPointF centerScene = projectedVolumeToScene(viewer, slice, crossing->point);
            if (!finiteScenePoint(centerScene)) {
                continue;
            }

            double rotation = 0.0;
            if (cv::norm(projectedTangent) > 0.0) {
                const QPointF tangentScene =
                    projectedVolumeToScene(viewer, slice, crossing->point + projectedTangent);
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

    if (slice.connectionSegment) {
        const auto& connector = *slice.connectionSegment;
        const cv::Vec3d delta = connector.targetPoint - connector.sourcePoint;
        const double length = cv::norm(delta);
        if (fslice::isFinitePoint(connector.sourcePoint) &&
            fslice::isFinitePoint(connector.targetPoint) &&
            std::isfinite(length)) {
            OverlayStyle connectorStyle;
            connectorStyle.penColor = QColor(255, 80, 70, 220);
            connectorStyle.brushColor = QColor(255, 80, 70, 180);
            connectorStyle.penCap = Qt::RoundCap;
            connectorStyle.penJoin = Qt::RoundJoin;
            connectorStyle.z = 50.0;

            constexpr int kSteps = 16;
            QPointF previousScene;
            double previousSize = 0.0;
            bool hasPreviousConnector = false;
            for (int step = 0; step <= kSteps; ++step) {
                const double t = static_cast<double>(step) / static_cast<double>(kSteps);
                const cv::Vec3d point = connector.sourcePoint + delta * t;
                const QPointF scenePoint = projectedVolumeToScene(viewer, slice, point);
                if (!finiteScenePoint(scenePoint)) {
                    hasPreviousConnector = false;
                    previousSize = 0.0;
                    continue;
                }
                const double distance =
                    std::abs(fslice::signedDistanceToPlane(point, slice.plane));
                const double size = fslice::connectorNormalizedThickness(distance,
                                                                         connector.maxDistanceVx,
                                                                         5.0,
                                                                         1.0);
                if (hasPreviousConnector) {
                    const QPointF deltaScene = scenePoint - previousScene;
                    auto style = connectorStyle;
                    style.penWidth = (previousSize + size) * 0.5;
                    if (std::hypot(deltaScene.x(), deltaScene.y()) > 1.0e-6) {
                        builder.addLineStrip({previousScene, scenePoint}, false, style);
                    } else if (step == kSteps) {
                        builder.addPoint(scenePoint, style.penWidth * 0.6, style);
                    }
                }
                previousScene = scenePoint;
                previousSize = size;
                hasPreviousConnector = true;
            }

            auto drawEndpointCross = [&](const cv::Vec3d& point, const QColor& color) {
                const QPointF center = projectedVolumeToScene(viewer, slice, point);
                if (!finiteScenePoint(center)) {
                    return;
                }
                OverlayStyle crossStyle;
                crossStyle.penColor = color;
                crossStyle.brushColor = Qt::transparent;
                crossStyle.penCap = Qt::RoundCap;
                crossStyle.penJoin = Qt::RoundJoin;
                crossStyle.penWidth = 1.5;
                crossStyle.z = 55.0;
                constexpr qreal kRadius = 7.0;
                builder.addLineStrip({
                    center + QPointF{-kRadius, -kRadius},
                    center + QPointF{kRadius, kRadius},
                }, false, crossStyle);
                builder.addLineStrip({
                    center + QPointF{-kRadius, kRadius},
                    center + QPointF{kRadius, -kRadius},
                }, false, crossStyle);
            };
            drawEndpointCross(connector.sourcePoint, QColor(255, 230, 70, 245));
            drawEndpointCross(connector.targetPoint, QColor(80, 230, 255, 245));
        }
    }
}
