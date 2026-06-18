#include "FiberSliceOverlayController.hpp"

#include "../LineAnnotationGeneratedViews.hpp"
#include "../volume_viewers/VolumeViewerBase.hpp"
#include "vc/core/util/PlaneSurface.hpp"

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

ViewerOverlayControllerBase::OverlayStyle lineStyle(QColor color, qreal width = 1.6)
{
    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = color;
    style.brushColor = Qt::transparent;
    style.penWidth = width;
    style.penCap = Qt::RoundCap;
    style.penJoin = Qt::RoundJoin;
    style.z = 40.0;
    return style;
}

ViewerOverlayControllerBase::OverlayStyle filledPointStyle(QColor pen,
                                                           QColor brush,
                                                           qreal width,
                                                           qreal z)
{
    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = pen;
    style.brushColor = brush;
    style.penWidth = width;
    style.z = z;
    return style;
}

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

cv::Vec3d toVec3d(const cv::Vec3f& point)
{
    return cv::Vec3d{
        static_cast<double>(point[0]),
        static_cast<double>(point[1]),
        static_cast<double>(point[2]),
    };
}

QColor fadedLineTailColor(const QColor& baseColor)
{
    QColor color;
    color.setRed(std::clamp(static_cast<int>(std::round(baseColor.red() * 0.75)), 0, 255));
    color.setGreen(std::clamp(static_cast<int>(std::round(baseColor.green() * 0.75)), 0, 255));
    color.setBlue(std::clamp(static_cast<int>(std::round(baseColor.blue() * 0.75)), 0, 255));
    color.setAlpha(std::clamp(static_cast<int>(std::round(baseColor.alpha() * 0.75)), 0, 255));
    return color;
}
} // namespace

FiberSliceOverlayController::FiberSliceOverlayController(QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroup, parent)
{
}

FiberSliceOverlayController::FiberStyle FiberSliceOverlayController::sourceFiberStyle()
{
    FiberStyle style;
    style.lineStyle = lineStyle(QColor(255, 185, 35, 215), 1.7);
    style.controlStyle = filledPointStyle(QColor(255, 95, 35, 255),
                                          QColor(255, 155, 35, 225),
                                          1.1,
                                          45.0);
    style.markerStyle = filledPointStyle(QColor(255, 235, 65, 250),
                                         QColor(255, 80, 55, 180),
                                         1.7,
                                         55.0);
    return style;
}

FiberSliceOverlayController::FiberStyle FiberSliceOverlayController::targetFiberStyle()
{
    FiberStyle style;
    style.lineStyle = lineStyle(QColor(35, 220, 255, 215), 1.7);
    style.controlStyle = filledPointStyle(QColor(0, 150, 255, 255),
                                          QColor(65, 225, 255, 225),
                                          1.1,
                                          45.0);
    style.markerStyle = filledPointStyle(QColor(85, 235, 255, 250),
                                         QColor(0, 150, 255, 180),
                                         1.7,
                                         55.0);
    return style;
}

bool FiberSliceOverlayController::focusMarkerVisible(double distanceToPlane,
                                                     double minVisibleViewportSpanVx,
                                                     double viewportFraction)
{
    if (!std::isfinite(distanceToPlane) ||
        !std::isfinite(minVisibleViewportSpanVx) ||
        !std::isfinite(viewportFraction) ||
        viewportFraction < 0.0) {
        return false;
    }
    return vc3d::fiber_slice::focusedIntersectionMarkerVisible(distanceToPlane,
                                                               minVisibleViewportSpanVx,
                                                               viewportFraction);
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

vc3d::fiber_slice::Plane FiberSliceOverlayController::currentPlaneForViewer(
    VolumeViewerBase* viewer,
    const SliceData& slice) const
{
    if (viewer) {
        if (auto* plane = dynamic_cast<PlaneSurface*>(viewer->currentSurface())) {
            const cv::Vec3d origin = toVec3d(plane->origin());
            const cv::Vec3d normal = toVec3d(plane->normal({0, 0, 0}));
            if (vc3d::fiber_slice::isFinitePoint(origin) &&
                vc3d::fiber_slice::isFinitePoint(normal) &&
                cv::norm(normal) > 1.0e-10) {
                return vc3d::fiber_slice::Plane{origin, normal};
            }
        }
    }
    return slice.plane;
}

QPointF FiberSliceOverlayController::projectedVolumeToScene(VolumeViewerBase* viewer,
                                                            const SliceData& slice,
                                                            const cv::Vec3d& point) const
{
    if (!viewer || !vc3d::fiber_slice::isFinitePoint(point)) {
        return {};
    }

    const cv::Vec3d projected =
        vc3d::fiber_slice::projectPointToPlane(point, currentPlaneForViewer(viewer, slice));
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
    const fslice::Plane plane = currentPlaneForViewer(viewer, slice);

    std::vector<uint64_t> fullLineFiberIds = slice.fullLineFiberIds;
    if (fullLineFiberIds.empty() && slice.selectedFiberId != 0) {
        fullLineFiberIds.push_back(slice.selectedFiberId);
    }
    auto isFullLineFiber = [&fullLineFiberIds](uint64_t fiberId) {
        return std::find(fullLineFiberIds.begin(), fullLineFiberIds.end(), fiberId) !=
               fullLineFiberIds.end();
    };

    const double minViewportSpan = currentViewportMinSpan(viewer, slice);

    for (size_t fullIndex = 0; fullIndex < fullLineFiberIds.size(); ++fullIndex) {
        const uint64_t fullFiberId = fullLineFiberIds[fullIndex];
        const auto fiberIt = std::find_if(slice.fibers.begin(), slice.fibers.end(),
                                          [fullFiberId](const FiberData& fiber) {
                                              return fiber.id == fullFiberId;
                                          });
        if (fiberIt == slice.fibers.end()) {
            continue;
        }

        auto thisLineStyle = fiberIt->style.lineStyle;
        auto thisControlStyle = fiberIt->style.controlStyle;
        auto tailLineStyle = thisLineStyle;
        tailLineStyle.penColor = fadedLineTailColor(thisLineStyle.penColor);
        tailLineStyle.z = thisLineStyle.z - 1.0;

        std::optional<std::pair<double, double>> controlRange;
        const fslice::ControlSpanSelection span =
            fslice::selectControlSpan(fiberIt->linePoints, fiberIt->controlPoints);
        if (span.valid) {
            controlRange = std::make_pair(static_cast<double>(span.firstLineIndex),
                                          static_cast<double>(span.lastLineIndex));
        }

        QPointF previousScene;
        double previousSize = 0.0;
        double previousLinePosition = 0.0;
        bool hasPrevious = false;
        for (size_t pointIndex = 0; pointIndex < fiberIt->linePoints.size(); ++pointIndex) {
            const cv::Vec3d& point = fiberIt->linePoints[pointIndex];
            if (!fslice::isFinitePoint(point)) {
                hasPrevious = false;
                previousSize = 0.0;
                continue;
            }

            const double distance = std::abs(fslice::signedDistanceToPlane(point, plane));
            const double size = fslice::distanceScaledSize(distance, minViewportSpan, 3.0, 0.75);
            const QPointF scenePoint = projectedVolumeToScene(viewer, slice, point);
            if (!finiteScenePoint(scenePoint)) {
                hasPrevious = false;
                previousSize = 0.0;
                continue;
            }

            if (hasPrevious) {
                auto segmentStyle =
                    vc3d::line_annotation::generatedLineSegmentIsTail(
                        previousLinePosition,
                        static_cast<double>(pointIndex),
                        controlRange)
                    ? tailLineStyle
                    : thisLineStyle;
                segmentStyle.penWidth = (previousSize + size) * 0.5;
                builder.addLineStrip({previousScene, scenePoint}, false, segmentStyle);
            }

            previousScene = scenePoint;
            previousSize = size;
            previousLinePosition = static_cast<double>(pointIndex);
            hasPrevious = true;
        }

        for (const cv::Vec3d& control : fiberIt->controlPoints) {
            if (!fslice::isFinitePoint(control)) {
                continue;
            }
            const QPointF scenePoint = projectedVolumeToScene(viewer, slice, control);
            if (!finiteScenePoint(scenePoint)) {
                continue;
            }
            const double distance = std::abs(fslice::signedDistanceToPlane(control, plane));
            const double radius = fslice::distanceScaledSize(distance, minViewportSpan, 7.0, 4.0);
            builder.addPoint(scenePoint, radius, thisControlStyle);
        }
    }

    if (slice.showGenericCrossings) {
        for (const FiberData& other : slice.fibers) {
            if (isFullLineFiber(other.id) || other.linePoints.size() < 2) {
                continue;
            }
            for (size_t i = 1; i < other.linePoints.size(); ++i) {
                const auto crossing =
                    fslice::segmentPlaneIntersection(other.linePoints[i - 1], other.linePoints[i], plane);
                if (!crossing) {
                    continue;
                }
                const fslice::EllipseStyle ellipse =
                    fslice::ellipseStyleForAngle(crossing->angleDegrees, kIntersectionMarkerBaseRadius);
                if (ellipse.opacity <= 0.01) {
                    continue;
                }

                cv::Vec3d projectedTangent =
                    crossing->tangent - plane.normal * crossing->tangent.dot(plane.normal);
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

                auto style = other.style.markerStyle;
                style.penWidth = 0.75;
                style.z = 38.0;
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

    auto styleForFiber = [&slice](uint64_t fiberId) -> FiberStyle {
        const auto it = std::find_if(slice.fibers.begin(), slice.fibers.end(),
                                     [fiberId](const FiberData& fiber) {
                                         return fiber.id == fiberId;
                                     });
        if (it != slice.fibers.end()) {
            return it->style;
        }
        return sourceFiberStyle();
    };

    auto drawEndpointCross = [&](const cv::Vec3d& point, const OverlayStyle& markerStyle) {
        const QPointF center = projectedVolumeToScene(viewer, slice, point);
        if (!finiteScenePoint(center)) {
            return;
        }
        auto crossStyle = markerStyle;
        crossStyle.brushColor = Qt::transparent;
        crossStyle.penCap = Qt::RoundCap;
        crossStyle.penJoin = Qt::RoundJoin;
        crossStyle.z = std::max<qreal>(crossStyle.z, 55.0);
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

    for (const FocusMarker& marker : slice.focusMarkers) {
        if (!fslice::isFinitePoint(marker.point)) {
            continue;
        }
        const double distance = fslice::signedDistanceToPlane(marker.point, plane);
        if (marker.requirePlaneProximity &&
            !focusMarkerVisible(distance, minViewportSpan, slice.focusMarkerViewportFraction)) {
            continue;
        }
        const QPointF center = projectedVolumeToScene(viewer, slice, marker.point);
        if (!finiteScenePoint(center)) {
            continue;
        }
        auto style = styleForFiber(marker.fiberId).markerStyle;
        style.brushColor = Qt::transparent;
        style.penCap = Qt::RoundCap;
        style.penJoin = Qt::RoundJoin;
        style.penWidth = std::max<qreal>(style.penWidth, 2.0);
        style.z = std::max<qreal>(style.z, 180.0);
        const qreal radius = static_cast<qreal>(marker.radius);
        builder.addLineStrip({
            center + QPointF{-radius, -radius},
            center + QPointF{radius, radius},
        }, false, style);
        builder.addLineStrip({
            center + QPointF{-radius, radius},
            center + QPointF{radius, -radius},
        }, false, style);
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
                    std::abs(fslice::signedDistanceToPlane(point, plane));
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

            drawEndpointCross(connector.sourcePoint,
                              styleForFiber(connector.sourceFiberId).markerStyle);
            drawEndpointCross(connector.targetPoint,
                              styleForFiber(connector.targetFiberId).markerStyle);
        }
    }
}
