#include "ViewerOverlayControllerBase.hpp"

#include "../CVolumeViewer.hpp"
#include "../ViewerManager.hpp"

#include <QGraphicsEllipseItem>
#include <QGraphicsPathItem>
#include <QGraphicsRectItem>
#include <QGraphicsScene>
#include <QGraphicsSimpleTextItem>
#include <QPainterPath>
#include <QPen>
#include <QBrush>
#include <QVector>

#include <algorithm>
#include <utility>
#include <cmath>

ViewerOverlayControllerBase::PathPrimitive
ViewerOverlayControllerBase::PathPrimitive::densify(float samplingInterval) const
{
    if (points.size() < 2) {
        return *this;
    }

    QPainterPath painterPath;
    bool firstPoint = true;
    for (const auto& pt : points) {
        if (firstPoint) {
            painterPath.moveTo(pt[0], pt[1]);
            firstPoint = false;
        } else {
            painterPath.lineTo(pt[0], pt[1]);
        }
    }

    float totalLength = painterPath.length();
    if (totalLength <= 0.0f) {
        return *this;
    }

    int numSamples = static_cast<int>(std::ceil(totalLength / samplingInterval));
    if (numSamples < 2) {
        return *this;
    }

    std::vector<cv::Vec3f> densifiedPoints;
    densifiedPoints.reserve(static_cast<size_t>(numSamples));

    for (int i = 0; i < numSamples; ++i) {
        float percent = static_cast<float>(i) / static_cast<float>(numSamples - 1);
        QPointF sampledPoint = painterPath.pointAtPercent(percent);
        float z = interpolateZ(percent, totalLength, painterPath);
        densifiedPoints.emplace_back(sampledPoint.x(), sampledPoint.y(), z);
    }

    PathPrimitive result = *this;
    result.points = std::move(densifiedPoints);
    return result;
}

float ViewerOverlayControllerBase::PathPrimitive::interpolateZ(float percent,
                                                               float totalLength,
                                                               const QPainterPath& path) const
{
    if (points.size() < 2) {
        return points.empty() ? 0.0f : points.front()[2];
    }

    float targetLength = percent * totalLength;
    float accumulatedLength = 0.0f;

    for (size_t i = 1; i < points.size(); ++i) {
        const cv::Vec3f& p1 = points[i - 1];
        const cv::Vec3f& p2 = points[i];
        float segmentLength = std::sqrt(std::pow(p2[0] - p1[0], 2.0f) + std::pow(p2[1] - p1[1], 2.0f));

        if (accumulatedLength + segmentLength >= targetLength && segmentLength > 0.0f) {
            float segmentPercent = (targetLength - accumulatedLength) / segmentLength;
            return p1[2] + segmentPercent * (p2[2] - p1[2]);
        }

        accumulatedLength += segmentLength;
    }

    return points.back()[2];
}

ViewerOverlayControllerBase::OverlayBuilder::OverlayBuilder(CVolumeViewer* viewer)
    : _viewer(viewer)
{
}

void ViewerOverlayControllerBase::OverlayBuilder::addPoint(const QPointF& position,
                                                           qreal radius,
                                                           OverlayStyle style)
{
    PointPrimitive prim;
    prim.position = position;
    prim.radius = radius;
    prim.style = style;
    _primitives.emplace_back(std::move(prim));
}

void ViewerOverlayControllerBase::OverlayBuilder::addCircle(const QPointF& center,
                                                            qreal radius,
                                                            bool filled,
                                                            OverlayStyle style)
{
    CirclePrimitive prim;
    prim.center = center;
    prim.radius = radius;
    prim.filled = filled;
    prim.style = style;
    _primitives.emplace_back(std::move(prim));
}

void ViewerOverlayControllerBase::OverlayBuilder::addLineStrip(const std::vector<QPointF>& points,
                                                               bool closed,
                                                               OverlayStyle style)
{
    if (points.empty()) {
        return;
    }
    LineStripPrimitive prim;
    prim.points = points;
    prim.closed = closed;
    prim.style = style;
    _primitives.emplace_back(std::move(prim));
}

void ViewerOverlayControllerBase::OverlayBuilder::addRect(const QRectF& rect,
                                                          bool filled,
                                                          OverlayStyle style)
{
    RectPrimitive prim;
    prim.rect = rect;
    prim.filled = filled;
    prim.style = style;
    _primitives.emplace_back(std::move(prim));
}

void ViewerOverlayControllerBase::OverlayBuilder::addText(const QPointF& position,
                                                          const QString& text,
                                                          const QFont& font,
                                                          OverlayStyle style)
{
    TextPrimitive prim;
    prim.position = position;
    prim.text = text;
    prim.font = font;
    prim.style = style;
    _primitives.emplace_back(std::move(prim));
}

void ViewerOverlayControllerBase::OverlayBuilder::addPath(const PathPrimitive& path)
{
    _primitives.emplace_back(path);
}

void ViewerOverlayControllerBase::OverlayBuilder::addArrow(const QPointF& start,
                                                           const QPointF& end,
                                                           qreal headLength,
                                                           qreal headWidth,
                                                           OverlayStyle style)
{
    ArrowPrimitive prim;
    prim.start = start;
    prim.end = end;
    prim.headLength = headLength;
    prim.headWidth = headWidth;
    prim.style = style;
    _primitives.emplace_back(std::move(prim));
}

std::vector<ViewerOverlayControllerBase::OverlayPrimitive>
ViewerOverlayControllerBase::OverlayBuilder::takePrimitives()
{
    return std::exchange(_primitives, {});
}

ViewerOverlayControllerBase::ViewerOverlayControllerBase(std::string overlayGroupKey, QObject* parent)
    : QObject(parent)
    , _overlayGroupKey(std::move(overlayGroupKey))
{
}

ViewerOverlayControllerBase::~ViewerOverlayControllerBase()
{
    detachAllViewers();
    if (_manager) {
        QObject::disconnect(_managerCreatedConn);
        QObject::disconnect(_managerDestroyedConn);
    }
}

void ViewerOverlayControllerBase::attachViewer(CVolumeViewer* viewer)
{
    if (!viewer) {
        return;
    }

    auto existing = std::find_if(_viewers.begin(), _viewers.end(), [viewer](const ViewerEntry& entry) {
        return entry.viewer == viewer;
    });
    if (existing != _viewers.end()) {
        rebuildOverlay(viewer);
        return;
    }

    ViewerEntry entry;
    entry.viewer = viewer;
    entry.overlaysUpdatedConn = QObject::connect(viewer, &CVolumeViewer::overlaysUpdated,
                                                 this, [this, viewer]() { rebuildOverlay(viewer); });
    entry.destroyedConn = QObject::connect(viewer, &QObject::destroyed,
                                           this, [this, viewer]() { detachViewer(viewer); });

    _viewers.push_back(entry);
    rebuildOverlay(viewer);
}

void ViewerOverlayControllerBase::detachViewer(CVolumeViewer* viewer)
{
    auto it = std::remove_if(_viewers.begin(), _viewers.end(), [viewer](const ViewerEntry& entry) {
        return entry.viewer == viewer;
    });

    for (auto iter = it; iter != _viewers.end(); ++iter) {
        QObject::disconnect(iter->overlaysUpdatedConn);
        QObject::disconnect(iter->destroyedConn);
        if (iter->viewer) {
            iter->viewer->clearOverlayGroup(_overlayGroupKey);
        }
    }

    _viewers.erase(it, _viewers.end());
}

void ViewerOverlayControllerBase::bindToViewerManager(ViewerManager* manager)
{
    if (_manager == manager) {
        return;
    }

    if (_manager) {
        QObject::disconnect(_managerCreatedConn);
        QObject::disconnect(_managerDestroyedConn);
    }

    _manager = manager;
    if (!_manager) {
        return;
    }

    _managerCreatedConn = QObject::connect(_manager, &ViewerManager::viewerCreated,
                                           this, [this](CVolumeViewer* viewer) {
                                               attachViewer(viewer);
                                           });

    QObject::disconnect(_managerDestroyedConn);
    _managerDestroyedConn = QObject::connect(_manager, &QObject::destroyed,
                                             this, [this]() {
                                                 _manager = nullptr;
                                             });

    _manager->forEachViewer([this](CVolumeViewer* viewer) {
        attachViewer(viewer);
    });
}

void ViewerOverlayControllerBase::refreshAll()
{
    for (const auto& entry : _viewers) {
        rebuildOverlay(entry.viewer);
    }
}

void ViewerOverlayControllerBase::refreshViewer(CVolumeViewer* viewer)
{
    rebuildOverlay(viewer);
}

bool ViewerOverlayControllerBase::isOverlayEnabledFor(CVolumeViewer* /*viewer*/) const
{
    return true;
}

void ViewerOverlayControllerBase::clearOverlay(CVolumeViewer* viewer) const
{
    if (viewer) {
        viewer->clearOverlayGroup(_overlayGroupKey);
    }
}

QPointF ViewerOverlayControllerBase::volumeToScene(CVolumeViewer* viewer, const cv::Vec3f& volumePoint) const
{
    if (!viewer) {
        return QPointF();
    }
    return viewer->volumePointToScene(volumePoint);
}

cv::Vec3f ViewerOverlayControllerBase::sceneToVolume(CVolumeViewer* viewer, const QPointF& scenePoint) const
{
    if (!viewer) {
        return cv::Vec3f();
    }
    return viewer->sceneToVolume(scenePoint);
}

std::vector<QPointF> ViewerOverlayControllerBase::volumeToScene(CVolumeViewer* viewer,
                                                                const std::vector<cv::Vec3f>& volumePoints) const
{
    std::vector<QPointF> results;
    results.reserve(volumePoints.size());
    for (const auto& p : volumePoints) {
        results.emplace_back(volumeToScene(viewer, p));
    }
    return results;
}

ViewerOverlayControllerBase::FilteredPoints
ViewerOverlayControllerBase::filterPoints(CVolumeViewer* viewer,
                                          const std::vector<cv::Vec3f>& points,
                                          const PointFilterOptions& options) const
{
    FilteredPoints result;
    if (!viewer || points.empty()) {
        return result;
    }

    result.volumePoints.reserve(points.size());
    if (options.computeScenePoints) {
        result.scenePoints.reserve(points.size());
    }
    result.sourceIndices.reserve(points.size());

    auto* surface = viewer->currentSurface();
    auto* planeSurface = options.clipToSurface ? dynamic_cast<PlaneSurface*>(surface) : nullptr;
    auto* quadSurface = options.clipToSurface ? dynamic_cast<QuadSurface*>(surface) : nullptr;

    QRectF visibleRect;
    if (options.requireSceneVisibility) {
        visibleRect = visibleSceneRect(viewer);
    }

    size_t index = 0;
    for (const auto& point : points) {
        const size_t srcIndex = index++;
        bool keep = true;

        if (planeSurface) {
            float dist = planeSurface->pointDist(point);
            if (std::fabs(dist) > options.planeDistanceTolerance) {
                keep = false;
            }
        }

        if (keep && quadSurface) {
            auto ptr = quadSurface->pointer();
            float res = quadSurface->pointTo(ptr, point, options.quadDistanceTolerance, 100);
            if (res > options.quadDistanceTolerance) {
                keep = false;
            }
        }

        if (keep && options.volumePredicate) {
            keep = options.volumePredicate(point, srcIndex);
        }

        QPointF scenePoint;
        bool sceneComputed = false;
        if (keep && (options.requireSceneVisibility || options.customSceneRect.has_value() || options.scenePredicate || options.computeScenePoints)) {
            scenePoint = volumeToScene(viewer, point);
            sceneComputed = true;

            bool visibleOk = true;
            if (options.requireSceneVisibility) {
                visibleOk = visibleRect.contains(scenePoint);
            }
            if (visibleOk && options.customSceneRect) {
                visibleOk = options.customSceneRect->contains(scenePoint);
            }
            if (!visibleOk) {
                keep = false;
            }

            if (keep && options.scenePredicate) {
                keep = options.scenePredicate(scenePoint, srcIndex);
            }
        }

        if (keep) {
            result.volumePoints.push_back(point);
            if (options.computeScenePoints) {
                if (!sceneComputed) {
                    scenePoint = volumeToScene(viewer, point);
                }
                result.scenePoints.push_back(scenePoint);
            }
            result.sourceIndices.push_back(srcIndex);
        }
    }

    return result;
}

QGraphicsScene* ViewerOverlayControllerBase::viewerScene(CVolumeViewer* viewer) const
{
    if (!viewer || !viewer->fGraphicsView) {
        return nullptr;
    }
    return viewer->fGraphicsView->scene();
}

QRectF ViewerOverlayControllerBase::visibleSceneRect(CVolumeViewer* viewer) const
{
    if (!viewer || !viewer->fGraphicsView) {
        return QRectF();
    }
    auto* view = viewer->fGraphicsView;
    return view->mapToScene(view->viewport()->rect()).boundingRect();
}

bool ViewerOverlayControllerBase::isScenePointVisible(CVolumeViewer* viewer, const QPointF& scenePoint) const
{
    return visibleSceneRect(viewer).contains(scenePoint);
}

Surface* ViewerOverlayControllerBase::viewerSurface(CVolumeViewer* viewer) const
{
    return viewer ? viewer->currentSurface() : nullptr;
}

namespace
{
void applyStyle(QGraphicsItem* item, const ViewerOverlayControllerBase::OverlayStyle& style)
{
    if (!item) {
        return;
    }

    item->setZValue(style.z);

    QPen pen(style.penColor);
    pen.setWidthF(style.penWidth);
    pen.setStyle(style.penStyle);
    pen.setCapStyle(style.penCap);
    pen.setJoinStyle(style.penJoin);
    if (!style.dashPattern.empty()) {
        QVector<qreal> pattern;
        pattern.reserve(static_cast<int>(style.dashPattern.size()));
        for (qreal value : style.dashPattern) {
            pattern.append(value);
        }
        pen.setDashPattern(pattern);
    }

    if (auto* pathItem = qgraphicsitem_cast<QGraphicsPathItem*>(item)) {
        pathItem->setPen(pen);
        pathItem->setBrush(QBrush(style.brushColor));
    } else if (auto* rectItem = qgraphicsitem_cast<QGraphicsRectItem*>(item)) {
        rectItem->setPen(pen);
        rectItem->setBrush(QBrush(style.brushColor));
    } else if (auto* ellipseItem = qgraphicsitem_cast<QGraphicsEllipseItem*>(item)) {
        ellipseItem->setPen(pen);
        ellipseItem->setBrush(QBrush(style.brushColor));
    } else if (auto* textItem = qgraphicsitem_cast<QGraphicsSimpleTextItem*>(item)) {
        textItem->setBrush(QBrush(style.penColor));
        textItem->setPen(pen);
    }
}
} // namespace

void ViewerOverlayControllerBase::applyPrimitives(CVolumeViewer* viewer,
                                                  const std::string& overlayKey,
                                                  std::vector<OverlayPrimitive> primitives)
{
    if (!viewer) {
        return;
    }

    if (primitives.empty()) {
        viewer->clearOverlayGroup(overlayKey);
        return;
    }

    auto* scene = viewer->fGraphicsView ? viewer->fGraphicsView->scene() : nullptr;
    if (!scene) {
        viewer->clearOverlayGroup(overlayKey);
        return;
    }

    std::vector<QGraphicsItem*> items;
    items.reserve(primitives.size());

    auto addItem = [&](QGraphicsItem* item, const OverlayStyle& style) {
        if (!item) {
            return;
        }
        applyStyle(item, style);
        scene->addItem(item);
        items.push_back(item);
    };

    struct PointGroup
    {
        qreal radius{0.0};
        OverlayStyle style{};
        QPainterPath path;
    };

    auto fuzzyEqual = [](qreal lhs, qreal rhs) {
        return std::fabs(lhs - rhs) <= 1e-6;
    };

    auto dashPatternEqual = [&](const OverlayStyle& a, const OverlayStyle& b) {
        if (a.dashPattern.size() != b.dashPattern.size()) {
            return false;
        }
        for (std::size_t i = 0; i < a.dashPattern.size(); ++i) {
            if (!fuzzyEqual(a.dashPattern[i], b.dashPattern[i])) {
                return false;
            }
        }
        return true;
    };

    auto styleEquals = [&](const OverlayStyle& a, const OverlayStyle& b) {
        return a.penColor == b.penColor &&
               a.brushColor == b.brushColor &&
               fuzzyEqual(a.penWidth, b.penWidth) &&
               a.penStyle == b.penStyle &&
               a.penCap == b.penCap &&
               a.penJoin == b.penJoin &&
               dashPatternEqual(a, b) &&
               fuzzyEqual(a.z, b.z);
    };

    std::vector<PointGroup> pointGroups;
    pointGroups.reserve(4);

    auto groupForPoint = [&](const PointPrimitive& prim) -> PointGroup& {
        for (auto& group : pointGroups) {
            if (fuzzyEqual(group.radius, prim.radius) && styleEquals(group.style, prim.style)) {
                return group;
            }
        }
        PointGroup group;
        group.radius = prim.radius;
        group.style = prim.style;
        group.path = QPainterPath();
        pointGroups.push_back(group);
        return pointGroups.back();
    };

    auto flushPointGroups = [&]() {
        for (auto& group : pointGroups) {
            if (group.path.isEmpty()) {
                continue;
            }
            auto* item = new QGraphicsPathItem(group.path);
            addItem(item, group.style);
        }
        pointGroups.clear();
    };

    for (const auto& primitive : primitives) {
        std::visit(
            [&](const auto& prim) {
                using T = std::decay_t<decltype(prim)>;
                if constexpr (std::is_same_v<T, PointPrimitive>) {
                    PointGroup& group = groupForPoint(prim);
                    group.path.addEllipse(prim.position, prim.radius, prim.radius);
                } else if constexpr (std::is_same_v<T, CirclePrimitive>) {
                    flushPointGroups();
                    auto* item = new QGraphicsEllipseItem(
                        prim.center.x() - prim.radius,
                        prim.center.y() - prim.radius,
                        prim.radius * 2.0,
                        prim.radius * 2.0);
                    auto style = prim.style;
                    if (!prim.filled) {
                        style.brushColor = Qt::transparent;
                    }
                    addItem(item, style);
                } else if constexpr (std::is_same_v<T, LineStripPrimitive>) {
                    flushPointGroups();
                    if (prim.points.size() < 2) {
                        return;
                    }
                    QPainterPath path(prim.points.front());
                    for (size_t i = 1; i < prim.points.size(); ++i) {
                        path.lineTo(prim.points[i]);
                    }
                    if (prim.closed) {
                        path.closeSubpath();
                    }
                    auto* item = new QGraphicsPathItem(path);
                    addItem(item, prim.style);
                } else if constexpr (std::is_same_v<T, RectPrimitive>) {
                    flushPointGroups();
                    auto* item = new QGraphicsRectItem(prim.rect);
                    auto style = prim.style;
                    if (!prim.filled) {
                        style.brushColor = Qt::transparent;
                    }
                    addItem(item, style);
                } else if constexpr (std::is_same_v<T, TextPrimitive>) {
                    flushPointGroups();
                    auto* item = new QGraphicsSimpleTextItem(prim.text);
                    item->setFont(prim.font);
                    item->setPos(prim.position);
                    addItem(item, prim.style);
                } else if constexpr (std::is_same_v<T, PathPrimitive>) {
                    flushPointGroups();
                    if (prim.points.empty()) {
                        return;
                    }

                    std::vector<QPointF> scenePoints;
                    scenePoints.reserve(prim.points.size());
                    for (const auto& p : prim.points) {
                        scenePoints.emplace_back(viewer->volumePointToScene(p));
                    }

                    OverlayStyle style;
                    style.penColor = prim.color;
                    style.penColor.setAlphaF(std::clamp(prim.opacity, 0.0, 1.0));
                    style.brushColor = Qt::transparent;
                    style.penWidth = prim.lineWidth;
                    style.penCap = prim.brushShape == PathBrushShape::Square ? Qt::SquareCap : Qt::RoundCap;
                    style.penJoin = prim.brushShape == PathBrushShape::Square ? Qt::MiterJoin : Qt::RoundJoin;
                    style.z = prim.z;

                    if (prim.isEraser) {
                        style.penColor = QColor(255, 50, 50);
                        style.penColor.setAlphaF(std::clamp(prim.opacity, 0.0, 1.0));
                        style.penStyle = Qt::DashLine;
                        style.dashPattern = {4.0, 4.0};
                    }

                    if (prim.renderMode == PathRenderMode::LineStrip) {
                        if (scenePoints.size() < 2) {
                            return;
                        }
                        QPainterPath path(scenePoints.front());
                        for (size_t i = 1; i < scenePoints.size(); ++i) {
                            path.lineTo(scenePoints[i]);
                        }
                        if (prim.closed) {
                            path.closeSubpath();
                        }
                        auto* item = new QGraphicsPathItem(path);
                        addItem(item, style);
                    } else {
                        QPainterPath path;
                        for (const auto& pt : scenePoints) {
                            if (prim.brushShape == PathBrushShape::Square) {
                                path.addRect(pt.x() - prim.pointRadius,
                                             pt.y() - prim.pointRadius,
                                             prim.pointRadius * 2.0,
                                             prim.pointRadius * 2.0);
                            } else {
                                path.addEllipse(pt, prim.pointRadius, prim.pointRadius);
                            }
                        }

                        style.brushColor = prim.color;
                        style.brushColor.setAlphaF(std::clamp(prim.opacity, 0.0, 1.0));
                        auto* item = new QGraphicsPathItem(path);
                        addItem(item, style);
                    }
                } else if constexpr (std::is_same_v<T, ArrowPrimitive>) {
                    flushPointGroups();
                    QPainterPath path;
                    path.moveTo(prim.start);
                    path.lineTo(prim.end);

                    QPointF dir = prim.end - prim.start;
                    double mag = std::hypot(dir.x(), dir.y());
                    if (mag < 1e-5) {
                        return;
                    }
                    dir.setX(dir.x() / mag);
                    dir.setY(dir.y() / mag);
                    QPointF perp(-dir.y(), dir.x());
                    QPointF tip = prim.end;
                    QPointF base = tip - dir * prim.headLength;
                    QPointF left = base + perp * prim.headWidth;
                    QPointF right = base - perp * prim.headWidth;
                    path.moveTo(tip);
                    path.lineTo(left);
                    path.moveTo(tip);
                    path.lineTo(right);

                    auto* item = new QGraphicsPathItem(path);
                    addItem(item, prim.style);
                }
            },
            primitive);
    }

    flushPointGroups();

    if (items.empty()) {
        viewer->clearOverlayGroup(overlayKey);
        return;
    }

    viewer->setOverlayGroup(overlayKey, items);
}

void ViewerOverlayControllerBase::rebuildOverlay(CVolumeViewer* viewer)
{
    if (!viewer) {
        return;
    }

    if (!isOverlayEnabledFor(viewer)) {
        viewer->clearOverlayGroup(_overlayGroupKey);
        return;
    }

    OverlayBuilder builder(viewer);
    collectPrimitives(viewer, builder);
    auto primitives = builder.takePrimitives();
    applyPrimitives(viewer, _overlayGroupKey, std::move(primitives));
}

void ViewerOverlayControllerBase::detachAllViewers()
{
    for (auto& entry : _viewers) {
        QObject::disconnect(entry.overlaysUpdatedConn);
        QObject::disconnect(entry.destroyedConn);
        if (entry.viewer) {
            entry.viewer->clearOverlayGroup(_overlayGroupKey);
        }
    }
    _viewers.clear();
}
