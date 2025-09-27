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

#include <algorithm>
#include <utility>

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
    if (primitives.empty()) {
        viewer->clearOverlayGroup(_overlayGroupKey);
        return;
    }

    auto* scene = viewer->fGraphicsView ? viewer->fGraphicsView->scene() : nullptr;
    if (!scene) {
        viewer->clearOverlayGroup(_overlayGroupKey);
        return;
    }

    std::vector<QGraphicsItem*> items;
    items.reserve(primitives.size());

    for (const auto& primitive : primitives) {
        std::visit(
            [&](const auto& prim) {
                using T = std::decay_t<decltype(prim)>;
                if constexpr (std::is_same_v<T, PointPrimitive>) {
                    auto* item = new QGraphicsEllipseItem(
                        prim.position.x() - prim.radius,
                        prim.position.y() - prim.radius,
                        prim.radius * 2.0,
                        prim.radius * 2.0);
                    applyStyle(item, prim.style);
                    scene->addItem(item);
                    items.push_back(item);
                } else if constexpr (std::is_same_v<T, LineStripPrimitive>) {
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
                    applyStyle(item, prim.style);
                    scene->addItem(item);
                    items.push_back(item);
                } else if constexpr (std::is_same_v<T, RectPrimitive>) {
                    auto* item = new QGraphicsRectItem(prim.rect);
                    auto style = prim.style;
                    if (!prim.filled) {
                        style.brushColor = Qt::transparent;
                    }
                    applyStyle(item, style);
                    scene->addItem(item);
                    items.push_back(item);
                } else if constexpr (std::is_same_v<T, TextPrimitive>) {
                    auto* item = new QGraphicsSimpleTextItem(prim.text);
                    item->setFont(prim.font);
                    item->setPos(prim.position);
                    applyStyle(item, prim.style);
                    scene->addItem(item);
                    items.push_back(item);
                }
            },
            primitive);
    }

    if (items.empty()) {
        viewer->clearOverlayGroup(_overlayGroupKey);
        return;
    }

    viewer->setOverlayGroup(_overlayGroupKey, items);
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
