#include "IntersectionLayerItem.hpp"

#include <QWidget>

#include <algorithm>
#include <cmath>

IntersectionLayerItem::IntersectionLayerItem(QGraphicsItem* parent)
    : QGraphicsItem(parent)
{
    setAcceptedMouseButtons(Qt::NoButton);
    setAcceptHoverEvents(false);
}

void IntersectionLayerItem::setEntries(std::vector<Entry> entries)
{
    prepareGeometryChange();
    _entries = std::move(entries);
    _bounds = QRectF();
    _elementCount = 0;
    for (const auto& entry : _entries) {
        if (entry.path.isEmpty()) {
            continue;
        }
        // Cosmetic pen widths are device pixels; the intersection scene is in
        // viewport pixels, so the width doubles as the item-space margin.
        const qreal margin = std::max<qreal>(1.0, entry.pen.widthF()) + 1.0;
        _bounds |= entry.path.controlPointRect().adjusted(-margin, -margin, margin, margin);
        _elementCount += entry.path.elementCount();
    }
    dropCache();
    update();
}

void IntersectionLayerItem::setMinCachedElements(int elements)
{
    _minCachedElements = std::max(0, elements);
    dropCache();
    update();
}

QRectF IntersectionLayerItem::boundingRect() const
{
    return _bounds;
}

QPainterPath IntersectionLayerItem::shape() const
{
    return {};
}

void IntersectionLayerItem::paintEntries(QPainter* painter) const
{
    for (const auto& entry : _entries) {
        if (entry.path.isEmpty()) {
            continue;
        }
        painter->setPen(entry.pen);
        painter->setBrush(entry.brush);
        painter->drawPath(entry.path);
    }
}

void IntersectionLayerItem::dropCache()
{
    _cache = QImage();
    _cacheTransform = QTransform();
    _cacheSize = QSize();
    _cacheDpr = 0.0;
}

void IntersectionLayerItem::paint(QPainter* painter,
                                  const QStyleOptionGraphicsItem* /*option*/,
                                  QWidget* widget)
{
    if (_entries.empty()) {
        return;
    }

    // Without a target widget (QGraphicsScene::render into an image, printing)
    // there is no stable viewport to cache against.
    if (!widget || _elementCount < _minCachedElements ||
        painter->opacity() < 1.0) {
        dropCache();
        paintEntries(painter);
        return;
    }

    const QTransform transform = painter->worldTransform();
    const QSize size = widget->size();
    const qreal dpr = widget->devicePixelRatioF();
    const QPainter::RenderHints hints = painter->renderHints();
    if (size.isEmpty()) {
        return;
    }

    if (_cache.isNull() || transform != _cacheTransform || size != _cacheSize ||
        dpr != _cacheDpr || hints != _cacheHints) {
        const QSize pixels(int(std::ceil(size.width() * dpr)),
                           int(std::ceil(size.height() * dpr)));
        _cache = QImage(pixels, QImage::Format_ARGB32_Premultiplied);
        if (_cache.isNull()) {
            paintEntries(painter);
            return;
        }
        _cache.setDevicePixelRatio(dpr);
        _cache.fill(Qt::transparent);
        QPainter imagePainter(&_cache);
        imagePainter.setRenderHints(hints);
        imagePainter.setWorldTransform(transform);
        paintEntries(&imagePainter);
        imagePainter.end();

        _cacheTransform = transform;
        _cacheSize = size;
        _cacheDpr = dpr;
        _cacheHints = hints;
        ++_rasterizationCount;
    }

    // The cache is in widget coordinates; any redirection offset (backing
    // store, QWidget::grab) lives below the world transform and still applies.
    painter->save();
    painter->setWorldTransform(QTransform());
    painter->drawImage(QPointF(0.0, 0.0), _cache);
    painter->restore();
}
