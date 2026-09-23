#pragma once

#include <QBrush>
#include <QGraphicsItem>
#include <QImage>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QTransform>

#include <vector>

// Scene item that draws a group of stroked/filled paths sharing one z value
// (the plane-viewer surface intersection lines).
//
// The viewers use QGraphicsView::FullViewportUpdate, so every crosshair move,
// chunk-ready frame or status refresh repaints every item. Re-stroking a few
// hundred thousand intersection segments per repaint costs 50-250 ms on the UI
// thread. Once the group is large enough, this item rasterises it into one
// viewport-sized image and re-blits that image while the painter's world
// transform, the viewport size and the paths are unchanged. The image is
// rasterised with the same transform and render hints as a direct draw, so the
// displayed pixels match drawing the paths directly.
//
// shape() is empty: these lines never take mouse input, and a stroked shape of
// a huge path is expensive for QGraphicsView::itemAt().
class IntersectionLayerItem final : public QGraphicsItem
{
public:
    struct Entry {
        QPainterPath path;
        QPen pen;
        QBrush brush = Qt::NoBrush;
    };

    // Path groups smaller than this (in QPainterPath elements) are drawn
    // directly; caching them would cost a viewport-sized image for no gain.
    static constexpr int kDefaultMinCachedElements = 4096;

    explicit IntersectionLayerItem(QGraphicsItem* parent = nullptr);

    // Replaces the paths, drawn in order. Invalidates the cached raster.
    void setEntries(std::vector<Entry> entries);
    const std::vector<Entry>& entries() const { return _entries; }

    void setMinCachedElements(int elements);

    QRectF boundingRect() const override;
    QPainterPath shape() const override;
    void paint(QPainter* painter,
               const QStyleOptionGraphicsItem* option,
               QWidget* widget) override;

    // Number of times the paths were stroked into the cache image.
    int rasterizationCount() const { return _rasterizationCount; }
    bool hasCachedRaster() const { return !_cache.isNull(); }

private:
    void paintEntries(QPainter* painter) const;
    void dropCache();

    std::vector<Entry> _entries;
    QRectF _bounds;
    qsizetype _elementCount = 0;
    int _minCachedElements = kDefaultMinCachedElements;

    QImage _cache;
    QTransform _cacheTransform;
    QSize _cacheSize;
    qreal _cacheDpr = 0.0;
    QPainter::RenderHints _cacheHints;
    int _rasterizationCount = 0;
};
