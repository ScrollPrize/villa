#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <QBrush>
#include <QGraphicsObject>
#include <QPainterPath>
#include <QPen>
#include <QPointF>
#include <QRectF>

#include <vector>

// A single QGraphicsItem that paints many overlay primitives itself.
//
// The default primitive materialization creates one QGraphicsItem per style
// group and deletes the lot on every rebuild. Overlays that redraw on every
// pan/zoom tick and carry thousands of primitives cannot afford that: the
// scene's spatial index is rebuilt each time, and per-point styling (the
// distance fade gives nearly every point its own brush alpha) defeats the
// grouping, so "per style group" degenerates to "per point".
//
// This item is retained across refreshes and re-filled in place instead.
// Primitives keep their original order and their own pen and brush, so
// overlapping semi-transparent marks blend exactly as they did when each was
// its own item -- this is a batching change, not a rendering change.

// A line strip, already flattened to scene coordinates.
struct OverlayLineCommand {
    QPainterPath path;
    QPen pen;
    QBrush brush;
    QRectF bounds;
};

// One dot. Kept as a bare centre and radius rather than a QPainterPath:
// building a path per point per tick dominated the rebuild, and drawEllipse()
// renders identically to drawPath() of a single addEllipse().
struct OverlayPointCommand {
    QPointF center;
    qreal radius{0.0};
    QPen pen;
    QBrush brush;
    QRectF bounds;
};

QPen overlayPenForStyle(const ViewerOverlayControllerBase::OverlayStyle& style);

// Splits a primitive list into the two draw lists in one pass. Returns false
// if the list holds anything other than line strips and points, in which case
// the caller must fall back to the general materialization path -- this item
// deliberately does not reimplement text, image or path rendering.
bool buildOverlayBatchCommands(
    const std::vector<ViewerOverlayControllerBase::OverlayPrimitive>& primitives,
    std::vector<OverlayLineCommand>& lineCommands,
    std::vector<OverlayPointCommand>& pointCommands);

class OverlayBatchItem final : public QGraphicsObject
{
public:
    OverlayBatchItem();

    QRectF boundingRect() const override { return _bounds; }

    // One item carries one kind, so that lines and points can sit at their own
    // z values: each setter replaces the item's contents, dropping the other
    // kind. Callers that draw both use two items.
    void setLineCommands(std::vector<OverlayLineCommand> commands);
    void setPointCommands(std::vector<OverlayPointCommand> commands);

    void paint(QPainter* painter,
               const QStyleOptionGraphicsItem* option,
               QWidget* widget) override;

private:
    void recomputeBounds();

    std::vector<OverlayLineCommand> _lines;
    std::vector<OverlayPointCommand> _points;
    QRectF _bounds;
};
