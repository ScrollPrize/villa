#include "OverlayBatchItem.hpp"

#include <QPainter>
#include <QStyleOptionGraphicsItem>
#include <QVector>

#include <algorithm>

namespace
{

QRectF paddedBounds(const QRectF& rect, const QPen& pen)
{
    const qreal padding = std::max<qreal>(0.5, pen.widthF() * 0.5) + 1.0;
    return rect.adjusted(-padding, -padding, padding, padding);
}

} // namespace

QPen overlayPenForStyle(const ViewerOverlayControllerBase::OverlayStyle& style)
{
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
    return pen;
}

bool buildOverlayBatchCommands(
    const std::vector<ViewerOverlayControllerBase::OverlayPrimitive>& primitives,
    std::vector<OverlayLineCommand>& lineCommands,
    std::vector<OverlayPointCommand>& pointCommands)
{
    lineCommands.clear();
    pointCommands.clear();
    lineCommands.reserve(primitives.size());
    pointCommands.reserve(primitives.size());

    for (const auto& primitive : primitives) {
        if (const auto* line =
                std::get_if<ViewerOverlayControllerBase::LineStripPrimitive>(&primitive)) {
            if (line->points.size() < 2) {
                continue;
            }
            OverlayLineCommand command;
            command.path = QPainterPath(line->points.front());
            for (std::size_t i = 1; i < line->points.size(); ++i) {
                command.path.lineTo(line->points[i]);
            }
            if (line->closed) {
                command.path.closeSubpath();
            }
            if (command.path.isEmpty()) {
                continue;
            }
            command.pen = overlayPenForStyle(line->style);
            command.brush = QBrush(line->style.brushColor);
            command.bounds = paddedBounds(command.path.boundingRect(), command.pen);
            lineCommands.push_back(std::move(command));
        } else if (const auto* point =
                       std::get_if<ViewerOverlayControllerBase::PointPrimitive>(&primitive)) {
            if (point->radius <= 0.0) {
                continue;
            }
            OverlayPointCommand command;
            command.center = point->position;
            command.radius = point->radius;
            command.pen = overlayPenForStyle(point->style);
            command.brush = QBrush(point->style.brushColor);
            command.bounds = paddedBounds(
                QRectF(command.center.x() - command.radius,
                       command.center.y() - command.radius,
                       command.radius * 2.0, command.radius * 2.0),
                command.pen);
            pointCommands.push_back(std::move(command));
        } else {
            return false;
        }
    }
    return true;
}

OverlayBatchItem::OverlayBatchItem()
{
    setAcceptedMouseButtons(Qt::NoButton);
    setFlag(QGraphicsItem::ItemUsesExtendedStyleOption, true);
}

void OverlayBatchItem::setLineCommands(std::vector<OverlayLineCommand> commands)
{
    prepareGeometryChange();
    _lines = std::move(commands);
    _points.clear();
    recomputeBounds();
    update();
}

void OverlayBatchItem::setPointCommands(std::vector<OverlayPointCommand> commands)
{
    prepareGeometryChange();
    _points = std::move(commands);
    _lines.clear();
    recomputeBounds();
    update();
}

void OverlayBatchItem::paint(QPainter* painter,
                             const QStyleOptionGraphicsItem* option,
                             QWidget* /*widget*/)
{
    if (!painter) {
        return;
    }
    painter->save();
    const QRectF exposed = option ? option->exposedRect : _bounds;
    const bool cull = !exposed.isEmpty();
    for (const OverlayLineCommand& command : _lines) {
        if (cull && !command.bounds.intersects(exposed)) {
            continue;
        }
        painter->setPen(command.pen);
        painter->setBrush(command.brush);
        painter->drawPath(command.path);
    }
    for (const OverlayPointCommand& command : _points) {
        if (cull && !command.bounds.intersects(exposed)) {
            continue;
        }
        painter->setPen(command.pen);
        painter->setBrush(command.brush);
        painter->drawEllipse(command.center, command.radius, command.radius);
    }
    painter->restore();
}

void OverlayBatchItem::recomputeBounds()
{
    QRectF bounds;
    bool haveBounds = false;
    auto unite = [&](const QRectF& rect) {
        if (!haveBounds) {
            bounds = rect;
            haveBounds = true;
        } else {
            bounds = bounds.united(rect);
        }
    };
    for (const OverlayLineCommand& command : _lines) unite(command.bounds);
    for (const OverlayPointCommand& command : _points) unite(command.bounds);
    _bounds = haveBounds ? bounds : QRectF{};
}
