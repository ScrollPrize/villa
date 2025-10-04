#include "elements/COutlinedTextItem.hpp"
#include <QPainterPath>
#include <QPen>



COutlinedTextItem::COutlinedTextItem(QGraphicsItem *parent)
    : QGraphicsTextItem(parent)
{
}

void COutlinedTextItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    // To prevent the base class from drawing the text, we handle all painting manually.
    Q_UNUSED(option);
    Q_UNUSED(widget);

    painter->save();

    // 1. Create a path from the text
    QPainterPath path;
    path.addText(0, 0, font(), toPlainText());

    // 2. Draw the outline
    QPen pen(Qt::black);
    pen.setWidth(10);
    pen.setCosmetic(true); // Ensures the outline is always 2 pixels regardless of zoom
    painter->setPen(pen);
    painter->setBrush(Qt::NoBrush);
    painter->drawPath(path);

    // 3. Draw the fill
    painter->setPen(Qt::NoPen);
    painter->setBrush(defaultTextColor());
    painter->drawPath(path);

    painter->restore();
}

QRectF COutlinedTextItem::boundingRect() const
{
    return QGraphicsTextItem::boundingRect().adjusted(-5,-5,5,5);
}

