#include "SpiralBrushCursorWidget.hpp"
#include <QPainter>

namespace {
constexpr qreal kPointPlacementDotRadius = 3.0;
}

SpiralBrushCursorWidget::SpiralBrushCursorWidget(QWidget* parent)
    : QWidget(parent)
{
    setAttribute(Qt::WA_TransparentForMouseEvents);
    setAttribute(Qt::WA_NoSystemBackground);
    setAttribute(Qt::WA_TranslucentBackground);
}

void SpiralBrushCursorWidget::setCursorState(
    const QPointF& position, int diameter, bool brushDiameterVisible,
    bool pointPlacementVisible, const QColor& pointPlacementColor)
{
    _position = position;
    _diameter = diameter;
    _brushDiameterVisible = brushDiameterVisible;
    _pointPlacementVisible = pointPlacementVisible;
    _pointPlacementColor = pointPlacementColor;
    update();
}

void SpiralBrushCursorWidget::setEditablePclHover(
    const std::optional<QPointF>& position, const QColor& color,
    bool sourceMarker, qreal radiusX, qreal radiusY, qreal penWidth)
{
    if (_editablePclHoverPosition == position
        && _editablePclHoverColor == color
        && _editablePclHoverSourceMarker == sourceMarker
        && _editablePclHoverRadiusX == radiusX
        && _editablePclHoverRadiusY == radiusY
        && _editablePclHoverPenWidth == penWidth) return;
    _editablePclHoverPosition = position;
    _editablePclHoverColor = color;
    _editablePclHoverSourceMarker = sourceMarker;
    _editablePclHoverRadiusX = radiusX;
    _editablePclHoverRadiusY = radiusY;
    _editablePclHoverPenWidth = penWidth;
    update();
}

void SpiralBrushCursorWidget::paintEvent(QPaintEvent*)
{
    if (!_brushDiameterVisible && !_pointPlacementVisible
        && !_editablePclHoverPosition) return;

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    if (_editablePclHoverPosition) {
        QPen hoverPen(_editablePclHoverSourceMarker
                          ? QColor(255, 255, 255, 200)
                          : _editablePclHoverColor);
        hoverPen.setWidthF(_editablePclHoverPenWidth);
        painter.setPen(hoverPen);
        painter.setBrush(_editablePclHoverColor);
        painter.drawEllipse(*_editablePclHoverPosition,
                            _editablePclHoverRadiusX,
                            _editablePclHoverRadiusY);
    }
    if (_pointPlacementVisible) {
        painter.setPen(Qt::NoPen);
        painter.setBrush(_pointPlacementColor);
        painter.drawEllipse(_position, kPointPlacementDotRadius,
                            kPointPlacementDotRadius);
        return;
    }

    if (!_brushDiameterVisible) return;

    QPen pen(QColor(255, 255, 255, 220));
    pen.setWidthF(1.5);
    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);
    const qreal radius = _diameter * 0.5;
    painter.drawEllipse(_position, radius, radius);
}
