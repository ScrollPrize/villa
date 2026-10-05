#include "SpiralBrushCursorWidget.hpp"
#include <QPainter>
#include <QRegion>

namespace {
constexpr qreal kPointPlacementDotRadius = 3.0;
constexpr qreal kBrushRingPenWidth = 1.5;
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
    if (_position == position && _diameter == diameter
        && _brushDiameterVisible == brushDiameterVisible
        && _pointPlacementVisible == pointPlacementVisible
        && _pointPlacementColor == pointPlacementColor) return;
    const QRect previous = cueBounds();
    _position = position;
    _diameter = diameter;
    _brushDiameterVisible = brushDiameterVisible;
    _pointPlacementVisible = pointPlacementVisible;
    _pointPlacementColor = pointPlacementColor;
    repaintCues(previous);
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
    const QRect previous = cueBounds();
    _editablePclHoverPosition = position;
    _editablePclHoverColor = color;
    _editablePclHoverSourceMarker = sourceMarker;
    _editablePclHoverRadiusX = radiusX;
    _editablePclHoverRadiusY = radiusY;
    _editablePclHoverPenWidth = penWidth;
    repaintCues(previous);
}

QRect SpiralBrushCursorWidget::cueBounds() const
{
    QRectF bounds;
    const auto include = [&bounds](const QPointF& center, qreal radiusX,
                                   qreal radiusY, qreal penWidth) {
        // Half the pen plus slack for antialiased edge coverage.
        const qreal margin = penWidth * 0.5 + 2.0;
        const QRectF rect(center.x() - radiusX - margin,
                          center.y() - radiusY - margin,
                          2.0 * (radiusX + margin), 2.0 * (radiusY + margin));
        bounds = bounds.isNull() ? rect : bounds.united(rect);
    };
    // Mirrors paintEvent: the hover marker always draws; the placement dot
    // takes precedence over the brush ring.
    if (_editablePclHoverPosition) {
        include(*_editablePclHoverPosition, _editablePclHoverRadiusX,
                _editablePclHoverRadiusY, _editablePclHoverPenWidth);
    }
    if (_pointPlacementVisible) {
        include(_position, kPointPlacementDotRadius, kPointPlacementDotRadius, 0.0);
    } else if (_brushDiameterVisible) {
        include(_position, _diameter * 0.5, _diameter * 0.5, kBrushRingPenWidth);
    }
    return bounds.isNull() ? QRect{} : bounds.toAlignedRect();
}

void SpiralBrushCursorWidget::repaintCues(const QRect& previous)
{
    // This widget covers the whole viewport of a QGraphicsView, and a
    // translucent child repaint forces the view to redraw the scene beneath
    // the dirty region. A plane view with surface intersections shown holds
    // thousands of items, so a full-widget update() on every mouse move
    // froze the UI. Only touch the pixels the cues occupied and now occupy,
    // and nothing at all when neither state draws anything.
    QRegion region;
    if (!previous.isEmpty()) region += previous;
    const QRect current = cueBounds();
    if (!current.isEmpty()) region += current;
    if (!region.isEmpty()) update(region);
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
    pen.setWidthF(kBrushRingPenWidth);
    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);
    const qreal radius = _diameter * 0.5;
    painter.drawEllipse(_position, radius, radius);
}
