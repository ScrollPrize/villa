#pragma once

#include <QColor>
#include <QPointF>
#include <QWidget>

#include <optional>

class QPaintEvent;

class SpiralBrushCursorWidget final : public QWidget
{
public:
    explicit SpiralBrushCursorWidget(QWidget* parent = nullptr);

    // `pointPlacementColor` is the accent of the active placement role; it is
    // only drawn while `pointPlacementVisible`.
    void setCursorState(const QPointF& position, int diameter,
                        bool brushDiameterVisible, bool pointPlacementVisible,
                        const QColor& pointPlacementColor = QColor(50, 255, 215));
    void setEditablePclHover(const std::optional<QPointF>& position,
                             const QColor& color, bool sourceMarker,
                             qreal radiusX, qreal radiusY, qreal penWidth);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QPointF _position;
    int _diameter = 32;
    bool _brushDiameterVisible = false;
    bool _pointPlacementVisible = false;
    QColor _pointPlacementColor{50, 255, 215};
    std::optional<QPointF> _editablePclHoverPosition;
    QColor _editablePclHoverColor;
    bool _editablePclHoverSourceMarker = false;
    qreal _editablePclHoverRadiusX = 0.0;
    qreal _editablePclHoverRadiusY = 0.0;
    qreal _editablePclHoverPenWidth = 0.0;
};
