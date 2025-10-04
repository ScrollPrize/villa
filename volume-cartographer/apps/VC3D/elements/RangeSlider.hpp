#pragma once

#include <QColor>
#include <QWidget>

class RangeSlider : public QWidget
{
    Q_OBJECT

public:
    explicit RangeSlider(Qt::Orientation orientation = Qt::Horizontal, QWidget* parent = nullptr);

    void setRange(int minimum, int maximum);
    void setValues(int low, int high);
    void setLowValue(int value);
    void setHighValue(int value);
    void setHandleBorderColor(const QColor& color);

    int minimum() const { return _minimum; }
    int maximum() const { return _maximum; }
    int lowValue() const { return _lowValue; }
    int highValue() const { return _highValue; }
    QColor handleBorderColor() const { return _handleBorderColor; }

    void setMinimumSeparation(int separation);
    int minimumSeparation() const { return _minimumSeparation; }

signals:
    void lowValueChanged(int value);
    void highValueChanged(int value);
    void valuesChanged(int low, int high);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

private:
    enum class DragTarget {
        None,
        Low,
        High
    };

    void updateValues(int low, int high, bool emitSignals);
    int valueFromPosition(int position) const;
    int positionFromValue(int value) const;
    QRect handleRectForValue(int value) const;
    QRect trackRect() const;

    Qt::Orientation _orientation;
    int _minimum;
    int _maximum;
    int _lowValue;
    int _highValue;
    int _minimumSeparation;

    DragTarget _dragTarget;
    int _dragOffset{}; // pixel delta between cursor and handle center
    QColor _handleBorderColor;
};
