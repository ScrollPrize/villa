#include "RangeSlider.hpp"

#include "../Keybinds.hpp"

#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QStyle>
#include <QStyleOption>

#include <algorithm>
#include <cmath>

namespace
{
constexpr int kHandleRadius = 6;
constexpr int kTrackThickness = 4;
constexpr int kHorizontalPadding = kHandleRadius;
constexpr int kHandleBorderWidth = 2;
}

RangeSlider::RangeSlider(Qt::Orientation orientation, QWidget* parent)
    : QWidget(parent)
    , _orientation(orientation)
    , _minimum(0)
    , _maximum(100)
    , _lowValue(0)
    , _highValue(100)
    , _minimumSeparation(1)
    , _dragTarget(DragTarget::None)
    , _handleBorderColor(palette().color(QPalette::Highlight))
{
    setFocusPolicy(Qt::StrongFocus);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setCursor(Qt::PointingHandCursor);
}

void RangeSlider::setRange(int minimum, int maximum)
{
    if (minimum > maximum) {
        std::swap(minimum, maximum);
    }

    if (_minimum == minimum && _maximum == maximum) {
        return;
    }

    _minimum = minimum;
    _maximum = maximum;

    const int span = std::max(_minimumSeparation, 1);
    const int clampedLow = std::clamp(_lowValue, _minimum, _maximum - span);
    const int clampedHigh = std::clamp(_highValue, _minimum + span, _maximum);
    updateValues(clampedLow, clampedHigh, true);
}

void RangeSlider::setValues(int low, int high)
{
    updateValues(low, high, true);
}

void RangeSlider::setLowValue(int value)
{
    updateValues(value, _highValue, true);
}

void RangeSlider::setHighValue(int value)
{
    updateValues(_lowValue, value, true);
}

void RangeSlider::setMinimumSeparation(int separation)
{
    _minimumSeparation = std::max(0, separation);
    updateValues(_lowValue, _highValue, true);
}

void RangeSlider::setHandleBorderColor(const QColor& color)
{
    if (_handleBorderColor == color) {
        return;
    }
    _handleBorderColor = color;
    update();
}

void RangeSlider::updateValues(int low, int high, bool emitSignals)
{
    if (_minimum >= _maximum) {
        return;
    }

    const int span = std::max(_minimumSeparation, 0);
    int clampedLow = std::clamp(low, _minimum, _maximum);
    int clampedHigh = std::clamp(high, _minimum, _maximum);

    if (clampedHigh - clampedLow < span) {
        if (_dragTarget == DragTarget::High) {
            clampedLow = std::clamp(clampedHigh - span, _minimum, _maximum - span);
        } else {
            clampedHigh = std::clamp(clampedLow + span, _minimum + span, _maximum);
        }
    }

    clampedLow = std::clamp(clampedLow, _minimum, _maximum - span);
    clampedHigh = std::clamp(clampedHigh, _minimum + span, _maximum);

    const bool changed = clampedLow != _lowValue || clampedHigh != _highValue;
    if (!changed) {
        return;
    }

    _lowValue = clampedLow;
    _highValue = clampedHigh;

    if (_lowValue > _highValue) {
        std::swap(_lowValue, _highValue);
    }

    update();

    if (emitSignals) {
        emit lowValueChanged(_lowValue);
        emit highValueChanged(_highValue);
        emit valuesChanged(_lowValue, _highValue);
    }
}

QSize RangeSlider::sizeHint() const
{
    if (_orientation == Qt::Horizontal) {
        return {200, 24};
    }
    return {24, 200};
}

QSize RangeSlider::minimumSizeHint() const
{
    if (_orientation == Qt::Horizontal) {
        return {2 * (kHorizontalPadding + kHandleRadius), 18};
    }
    return {18, 2 * (kHorizontalPadding + kHandleRadius)};
}

void RangeSlider::paintEvent(QPaintEvent* event)
{
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    const QRect track = trackRect();

    painter.setPen(Qt::NoPen);
    painter.setBrush(palette().color(QPalette::Mid));
    painter.drawRoundedRect(track, kTrackThickness / 2.0, kTrackThickness / 2.0);

    const int lowPos = positionFromValue(_lowValue);
    const int highPos = positionFromValue(_highValue);
    QRect selectedRect(track);
    selectedRect.setLeft(lowPos);
    selectedRect.setRight(highPos);

    painter.setBrush(palette().color(QPalette::Highlight));
    painter.drawRoundedRect(selectedRect, kTrackThickness / 2.0, kTrackThickness / 2.0);

    const QRect lowHandle = handleRectForValue(_lowValue);
    const QRect highHandle = handleRectForValue(_highValue);

    painter.setBrush(palette().color(QPalette::Base));
    QPen handlePen(_handleBorderColor);
    handlePen.setWidth(kHandleBorderWidth);
    painter.setPen(handlePen);
    painter.drawEllipse(lowHandle);
    painter.drawEllipse(highHandle);
}

void RangeSlider::mousePressEvent(QMouseEvent* event)
{
    if (event->button() != Qt::LeftButton) {
        QWidget::mousePressEvent(event);
        return;
    }

    const QPoint pos = event->position().toPoint();
    const QRect lowHandle = handleRectForValue(_lowValue);
    const QRect highHandle = handleRectForValue(_highValue);

    if (lowHandle.contains(pos)) {
        _dragTarget = DragTarget::Low;
        _dragOffset = pos.x() - lowHandle.center().x();
    } else if (highHandle.contains(pos)) {
        _dragTarget = DragTarget::High;
        _dragOffset = pos.x() - highHandle.center().x();
    } else {
        const int lowDistance = std::abs(pos.x() - lowHandle.center().x());
        const int highDistance = std::abs(pos.x() - highHandle.center().x());
        if (lowDistance <= highDistance) {
            _dragTarget = DragTarget::Low;
            _dragOffset = 0;
            const int value = valueFromPosition(pos.x());
            updateValues(value, _highValue, true);
        } else {
            _dragTarget = DragTarget::High;
            _dragOffset = 0;
            const int value = valueFromPosition(pos.x());
            updateValues(_lowValue, value, true);
        }
    }

    setFocus(Qt::MouseFocusReason);
    update();
}

void RangeSlider::mouseMoveEvent(QMouseEvent* event)
{
    if (_dragTarget == DragTarget::None) {
        QWidget::mouseMoveEvent(event);
        return;
    }

    const QPoint pos = event->position().toPoint();
    const int value = valueFromPosition(pos.x() - _dragOffset);

    if (_dragTarget == DragTarget::Low) {
        updateValues(value, _highValue, true);
    } else if (_dragTarget == DragTarget::High) {
        updateValues(_lowValue, value, true);
    }
}

void RangeSlider::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton && _dragTarget != DragTarget::None) {
        _dragTarget = DragTarget::None;
        _dragOffset = 0;
        update();
    }
    QWidget::mouseReleaseEvent(event);
}

void RangeSlider::keyPressEvent(QKeyEvent* event)
{
    int step = 0;
    const int key = event->key();
    if (key == vc3d::keybinds::range_slider::StepDownLeft.key ||
        key == vc3d::keybinds::range_slider::StepDownDown.key) {
        step = -1;
    } else if (key == vc3d::keybinds::range_slider::StepUpRight.key ||
               key == vc3d::keybinds::range_slider::StepUpUp.key) {
        step = 1;
    } else if (key == vc3d::keybinds::range_slider::PageDown.key) {
        step = -5;
    } else if (key == vc3d::keybinds::range_slider::PageUp.key) {
        step = 5;
    } else {
        QWidget::keyPressEvent(event);
        return;
    }

    if (event->modifiers() & Qt::ShiftModifier) {
        step *= 10;
    }

    updateValues(_lowValue + step, _highValue + step, true);
}

int RangeSlider::valueFromPosition(int position) const
{
    const QRect track = trackRect();
    if (track.width() <= 0) {
        return _minimum;
    }

    const int clamped = std::clamp(position, track.left(), track.right());
    const double ratio = static_cast<double>(clamped - track.left()) / track.width();
    const int value = static_cast<int>(std::round(ratio * (_maximum - _minimum))) + _minimum;
    return std::clamp(value, _minimum, _maximum);
}

int RangeSlider::positionFromValue(int value) const
{
    const QRect track = trackRect();
    if (track.width() <= 0) {
        return track.left();
    }
    const double ratio = static_cast<double>(value - _minimum) / (_maximum - _minimum);
    return track.left() + static_cast<int>(std::round(ratio * track.width()));
}

QRect RangeSlider::handleRectForValue(int value) const
{
    const int centerX = positionFromValue(value);
    const int centerY = height() / 2;
    return QRect(centerX - kHandleRadius, centerY - kHandleRadius, kHandleRadius * 2, kHandleRadius * 2);
}

QRect RangeSlider::trackRect() const
{
    const int left = kHorizontalPadding;
    const int right = width() - kHorizontalPadding;
    const int centerY = height() / 2;
    return QRect(left, centerY - kTrackThickness / 2, right - left, kTrackThickness);
}
