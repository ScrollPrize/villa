#include "WindowRangeWidget.hpp"

#include "elements/RangeSlider.hpp"

#include <QHBoxLayout>
#include <QSignalBlocker>
#include <QSpinBox>

#include <algorithm>

WindowRangeWidget::WindowRangeWidget(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);

    _slider = new RangeSlider(Qt::Horizontal, this);
    {
        QSignalBlocker blocker(_slider);
        _slider->setRange(_minimum, _maximum);
    }
    _slider->setMinimumSeparation(_minimumSeparation);
    _slider->setValues(_minimum, _maximum);

    connect(_slider, &RangeSlider::valuesChanged, this, [this](int low, int high) {
        if (_suppressSignals) {
            return;
        }
        syncControls(low, high, true);
    });

    _lowSpin = new QSpinBox(this);
    _lowSpin->setRange(_minimum, _maximum);
    _lowSpin->setValue(_minimum);
    _lowSpin->setSuffix(QStringLiteral(" L"));

    _highSpin = new QSpinBox(this);
    _highSpin->setRange(_minimum, _maximum);
    _highSpin->setValue(_maximum);
    _highSpin->setSuffix(QStringLiteral(" H"));

    connect(_lowSpin, qOverload<int>(&QSpinBox::valueChanged), this, [this](int value) {
        if (_suppressSignals) {
            return;
        }
        syncControls(value, _slider->highValue(), true);
    });

    connect(_highSpin, qOverload<int>(&QSpinBox::valueChanged), this, [this](int value) {
        if (_suppressSignals) {
            return;
        }
        syncControls(_slider->lowValue(), value, true);
    });

    layout->addWidget(_slider, /*stretch=*/1);
    layout->addWidget(_lowSpin);
    layout->addWidget(_highSpin);
}

void WindowRangeWidget::setRange(int minimum, int maximum)
{
    if (minimum > maximum) {
        std::swap(minimum, maximum);
    }

    if (_minimum == minimum && _maximum == maximum) {
        return;
    }

    _minimum = minimum;
    _maximum = maximum;

    {
        QSignalBlocker blocker(_slider);
        _slider->setRange(_minimum, _maximum);
    }
    _lowSpin->setRange(_minimum, _maximum);
    _highSpin->setRange(_minimum, _maximum);

    syncControls(_slider->lowValue(), _slider->highValue(), false);
}

void WindowRangeWidget::setWindowValues(int low, int high)
{
    syncControls(low, high, false);
}

void WindowRangeWidget::setControlsEnabled(bool enabled)
{
    setEnabled(enabled);
    if (_slider) {
        _slider->setEnabled(enabled);
    }
    if (_lowSpin) {
        _lowSpin->setEnabled(enabled);
    }
    if (_highSpin) {
        _highSpin->setEnabled(enabled);
    }
}

void WindowRangeWidget::setMinimumSeparation(int separation)
{
    _minimumSeparation = std::max(0, separation);
    if (_slider) {
        QSignalBlocker blocker(_slider);
        _slider->setMinimumSeparation(_minimumSeparation);
    }
    syncControls(_slider->lowValue(), _slider->highValue(), false);
}

void WindowRangeWidget::syncControls(int low, int high, bool emitSignal)
{
    if (!_slider || !_lowSpin || !_highSpin) {
        return;
    }

    const int minGap = std::max(_minimumSeparation, 0);
    int clampedLow = std::clamp(low, _minimum, _maximum - minGap);
    int clampedHigh = std::clamp(high, _minimum + minGap, _maximum);
    if (clampedHigh - clampedLow < minGap) {
        if (emitSignal && clampedLow != low) {
            clampedHigh = std::min(_maximum, clampedLow + minGap);
        } else {
            clampedLow = std::max(_minimum, clampedHigh - minGap);
        }
    }

    _suppressSignals = true;
    {
        QSignalBlocker sliderBlocker(_slider);
        _slider->setValues(clampedLow, clampedHigh);
    }
    {
        QSignalBlocker lowBlocker(_lowSpin);
        _lowSpin->setValue(clampedLow);
    }
    {
        QSignalBlocker highBlocker(_highSpin);
        _highSpin->setValue(clampedHigh);
    }
    _suppressSignals = false;

    if (emitSignal) {
        emit windowValuesChanged(clampedLow, clampedHigh);
    }
}
