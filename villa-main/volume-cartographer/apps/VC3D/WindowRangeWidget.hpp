#pragma once

#include <QWidget>

class QSpinBox;
class RangeSlider;

class WindowRangeWidget : public QWidget
{
    Q_OBJECT

public:
    explicit WindowRangeWidget(QWidget* parent = nullptr);

    void setRange(int minimum, int maximum);
    void setWindowValues(int low, int high);
    void setControlsEnabled(bool enabled);
    void setMinimumSeparation(int separation);

signals:
    void windowValuesChanged(int low, int high);

private:
    void syncControls(int low, int high, bool emitSignal);

    RangeSlider* _slider{nullptr};
    QSpinBox* _lowSpin{nullptr};
    QSpinBox* _highSpin{nullptr};
    int _minimum{0};
    int _maximum{255};
    int _minimumSeparation{1};
    bool _suppressSignals{false};
};
