#pragma once

#include <QWidget>

class QHBoxLayout;
class QLabel;

class LabeledControlRow : public QWidget
{
    Q_OBJECT

public:
    explicit LabeledControlRow(const QString& labelText, QWidget* parent = nullptr);

    QLabel* label() const { return _label; }
    QHBoxLayout* rowLayout() const { return _layout; }

    void addControl(QWidget* widget, int stretch = 0);
    void addStretch(int stretch = 1);
    void setLabelToolTip(const QString& tooltip);

private:
    QLabel* _label{nullptr};
    QHBoxLayout* _layout{nullptr};
};
