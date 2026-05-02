#include "elements/LabeledControlRow.hpp"

#include <QHBoxLayout>
#include <QLabel>

LabeledControlRow::LabeledControlRow(const QString& labelText, QWidget* parent)
    : QWidget(parent)
    , _label(new QLabel(labelText, this))
    , _layout(new QHBoxLayout(this))
{
    _layout->setContentsMargins(2, 2, 2, 2);
    _layout->setSpacing(8);
    _layout->addWidget(_label);
}

void LabeledControlRow::addControl(QWidget* widget, int stretch)
{
    if (!widget) {
        return;
    }
    _layout->addWidget(widget, stretch);
}

void LabeledControlRow::addStretch(int stretch)
{
    _layout->addStretch(stretch);
}

void LabeledControlRow::setLabelToolTip(const QString& tooltip)
{
    if (_label) {
        _label->setToolTip(tooltip);
    }
}
