#pragma once

#include <QLabel>

#include <algorithm>

// A label whose size hints never shrink, for text that is rewritten many times
// a second (status-bar cache/network figures). Sizing to each new text makes
// the label and its status-bar neighbours jump around too fast to read.
class GrowOnlyLabel : public QLabel
{
public:
    using QLabel::QLabel;

    QSize sizeHint() const override { return grow(QLabel::sizeHint()); }
    QSize minimumSizeHint() const override { return grow(QLabel::minimumSizeHint()); }

private:
    QSize grow(QSize size) const
    {
        _maxWidth = std::max(_maxWidth, size.width());
        size.setWidth(_maxWidth);
        return size;
    }

    mutable int _maxWidth{0};
};
