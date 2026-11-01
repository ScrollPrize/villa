#include "FlowLayout.hpp"

#include <QWidget>
#include <QWidgetItem>

#include <algorithm>

FlowLayout::FlowLayout(QWidget* parent, int hSpacing, int vSpacing)
    : QLayout(parent), _hSpace(hSpacing), _vSpace(vSpacing)
{
}

FlowLayout::~FlowLayout()
{
    while (QLayoutItem* item = takeAt(0)) {
        delete item;
    }
}

void FlowLayout::addItem(QLayoutItem* item)
{
    _items.append(item);
}

void FlowLayout::insertWidget(int index, QWidget* widget)
{
    if (!widget) {
        return;
    }
    addChildWidget(widget);
    index = std::clamp(index, 0, static_cast<int>(_items.size()));
    _items.insert(index, new QWidgetItem(widget));
    invalidate();
}

int FlowLayout::count() const
{
    return static_cast<int>(_items.size());
}

QLayoutItem* FlowLayout::itemAt(int index) const
{
    return _items.value(index, nullptr);
}

QLayoutItem* FlowLayout::takeAt(int index)
{
    if (index < 0 || index >= _items.size()) {
        return nullptr;
    }
    return _items.takeAt(index);
}

Qt::Orientations FlowLayout::expandingDirections() const
{
    return {};
}

bool FlowLayout::hasHeightForWidth() const
{
    return true;
}

int FlowLayout::heightForWidth(int width) const
{
    return doLayout(QRect(0, 0, width, 0), true);
}

QSize FlowLayout::minimumSize() const
{
    // Narrowest useful width: the widest single item (everything else wraps).
    QSize size;
    for (const QLayoutItem* item : _items) {
        if (!item->isEmpty()) {
            size = size.expandedTo(item->minimumSize().expandedTo(item->sizeHint()));
        }
    }
    const QMargins m = contentsMargins();
    return size + QSize(m.left() + m.right(), m.top() + m.bottom());
}

QSize FlowLayout::sizeHint() const
{
    // Preferred: everything on one row.
    int width = 0;
    int height = 0;
    int visible = 0;
    for (const QLayoutItem* item : _items) {
        if (item->isEmpty()) {
            continue;
        }
        const QSize hint = item->sizeHint();
        width += hint.width();
        height = std::max(height, hint.height());
        ++visible;
    }
    if (visible > 1) {
        width += _hSpace * (visible - 1);
    }
    const QMargins m = contentsMargins();
    return QSize(width + m.left() + m.right(), height + m.top() + m.bottom());
}

void FlowLayout::setGeometry(const QRect& rect)
{
    QLayout::setGeometry(rect);
    doLayout(rect, false);
}

int FlowLayout::doLayout(const QRect& rect, bool testOnly) const
{
    const QMargins m = contentsMargins();
    const QRect area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom());
    int x = area.x();
    int y = area.y();
    int lineHeight = 0;

    for (QLayoutItem* item : _items) {
        if (item->isEmpty()) {
            continue;
        }
        const QSize hint = item->sizeHint();
        if (x > area.x() && x + hint.width() > area.right() + 1) {
            x = area.x();
            y += lineHeight + _vSpace;
            lineHeight = 0;
        }
        if (!testOnly) {
            item->setGeometry(QRect(QPoint(x, y), hint));
        }
        x += hint.width() + _hSpace;
        lineHeight = std::max(lineHeight, hint.height());
    }
    return y + lineHeight - rect.y() + m.bottom();
}
