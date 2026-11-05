#pragma once

#include <QLayout>
#include <QLayoutItem>
#include <QWidget>

// Gives a layout item's widgets, nested layouts included, a new parent.
inline void reparentItemWidgets(QLayoutItem* item, QWidget* newParent)
{
    if (!item || !newParent) {
        return;
    }
    if (auto* widget = item->widget()) {
        widget->setParent(newParent);
        return;
    }
    if (auto* layout = item->layout()) {
        for (int i = 0; i < layout->count(); ++i) {
            reparentItemWidgets(layout->itemAt(i), newParent);
        }
    }
}
