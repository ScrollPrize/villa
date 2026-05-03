#include "elements/ViewerStatsBar.hpp"

ViewerStatsBar::ViewerStatsBar(QWidget* parent)
    : QLabel(parent)
{
    setStyleSheet("QLabel { color : #00FF00; background-color: rgba(0,0,0,128); padding: 2px 4px; }");
    setMinimumWidth(520);
}

void ViewerStatsBar::setItems(const QStringList& items)
{
    QStringList visibleItems;
    for (const auto& item : items) {
        if (!item.isEmpty())
            visibleItems.push_back(item);
    }

    setText(visibleItems.join("  "));
    adjustSize();
}
