#pragma once

#include <QTreeWidget>

#define SURFACE_ID_COLUMN 1

class SurfaceTreeWidgetItem : public QTreeWidgetItem
{
public:
    SurfaceTreeWidgetItem(QTreeWidget* parent) : QTreeWidgetItem(parent) {}

    void updateItemIcon(bool approved, bool defective);

private:
    bool operator<(const QTreeWidgetItem& other) const
    {
        int column = treeWidget()->sortColumn();
        // Column 0 = icon (sort entries without one at the bottom)
        if (column == 0)
            return data(column, Qt::UserRole).toString() < other.data(column, Qt::UserRole).toString();
        else if (column == SURFACE_ID_COLUMN)
            return text(column).toLower() < other.text(column).toLower();
        else
            return text(column).toDouble() < other.text(column).toDouble();
    }
};

class SurfaceTreeWidget : public QTreeWidget
{
    Q_OBJECT

public:
    SurfaceTreeWidget(QWidget* parent = nullptr) : QTreeWidget(parent) {
        setContextMenuPolicy(Qt::CustomContextMenu);
    }
    
    SurfaceTreeWidgetItem* findItemForSurface(std::string id);
};