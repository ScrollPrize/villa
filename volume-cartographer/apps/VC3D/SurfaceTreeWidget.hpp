#pragma once

#include <QTreeWidget>

#define SURFACE_ID_COLUMN 1
#define TIMESTAMP_COLUMN 5  // New column for timestamps

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
        if (column == 0) {
            return data(column, Qt::UserRole).toString() < other.data(column, Qt::UserRole).toString();
        }
        // Column 1 = Surface ID (case-insensitive string comparison)
        else if (column == SURFACE_ID_COLUMN) {
            return text(column).toLower() < other.text(column).toLower();
        }
        // Column 5 = Timestamp (string comparison works due to YYYYMMDDHHMMSSmmm format)
        else if (column == TIMESTAMP_COLUMN) {
            // Empty timestamps sort to the bottom
            QString thisTime = text(column);
            QString otherTime = other.text(column);

            if (thisTime.isEmpty() && otherTime.isEmpty()) return false;
            if (thisTime.isEmpty()) return false;  // Empty sorts after non-empty
            if (otherTime.isEmpty()) return true;   // Non-empty sorts before empty

            return thisTime < otherTime;
        }
        // All other columns treated as numbers
        else {
            return text(column).toDouble() < other.text(column).toDouble();
        }
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