#include "SurfaceTreeWidget.hpp"

#include <QObject>
#include <QApplication>

SurfaceTreeWidgetItem* SurfaceTreeWidget::findItemForSurface(std::string id)
{
    QTreeWidgetItemIterator it(this);
    while (*it) {
        if (id == (*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString()) {
            return static_cast<SurfaceTreeWidgetItem*>(*it);
        }

        ++it;
    }

    return nullptr;
}

void SurfaceTreeWidgetItem::updateItemIcon(bool approved, bool defective)
{
    if (approved) {        
        setData(0, Qt::UserRole, "1");
        setIcon(0, qApp->style()->standardIcon(QStyle::SP_DialogOkButton));
        setToolTip(0, QObject::tr("Approved"));
    } else if (defective) {
        setData(0, Qt::UserRole, "2");
        setIcon(0, qApp->style()->standardIcon(QStyle::SP_MessageBoxWarning));
        setToolTip(0, QObject::tr("Defective"));
    } else {            
        setData(0, Qt::UserRole, "3");
        setIcon(0, QIcon());
        setToolTip(0, QObject::tr("Unknown"));
    }
}