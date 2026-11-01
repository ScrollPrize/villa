#include "SurfaceTreeWidget.hpp"

#include <QObject>
#include <QApplication>

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