#pragma once

#include <QJsonObject>
#include <QString>

namespace vc3d::spiral {

inline bool inputVisible(const QJsonObject& row, const QString& label,
                         const QString& search, bool showOriginal)
{
    const bool changed = row.value(QStringLiteral("session_changed")).toBool()
        || row.value(QStringLiteral("dirty")).toBool()
        || !row.value(QStringLiteral("error")).toString().isEmpty();
    return (showOriginal || changed) && label.contains(search, Qt::CaseInsensitive);
}

} // namespace vc3d::spiral
