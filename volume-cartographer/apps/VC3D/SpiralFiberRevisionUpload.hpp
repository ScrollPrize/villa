#pragma once

#include <QJsonObject>
#include <QString>

namespace vc3d {

inline QString spiralFiberConflictRevision(const QJsonObject& failure)
{
    return failure.value(QStringLiteral("current_revision")).toString();
}

inline bool spiralFiberUploadNeedsCasRetry(const QString& revision,
                                           const QString& error)
{
    return !error.isEmpty() && !revision.isEmpty();
}

}  // namespace vc3d
