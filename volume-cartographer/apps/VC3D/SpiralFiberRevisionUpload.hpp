#pragma once

#include <QFileInfo>
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

// The Spiral input id of a fiber JSON file. The fitter identifies dataset
// fibers by file stem and the service commits an uploaded fiber to
// paths.fibers/<id>.json, so the stem is the id under which a re-upload
// replaces the fiber instead of adding a second copy.
inline QString spiralFiberInputId(const QString& fiberJsonPath)
{
    return QFileInfo(fiberJsonPath).completeBaseName();
}

}  // namespace vc3d
