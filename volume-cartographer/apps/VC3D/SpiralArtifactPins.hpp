#pragma once

#include <QString>
#include <QStringList>

namespace vc3d {

// Cache entries the preview pruner must keep: the installed preview, its
// diagnostics, and the other artifacts the workspace still reads (the
// session's input manifest).
inline QStringList spiralArtifactCachePins(
    const QString& preview, const QString& diagnostics,
    const QStringList& otherArtifacts)
{
    QStringList pins{preview, diagnostics};
    pins += otherArtifacts;
    return pins;
}

} // namespace vc3d
