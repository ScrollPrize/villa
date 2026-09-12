#pragma once

#include <QString>
#include <QStringList>

namespace vc3d {

// Cache entries the preview pruner must keep: the installed preview, its
// diagnostics, and every display-only PCL artifact (one per editable role).
inline QStringList spiralArtifactCachePins(
    const QString& preview, const QString& diagnostics,
    const QStringList& pclArtifacts)
{
    QStringList pins{preview, diagnostics};
    pins += pclArtifacts;
    return pins;
}

} // namespace vc3d
