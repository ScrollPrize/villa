#pragma once

#include <QString>
#include <QStringList>

namespace vc3d {

inline QStringList spiralArtifactCachePins(
    const QString& preview, const QString& diagnostics,
    const QString& sameWinding)
{
    return {preview, diagnostics, sameWinding};
}

} // namespace vc3d
