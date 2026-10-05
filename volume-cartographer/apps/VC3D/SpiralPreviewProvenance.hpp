#pragma once

#include <QJsonObject>
#include <QString>
#include <QtGlobal>

namespace vc3d {

struct SpiralPreviewProvenance {
    qint64 sourceIteration = -1;
};

inline SpiralPreviewProvenance spiralPreviewProvenance(
    const QJsonObject& manifest)
{
    return {
        manifest.value(QStringLiteral("source_fit_iteration")).toInteger(-1),
    };
}

} // namespace vc3d
