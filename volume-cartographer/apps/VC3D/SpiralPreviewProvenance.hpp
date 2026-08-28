#pragma once

#include <QJsonObject>
#include <QString>
#include <QtGlobal>

namespace vc3d {

struct SpiralPreviewProvenance {
    qint64 sourceIteration = -1;
    QString initializationMode;
};

inline SpiralPreviewProvenance spiralPreviewProvenance(
    const QJsonObject& manifest)
{
    return {
        manifest.value(QStringLiteral("source_fit_iteration")).toInteger(-1),
        manifest.value(QStringLiteral("initialization_mode")).toString(),
    };
}

} // namespace vc3d
