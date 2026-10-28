#pragma once

#include <QJsonObject>
#include <QString>
#include <QtGlobal>

namespace vc3d {

struct SpiralPreviewProvenance {
    qint64 sourceIteration = -1;
    // The object-store inputs the previewed model was fitted to; empty for
    // previews published before they were recorded.
    QString inputDigest;
    // The hub position they were read at; -1 when not read at one.
    qint64 inputPosition = -1;
};

inline SpiralPreviewProvenance spiralPreviewProvenance(
    const QJsonObject& manifest)
{
    return {
        manifest.value(QStringLiteral("source_fit_iteration")).toInteger(-1),
        manifest.value(QStringLiteral("input_digest")).toString(),
        manifest.value(QStringLiteral("input_position")).toObject()
            .value(QStringLiteral("sequence")).toInteger(-1),
    };
}

// How a preview's inputs relate to the session's: empty when they are the
// same inputs (or either is unknown), otherwise a label for the preview,
// naming the hub position it was fitted at when known. Equal inputs read at
// two positions are the same inputs.
inline QString spiralPreviewInputNote(const SpiralPreviewProvenance& preview,
                                      const QString& sessionDigest)
{
    if (preview.inputDigest.isEmpty() || sessionDigest.isEmpty()
        || preview.inputDigest == sessionDigest)
        return {};
    if (preview.inputPosition >= 0)
        return QStringLiteral("fitted to the inputs of hub position %1 (%2, not %3)")
            .arg(preview.inputPosition).arg(preview.inputDigest.left(12), sessionDigest.left(12));
    return QStringLiteral("fitted to other inputs (%1, not %2)")
        .arg(preview.inputDigest.left(12), sessionDigest.left(12));
}

} // namespace vc3d
