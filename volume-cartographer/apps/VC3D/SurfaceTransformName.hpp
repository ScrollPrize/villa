#pragma once

#include <QRegularExpression>
#include <QString>

inline QString suggestedTransformedSurfaceName(QString sourceName)
{
    static const QRegularExpression unsupported(QStringLiteral("[^a-zA-Z0-9_-]+"));
    sourceName = sourceName.trimmed();
    sourceName.replace(unsupported, QStringLiteral("_"));
    if (sourceName.isEmpty()) sourceName = QStringLiteral("surface");
    return sourceName + QStringLiteral("_transformed");
}
