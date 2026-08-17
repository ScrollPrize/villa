#pragma once

#include <QJsonObject>

namespace vc3d {

inline QJsonObject effectiveSpiralSessionConfig(
    const QJsonObject& sessionRequest,
    const QJsonObject& defaultAdvancedConfig,
    const QJsonObject& activeRunConfig)
{
    QJsonObject effective = defaultAdvancedConfig;
    const QJsonObject run =
        sessionRequest.value(QStringLiteral("run")).toObject();
    const QJsonObject requested =
        run.value(QStringLiteral("config")).toObject();
    for (auto it = requested.begin(); it != requested.end(); ++it)
        effective.insert(it.key(), it.value());
    for (auto it = activeRunConfig.begin(); it != activeRunConfig.end(); ++it)
        effective.insert(it.key(), it.value());
    return effective;
}

} // namespace vc3d
