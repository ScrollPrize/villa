#pragma once

#include <QJsonObject>
#include <QSet>
#include <QString>

namespace vc3d {

inline QJsonObject normalizedSpiralReloadRequest(
    QJsonObject request,
    const QJsonObject& defaultAdvancedConfig,
    const QSet<QString>& runConfigKeys)
{
    QJsonObject run = request.value(QStringLiteral("run")).toObject();
    const QJsonObject requestedConfig =
        run.value(QStringLiteral("config")).toObject();

    // Compare effective values, not profile representation. Default is sent
    // sparsely, whereas Custom and saved profiles commonly contain the
    // service's complete expanded defaults.
    QJsonObject effectiveConfig = defaultAdvancedConfig;
    for (auto it = requestedConfig.begin(); it != requestedConfig.end(); ++it)
        effectiveConfig.insert(it.key(), it.value());
    for (const QString& key : runConfigKeys) effectiveConfig.remove(key);
    run[QStringLiteral("config")] = effectiveConfig;
    request[QStringLiteral("run")] = run;
    return request;
}

} // namespace vc3d
