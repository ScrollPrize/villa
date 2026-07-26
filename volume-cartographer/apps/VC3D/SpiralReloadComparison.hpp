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

    // The service expands preview defaults in its canonical session request,
    // while older panel requests only sent first_winding. Treat both wire
    // representations as the same loaded preview configuration.
    QJsonObject preview =
        request.value(QStringLiteral("preview")).toObject();
    if (!preview.contains(QStringLiteral("first_winding")))
        preview[QStringLiteral("first_winding")] = 10;
    if (!preview.contains(QStringLiteral("variant")))
        preview[QStringLiteral("variant")] = QStringLiteral("raw");
    request[QStringLiteral("preview")] = preview;
    return request;
}

} // namespace vc3d
