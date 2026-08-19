#pragma once

#include <QJsonObject>
#include <QSet>

namespace vc3d {

inline bool spiralCheckpointLoadAvailable(
    bool connected, const QString& sessionState, bool checkpointSelected)
{
    return connected && checkpointSelected
        && (sessionState == QStringLiteral("Uninitialized")
            || sessionState == QStringLiteral("Idle"));
}

inline QJsonObject spiralSessionRequestWithCheckpoint(
    QJsonObject request, const QString& checkpoint)
{
    QJsonObject paths =
        request.value(QStringLiteral("paths")).toObject();
    paths[QStringLiteral("checkpoint")] = checkpoint;
    request[QStringLiteral("paths")] = paths;
    return request;
}

inline QJsonObject spiralCheckpointInitializationRequest(
    QJsonObject request, const QString& checkpoint)
{
    request = spiralSessionRequestWithCheckpoint(request, checkpoint);

    // The checkpoint is the durable configuration source. In particular, a
    // profile that happened to be selected before Load must not be layered
    // onto it while constructing the model.
    QJsonObject run = request.value(QStringLiteral("run")).toObject();
    run[QStringLiteral("config")] = QJsonObject{};
    request[QStringLiteral("run")] = run;
    return request;
}

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
    // The z window is represented by run.z_begin/run.z_end and edited in the
    // panel dock. Older profiles and resolved checkpoint configurations may
    // still contain durable copies, but they are not advanced JSON settings.
    effective.remove(QStringLiteral("z_begin"));
    effective.remove(QStringLiteral("z_end"));
    return effective;
}

inline QJsonObject spiralRunBoundaryConfig(
    const QJsonObject& config,
    const QSet<QString>& runConfigKeys)
{
    QJsonObject result;
    for (auto it = config.begin(); it != config.end(); ++it)
        if (runConfigKeys.contains(it.key()))
            result.insert(it.key(), it.value());
    return result;
}

} // namespace vc3d
