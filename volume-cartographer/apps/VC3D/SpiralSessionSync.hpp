#pragma once

#include <QJsonObject>
#include <QSet>

namespace vc3d {

inline bool spiralCheckpointLoadAvailable(
    bool connected, const QString& sessionState, bool checkpointSelected)
{
    return connected && checkpointSelected
        && (sessionState == QStringLiteral("Uninitialized")
            || sessionState == QStringLiteral("Idle")
            || sessionState == QStringLiteral("Error"));
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

inline QJsonObject completeSpiralRunConfiguration(
    const QJsonObject& defaults,
    const QJsonObject& appliedConfig,
    const QJsonObject& runConfig)
{
    QJsonObject result = defaults;
    auto applyKnown = [&result](const QJsonObject& values) {
        for (auto it = values.begin(); it != values.end(); ++it)
            if (result.contains(it.key()))
                result.insert(it.key(), it.value());
    };
    // The catalog defaults define the exact /session/run schema. Resident
    // status also carries run-block values such as z_begin/z_end; those must
    // not leak back into the advanced configuration payload.
    applyKnown(appliedConfig);
    applyKnown(runConfig);
    return result;
}

inline QJsonObject spiralRunRequest(
    const QJsonObject& configuration,
    int iterations,
    const QJsonObject& influence,
    const QJsonObject& dtLossSchedule,
    qint64 expectedSessionRevision,
    const QJsonObject& previewSchedule = {})
{
    QJsonObject result{
        {QStringLiteral("configuration"), configuration},
        {QStringLiteral("iterations"), iterations},
        {QStringLiteral("influence"), influence},
        {QStringLiteral("dt_loss_schedule"), dtLossSchedule},
        {QStringLiteral("expected_session_revision"), expectedSessionRevision},
    };
    if (!previewSchedule.isEmpty())
        result[QStringLiteral("preview_schedule")] = previewSchedule;
    return result;
}

} // namespace vc3d
