#pragma once

#include <QJsonObject>
#include <QSet>
#include <QString>

namespace vc3d {

inline QJsonObject normalizedSpiralReloadRequest(
    QJsonObject request,
    const QJsonObject& defaultAdvancedConfig,
    const QSet<QString>& runConfigKeys,
    const QSet<QString>& runMutablePaths = {})
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

    QJsonObject paths = request.value(QStringLiteral("paths")).toObject();
    for (const QString& key : runMutablePaths) paths.remove(key);
    request[QStringLiteral("paths")] = paths;

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

// Which build stage a rebuild from `current` would need, mirroring
// ServiceState._rebuild_stage_locked: everything outside run.config forces
// "all", and within run.config the answer is "model" only when every key
// whose requested value differs is on the service-advertised allowlist
// (schema.model_stage_keys). Returns "model" or "all".
//
// Deliberately compares the requests as sent, not their effective values,
// because that is what the service diffs; an advisory answer that disagreed
// with the service could tell the user their uncommitted inputs survive a
// rebuild that in fact discards them.
inline QString spiralRebuildStage(const QJsonObject& current,
                                  const QJsonObject& loaded,
                                  const QSet<QString>& modelStageKeys)
{
    // Empty defaults and no run-mutable keys leave run.config exactly as each
    // request carries it; the call is here for the preview-default expansion,
    // which is a wire-representation difference rather than a real one.
    QJsonObject left = normalizedSpiralReloadRequest(current, {}, {});
    QJsonObject right = normalizedSpiralReloadRequest(loaded, {}, {});
    QJsonObject leftRun = left.value(QStringLiteral("run")).toObject();
    QJsonObject rightRun = right.value(QStringLiteral("run")).toObject();
    const QJsonObject leftConfig =
        leftRun.take(QStringLiteral("config")).toObject();
    const QJsonObject rightConfig =
        rightRun.take(QStringLiteral("config")).toObject();
    left[QStringLiteral("run")] = leftRun;
    right[QStringLiteral("run")] = rightRun;
    if (left != right) return QStringLiteral("all");

    QSet<QString> changed;
    for (const QString& key : leftConfig.keys())
        if (leftConfig.value(key) != rightConfig.value(key))
            changed.insert(key);
    for (const QString& key : rightConfig.keys())
        if (!leftConfig.contains(key)) changed.insert(key);
    for (const QString& key : changed)
        if (!modelStageKeys.contains(key)) return QStringLiteral("all");
    return QStringLiteral("model");
}

} // namespace vc3d
