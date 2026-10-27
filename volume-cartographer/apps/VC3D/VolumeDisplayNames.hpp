#pragma once

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Volume.hpp"
#include "VolumePathLabels.hpp"
#include <QString>
#include <QHash>
#include <algorithm>
#include <string>
#include <vector>

namespace vc3d {

// A Lasagna presence channel, a catalog surface prediction, or a catalog scan
// (at any level); empty for other volumes.
inline QString volumeDisplayAlias(const VolumePkg& pkg, const std::string& id)
{
    const auto tags = pkg.volumeTags(id);
    if (std::find(tags.begin(), tags.end(), "vc-lasagna-group:presence") != tags.end()) return QStringLiteral("fiber");
    const auto view = pkg.catalogView(id);
    if (!view) return {};
    if (view->source) return QStringLiteral("scan");
    if (view->artifact->type == "surface-prediction-zarr") return QStringLiteral("surf");
    return {};
}

inline QString volumeDisplayPath(const VolumePkg& pkg, const std::string& id)
{
    const auto volume = pkg.volume(id);
    return volumePathForDisplay(QString::fromStdString(volume->isRemote()
        ? volume->remoteLocator() : volume->path().string()));
}

// One snapshot per dropdown refresh, shared by all labels in that refresh.
inline QHash<QString, QString> volumeDisplaySuffixes(const VolumePkg& pkg)
{
    const auto ids = pkg.volumeIDs();
    QStringList paths;
    for (const auto& candidate : ids) {
        paths.push_back(volumeDisplayPath(pkg, candidate));
    }
    const auto suffixes = distinguishingVolumePaths(paths);
    QHash<QString, QString> result;
    qsizetype index = 0;
    for (const auto& id : ids)
        result.insert(QString::fromStdString(id), suffixes[index++]);
    return result;
}

inline QString volumeDisplayLabel(const VolumePkg& pkg, const std::string& id,
                                  const QString& original, const QString& suffix)
{
    QString label = original;
    if (suffix.contains('/'))
        label += QStringLiteral(" [%1]").arg(suffix);
    const auto alias = volumeDisplayAlias(pkg, id);
    return alias.isEmpty() ? label : alias + QStringLiteral(" - ") + label;
}

inline auto orderedDisplayVolumeIds(const VolumePkg& pkg)
{
    auto ids = pkg.volumeIDs();
    std::stable_sort(ids.begin(), ids.end(), [&](const auto& a, const auto& b) {
        return !volumeDisplayAlias(pkg, a).isEmpty() && volumeDisplayAlias(pkg, b).isEmpty();
    });
    return ids;
}

} // namespace vc3d
