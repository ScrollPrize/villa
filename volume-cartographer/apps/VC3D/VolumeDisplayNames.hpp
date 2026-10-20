#pragma once

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Volume.hpp"
#include "VolumePathLabels.hpp"
#include <QString>
#include <algorithm>
#include <string>
#include <vector>

namespace vc3d {

inline QString volumeDisplayAlias(const std::vector<std::string>& tags)
{
    const auto has = [&](const char* tag) {
        return std::find(tags.begin(), tags.end(), tag) != tags.end();
    };
    if (has("vc-lasagna-group:presence")) return QStringLiteral("fiber");
    if (has("surface-prediction")) return QStringLiteral("surf");
    if (has("vc-open-data-preferred-source") || has("vc-open-data-virtual-source"))
        return QStringLiteral("scan");
    return {};
}

inline QString volumeDisplayPath(const VolumePkg& pkg, const std::string& id)
{
    const auto volume = pkg.volume(id);
    return volumePathForDisplay(QString::fromStdString(volume->isRemote()
        ? volume->remoteLocator() : volume->path().string()));
}

inline QString volumeDisplayLabel(const VolumePkg& pkg, const std::string& id,
                                  const QString& original)
{
    const auto ids = pkg.volumeIDs();
    QStringList paths;
    qsizetype selected = -1;
    for (const auto& candidate : ids) {
        if (candidate == id) selected = paths.size();
        paths.push_back(volumeDisplayPath(pkg, candidate));
    }
    QString label = original;
    if (selected >= 0) {
        const auto suffix = distinguishingVolumePaths(paths)[selected];
        if (suffix.contains('/'))
            label += QStringLiteral(" [%1]").arg(suffix);
    }
    const auto alias = volumeDisplayAlias(pkg.volumeTags(id));
    return alias.isEmpty() ? label : alias + QStringLiteral(" - ") + label;
}

inline auto orderedDisplayVolumeIds(const VolumePkg& pkg)
{
    auto ids = pkg.volumeIDs();
    std::stable_sort(ids.begin(), ids.end(), [&](const auto& a, const auto& b) {
        return !volumeDisplayAlias(pkg.volumeTags(a)).isEmpty() &&
                volumeDisplayAlias(pkg.volumeTags(b)).isEmpty();
    });
    return ids;
}

} // namespace vc3d
