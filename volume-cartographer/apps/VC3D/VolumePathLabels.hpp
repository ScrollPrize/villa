#pragma once

#include <QStringList>
#include <QUrl>

namespace vc3d {

inline QString volumePathForDisplay(QString path)
{
    if (path.contains(QStringLiteral("://"))) {
        QUrl url(path);
        url.setPassword({});
        url.setQuery(QString());
        return url.toString();
    }
    return path.replace('\\', '/');
}

// Expand colliding filenames together, stopping at the first unique suffix.
// Identical locations stay identical; the existing volume IDs distinguish them.
inline QStringList distinguishingVolumePaths(const QStringList& locations)
{
    QList<QStringList> parts;
    for (const auto& location : locations)
        parts.push_back(volumePathForDisplay(location).split('/', Qt::SkipEmptyParts));
    QStringList result;
    for (qsizetype i = 0; i < parts.size(); ++i) {
        QString suffix;
        for (qsizetype depth = 1; depth <= parts[i].size(); ++depth) {
            suffix = parts[i].mid(parts[i].size() - depth).join('/');
            bool collision = false;
            for (qsizetype j = 0; j < parts.size(); ++j) {
                if (i == j || parts[i] == parts[j]) continue;
                if (parts[j].size() >= depth &&
                    suffix == parts[j].mid(parts[j].size() - depth).join('/')) {
                    collision = true;
                    break;
                }
            }
            if (!collision) break;
        }
        result.push_back(suffix);
    }
    return result;
}


} // namespace vc3d
