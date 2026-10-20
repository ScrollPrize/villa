#pragma once

#include <QComboBox>
#include <QEvent>
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

class VolumePathTooltipFilter final : public QObject
{
public:
    explicit VolumePathTooltipFilter(QComboBox* combo) : QObject(combo) {}
protected:
    bool eventFilter(QObject* watched, QEvent* event) override
    {
        if (event->type() == QEvent::ToolTip) {
            auto* combo = static_cast<QComboBox*>(watched);
            combo->setToolTip(combo->currentData(Qt::ToolTipRole).toString());
        }
        return false;
    }
};

inline void refreshVolumeComboTooltip(QComboBox* combo)
{
    if (!combo->property("vcVolumePathTooltip").toBool()) {
        combo->setProperty("vcVolumePathTooltip", true);
        combo->installEventFilter(new VolumePathTooltipFilter(combo));
        QObject::connect(combo, &QComboBox::currentIndexChanged, combo, [combo] {
            combo->setToolTip(combo->currentData(Qt::ToolTipRole).toString());
        });
    }
    // Population often blocks signals, so initialize explicitly as well.
    combo->setToolTip(combo->currentData(Qt::ToolTipRole).toString());
}

} // namespace vc3d
