#pragma once

#include <QHash>
#include <QObject>
#include <QString>

class QFileSystemWatcher;

// Hover text for the Spiral tab, read from spiral-fitting/descriptions.json so
// the wording can change without rebuilding VC3D. The "panel" section is keyed
// by Spiral panel control ids, the "config" section by configuration keys (the
// service serves the same section as catalog field descriptions). The file is
// watched; edits are picked up while VC3D runs and announced by changed().
class SpiralDescriptions final : public QObject
{
    Q_OBJECT
public:
    explicit SpiralDescriptions(const QString& path, QObject* parent = nullptr);

    QString path() const { return _path; }
    // A panel control's description, or the configuration key's description
    // when the panel section has no entry for id. Empty when neither has one.
    QString describe(const QString& id) const;

signals:
    void changed();

private:
    void reload();

    QString _path;
    QFileSystemWatcher* _watcher = nullptr;
    QHash<QString, QString> _panel;
    QHash<QString, QString> _config;
};

namespace vc3d::spiral {
// Rich-text tooltip of a description followed by an optional detail paragraph
// (a state note or a configuration key). Rich text lets Qt word-wrap it.
// Empty when both are empty.
QString descriptionToolTip(const QString& description, const QString& detail = {});
}
