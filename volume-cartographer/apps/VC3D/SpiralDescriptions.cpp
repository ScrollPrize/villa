#include "SpiralDescriptions.hpp"

#include <QFile>
#include <QFileInfo>
#include <QFileSystemWatcher>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QStringList>
#include <QtGlobal>

namespace {
QHash<QString, QString> textSection(const QJsonObject& document, const QString& name)
{
    QHash<QString, QString> result;
    const QJsonObject section = document.value(name).toObject();
    for (auto it = section.begin(); it != section.end(); ++it) {
        const QString text = it.value().toString().trimmed();
        if (!text.isEmpty()) result.insert(it.key(), text);
    }
    return result;
}
}

SpiralDescriptions::SpiralDescriptions(const QString& path, QObject* parent)
    : QObject(parent), _path(path), _watcher(new QFileSystemWatcher(this))
{
    // Editors commonly save by replacing the file, which drops it from the
    // watcher, so the directory is watched too and the file re-added on reload.
    connect(_watcher, &QFileSystemWatcher::fileChanged, this, &SpiralDescriptions::reload);
    connect(_watcher, &QFileSystemWatcher::directoryChanged, this, &SpiralDescriptions::reload);
    if (!_path.isEmpty()) {
        const QString directory = QFileInfo(_path).absolutePath();
        if (QFileInfo(directory).isDir()) _watcher->addPath(directory);
    }
    reload();
}

QString SpiralDescriptions::describe(const QString& id) const
{
    const auto panel = _panel.constFind(id);
    if (panel != _panel.cend()) return *panel;
    return _config.value(id);
}

void SpiralDescriptions::reload()
{
    QHash<QString, QString> panel;
    QHash<QString, QString> config;
    if (!_path.isEmpty() && QFileInfo(_path).isFile()) {
        if (!_watcher->files().contains(_path)) _watcher->addPath(_path);
        QFile file(_path);
        if (file.open(QIODevice::ReadOnly)) {
            QJsonParseError error;
            const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &error);
            if (error.error == QJsonParseError::NoError && document.isObject()) {
                panel = textSection(document.object(), QStringLiteral("panel"));
                config = textSection(document.object(), QStringLiteral("config"));
            } else {
                // Keep the last good text while a half-written edit is saved.
                qWarning("Spiral descriptions %s: %s", qPrintable(_path),
                         qPrintable(error.errorString()));
                return;
            }
        }
    }
    if (panel == _panel && config == _config) return;
    _panel = std::move(panel);
    _config = std::move(config);
    emit changed();
}

namespace vc3d::spiral {
QString descriptionToolTip(const QString& description, const QString& detail)
{
    QString result;
    for (const QString& text : {description, detail}) {
        const QString trimmed = text.trimmed();
        if (trimmed.isEmpty()) continue;
        result += QStringLiteral("<p>%1</p>").arg(
            trimmed.toHtmlEscaped().replace(QLatin1Char('\n'), QStringLiteral("<br>")));
    }
    return result;
}
}
