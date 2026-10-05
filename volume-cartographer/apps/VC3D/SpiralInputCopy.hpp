#pragma once

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QObject>
#include <QTemporaryDir>
#include <functional>
#include <memory>

namespace vc3d::spiral {
inline bool copyInput(const QString& source, const QString& destination, QString& error,
                      const std::function<bool(qint64)>& progress = {})
{
    if (progress && !progress(-1)) {
        error = QObject::tr("Input copy cancelled");
        return false;
    }
    const QFileInfo info(source);
    if (info.isSymLink()) {
        error = QObject::tr("Input working copies cannot contain symbolic links: %1").arg(source);
        return false;
    }
    if (info.isDir()) {
        if (!QDir().mkpath(destination)) {
            error = QObject::tr("Cannot create %1").arg(destination);
            return false;
        }
        const QDir directory(source);
        for (const auto& child : directory.entryInfoList(QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden))
            if (!copyInput(child.absoluteFilePath(), QDir(destination).filePath(child.fileName()), error, progress)) return false;
        return true;
    }
    QDir().mkpath(QFileInfo(destination).absolutePath());
    QFile input(source);
    if (!input.copy(destination)) {
        error = QObject::tr("Cannot copy %1 to %2: %3").arg(source, destination, input.errorString());
        return false;
    }
    if (progress) progress(info.size());
    return true;
}

struct InputCopyResult {
    std::shared_ptr<QTemporaryDir> directory;
    QString path;
    QString error;
};
}
