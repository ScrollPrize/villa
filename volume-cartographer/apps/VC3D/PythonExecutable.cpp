#include "PythonExecutable.hpp"

#include <QDir>
#include <QFileInfo>
#include <QStandardPaths>

namespace vc3d {

QString pythonExecutableForServices(const QString& preferred)
{
    const auto executable = [](const QString& path) {
        const QFileInfo info(path);
        return info.exists() && info.isExecutable();
    };
    if (!preferred.trimmed().isEmpty()) {
        return QDir::cleanPath(QFileInfo(preferred.trimmed()).absoluteFilePath());
    }
    for (const char* variable : {"VIRTUAL_ENV", "CONDA_PREFIX"}) {
        const QString prefix = qEnvironmentVariable(variable);
        if (prefix.isEmpty()) {
            continue;
        }
        for (const char* name : {"bin/python", "bin/python3"}) {
            const QString candidate = QDir(prefix).filePath(QString::fromLatin1(name));
            if (executable(candidate)) {
                return candidate;
            }
        }
    }
    for (const char* name : {"python3", "python"}) {
        const QString found = QStandardPaths::findExecutable(QString::fromLatin1(name));
        if (!found.isEmpty()) {
            return found;
        }
    }
    return QStringLiteral("python3");
}

}  // namespace vc3d
