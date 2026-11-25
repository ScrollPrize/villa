#include "PythonExecutable.hpp"

#include <QDir>
#include <QFileInfo>
#include <QStandardPaths>

namespace vc3d {

namespace {

bool isExecutableFile(const QString& path)
{
    const QFileInfo info(path);
    return info.isFile() && info.isExecutable();
}

}  // namespace

QString installedPythonToolsRoot()
{
    const QString configured = qEnvironmentVariable("VC3D_PYTHON_TOOLS");
    if (!configured.isEmpty()) {
        return QDir::cleanPath(configured);
    }
    const QString appData = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    return appData.isEmpty() ? QString() : QDir(appData).filePath(QStringLiteral("python-tools"));
}

QString installedPythonEnvironment(const QString& name)
{
    const QString root = installedPythonToolsRoot();
    if (root.isEmpty()) {
        return {};
    }
    const QDir environment(QDir(root).filePath(QStringLiteral("envs/") + name));
    // the installer writes this marker last, so a half-installed or
    // half-upgraded environment is never used
    if (!QFileInfo::exists(environment.filePath(QStringLiteral(".vc3d-installed")))) {
        return {};
    }
    // the venv's own interpreter, not the one it links to, so its
    // site-packages stay in effect
#ifdef Q_OS_WIN
    const QString python = environment.filePath(QStringLiteral("Scripts/python.exe"));
#else
    const QString python = environment.filePath(QStringLiteral("bin/python"));
#endif
    return isExecutableFile(python) ? python : QString();
}

QString installedPythonToolsSource(const QString& relative)
{
    const QString root = installedPythonToolsRoot();
    if (root.isEmpty()) {
        return {};
    }
    const QString path = QDir(root).filePath(QStringLiteral("villa/") + relative);
    return QFileInfo::exists(path) ? path : QString();
}

QString pythonExecutableForServices(const QString& preferred)
{
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
            if (isExecutableFile(candidate)) {
                return candidate;
            }
        }
    }
    // the neural tracer and fibre follow run from the vesuvius package
    const QString installed = installedPythonEnvironment(QStringLiteral("vesuvius"));
    if (!installed.isEmpty()) {
        return installed;
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
