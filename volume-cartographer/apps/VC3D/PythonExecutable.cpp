#include "PythonExecutable.hpp"

#include <QDir>
#include <QFileInfo>
#include <QStandardPaths>

#include <cstdlib>
#include <cstring>

namespace vc3d {

namespace {

// Launch values recorded by rememberLaunchEnvironmentValue(). Plain
// zero-initialized storage, so recording works before any static
// constructor has run; written only while VC3D starts, before other threads
struct LaunchValue {
    const char* name;
    char* value;  // null when the variable was unset
};
constexpr std::size_t kMaxLaunchValues = 16;
LaunchValue launchValues[kMaxLaunchValues];
std::size_t launchValueCount = 0;

bool isExecutableFile(const QString& path)
{
    const QFileInfo info(path);
    return info.isFile() && info.isExecutable();
}

// the python of the active virtualenv, else of the active conda env
QString activeEnvironmentPython()
{
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
    return {};
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
    const QString active = activeEnvironmentPython();
    if (!active.isEmpty()) {
        return active;
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

QString chosenPythonForServices(const QString& preferred)
{
    const QString trimmed = preferred.trimmed();
    if (trimmed.isEmpty()) {
        return activeEnvironmentPython();
    }
    const QString path = QDir::cleanPath(QFileInfo(trimmed).absoluteFilePath());
    return isExecutableFile(path) ? path : QString();
}

void rememberLaunchEnvironmentValue(const char* name)
{
    for (std::size_t i = 0; i < launchValueCount; ++i) {
        if (std::strcmp(launchValues[i].name, name) == 0) {
            return;
        }
    }
    if (launchValueCount == kMaxLaunchValues) {
        std::abort();  // a fixed list of startup overrides outgrew the table
    }
    char* copy = nullptr;
    if (const char* value = std::getenv(name)) {
        const std::size_t size = std::strlen(value) + 1;
        copy = static_cast<char*>(std::malloc(size));
        if (copy) {
            std::memcpy(copy, value, size);
        }
    }
    launchValues[launchValueCount++] = {name, copy};
}

QProcessEnvironment pythonServiceEnvironment()
{
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    for (std::size_t i = 0; i < launchValueCount; ++i) {
        const QString name = QString::fromLatin1(launchValues[i].name);
        if (launchValues[i].value) {
            environment.insert(name, QString::fromLocal8Bit(launchValues[i].value));
        } else {
            environment.remove(name);
        }
    }
    return environment;
}

}  // namespace vc3d
