#include "ToolExecutable.hpp"

#include "VCSettings.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QProcessEnvironment>
#include <QSettings>
#include <QStandardPaths>

namespace vc3d {
namespace {

bool isExecutableFile(const QString& path)
{
    QFileInfo fi(path);
    return fi.exists() && fi.isFile() && fi.isExecutable();
}

QStringList applicationRelativeExecutablePaths(const QString& name)
{
#ifdef _WIN32
    const QString executableName = name.endsWith(QStringLiteral(".exe"), Qt::CaseInsensitive)
        ? name
        : name + QStringLiteral(".exe");
#else
    const QString executableName = name;
#endif

    const QString appDir = QCoreApplication::applicationDirPath();
    QStringList candidates{
        QDir(appDir).filePath(executableName),
        QDir(appDir).filePath(QStringLiteral("../") + executableName),
        QDir(appDir).filePath(QStringLiteral("../bin/") + executableName),
        QDir(appDir).filePath(QStringLiteral("../../bin/") + executableName),
        QDir(appDir).filePath(QStringLiteral("../libexec/") + executableName),
        QDir(appDir).filePath(QStringLiteral("../Resources/bin/") + executableName),
    };
    candidates.removeDuplicates();
    return candidates;
}

} // namespace

QString findToolExecutable(const QString& name, const QStringList& extraPaths, const QString& envVar)
{
    // 1. Environment variable override (if provided)
    if (!envVar.isEmpty()) {
        const QByteArray envVal = qgetenv(envVar.toUtf8().constData());
        if (!envVal.isEmpty()) {
            const QString envPath = QString::fromLocal8Bit(envVal);
            if (isExecutableFile(envPath))
                return QFileInfo(envPath).absoluteFilePath();
        }
    }

    // 2. INI settings (tools/<name>_path, tools/<name>)
    {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        const QString key1 = QStringLiteral("tools/%1_path").arg(name);
        const QString key2 = QStringLiteral("tools/%1").arg(name);
        const QString iniPath =
            settings.value(key1, settings.value(key2)).toString().trimmed();
        if (!iniPath.isEmpty() && isExecutableFile(iniPath)) {
            return QFileInfo(iniPath).absoluteFilePath();
        }
    }

    // 3. Extra hard-coded paths (caller-supplied)
    for (const QString& p : extraPaths) {
        if (isExecutableFile(p))
            return QFileInfo(p).absoluteFilePath();
    }

    // 4. Binaries colocated with the GUI app or common install layouts
    for (const QString& p : applicationRelativeExecutablePaths(name)) {
        if (isExecutableFile(p))
            return QFileInfo(p).absoluteFilePath();
    }

    // 5. QStandardPaths / manual PATH walk
#if QT_VERSION >= QT_VERSION_CHECK(5, 10, 0)
    const QString onPath = QStandardPaths::findExecutable(name);
    if (!onPath.isEmpty()) return onPath;
#else
    const QStringList pathDirs =
        QProcessEnvironment::systemEnvironment().value("PATH")
            .split(QDir::listSeparator(), Qt::SkipEmptyParts);
    for (const QString& dir : pathDirs) {
        const QString candidate = QDir(dir).filePath(name);
        QFileInfo fi(candidate);
        if (fi.exists() && fi.isFile() && fi.isExecutable())
            return fi.absoluteFilePath();
    }
#endif
    return {};
}

} // namespace vc3d
