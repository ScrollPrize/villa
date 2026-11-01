#pragma once

#include "VCSettings.hpp"

#include <QDir>
#include <QSettings>
#include <QStandardPaths>
#include <QString>

namespace vc3d {

// Where VC3D keeps the projects it makes for the user (Open Data samples' among them).
inline QString projectsDirectory()
{
    const auto directory = QDir(QStandardPaths::writableLocation(QStandardPaths::AppDataLocation))
                               .filePath(QStringLiteral("projects"));
    QDir().mkpath(directory);
    return directory;
}

inline QString defaultNewProjectDirectory(QSettings& settings)
{
    const auto chosen = settings.value(vc3d::settings::project::DEFAULT_PATH).toString();
    return chosen.isEmpty() ? projectsDirectory() : chosen;
}

} // namespace vc3d
