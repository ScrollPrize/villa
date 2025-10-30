#pragma once

#include <QDir>
#include <QString>

namespace vc3d {

inline QString settingsFilePath()
{
    const QString homeDir = QDir::homePath();
    const QString configDir = homeDir + "/.VC3D";
    QDir dir;
    if (!dir.exists(configDir)) {
        dir.mkpath(configDir);
    }
    return configDir + "/VC3D.ini";
}

} // namespace vc3d
